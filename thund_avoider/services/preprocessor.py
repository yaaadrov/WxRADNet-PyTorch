from datetime import datetime
from pathlib import Path
from typing import Literal

import affine
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from concave_hull import concave_hull
from rasterio import Env
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union

from thund_avoider.schemas.preprocessor import RasterioCompressionSchema
from thund_avoider.services.dynamic_avoider.data_loader import DataLoader
from thund_avoider.settings import PreprocessorConfig


class Preprocessor:
    def __init__(self, config: PreprocessorConfig):
        """
        Initialize `Preprocessor` class

        Args:
            config (PreprocessorConfig): Preprocessor configuration:
                - base_url (str): Base URL to get data from
                - intensity_threshold_low (int): Threshold for minimum intensity value
                - intensity_threshold_high (int): Threshold for maximum intensity value
                - distance_between (int): Half distance between two thunderstorms to proceed between
                - distance_avoid (int): Minimum buffer to avoid thunderstorm with
        """
        self.base_url = config.base_url
        self.intensity_threshold_low = config.intensity_threshold_low
        self.intensity_threshold_high = config.intensity_threshold_high
        self.distance_between = config.distance_between
        self.distance_avoid = config.distance_avoid

    def generate_url(self, current_date: datetime) -> str:
        """Generate URL with correct datetime values"""
        return self.base_url.format(
            year=current_date.year,
            month=f"{current_date.month:02d}",
            day=f"{current_date.day:02d}",
            hour=f"{current_date.hour:02d}",
            minute=f"{current_date.minute:02d}"
        )

    def _apply_mask(self, band: np.ndarray) -> np.ndarray:
        """Apply mask based on intensity threshold"""
        return (band > self.intensity_threshold_low) & (band < self.intensity_threshold_high)

    @staticmethod
    def _convert_raster_to_polygons(
        mask: np.ndarray,
        transform: affine.Affine,
    ) -> list[Polygon]:
        """Convert raster mask to list of shapely Polygons"""
        shapes = list(rasterio.features.shapes(mask.astype(np.uint8), transform=transform))
        return [shape(geom) for geom, val in shapes if val == 1]

    @staticmethod
    def _create_geodataframe(
        polygons: list[Polygon],
        crs: rasterio.crs.CRS,
    ) -> gpd.GeoDataFrame:
        """Create GeoDataFrame from list of shapely Polygons with given CRS"""
        return gpd.GeoDataFrame(geometry=polygons, crs=crs)

    def _calculate_buffers(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Calculate buffers of `distance_between` and `distance_avoid` size"""
        gdf["buffer_between"] = gdf.geometry.buffer(self.distance_between)
        gdf["buffer_avoid"] = gdf.geometry.buffer(self.distance_avoid)
        return gdf

    @staticmethod
    def _find_overlaps(gdf_buffer: gpd.GeoDataFrame) -> list[tuple]:
        """Find pairs of indices of overlapping polygons"""
        overlaps = gpd.sjoin(gdf_buffer, gdf_buffer, how="inner", predicate="intersects")
        overlaps = overlaps.reset_index().rename(columns={"index": "index_left"})
        overlaps = overlaps[overlaps["index_left"] != overlaps["index_right"]]
        return list(zip(overlaps["index_left"], overlaps["index_right"]))

    @staticmethod
    def _handle_unassigned_groups(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Handle non-overlapping shapely Polygons without groups"""
        unassigned = gdf["group"].isna().sum()
        unique_negatives = -np.arange(1, unassigned + 1)
        gdf["group"] = gdf["group"].fillna(
            pd.Series(unique_negatives, index=gdf[gdf["group"].isna()].index)
        )
        return gdf

    def _assign_groups(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Assign groups to overlapping shapely Polygons"""
        gdf_buffer = gpd.GeoDataFrame(geometry=gdf["buffer_between"], crs=gdf.crs)
        overlaps = self._find_overlaps(gdf_buffer)
        G = nx.Graph()
        G.add_edges_from(overlaps)

        connected_components = list(nx.connected_components(G))
        index_to_group = {
            index: group_id
            for group_id, component in enumerate(connected_components)
            for index in component
        }

        gdf["group"] = gdf.index.map(index_to_group)
        gdf = self._handle_unassigned_groups(gdf)
        return gdf

    @staticmethod
    def _concave_from_multipolygon(
        multipolygon: Polygon | MultiPolygon,
    ) -> Polygon:
        """Create a concave hull from a multipolygon"""
        if isinstance(multipolygon, Polygon):
            return multipolygon
        elif isinstance(multipolygon, MultiPolygon):
            points = np.vstack(
                [np.array(line.coords) for line in multipolygon.boundary.geoms]
            )
            return Polygon(concave_hull(points, concavity=1.0))
        raise ValueError('Shapely Polygon or MultiPolygon should be passed as an argument')

    def _union_polygons(
        self,
        gdf: gpd.GeoDataFrame,
        crs: rasterio.crs.CRS,
        strategy: Literal["concave", "convex", "both"] = "both",
    ) -> gpd.GeoDataFrame:
        """
        Union polygons within groups and create hulls for specified strategy.

        Args:
            gdf: GeoDataFrame with grouped polygons.
            crs: Coordinate reference system.
            strategy: Hull strategy - "concave", "convex", or "both".

        Returns:
            gpd.GeoDataFrame: Unioned polygons with hull geometry column(s).
        """
        polygons_union = [
            unary_union(gdf[gdf["group"] == group].geometry)
            for group in gdf["group"].unique()
        ]
        polygons_union_buffer = [
            unary_union(gdf.loc[gdf["group"] == group, "buffer_avoid"])
            for group in gdf["group"].unique()
        ]
        gdf_union = gpd.GeoDataFrame(geometry=polygons_union, crs=crs)
        gdf_union["buffer"] = gpd.GeoDataFrame(geometry=polygons_union_buffer, crs=crs)

        match strategy:
            case "concave":
                gdf_union["concave"] = gdf_union["buffer"].apply(self._concave_from_multipolygon)
            case "convex":
                gdf_union["convex"] = gdf_union["buffer"].convex_hull
            case "both":
                gdf_union["convex"] = gdf_union["buffer"].convex_hull
                gdf_union["concave"] = gdf_union["buffer"].apply(self._concave_from_multipolygon)

        return gdf_union

    def prediction_to_polygons(
        self,
        prediction: np.ndarray,
        transform: affine.Affine,
        crs: rasterio.crs.CRS,
        strategy: Literal["concave", "convex"],
    ) -> list[Polygon]:
        """
        Convert a prediction raster frame to polygon obstacles.

        Args:
            prediction: Single prediction frame (2D or 3D with channel first).
            transform: Affine transform for coordinate mapping.
            crs: Coordinate reference system.
            strategy: Hull strategy ("concave" or "convex").

        Returns:
            list[Polygon]: List of obstacle polygons.
        """
        if prediction.ndim == 3:
            prediction = prediction[0]
        mask = self._apply_mask(prediction)  # (prediction > intensity_threshold_low) & (prediction < intensity_threshold_high)
        polygons = self._convert_raster_to_polygons(mask, transform)
        if not polygons:
            return []
        gdf = self._create_geodataframe(polygons, crs)
        gdf = self._calculate_buffers(gdf)
        gdf = self._assign_groups(gdf)
        gdf_union = self._union_polygons(gdf, crs, strategy=strategy)
        return gdf_union[strategy].tolist()

    def save_raster_from_url(self, current_date: datetime, output_path: Path):
        """Compress and save raster GeoTIFF file from URL"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        url = self.generate_url(current_date)
        with rasterio.open(url) as src:
            profile = src.profile
            profile.update(RasterioCompressionSchema().model_dump())
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(src.read())

    @staticmethod
    def get_res_raster(url: str | Path) -> int:
        """Get raster image resolution in meters/pixel."""
        with rasterio.open(url) as dataset:
            res = dataset.res[0]
        return res

    @staticmethod
    def load_raster_data(url: str | Path) -> tuple[np.ndarray, affine.Affine, rasterio.crs.CRS]:
        """Fetch data from geotiff image"""
        with rasterio.open(url) as dataset:
            band = dataset.read(1)
            transform = dataset.transform
            crs = dataset.crs
        return band, transform, crs

    @staticmethod
    def save_geodataframe_to_csv(gdf: gpd.GeoDataFrame, file_path: Path) -> None:
        """Save GeoDataFrame to a CSV file"""
        DataLoader.save_geodataframe_to_csv(gdf, file_path)

    @staticmethod
    def load_geodataframe_from_csv(file_path: Path) -> gpd.GeoDataFrame:
        """Load GeoDataFrame from a CSV file"""
        return DataLoader.load_geodataframe_from_csv(file_path)

    def get_gdf_for_current_date(self, current_date: datetime) -> gpd.GeoDataFrame:
        """Get gpd.GeoDataFrame for required timestamp"""
        url = self.generate_url(current_date)
        band, transform, crs = self.load_raster_data(url)
        mask = self._apply_mask(band)
        polygons = self._convert_raster_to_polygons(mask, transform)
        gdf = self._create_geodataframe(polygons, crs)
        gdf = self._calculate_buffers(gdf)
        gdf = self._assign_groups(gdf)
        return self._union_polygons(gdf, crs)
