from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import affine
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import shapely
from concave_hull import concave_hull
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union

from thund_avoider.schemas.preprocessor import RasterioCompressionSchema


class Preprocessor:
    def __init__(
        self,
        base_url: str,
        intensity_threshold_low: int,
        intensity_threshold_high: int,
        distance_between: int,
        distance_avoid: int,
    ):
        """
        Initialize `Preprocessor` class

        Args:
            base_url (str): Base URL to get data from
            intensity_threshold_low (int): Threshold for minimum intensity value
            intensity_threshold_high (int): Threshold for maximum intensity value
            distance_between (int): Half distance between two thunderstorms to proceed between
            distance_avoid (int): Minimum buffer to avoid thunderstorm with
        """
        self.base_url = base_url
        self.intensity_threshold_low = intensity_threshold_low
        self.intensity_threshold_high = intensity_threshold_high
        self.distance_between = distance_between
        self.distance_avoid = distance_avoid

    def _generate_url(self, current_date: datetime) -> str:
        """Generate URL with correct datetime values"""
        return self.base_url.format(
            year=current_date.year,
            month=f"{current_date.month:02d}",
            day=f"{current_date.day:02d}",
            hour=f"{current_date.hour:02d}",
            minute=f"{current_date.minute:02d}"
        )

    @staticmethod
    def _fetch_raster_data(url: str | Path) -> tuple[np.ndarray, affine.Affine, rasterio.crs.CRS]:
        """Fetch data from geotiff image"""
        with rasterio.open(url) as dataset:
            band = dataset.read(1)
            transform = dataset.transform
            crs = dataset.crs
        return band, transform, crs

    def _apply_mask(self, band: np.ndarray) -> np.ndarray:
        """Apply mask based on intensity threshold"""
        return (band > self.intensity_threshold_low) & (band < self.intensity_threshold_high)

    @staticmethod
    def _convert_raster_to_polygons(
        mask: np.ndarray,
        transform: affine.Affine,
    ) -> List[shapely.geometry.Polygon]:
        """Convert raster mask to list of shapely Polygons"""
        shapes = list(rasterio.features.shapes(mask.astype(np.uint8), transform=transform))
        return [shape(geom) for geom, val in shapes if val == 1]

    @staticmethod
    def _create_geodataframe(
        polygons: List[shapely.geometry.Polygon],
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
    def _find_overlaps(gdf_buffer: gpd.GeoDataFrame) -> List[Tuple]:
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
        multipolygon: shapely.geometry.Polygon | shapely.geometry.MultiPolygon,
    ) -> shapely.geometry.Polygon:
        """Create a concave hull from a multipolygon"""
        if isinstance(multipolygon, Polygon):
            return multipolygon
        elif isinstance(multipolygon, MultiPolygon):
            points = np.vstack(
                [np.array(line.coords) for line in multipolygon.boundary.geoms]
            )
            return Polygon(concave_hull(points, concavity=1.0))
        raise ValueError('Shapely Polygon or MultiPolygon should be passed as an argument')

    def _union_polygons(self, gdf: gpd.GeoDataFrame, crs: rasterio.crs.CRS) -> gpd.GeoDataFrame:
        """Union shapely Polygons within groups and create convex and concave hulls"""
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
        gdf_union["convex"] = gdf_union["buffer"].convex_hull
        gdf_union["concave"] = gdf_union["buffer"].apply(self._concave_from_multipolygon)
        return gdf_union

    @staticmethod
    def save_raster_from_url(url: str, output_path: Path):
        """Compress and save raster GeoTIFF file from URL"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(url) as src:
            profile = src.profile
            profile.update(RasterioCompressionSchema().model_dump())
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(src.read())

    @staticmethod
    def save_geodataframe_to_csv(gdf: gpd.GeoDataFrame, file_path: Path) -> None:
        """Save GeoDataFrame to a CSV file"""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        for col in gdf.columns:
            gdf[col] = gdf[col].apply(lambda geom: geom.wkt if geom is not None else None)
        gdf.to_csv(file_path, index=False)

    @staticmethod
    def load_geodataframe_from_csv(file_path: Path) -> gpd.GeoDataFrame:
        """Load GeoDataFrame from a CSV file"""
        df = pd.read_csv(file_path)
        for col in df.columns:
            df[col] = gpd.GeoSeries.from_wkt(df[col])
        gdf = gpd.GeoDataFrame(df, geometry="geometry")
        return gdf

    def get_gpd_for_current_date(self, current_date: datetime) -> gpd.GeoDataFrame:
        """Get gpd.GeoDataFrame for required timestamp"""
        url = self._generate_url(current_date)
        band, transform, crs = self._fetch_raster_data(url)
        mask = self._apply_mask(band)
        polygons = self._convert_raster_to_polygons(mask, transform)
        gdf = self._create_geodataframe(polygons, crs)
        gdf = self._calculate_buffers(gdf)
        gdf = self._assign_groups(gdf)
        return self._union_polygons(gdf, crs)
