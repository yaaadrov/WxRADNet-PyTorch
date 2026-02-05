import os
import re
import numpy as np
import pandas as pd
import networkx as nx
import warnings
import pickle
from pathlib import Path
from typing import List, Tuple
from datetime import datetime, timedelta
from pandas.errors import SettingWithCopyWarning

import geopandas as gpd
import shapely
import rasterio
import rasterio.features
from rasterio.errors import RasterioIOError
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union
import affine
from concave_hull import concave_hull
from settings import settings


warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


ROOT_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_PATH / 'data'
TIMESTAMPS_PATH = ROOT_PATH / 'config' / 'timestamps.pkl'
NUM_ITERATIONS: float = 25


class Parser:
    def __init__(
            self,
            base_url: str,
            intensity_threshold_low: int,
            intensity_threshold_high: int,
            distance_between: int,
            distance_avoid: int,
            delta_minutes: int,
            data_path: Path,
    ):
        """
        Initialize `Parser` class
        Args:
            base_url (str): Base URL to get data from
            intensity_threshold_low (int): Threshold for minimum intensity value
            intensity_threshold_high (int): Threshold for maximum intensity value
            distance_between (int): Half distance between two thunderstorms to proceed between
            distance_avoid (int): Minimum buffer to avoid thunderstorm with
            delta_minutes (int): Delta minutes between two sequential thunderstorm images
            data_path (str): Path to data directory
        """
        self.base_url = base_url
        self.intensity_threshold_low = intensity_threshold_low
        self.intensity_threshold_high = intensity_threshold_high
        self.distance_between = distance_between
        self.distance_avoid = distance_avoid
        self.delta_minutes = delta_minutes
        self.data_path = data_path

    def _generate_url(self, current_date: datetime) -> str:
        """Generate URL with correct datetime values"""
        return self.base_url.format(
            year=current_date.year,
            month=f"{current_date.month:02d}",
            day=f"{current_date.day:02d}",
            hour=f"{current_date.hour:02d}",
            minute=f"{current_date.minute:02d}"
        )

    def _fetch_raster_data(self, url: str) -> tuple[np.ndarray, affine.Affine, rasterio.crs.CRS]:
        """Fetch data from geotiff image"""
        with rasterio.open(url) as dataset:
            band = dataset.read(1)
            transform = dataset.transform
            crs = dataset.crs
        return band, transform, crs

    def _apply_mask(self, band: np.ndarray) -> np.ndarray:
        """Apply mask based on intensity threshold"""
        return (band > self.intensity_threshold_low) & (band < self.intensity_threshold_high)

    def _convert_raster_to_polygons(
            self,
            mask: np.ndarray,
            transform: affine.Affine
    ) -> List[shapely.geometry.Polygon]:
        """Convert raster mask to list of shapely Polygons"""
        shapes = list(rasterio.features.shapes(mask.astype(np.uint8), transform=transform))
        return [shape(geom) for geom, val in shapes if val == 1]

    def _create_geodataframe(
            self,
            polygons: List[shapely.geometry.Polygon],
            crs: rasterio.crs.CRS,
    ) -> gpd.GeoDataFrame:
        """Create GeoDataFrame from list of shapely Polygons with given CRS"""
        return gpd.GeoDataFrame(geometry=polygons, crs=crs)

    def _calculate_buffers(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        gdf["buffer_between"] = gdf.geometry.buffer(self.distance_between)
        gdf["buffer_avoid"] = gdf.geometry.buffer(self.distance_avoid)
        return gdf

    def _find_overlaps(self, gdf_buffer: gpd.GeoDataFrame) -> List[Tuple]:
        """Find pairs of indices of overlapping polygons"""
        overlaps = gpd.sjoin(gdf_buffer, gdf_buffer, how="inner", predicate="intersects")
        overlaps = overlaps.reset_index().rename(columns={"index": "index_left"})
        overlaps = overlaps[overlaps["index_left"] != overlaps["index_right"]]
        return list(zip(overlaps["index_left"], overlaps["index_right"]))

    def _handle_unassigned_groups(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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

    def _concave_from_multipolygon(
            self,
            multipolygon: shapely.geometry.Polygon | shapely.geometry.MultiPolygon
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

    def _save_geodataframe_to_csv(self, gdf: gpd.GeoDataFrame, file_path: Path) -> None:
        """Save a GeoDataFrame to a CSV file"""
        for col in gdf.columns:
            gdf[col] = gdf[col].apply(lambda geom: geom.wkt if geom is not None else None)
        gdf.to_csv(file_path, index=False)

    def _load_geodataframe_from_csv(self, file_path: Path) -> gpd.GeoDataFrame:
        df = pd.read_csv(file_path)
        for col in df.columns:
            df[col] = gpd.GeoSeries.from_wkt(df[col])
        gdf = gpd.GeoDataFrame(df, geometry='geometry')
        return gdf

    def prepare_data(self, current_date: datetime) -> gpd.GeoDataFrame:
        """Get gpd.GeoDataFrame for required timestamp"""
        url = self._generate_url(current_date)
        band, transform, crs = self._fetch_raster_data(url)
        mask = self._apply_mask(band)
        polygons = self._convert_raster_to_polygons(mask, transform)
        gdf = self._create_geodataframe(polygons, crs)
        gdf = self._calculate_buffers(gdf)
        gdf = self._assign_groups(gdf)
        return self._union_polygons(gdf, crs)

    def process_dates(self, timestamps: list[datetime], iterations: int):
        """Save GeoDataFrames for given timestamps"""
        for j, current_date in enumerate(timestamps):
            directory_path = Path()
            for i in range(iterations):
                file_name = '_'.join(re.split('[- :]', str(current_date)))
                length = len(str(iterations))

                # Create corresponding directory if necessary
                if i == 0:
                    directory_path = DATA_PATH / file_name
                    if directory_path.exists():
                        print(f'\n({j + 1}/{len(timestamps)}) Directory {directory_path} exists. Skipping {current_date}')
                        break
                    os.makedirs(directory_path, exist_ok=False)
                    print(f'\n({j + 1}/{len(timestamps)}) Directory {directory_path} created')

                try:
                    gdf_union = self.prepare_data(current_date)
                    self._save_geodataframe_to_csv(gdf_union, directory_path / f'{file_name}.csv')
                    print(f'{i + 1:<{length}}/{iterations}: {current_date} ready')
                except RasterioIOError:
                    previous_file_name = sorted(os.listdir(directory_path))[-1]
                    gdf_union = self._load_geodataframe_from_csv(directory_path / previous_file_name)
                    self._save_geodataframe_to_csv(gdf_union, directory_path / f'{file_name}.csv')
                    print(f'{i + 1:<{length}}/{iterations}: {current_date} not available; used previous file instead')

                current_date += timedelta(minutes=self.delta_minutes)


if __name__ == '__main__':
    parser = Parser(
        base_url=settings.base_url,
        intensity_threshold_low=settings.intensity_threshold_low,
        intensity_threshold_high=settings.intensity_threshold_high,
        distance_between=settings.distance_between,
        distance_avoid=settings.distance_avoid,
        delta_minutes=settings.delta_minutes,
        data_path=DATA_PATH,
    )
    with open(TIMESTAMPS_PATH, 'rb') as file_in:
        timestamps = pickle.load(file_in)
    parser.process_dates(timestamps, iterations=NUM_ITERATIONS)
