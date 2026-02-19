import os
from pathlib import Path

import geopandas as gpd
import pandas as pd

from thund_avoider.settings import DataLoaderConfig


class DataLoader:
    """Handles loading and saving of obstacle data from CSV files."""

    def __init__(self, config: DataLoaderConfig) -> None:
        """Initialize DataLoader with configuration."""
        self._config = config

    @staticmethod
    def load_geodataframe_from_csv(file_path: Path) -> gpd.GeoDataFrame:
        """Load GeoDataFrame from a CSV file."""
        df = pd.read_csv(file_path)
        for col in df.columns:
            df[col] = gpd.GeoSeries.from_wkt(df[col])
        gdf = gpd.GeoDataFrame(df, geometry="geometry")
        return gdf

    @staticmethod
    def save_geodataframe_to_csv(gdf: gpd.GeoDataFrame, file_path: Path) -> None:
        """Save GeoDataFrame to a CSV file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        gdf_copy = gdf.copy()
        for col in gdf_copy.columns:
            gdf_copy[col] = gdf_copy[col].apply(lambda geom: geom.wkt if geom is not None else None)
        gdf_copy.to_csv(file_path, index=False)

    @staticmethod
    def extract_time_keys(dir_path: Path) -> list[str]:
        """Extract sorted time keys from CSV filenames in a directory."""
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        files = os.listdir(dir_path)
        if not files:
            raise FileNotFoundError(f"No files found in directory: {dir_path}")
        return sorted(
            file_name.split(".")[0] for file_name in files
            if file_name.endswith(".csv") and file_name.split(".")[0]
        )

    def collect_obstacles(
        self,
        directory_path: Path,
        time_keys: list[str],
    ) -> dict[str, gpd.GeoDataFrame]:
        """
        Collect GeoDataFrames with obstacles data into a dictionary with corresponding timestamp keys.

        Args:
            directory_path (Path): Path to the directory with obstacles CSV files.
            time_keys (list[str]): All available time keys sorted chronologically.

        Returns:
            dict[str, gpd.GeoDataFrame]: Obstacle geometries for each time key.
        """
        return {
            time_key: self.load_geodataframe_from_csv(directory_path / f"{time_key}.csv")
            for time_key in time_keys
        }
