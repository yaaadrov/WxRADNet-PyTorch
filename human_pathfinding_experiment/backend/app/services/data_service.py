import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from app.config import settings


class DataService:
    """Service for loading and caching experiment data."""

    def __init__(self) -> None:
        """Initialize data service."""
        self._cache: dict[str, Any] = {}

    def get_available_timestamps(self) -> list[str]:
        """Get list of available timestamp directories from data path."""
        if not settings.data_path.exists():
            return []

        timestamps = sorted(
            d.name
            for d in settings.data_path.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        return timestamps

    @lru_cache(maxsize=1)
    def _load_ab_points(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Load AB points from pickle file (cached)."""
        if not settings.ab_points_path.exists():
            raise FileNotFoundError(f"AB points file not found: {settings.ab_points_path}")

        with open(settings.ab_points_path, "rb") as f:
            ab_points = pickle.load(f)

        # Convert Points to coordinate tuples
        return [
            ((a.x, a.y), (b.x, b.y))
            for a, b in ab_points
        ]

    @lru_cache(maxsize=1)
    def _load_timestamps(self) -> list[str]:
        """Load timestamps from pickle file (cached)."""
        if not settings.timestamps_path.exists():
            raise FileNotFoundError(f"Timestamps file not found: {settings.timestamps_path}")

        with open(settings.timestamps_path, "rb") as f:
            timestamps = pickle.load(f)

        return timestamps

    def get_ab_points_for_timestamp(self, timestamp: str) -> tuple[tuple[float, float], tuple[float, float]]:
        """Get A and B points for a specific timestamp."""
        timestamps = self._load_timestamps()
        ab_points = self._load_ab_points()

        try:
            # Format timestamp to match the format in timestamps.pkl
            formatted_ts = timestamp.replace("_", " ")
            # Find index in timestamps list
            for i, ts in enumerate(timestamps):
                ts_str = str(ts).replace("-", "_").replace(" ", "_").replace(":", "_")
                if ts_str == timestamp or str(ts).startswith(formatted_ts):
                    return ab_points[i]

            # If not found by exact match, try to parse timestamp and match
            for i, ts in enumerate(timestamps):
                ts_formatted = "_".join(str(ts).split())
                if timestamp in ts_formatted or ts_formatted.startswith(timestamp):
                    return ab_points[i]

            raise ValueError(f"Timestamp not found: {timestamp}")
        except Exception as e:
            raise ValueError(f"Error finding timestamp {timestamp}: {e}")

    def get_data_dir(self, timestamp: str) -> Path:
        """Get the data directory path for a timestamp."""
        return settings.data_path / timestamp

    def extract_time_keys(self, data_dir: Path) -> list[str]:
        """Extract sorted time keys from CSV filenames in a directory."""
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {data_dir}")

        files = list(data_dir.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in directory: {data_dir}")

        return sorted(f.stem for f in files if f.stem)

    def load_geodataframe_from_csv(self, file_path: Path) -> gpd.GeoDataFrame:
        """Load GeoDataFrame from a CSV file with WKT geometry."""
        df = pd.read_csv(file_path)
        for col in df.columns:
            df[col] = gpd.GeoSeries.from_wkt(df[col])
        gdf = gpd.GeoDataFrame(df, geometry="geometry")
        return gdf

    def collect_obstacles(
        self,
        directory_path: Path,
        time_keys: list[str],
    ) -> dict[str, gpd.GeoDataFrame]:
        """Collect GeoDataFrames with obstacles data into a dictionary."""
        return {
            time_key: self.load_geodataframe_from_csv(directory_path / f"{time_key}.csv")
            for time_key in time_keys
        }

    def get_obstacles_for_timestamp(
        self,
        timestamp: str,
    ) -> tuple[list[str], dict[str, gpd.GeoDataFrame]]:
        """Load obstacles for a timestamp (reuses logic from run_masked_dynamic_avoider)."""
        data_dir = self.get_data_dir(timestamp)
        time_keys = self.extract_time_keys(data_dir)
        dict_obstacles = self.collect_obstacles(data_dir, time_keys)
        return time_keys, dict_obstacles
