import itertools as it
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pydantic import BaseModel

from thund_avoider.schemas.interpretation.config import (
    DataSourceConfig,
    OutlierConfig,
    ReportConfig,
)


class DataPreprocessor:
    """
    Handles data loading and preprocessing for statistical analysis.

    This class is responsible for loading parquet files, applying outlier
    filters, and preparing data for statistical analysis.
    """

    def __init__(self, config: ReportConfig) -> None:
        self._config = config
        self._data: dict[str, gpd.GeoDataFrame] = {}

    @property
    def data(self) -> dict[str, gpd.GeoDataFrame]:
        """Get loaded data dictionary."""
        return self._data

    @property
    def algorithm_names(self) -> list[str]:
        """Get list of loaded algorithm names."""
        return list(self._data.keys())

    def load_data(self) -> dict[str, gpd.GeoDataFrame]:
        """
        Load all configured parquet files.

        Returns:
            Dictionary mapping algorithm names to GeoDataFrames.
        """
        for source in self._config.data_sources:
            if source.file_path.exists():
                self._data[source.russian_name] = gpd.read_parquet(source.file_path)
        return self._data

    def preprocess(
        self,
        gdf: gpd.GeoDataFrame,
        outlier_config: OutlierConfig | None = None,
    ) -> gpd.GeoDataFrame:
        """
        Apply preprocessing and outlier filtering to a GeoDataFrame.

        Args:
            gdf: Input GeoDataFrame with pathfinding results.
            outlier_config: Outlier filtering configuration.

        Returns:
            Preprocessed GeoDataFrame.
        """
        config = outlier_config or self._config.outlier_config

        # Make a copy to avoid modifying original
        df = gdf.copy()

        # Add success column if not present (assume success for existing paths)
        if "success" not in df.columns:
            df["success"] = True

        # Filter by length outliers
        df = df[
            (df["length"] >= config.min_length) &
            (df["length"] <= config.max_length)
        ]

        # Filter suspicious timestamps
        if config.suspicious_timestamps:
            df = df[~df["timestamp"].isin(config.suspicious_timestamps)]

        # Adjust window size (subtract 1 for consistency)
        df["window_size"] = df["window_size"] - 1

        # Create subject_id for paired analysis
        df["subject_id"] = df["timestamp"] + "_" + df["direction"]

        return df.reset_index(drop=True)

    def preprocess_all(self) -> dict[str, gpd.GeoDataFrame]:
        """
        Load and preprocess all configured data sources.

        Returns:
            Dictionary of preprocessed GeoDataFrames.
        """
        self.load_data()
        return {
            name: self.preprocess(gdf)
            for name, gdf in self._data.items()
        }

    @staticmethod
    def compute_relative_differences(
        df: pd.DataFrame,
        col1: str,
        col2: str,
    ) -> pd.Series:
        """
        Compute relative differences between two columns.

        Args:
            df: DataFrame with the columns.
            col1: First column name.
            col2: Second column name.

        Returns:
            Series of relative differences.
        """
        return (df[col1] - df[col2]) / ((df[col1] + df[col2]) / 2)

    @staticmethod
    def get_balanced_subset(
        dfs: dict[str, pd.DataFrame],
        on_columns: list[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Get balanced subset where all datasets share the same subjects.

        Args:
            dfs: Dictionary of DataFrames.
            on_columns: Columns to merge on (default: subject_id).

        Returns:
            Dictionary of balanced DataFrames.
        """
        if on_columns is None:
            on_columns = ["subject_id"]

        # Find common subject_ids across all DataFrames
        common_ids = set.intersection(
            *[set(df["subject_id"].unique()) for df in dfs.values()]
        )

        return {
            name: df[df["subject_id"].isin(common_ids)].copy()
            for name, df in dfs.items()
        }

    @staticmethod
    def merge_for_comparison(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        name1: str,
        name2: str,
        merge_on: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Merge two DataFrames for pairwise comparison.

        Args:
            df1: First DataFrame.
            df2: Second DataFrame.
            name1: Name suffix for first DataFrame.
            name2: Name suffix for second DataFrame.
            merge_on: Columns to merge on.

        Returns:
            Merged DataFrame.
        """
        if merge_on is None:
            merge_on = ["subject_id", "window_size"]

        return pd.merge(
            df1,
            df2,
            on=merge_on,
            suffixes=(f"_{name1}", f"_{name2}"),
        )

    def get_all_pairwise_combinations(self) -> list[tuple[str, str]]:
        """
        Get all pairwise combinations of loaded algorithms.

        Returns:
            List of algorithm name pairs.
        """
        names = self.algorithm_names
        return list(it.combinations(names, 2))
