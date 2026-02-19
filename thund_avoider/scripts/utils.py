from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Final

import geopandas as gpd
import pandas as pd

from thund_avoider.settings import RESULT_PATH

WINDOW_SIZES: Final = [1, 2, 3, 4, 5, 6, 7]


def format_timestamp(ts: datetime) -> str:
    """Format datetime as filename-safe string."""
    return "_".join(str(ts).split())


def save_combined_results(
    result_dir: Path,
    result_name: str,
    logger: Logger,
) -> None:
    """
    Combine all parquet files into a single result file.

    Args:
        result_dir: Directory containing individual parquet files.
        result_name: Name for the final parquet file (without extension).
        logger: Logger instance.
    """
    parquet_files = list(result_dir.glob("*.parquet"))
    if not parquet_files:
        logger.warning(f"No parquet files found in {result_dir}")
        return

    dfs = [gpd.read_parquet(f) for f in parquet_files]
    combined = gpd.GeoDataFrame(pd.concat(dfs, ignore_index=True), crs="EPSG:3067")
    output_path = RESULT_PATH / f"{result_name}.parquet"
    combined.to_parquet(output_path, index=False)
    logger.info(f"Final DataFrame saved successfully to '{output_path}'")
