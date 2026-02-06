import os
import pickle
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import Point, LineString

from ThundAvoider.thund_avoider.static_avoider import StaticAvoider
from thund_avoider.settings import settings


def load_geodataframe_from_csv(file_path: Path) -> gpd.GeoDataFrame:
    """
    Load GeoDataFrame from a CSV file

    Args:
        file_path (Path): Path to the CSV file

    Returns:
        gpd.GeoDataFrame: Loaded GeoDataFrame
    """
    df = pd.read_csv(file_path)
    for col in df.columns:
        df[col] = gpd.GeoSeries.from_wkt(df[col])
    gdf = gpd.GeoDataFrame(df, geometry="geometry")
    return gdf


def points_to_df(
    static_avoider: StaticAvoider,
    timestamp: str,
    gdf: gpd.GeoDataFrame,
    points: dict[str, Point],
    strategy: str,
) -> pd.DataFrame:
    """
    Get pd.DataFrame with results for a particular points set and strategy

    Args:
        static_avoider (StaticAvoider): StaticAvoider object
        timestamp (str): Timestamp string
        gdf (gpd.GeoDataFrame): GeoDataFrame containing obstacle data
        points (dict[str, Point]): Dictionary of Point objects
        strategy (str): Strategy to apply, either 'convex' or 'concave'

    Returns:
        pd.DataFrame: Dataframe with results for a particular points set and strategy
    """
    paths = [
        LineString(static_avoider.find_shortest_path(gdf=gdf, strategy=strategy, A=points[start], B=points[end]))
        for start, end in [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")]
    ]
    lengths = [path.length for path in paths]
    return pd.DataFrame({
        "timestamp": [timestamp] * len(paths),
        "strategy": [strategy] * len(paths),
        "paths": paths,
        "length": lengths,
    })


def process_data(buffer: int, tolerance: int, bbox_buffer: int, timestamps: list[datetime]) -> None:
    """
    Calculate path length for given number of timestamps for both strategies

    Args:
        buffer (int): Buffer distance for geometry simplification
        tolerance (int): Simplification tolerance for geometry
        bbox_buffer (int): Buffer distance for bounding box when choosing A and B points
        timestamps (list[datetime]): Timestamps
    """
    static_avoider = StaticAvoider(
        buffer=buffer,
        tolerance=tolerance,
        bbox_buffer=bbox_buffer,
    )

    length = len(str(len(timestamps)))
    dir_path_result = RESULT_PATH / "static_avoider"
    if not dir_path_result.exists():
        dir_path_result.mkdir(parents=True, exist_ok=True)

    for i, current_date in enumerate(timestamps):
        file_name = "_".join(re.split(r"[- :]", str(current_date)))
        if f"{file_name}.csv" in os.listdir(dir_path_result):
            print(f"{i + 1:<{length}}/{len(timestamps)}: File {file_name}.csv exists. Skipping {current_date}")
            continue

        file_path = DATA_PATH / file_name / f"{file_name}.csv"
        gdf = load_geodataframe_from_csv(file_path)

        minx, miny, maxx, maxy = gdf["convex"].buffer(static_avoider.bbox_buffer).total_bounds
        points = {
            "A": Point(minx, miny),
            "B": Point(maxx, maxy),
            "C": Point(minx, maxy),
            "D": Point(maxx, miny),
            "E": Point((minx + maxx) / 2, miny),
            "F": Point((minx + maxx) / 2, maxy),
            "G": Point(minx, (miny + maxy) / 2),
            "H": Point(maxx, (miny + maxy) / 2),
        }
        df = pd.concat(
            [
                points_to_df(
                    static_avoider=static_avoider,
                    timestamp=file_name,
                    gdf=gdf,
                    points=points,
                    strategy="convex",
                ),
                points_to_df(
                    static_avoider=static_avoider,
                    timestamp=file_name,
                    gdf=gdf,
                    points=points,
                    strategy="concave",
                ),
            ],
        )
        df.to_csv(dir_path_result / f"{file_name}.csv", index=None)
        print(f'{i + 1:<{length}}/{len(timestamps)}: {current_date} saved to "{file_name}.csv"')

    df_result = pd.concat(
        [
            pd.read_csv(dir_path_result / file_name)
            for file_name in os.listdir(dir_path_result)
        ],
        ignore_index=True,
    )
    df_result.to_csv(RESULT_PATH / "static_avoider.csv", index=None)
    print("Final DataFramed saved successfully to 'results/static_avoider.csv'")


if __name__ == "__main__":
    with open(TIMESTAMPS_PATH, "rb") as file_in:
        timestamps = pickle.load(file_in)
    process_data(
        buffer=settings.buffer,
        tolerance=settings.tolerance,
        bbox_buffer=settings.bbox_buffer,
        timestamps=timestamps,
    )
