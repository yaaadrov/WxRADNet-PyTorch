import os
import pickle
import re
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import LineString, Point, Polygon


BUFFER: int = 5_000  # Additional 5 km buffer
TOLERANCE: int = 5_000  # Simplification tolerance
BBOX_BUFFER: int = 50_000  # Additional 50 km buffer for A and B points

ROOT_PATH: Path = Path(__file__).resolve().parents[1]
DATA_PATH: Path = ROOT_PATH / "data"
RESULT_PATH: Path = ROOT_PATH / "results"
TIMESTAMPS_PATH: Path = ROOT_PATH / "config" / "timestamps.pkl"


class StaticAvoider:
    def __init__(
            self,
            buffer: int,
            tolerance: int,
            bbox_buffer: int,
    ):
        """
        Initialize `StaticAvoider` class
        Args:
            buffer (int): Buffer distance for geometry simplification
            tolerance (int): Simplification tolerance for geometry
            bbox_buffer (int): Buffer distance for bounding box when choosing A and B points
        """
        self.buffer = buffer
        self.tolerance = tolerance
        self.bbox_buffer = bbox_buffer

    def _process_obstacles(
            self,
            gdf: gpd.GeoDataFrame,
            strategy: str,
    ) -> tuple[list[Polygon], list[Polygon]]:
        """
        Simplify obstacles based on the chosen strategy
        Args:
            gdf: GeoDataFrame containing obstacle data
            strategy (str): Strategy to apply, either 'convex' or 'concave'
        Returns:
            tuple[list[Polygon], list[Polygon]]: List of obstacle geometries and list of simplified obstacle geometries in view of given strategy
        """
        obstacles = gdf[strategy]
        simplified_obstacles = obstacles.buffer(self.buffer).simplify(self.tolerance, preserve_topology=True)
        return obstacles, list(simplified_obstacles)

    def _extract_vertices(self, simplified_obstacles: list[Polygon], points: list[Point]) -> list[Point]:
        """
        Extract unique vertices from obstacles and additional points
        Args:
            obstacles (list[Polygon]): List of obstacle geometries
            points (list[Point]): A and B points to include
        Returns:
            list[Point]: Deduplicated list of Point objects
        """
        all_vertices = []
        for polygon in simplified_obstacles:
            all_vertices.extend(polygon.exterior.coords)

        all_vertices = list(set(all_vertices))  # Deduplicate
        all_vertices = [Point(p) for p in all_vertices]
        all_vertices.extend(points)  # Add A and B points
        return all_vertices

    def _is_line_valid(self, line: LineString, obstacles: list[Polygon]) -> bool:
        """
        Check if a line intersects any obstacle
        Args:
            line (LineString): Line to check
            obstacles (list[Polygon]): List of obstacle geometries
        Returns:
            bool: True if the line is valid, False otherwise
        """
        for obstacle in obstacles:
            if line.intersects(obstacle):
                return False
        return True

    def _build_visibility_graph(self, vertices: list[Point], obstacles: list[Polygon]) -> nx.Graph:
        """
        Build a visibility graph for the given vertices and obstacles
        Args:
            vertices (list[Point]): List of vertex Point objects
            obstacles (list[Point]): List of obstacle geometries
        Returns:
            nx.Graph: Visibility graph with weighted edges
        """
        G = nx.Graph()

        # Add nodes
        for i, point in enumerate(vertices):
            G.add_node(i, pos=(point.x, point.y), point=point)

        # Add edges if visible
        for i, p1 in enumerate(vertices):
            for j, p2 in enumerate(vertices):
                if i != j:  # Avoid self-loops
                    line = LineString([p1, p2])
                    if self._is_line_valid(line, obstacles):
                        distance = p1.distance(p2)
                        G.add_edge(i, j, weight=distance)

        return G

    def find_shortest_path(
            self,
            gdf: gpd.GeoDataFrame,
            strategy: str,
            A: Point,
            B: Point,
    ) -> list[Point] | ValueError:
        """
        Find the shortest path between two points avoiding obstacles
        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame containing obstacle data
            strategy (str): Strategy to apply, either 'convex' or 'concave'
            A (Point): Starting point
            B (Point): Ending point
        Returns:
            list[Point]: List of Point objects representing the shortest path
        Raises:
            ValueError: If strategy is not 'convex' or 'concave'
        """
        if strategy not in ["convex", "concave"]:
            raise ValueError("Strategy must be 'convex' or 'concave'")

        obstacles, simplified_obstacles = self._process_obstacles(gdf, strategy=strategy)
        all_vertices = self._extract_vertices(simplified_obstacles, [A, B])
        G = self._build_visibility_graph(all_vertices, obstacles)
        source_node = len(all_vertices) - 2  # Index of start_point
        target_node = len(all_vertices) - 1  # Index of end_point
        shortest_path_nodes = nx.shortest_path(G, source=source_node, target=target_node, weight="weight")
        return [all_vertices[node] for node in shortest_path_nodes]


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
        buffer=BUFFER,
        tolerance=TOLERANCE,
        bbox_buffer=BBOX_BUFFER,
        timestamps=timestamps,
    )
