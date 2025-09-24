import itertools as it
import logging
import math
import os
import pickle
import re
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Final, Literal

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from pyproj import CRS
from scipy.spatial import Delaunay, KDTree
from shapely import STRtree
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import substring, unary_union


WINDOW_SIZE_INITIAL: Final = [1, 2, 3, 4, 5, 6, 7]
WINDOW_SIZE_FINE_TUNE: Final = [2, 3]

PROJECTED_CRS: Final = CRS(3067)  # ETRS89 / TM35FIN(E,N)
VELOCITY_KMH: Final = 900  # Velocity in km/hour
DELTA_MINUTES: Final = 5  # Forecast frequency
BUFFER: Final = 5_000  # Additional 5 km buffer
TOLERANCE: Final = 5_000  # Simplification tolerance
K_NEIGHBORS: Final = 10  # Number of neighbors for master graph
MAX_DISTANCE: Final = 20_000  # Split each segment into several subsegments for greedy fine-tuning
SIMPLIFICATION_TOLERANCE: Final = 1e-9  # Tolerance to simplify paths after densifying
SMOOTH_TOLERANCE: Final = TOLERANCE * 5  # Tolerance for smoothing fine-tuning
MAX_ITER: Final = 300  # Maximum number of iterations for smooth fine-tuning
DELTA_LENGTH: Final = 1.0  # Smooth fine-tuning length sensitivity

ROOT_PATH: Path = Path(__file__).resolve().parents[1]
DATA_PATH: Path = ROOT_PATH / "data"
RESULT_PATH: Path = ROOT_PATH / "results"
BASE_PATH: Path = RESULT_PATH / "dynamic_avoider_base"
TIMESTAMPS_PATH: Path = ROOT_PATH / "config" / "timestamps.pkl"
AB_POINTS_PATH: Path = ROOT_PATH / "config" / "ab_points.pkl"


class SlidingWindowPath(BaseModel):
    strategy: str
    path: list[list[Point]]
    all_paths: list[list[Point]]
    all_graphs: list[nx.Graph]
    success: bool
    success_intermediate: bool
    num_segments: int

    model_config = ConfigDict(arbitrary_types_allowed=True)


class FineTunedPath(BaseModel):
    strategy: str
    path: list[list[Point]]
    all_paths_initial: list[list[Point]]
    all_paths_fine_tuned: list[list[Point]]
    all_paths_initial_raw: list[SlidingWindowPath]
    fine_tuning_iters: list[int]
    fine_tuning_times: list[float]
    success: bool
    success_intermediate: bool
    num_segments: int

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DynamicAvoider:
    def __init__(
        self,
        crs: CRS,
        velocity_kmh: float,
        delta_minutes: float,
        buffer: float,
        tolerance: float,
        k_neighbors: int,
        max_distance: float,
        simplification_tolerance: float,
        smooth_tolerance: float,
        max_iter: int,
        delta_length: float,
        strategy: Literal["concave", "convex"] = "concave",
        tuning_strategy: Literal["greedy", "smooth"] = "greedy",
    ) -> None:
        """
        Initialize `DynamicAvoider` class

        Args:
            crs (CRS): Coordinate Reference System
            velocity_kmh (float): Velocity in km/h
            delta_minutes (float): Forecast frequency in minutes
            buffer (float): Buffer distance for geometry simplification
            tolerance (float): Simplification tolerance for geometry
            k_neighbors (int): Number of neighbors for master graph
            max_distance (float): Split each segment into several subsegments for greedy fine-tuning
            simplification_tolerance (float): Tolerance to simplify paths after densifying
            smooth_tolerance (float): Tolerance for smoothing fine-tuning
            max_iter (int): Maximum number of iterations for smooth fine-tuning
            delta_length (float): Smooth fine-tuning length sensitivity
            strategy (Literal["concave", "convex"]): Path-finding strategy to apply
            tuning_strategy (Literal["greedy", "smooth"]): Fine-tuning strategy to apply
        """
        self.crs = crs
        self.velocity_mpm = velocity_kmh * 1000 / 60  # Velocity in meters/minute
        self.delta_minutes = delta_minutes
        self.buffer = buffer
        self.tolerance = tolerance
        self.k_neighbors = k_neighbors
        self.max_distance = max_distance
        self.simplification_tolerance = simplification_tolerance
        self.smooth_tolerance = smooth_tolerance
        self.max_iter = max_iter
        self.delta_length = delta_length
        self.strategy = strategy
        self.tuning_strategy = tuning_strategy

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _load_geodataframe_from_csv(file_path: Path) -> gpd.GeoDataFrame:
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

    @staticmethod
    def extract_time_keys(dir_path: Path) -> list[str]:
        """
        Extract sorted time keys from CSV filenames in a directory

        Args:
            dir_path (Path): Path to the directory containing obstacle CSV files

        Returns:
            list[str]: All available time keys sorted chronologically

        Raises:
            FileNotFoundError: If the directory does not exist or is empty.
        """
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
        Collect GeoDataFrames with obstacles data into a dictionary with corresponding timestamp keys

        Args:
            directory_path (Path): Path to the directory with obstacles CSV files
            time_keys (list[str]): All available time keys sorted chronologically

        Returns:
            dict[str, gpd.GeoDataFrame]: Obstacle geometries for each time key
        """
        return {
            time_key: self._load_geodataframe_from_csv(directory_path / f"{time_key}.csv")
            for time_key in time_keys
        }

    def _collect_vertices(self, flat_obstacles: list[Polygon]) -> list[Point]:
        """
        Collect vertices from the current obstacle set

        Args:
            flat_obstacles (list[Polygon]): List of previously flattened obstacles

        Returns:
            list[Point]: List of vertices collected from the current obstacle set

        Raises:
            TypeError: If `obstacles_simplified` is none of Polygon, MultiPolygon
        """
        vertices = []
        obstacles_simplified = (
            unary_union(gpd.GeoDataFrame(geometry=flat_obstacles, crs=self.crs))
            .buffer(self.buffer)
            .simplify(self.tolerance, preserve_topology=True)
        )
        if isinstance(obstacles_simplified, Polygon):
            vertices.extend(obstacles_simplified.exterior.coords)
            return list(set(Point(coord) for coord in vertices))
        if isinstance(obstacles_simplified, MultiPolygon):
            for polygon in obstacles_simplified.geoms:
                vertices.extend(polygon.exterior.coords)
            return sorted({Point(coord) for coord in vertices}, key=lambda p: (p.x, p.y))
        raise TypeError(f"Unexpected type {type(obstacles_simplified)}")

    @staticmethod
    def _is_line_valid(line: LineString, obstacles: list[Polygon]) -> bool:
        """
        Check if a line intersects any obstacle

        Args:
            line (LineString): Line to check
            obstacles (list[Polygon]): List of obstacle geometries

        Returns:
            bool: True if the line is valid, False otherwise
        """
        return not any(line.intersects(obstacle) for obstacle in obstacles)

    def _simplify_vertices(self, vertices: list[Point]) -> list[Point]:
        """
        Simplify vertices under given tolerance

        Args:
            vertices (list[Point]): List of vertex Point objects

        Returns:
            list[Point]: List of vertices (centroids) after simplification
        """
        visited = set()
        clusters = []
        tree = STRtree(vertices)
        for i, point in enumerate(vertices):
            if i in visited:
                continue
            buffer = point.buffer(self.tolerance)
            idxs = tree.query(buffer)
            neighbors = [vertices[j] for j in idxs if vertices[j].distance(point) <= self.tolerance]
            for j in idxs:
                visited.add(j)
            clusters.append(neighbors)
        return [unary_union(cluster).centroid for cluster in clusters]

    def _validate_candidates(
        self,
        G: nx.Graph,
        centroids: list[Point],
        candidates: set[tuple[int, int]],
        strtrees: dict[str, STRtree],
    ) -> tuple[nx.Graph, dict[str, list[tuple[int, int, float]]]]:
        """
        Validate candidate edges of the master graph for each time keys

        Args:
            G (nx.Graph): Master graph to validate candidate edges
            centroids (list[Point]): List of master centroids
            candidates (set[tuple[int, int]]): List of candidate edges
            strtrees (dict[str, STRtree]): Dictionary of strtrees for each time key

        Returns:
            tuple:
                - dict[str, list[tuple[int, int, float]]]: List of valid edges for each time key
                - nx.Graph: Master graph with all possible valid edges
        """
        G_master = G.copy()
        length = len(str(len(strtrees)))
        time_valid_edges: defaultdict[str, list[tuple[int, int, float]]] = defaultdict(list)
        for num, (time_key, tree) in enumerate(strtrees.items()):
            for i, j in candidates:
                p_i, p_j = centroids[i], centroids[j]
                line = LineString([p_i, p_j])
                possible_idxs = tree.query(line)
                possible_obs = [tree.geometries[i] for i in possible_idxs]
                if self._is_line_valid(line, possible_obs):
                    w = p_i.distance(p_j)
                    time_valid_edges[time_key].append((i, j, w))
                    if not G_master.has_edge(i, j):
                        G_master.add_edge(i, j, weight=w)
            self.logger.info(
                f"\t{num + 1:<{length}}/{len(strtrees)}: Time key {time_key} processed"
            )
        return G_master, time_valid_edges

    def _build_master_graph(
        self,
        vertices: list[Point],
        time_keys: list[str],
        dict_obstacles: dict[str, gpd.GeoDataFrame],
    ) -> tuple[nx.Graph, dict[str, list[tuple[int, int, float]]]]:
        """
        Build master graph for given vertices from all available obstacles

        Args:
            vertices (list[Point]): List of all available vertex Point objects
            time_keys (list[str]): List of master time keys
            dict_obstacles (dict[str, gpd.GeoDataFrame]): Obstacle geometries for each time key

        Returns:
            tuple:
                - nx.Graph: Master graph with all possible valid edges
                - dict[str, list[tuple[int, int, float]]]: List of valid edges for each time key
        """
        self.logger.info("Building master graph:")
        centroids = self._simplify_vertices(vertices)
        G_master = nx.Graph()
        for i, p in enumerate(centroids):
            G_master.add_node(i, pos=(p.x, p.y), point=p)

        coords = [(p.x, p.y) for p in centroids]
        tri = Delaunay(coords)
        delaunay_edges = {
            tuple(sorted((i, j)))
            for simplex in tri.simplices
            for i, j in it.combinations(simplex, 2)
        }

        kd = KDTree(coords)
        knn_edges = {
            tuple(sorted((i, j)))
            for i, coord in enumerate(coords)
            for j in kd.query(coord, k=self.k_neighbors + 1)[1]
            if i != j
        }

        candidates = delaunay_edges | knn_edges
        strtrees = {
            time_key: STRtree(list(dict_obstacles[time_key][self.strategy]))
            for time_key in time_keys
        }
        return self._validate_candidates(
            G=G_master,
            centroids=centroids,
            candidates=candidates,
            strtrees=strtrees,
        )

    @staticmethod
    def _subgraph_for_time_keys(
        G_master: nx.Graph,
        time_keys: list[str],
        time_valid_edges: dict[str, list[tuple[int, int, float]]],
    ) -> nx.Graph:
        """
        Create a subgraph with edges valid across specified time keys

        Args:
            G_master (nx.Graph): Master graph
            time_keys (list[str]): List of time keys (within a window) to validate edges for
            time_valid_edges (dict[str, list[tuple[int, int, float]]]): Valid edges for time keys

        Returns:
            nx.Graph: Subgraph with edges valid across specified time keys
        """
        edge_sets = [
            set(
                (i, j, w)
                for i, j, w in time_valid_edges[time_key]
            )
            for time_key in time_keys
        ]
        common_edges = set.intersection(*edge_sets)
        G_sub = nx.Graph()
        G_sub.add_nodes_from(G_master.nodes(data=True))
        for i, j, w in common_edges:
            G_sub.add_edge(i, j, weight=w)
        return G_sub

    def _add_points_to_subgraph(
        self,
        points: list[tuple[Literal["start", "end"], Point]],
        G_master: nx.Graph,
        G_sub: nx.Graph,
        obstacles: list[Polygon],
    ) -> nx.Graph:
        """
        Adds start and end points to the graph and connect them to visible nearby nodes

        Args:
            points (list[tuple[Literal["start", "end"], Point]]): Start and end points
            G_master (nx.Graph): Master graph
            G_sub (nx.Graph): Subgraph for a given window
            obstacles (list[Polygon]): List of obstacle polygons for a given window

        Returns:
            nx.Graph: Subgraph with points added to the graph
        """
        vertices = [data["point"] for _, data in G_master.nodes(data=True)]
        coords = [(p.x, p.y) for p in vertices]
        kd_tree = KDTree(coords)
        strtree = STRtree(obstacles)

        G = G_sub.copy()
        for point_name, point in points:
            G.add_node(point_name, pos=(point.x, point.y), point=point)
            _, idxs = kd_tree.query([point.x, point.y], k=self.k_neighbors)
            for i in idxs:
                target_point = vertices[i]
                line = LineString([point, target_point])
                possible_idxs = strtree.query(line)
                possible_obs = [strtree.geometries[i] for i in possible_idxs]
                if self._is_line_valid(line, possible_obs):
                    w = point.distance(target_point)
                    G.add_edge(point_name, i, weight=w)
        return G

    def _find_shortest_path(
        self,
        G: nx.Graph,
        source_node: str | int = "start",
        target_node: str | int = "end",
    ) -> tuple[list[Point], list[int | str]] | None:
        """
        Find the shortest path in graph

        Args:
            G (nx.Graph): Graph to find the shortest path on
            source_node (str | int): Source node index
            target_node (str | int): Target node index

        Returns:
            tuple:
                - list[Point]: List of shortest path points
                - list [int | str]: List of shortest path indices
        """
        try:
            path_idx = nx.shortest_path(
                G,
                source=source_node,
                target=target_node,
                weight="weight",
            )
            return [G.nodes[node]["point"] for node in path_idx], path_idx
        except nx.NetworkXNoPath:
            self.logger.info("Graph seems to be unconnected")
            return None

    def _build_visibility_graph(
        self,
        G: nx.Graph,
        obstacles: list[Polygon],
    ) -> nx.Graph:
        """
        Build a visibility graph for the given vertices and obstacles

        Args:
            G (nx.Graph): Graph with vertices
            obstacles (list[Point]): List of obstacle geometries

        Returns:
            nx.Graph: Visibility graph with weighted edges
        """
        for i, node_i in enumerate(G.nodes):
            for j, node_j in enumerate(G.nodes):
                p1, p2 = G.nodes[node_i]["point"], G.nodes[node_j]["point"]
                line = LineString([p1, p2])
                if self._is_line_valid(line, obstacles):
                    w = p1.distance(p2)
                    G.add_edge(node_i, node_j, weight=w)
        return G

    def _tune_path(
        self,
        G_sub: nx.Graph,
        path_idx: list[int | str],
        window_obstacles: list[Polygon],
    ) -> tuple[list[Point], list[int | str]] | None:
        """
        Tune initially found path

        Args:
            G_sub (nx.Graph): Subgraph for a given window
            path_idx (list[int | str]): List of initial shortest path indices
            obstacles (list[Point]): List of obstacle geometries for a given window

        Returns:
            tuple:
                - list[Point]: List of shortest path points
                - list [int | str]: List of shortest path indices
        """
        G_tune = nx.Graph()
        G_tune.add_nodes_from((node, G_sub.nodes[node]) for node in path_idx)
        G_tune = self._build_visibility_graph(G_tune, window_obstacles)
        return self._find_shortest_path(G_tune)

    def _check_start_end_point(
        self,
        point: Point,
        point_type: Literal["start", "end"],
        flat_obstacles: list[Polygon],
        vertices: list[Point],
    ):
        for poly in flat_obstacles:
            if poly.contains(point):
                min_dist = float("inf")
                closest_point = point
                for vertex in vertices:
                    if any(poly.contains(vertex) for poly in flat_obstacles):
                        continue
                    dist = point.distance(vertex)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = vertex
                self.logger.info(
                    f"Current {point_type} was inside an obstacle. "
                    f"Replacing with closest vertex at {closest_point}"
                )
                return closest_point, False if point_type == "start" else True
        return point, True

    def _perform_pathfinding(
        self,
        current_pos: Point,
        end: Point,
        *,
        current_time_index: int,
        window_size: int,
        time_keys: list[str],
        dict_obstacles: dict[str, gpd.GeoDataFrame],
        G_master: nx.Graph,
        master_vertices: list[Point],
        time_valid_edges: dict[str, list[tuple[int, int, float]]],
    ) -> SlidingWindowPath:
        path = []
        all_paths = []
        all_graphs = []
        success = True
        success_intermediate = True
        length = len(str(len(time_keys)))
        self.logger.info("Start pathfinding:")

        while current_pos != end and current_time_index < len(time_keys) - window_size:
            # Extract time keys for the current window
            window_time_keys = time_keys[current_time_index:current_time_index + window_size]
            if not window_time_keys:
                self.logger.info("No more available time keys")
                break

            # Collect obstacles for current window
            window_obstacles = [
                dict_obstacles[time_key][self.strategy] for time_key in window_time_keys
            ]
            window_obstacles_flat = list(it.chain(*window_obstacles))

            # Check start and end points
            current_pos, success_start = self._check_start_end_point(
                point=current_pos,
                point_type="start",
                flat_obstacles=window_obstacles_flat,
                vertices=master_vertices,
            )
            current_end, _ = self._check_start_end_point(
                point=end,
                point_type="end",
                flat_obstacles=window_obstacles_flat,
                vertices=master_vertices,
            )
            success_intermediate = success_intermediate and success_start

            # Build subgraph and add start and end points
            G_sub = self._subgraph_for_time_keys(
                G_master=G_master,
                time_keys=window_time_keys,
                time_valid_edges=time_valid_edges,
            )
            G_sub = self._add_points_to_subgraph(
                points=[("start", current_pos), ("end", current_end)],
                G_master=G_master,
                G_sub=G_sub,
                obstacles=window_obstacles_flat,
            )

            # Find the shortest path
            shortest_path = self._find_shortest_path(G_sub)
            if shortest_path is None:
                success = False
                self.logger.info(
                    f"No path found for time window at {time_keys[current_time_index]}"
                )
                break
            _, path_idx = shortest_path

            # Tune shortest path
            shortest_path_tuned = self._tune_path(G_sub, path_idx, window_obstacles_flat)
            if shortest_path_tuned is None:
                success = False
                self.logger.info(
                    f"Unable to tune path for time window at {time_keys[current_time_index]}"
                )
                break
            path_tuned, _ = shortest_path_tuned

            # Add data
            all_paths.append(path_tuned)
            all_graphs.append(G_sub)

            # Update current position
            path_line = LineString(path_tuned)
            path_within_window = substring(path_line, 0, self.velocity_mpm * self.delta_minutes)
            points_within_window = [Point(coord) for coord in path_within_window.coords]
            current_pos = points_within_window[-1]
            current_time_index += 1

            # Extend the valid portion of the path
            path.append(points_within_window)
            self.logger.info(
                f"\t{current_time_index:<{length}}/{len(time_keys) - window_size}: Path created, "
                f"Num edges: {len(G_sub.edges)}",
            )

        # Add the end point if not reached yet
        num_segments = current_time_index - 1
        if path and list(it.chain(*path))[-1] != end:
            path.append([end])

        return SlidingWindowPath(
            strategy=self.strategy,
            path=path,
            all_paths=all_paths,
            all_graphs=all_graphs,
            success=success,
            success_intermediate=success_intermediate,
            num_segments=num_segments,
        )

    def _interpolate_points(self, p1: Point, p2: Point) -> list[Point]:
        """
        Interpolates points between p1 and p2 if the distance between them exceeds `max_distance`

        Args:
            p1 (Point): Starting point
            p2 (Point): Ending point

        Returns:
            list[Point]: Including p1, interpolated points, and p2
        """
        distance = p1.distance(p2)
        if distance <= self.max_distance:
            return [p1, p2]

        num_segments = int(np.ceil(distance / self.max_distance))
        interpolated_points = []
        for i in range(num_segments + 1):
            x = p1.x + i * (p2.x - p1.x) / num_segments
            y = p1.y + i * (p2.y - p1.y) / num_segments
            interpolated_points.append(Point(x, y))
        return interpolated_points

    def _densify_path(self, path: list[Point]) -> list[Point]:
        """
        Densify the path by interpolating points to ensure no segment exceeds `max_distance`

        Args:
            point_lists (list[Point]): Original path points

        Returns:
            list[Point]: Densified path points
        """
        dense_path = []
        for i in range(len(path) - 1):
            segment_points = self._interpolate_points(path[i], path[i + 1])
            dense_path.extend(segment_points[:-1])  # Avoid duplicating points
        dense_path.append(path[-1])
        return dense_path

    @staticmethod
    def _split_path_into_segments(path: list[Point], max_segment_length: float) -> list[LineString]:
        """
        Split a LineString into segments no longer than max_segment_length

        Args:
            path (list[Point]): Path to split into segments
            max_segment_length (float): Maximum available segment length

        Returns:
            list[LineString]: Segments
        """
        line = LineString(path)
        if len(path) < 2:
            return [line]
        total_length = line.length
        segments = []
        start_dist = 0.0
        while start_dist < total_length:
            end_dist = min(start_dist + max_segment_length, total_length)
            segment = substring(line, start_dist, end_dist)
            segments.append(segment)
            start_dist = end_dist
        return segments

    def _validate_segments(
        self,
        segments: list[LineString],
        time_keys: list[str],
        strtrees: dict[str, STRtree],
    ) -> bool:
        for i, segment in enumerate(segments):
            time_key = time_keys[i]
            tree = strtrees[time_key]
            possible_idxs = tree.query(segment)
            possible_obs = [tree.geometries[i] for i in possible_idxs]
            if not self._is_line_valid(segment, possible_obs):
                return False
        return True

    def _find_shortcut(
        self,
        dense_path: list[Point],
        i: int,
        time_keys: list[str],
        strtrees: dict[str, STRtree],
        num_segments_initial: int,
    ) -> list[Point]:
        for j, _ in reversed(list(enumerate(dense_path))):
            shortcut = [dense_path[i], dense_path[j]]
            path_suggested = dense_path[:i] + shortcut + dense_path[j + 1:]
            segments_suggested = self._split_path_into_segments(
                path=path_suggested,
                max_segment_length=self.velocity_mpm * self.delta_minutes,
            )
            if len(segments_suggested) > num_segments_initial:
                continue
            if self._validate_segments(
                segments=segments_suggested,
                time_keys=time_keys,
                strtrees=strtrees,
            ):
                shortcut_dense = self._densify_path(shortcut)
                return dense_path[:i] + shortcut_dense + dense_path[j + 1:]
        return dense_path

    @staticmethod
    def _linestring_to_points(linestring: LineString) -> list[Point]:
        return [Point(coord) for coord in linestring.coords]

    def _greedy_fine_tuning(
        self,
        path_flat: list[Point],
        time_keys: list[str],
        strtrees: dict[str, STRtree],
        quick: bool = True,
    ) -> tuple[list[Point], int, float]:
        dense_path = self._densify_path(path_flat)
        i = 0
        current_pos = dense_path[i]
        new_dense_path = dense_path.copy()
        tic = time.perf_counter()
        while current_pos != dense_path[-1]:
            num_segments_initial = len(
                self._split_path_into_segments(
                    path=new_dense_path,
                    max_segment_length=self.velocity_mpm * self.delta_minutes,
                )
            )
            new_dense_path = self._find_shortcut(
                dense_path=new_dense_path,
                i=i,
                time_keys=time_keys,
                strtrees=strtrees,
                num_segments_initial=num_segments_initial,
            )
            i += 1
            current_pos = new_dense_path[i]
            if (
                quick
                and len(new_dense_path[:i + 1]) > 1
                and LineString(new_dense_path[:i + 1]).length > self.velocity_mpm * self.delta_minutes  # noqa: E501
            ):
                break
        tuned_line = LineString(new_dense_path)
        tuned_line_simplified = tuned_line.simplify(
            tolerance=self.simplification_tolerance,
            preserve_topology=True,
        )
        return self._linestring_to_points(tuned_line_simplified), i, time.perf_counter() - tic

    @staticmethod
    def _compute_angle(A: Point, B: Point, C: Point) -> float:
        """
        Compute the angle (in degrees) between three shapely Points A, B, C with B as the vertex

        Args:
            A (Point): Shapely Point
            B (Point): Shapely Point – vertex
            C (Point): Shapely Point

        Returns:
            float: Angle in degrees between vectors BA and BC
        """
        try:
            ba_x = A.x - B.x
            ba_y = A.y - B.y

            bc_x = C.x - B.x
            bc_y = C.y - B.y

            dot_product = ba_x * bc_x + ba_y * bc_y
            mag_ba = math.hypot(ba_x, ba_y)
            mag_bc = math.hypot(bc_x, bc_y)

            if mag_ba == 0 or mag_bc == 0:
                raise ValueError("Cannot compute angle with zero-length vector")

            cos_theta = dot_product / (mag_ba * mag_bc)
            cos_theta = max(-1.0, min(1.0, cos_theta))
            angle_rad = math.acos(cos_theta)
            return math.degrees(angle_rad)
        except Exception:
            return 180.0

    def _compute_angles(self, path: list[Point]) -> list[tuple[int, float]]:
        """
        Compute all angles for a given path

        Args:
            path (list[Point]): Path points

        Returns:
            list[tuple[int, float]]: List of angles for each point index (ascending)
        """
        angles = [
            (i, self._compute_angle(path[i - 1], path[i], path[i + 1]))
            for i in range(1, len(path) - 1)
        ]
        return sorted(angles, key=lambda x: x[1])

    @staticmethod
    def _get_unit_direction(p1: Point, p2: Point) -> tuple[float, float]:
        """
        Compute the unit direction vector from point p1 to point p2

        Args:
            p1 (Point): The starting point
            p2 (Point): The ending point

        Returns:
            tuple[float, float]: The unit vector (dx, dy) pointing from p1 to p2
        """
        dx, dy = p2.x - p1.x, p2.y - p1.y
        length = np.hypot(dx, dy)
        if length == 0:
            return 0, 0
        return dx / length, dy / length

    @staticmethod
    def _shift_point(point: Point, direction: tuple[float, float], distance: float) -> Point:
        """
        Shift a point along a unit direction vector by a given distance

        Args:
            point (Point): The original point to be shifted
            direction (tuple[float, float]): A 2D unit direction vector (dx, dy)
            distance (float): Distance to shift along the direction vector

        Returns:
            Point: A new point shifted from the original by the specified distance
        """
        dx, dy = direction
        return Point(point.x + dx * distance, point.y + dy * distance)

    def _replace_point_with_shifted(
        self,
        points: list[Point],
        index: int,
        tolerance: float,
    ) -> list[Point]:
        """
        Replace the point at `index` with:
            - A point `tolerance` meters before it (if distance to previous > `tolerance`)
            - A point `tolerance` meters after it (if distance to next > `tolerance`)
            - If neither condition is met, the point is simply removed

        Args:
            points (list[Point]): List of points
            index (int): Index of the point to replace
            tolerance (float): Distance threshold for inserting shifted points

        Returns:
            List[Point]: Updated list of points with the point at `index` replaced

        Raises:
            ValueError: If `index` is the first or last element in the list
        """
        if index <= 0 or index >= len(points) - 1:
            raise ValueError("Point must not be first or last in the list to compute direction")

        prev_point = points[index - 1]
        next_point = points[index + 1]
        curr_point = points[index]
        new_points = []

        if curr_point.distance(prev_point) > tolerance:
            direction_prev = self._get_unit_direction(prev_point, curr_point)
            before_point = self._shift_point(
                point=curr_point,
                direction=direction_prev,
                distance=-tolerance,
            )
            new_points.append(before_point)

        if curr_point.distance(next_point) > tolerance:
            direction_next = self._get_unit_direction(curr_point, next_point)
            after_point = self._shift_point(
                point=curr_point,
                direction=direction_next,
                distance=tolerance,
            )
            new_points.append(after_point)

        return points[:index] + new_points + points[index + 1:]

    def _smooth_fine_tuning(
        self,
        path_flat: list[Point],
        time_keys: list[str],
        strtrees: dict[str, STRtree],
    ) -> tuple[list[Point], int, float]:
        num_iter = 0
        tic = time.perf_counter()
        while True:
            modified = False
            angles = self._compute_angles(path_flat)
            for i, _ in angles:
                path_smooth = self._replace_point_with_shifted(
                    points=path_flat,
                    index=i,
                    tolerance=self.smooth_tolerance,
                )
                segments = self._split_path_into_segments(
                    path=path_smooth,
                    max_segment_length=self.velocity_mpm * self.delta_minutes,
                )
                if self._validate_segments(
                    segments=segments,
                    time_keys=time_keys,
                    strtrees=strtrees,
                ):
                    length_initial = LineString(path_flat).length
                    path_smooth = [
                        Point(coord)
                        for coord in LineString(path_smooth).simplify(
                            tolerance=self.simplification_tolerance,
                            preserve_topology=True,
                        ).coords
                    ]
                    path_flat = path_smooth
                    modified = True
                    num_iter += 1
                    if length_initial - LineString(path_flat).length < self.delta_length:
                        modified = False
                        self.logger.info(
                            f"Smooth fine-tuning: Length change is less than {self.delta_length} m",
                        )
                    if num_iter == self.max_iter:
                        modified = False
                        self.logger.info(
                            f"Smooth fine-tuning: max iterations ({num_iter}) completed",
                        )
                    break  # Restart from the beginning with the new path
            if not modified:
                break  # Exit loop if no changes made in this iteration
        self.logger.info(f"{num_iter} iterations completed")
        return path_flat, num_iter, time.perf_counter() - tic

    @property
    def fine_tuning_function(self) -> Callable:
        match self.tuning_strategy:
            case "greedy":
                return self._greedy_fine_tuning
            case "smooth":
                return self._smooth_fine_tuning

    def _perform_pathfinding_with_finetuning(
        self,
        current_pos: Point,
        end: Point,
        *,
        current_time_index: int,
        window_size: int,
        time_keys: list[str],
        dict_obstacles: dict[str, gpd.GeoDataFrame],
        G_master: nx.Graph,
        master_vertices: list[Point],
        time_valid_edges: dict[str, list[tuple[int, int, float]]],
    ):
        path = []
        all_paths_initial = []
        all_paths_fine_tuned = []
        all_paths_initial_raw = []
        fine_tuning_iters = []
        fine_tuning_times = []
        success = True
        success_intermediate = True
        length = len(str(len(time_keys)))
        strtrees = {
            time_key: STRtree(list(dict_obstacles[time_key][self.strategy]))
            for time_key in time_keys
        }
        self.logger.info("Start pathfinding with fine-tuning:")

        while current_pos != end and current_time_index < len(time_keys) - window_size:
            # Find initial path
            path_initial = self._perform_pathfinding(
                current_pos=current_pos,
                end=end,
                current_time_index=current_time_index,
                window_size=window_size,
                time_keys=time_keys,
                dict_obstacles=dict_obstacles,
                G_master=G_master,
                master_vertices=master_vertices,
                time_valid_edges=time_valid_edges,
            )
            all_paths_initial_raw.append(path_initial)
            success_intermediate = success_intermediate and path_initial.success_intermediate
            if not path_initial.success:
                success = False
                self.logger.info("Unable to fine-tune non-success path")
                break

            # Fine-tune initial path according to selected strategy
            path_flat = list(it.chain(*path_initial.path))
            path_flat = [key for key, group in it.groupby(path_flat)]
            path_fine_tuned, num_iters, fine_tuning_time = self.fine_tuning_function(
                path_flat=path_flat,
                time_keys=time_keys[current_time_index:],
                strtrees=strtrees,
            )

            # Add data
            all_paths_initial.append(path_flat)
            all_paths_fine_tuned.append(path_fine_tuned)
            fine_tuning_iters.append(num_iters)
            fine_tuning_times.append(fine_tuning_time)

            # Update current position
            path_line = LineString(path_fine_tuned)
            path_within_window = substring(path_line, 0, self.velocity_mpm * self.delta_minutes)
            points_within_window = [Point(coord) for coord in path_within_window.coords]
            current_pos = points_within_window[-1]
            current_time_index += 1

            # Extend the valid portion of the path
            path.append(points_within_window)
            self.logger.info(
                f"\t{current_time_index:<{length}}/{len(time_keys) - window_size}: Path fine-tuned"
            )

        # Add the end point if not reached yet
        num_segments = current_time_index - 1
        if path and list(it.chain(*path))[-1] != end:
            path.append([end])

        return FineTunedPath(
            strategy=self.strategy,
            path=path,
            all_paths_initial=all_paths_initial,
            all_paths_fine_tuned=all_paths_fine_tuned,
            all_paths_initial_raw=all_paths_initial_raw,
            fine_tuning_iters=fine_tuning_iters,
            fine_tuning_times=fine_tuning_times,
            success=success,
            success_intermediate=success_intermediate,
            num_segments=num_segments,
        )

    def create_master_graph(
        self,
        time_keys: list[str],
        dict_obstacles: dict[str, gpd.GeoDataFrame],
    ) -> tuple[nx.Graph, dict[str, list[tuple[int, int, float]]]]:
        """
        Create a master graph from `dict_obstacles`

        Args:
            time_keys (list[str]): All available time keys sorted chronologically
            dict_obstacles (dict[str, gpd.GeoDataFrame]): Dictionary of obstacles GeoDataFrames

        Returns:
            tuple:
                - nx.Graph: Master graph with all possible valid edges
                - dict[str, list[tuple[int, int, float]]]: List of valid edges for each time key
        """
        master_obstacles = [
            dict_obstacles[time_key][self.strategy]
            for time_key in time_keys
        ]
        master_obstacles_flat = list(it.chain(*master_obstacles))
        all_vertices = sorted(
            set(
                it.chain(
                    *[self._collect_vertices([obstacle]) for obstacle in master_obstacles_flat]
                )
            ),
            key=lambda p: (p.x, p.y),
        )
        return self._build_master_graph(
            vertices=all_vertices,
            time_keys=time_keys,
            dict_obstacles=dict_obstacles,
        )

    def sliding_window_pathfinding(
        self,
        start: Point,
        end: Point,
        *,
        window_size: int,
        time_keys: list[str],
        dict_obstacles: dict[str, gpd.GeoDataFrame],
        G_master: nx.Graph,
        time_valid_edges: dict[str, list[tuple[int, int, float]]],
        with_fine_tuning: bool = False,
    ) -> SlidingWindowPath | FineTunedPath:
        """
        Create a path from start to end with dynamic obstacles

        Args:
            start (Point): Starting point
            end (Point): End point
            window_size (int): Sliding window size
            time_keys (list[str]): All available time keys sorted chronologically
            dict_obstacles (dict[str, gpd.GeoDataFrame]): Dictionary of obstacles GeoDataFrames
            G_master (nx.Graph): Master graph with all possible valid edges
            time_valid_edges (dict[str, list[tuple[int, int, float]]]): List of valid edges for each time key
            with_fine_tuning (bool): Weather to apply global fine-tuning

        Returns:
            SlidingWindowPath:
                - path (list[list[Point]]): Computed path divided by time keys
                - all_paths (list[list[Point]]): Full precomputed paths, starting from each time key
                - all_graphs (list[nx.Graph]): Visibility graphs for each time key
                - success (bool): Was pathfinding successful
                - success_intermediate (bool): Whether any start was inside an obstacle
                - num_segments (int): Number of segments in final path
            FineTunedPath:
                - path (list[list[Point]]): Computed fine-tuned path divided by time keys
                - all_paths_initial (list[list[Point]]): All precomputed paths
                - all_paths_fine_tuned (list[list[Point]]): All fine-tuned paths
                - all_paths_initial_raw (list[SlidingWindowPath]): All precomputed SlidingWindowPath
                - fine_tuning_iters (list[int]): Number of iterations required for each fine-tuning
                - fine_tuning_times (list[float]): Time required for each fine-tuning
                - success (bool): Was pathfinding successful
                - success_intermediate (bool): Whether any start was inside an obstacle
                - num_segments (int): Number of segments in final fine-tuned path
        """
        # Extract master graph vertices
        master_vertices = [data["point"] for _, data in G_master.nodes(data=True)]
        current_time_index = 0

        # Perform pathfinding with or without global fine-tuning
        function_pathfinding = self._perform_pathfinding
        if with_fine_tuning:
            function_pathfinding = self._perform_pathfinding_with_finetuning
        return function_pathfinding(
            current_pos=start,
            end=end,
            current_time_index=current_time_index,
            window_size=window_size,
            time_keys=time_keys,
            dict_obstacles=dict_obstacles,
            G_master=G_master,
            master_vertices=master_vertices,
            time_valid_edges=time_valid_edges,
        )


def pickle_to_df(pkl_file_path: Path) -> pd.DataFrame:
    """
    Load `result_dict` pickle data and create a result DataFrame

    Args:
        pkl_file_path (Path): Path to `result_dict` pickle file

    Returns:
        pd.DataFrame: Result data for further analysis
    """
    with open(pkl_file_path, "rb") as file_in:
        result_dicts: dict[str, dict[int, SlidingWindowPath | FineTunedPath]] = pickle.load(file_in)
    current_date = str(pkl_file_path).split("/")[-1].split(".")[0]
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "strategy": [
                        result_dicts[start_point][w_size].strategy
                        for w_size in result_dicts[start_point].keys()
                    ],
                    "timestamp": [current_date] * len(result_dicts[start_point]),
                    "window_size": list(result_dicts[start_point].keys()),
                    "start_point": [start_point] * len(result_dicts[start_point]),
                    "path": [
                        LineString(list(it.chain(*result_dicts[start_point][w_size].path)))
                        for w_size in result_dicts[start_point].keys()
                    ],
                    "length": [
                        LineString(
                            list(it.chain(*result_dicts[start_point][w_size].path))
                        ).length
                        for w_size in result_dicts[start_point].keys()
                    ],
                    "success": [
                        result_dicts[start_point][w_size].success
                        for w_size in result_dicts[start_point].keys()
                    ],
                    "success_intermediate": [
                        result_dicts[start_point][w_size].success_intermediate
                        for w_size in result_dicts[start_point].keys()
                    ],
                },
            )
            for start_point in result_dicts.keys()
        ],
        ignore_index=True,
    )


def process_data(
    dynamic_avoider: DynamicAvoider,
    *,
    timestamps: list[datetime],
    ab_points: list[tuple[Point, Point]],
    window_sizes: list[int],
    with_fine_tuning: bool = True,
    with_backward_pathfinding: bool = False,
) -> None:
    """
    Initial / fine-tuning (with or w/o backward) pathfinding for given `timestamps`, `ab_points` and `window_sizes`
    with saving intermediate (pickle) results to "result/dynamic_avoider_master/" or "result/dynamic_avoider_{tuning_strategy}/"
    and final results to "result/dynamic_avoider_master.csv" or "result/dynamic_avoider_{tuning_strategy}.csv"

    Args:
        dynamic_avoider (DynamicAvoider): DynamicAvoider object tp use for pathfinding
        timestamps (list[datetime]): Initial timestampt to process
        ab_points (list[tuple[Point, Point]]): Start and end points
        window_sizes (list[int]): Sliding window sizes
        with_fine_tuning (bool): Fine-tuning (True) or initial (False) pathfinding
        with_backward_pathfinding (bool): Perform B -> A pathfinding (True) or not (False)
    """
    length = len(str(len(timestamps)))
    type_pathfinding = "master"
    if with_fine_tuning:
        type_pathfinding = dynamic_avoider.tuning_strategy
    dir_path_result = RESULT_PATH / f"dynamic_avoider_{type_pathfinding}"
    if not dir_path_result.exists():
        dir_path_result.mkdir(parents=True, exist_ok=True)

    for i, current_date in enumerate(timestamps):
        A, B = ab_points[i]
        file_name = "_".join(re.split(r"[- :]", str(current_date)))
        if f"{file_name}.pkl" in os.listdir(dir_path_result):
            dynamic_avoider.logger.info(
                f"{i + 1:<{length}}/{len(timestamps)}: "
                f"File {file_name}.pkl exists. Skipping {current_date}"
            )
            continue

        # Get time keys and corresponding obstacles, build master graph
        dynamic_avoider.logger.info(
            f"START BUILDING MASTER GRAPH FOR {current_date}\n"
        )
        time_keys = dynamic_avoider.extract_time_keys(DATA_PATH / file_name)
        dict_obstacles = dynamic_avoider.collect_obstacles(DATA_PATH / file_name, time_keys)
        G_master, time_valid_edges = dynamic_avoider.create_master_graph(
            time_keys=time_keys,
            dict_obstacles=dict_obstacles,
        )
        dynamic_avoider.logger.info(
            f"\nMASTER GRAPH FOR {current_date} READY\n"
        )

        result_a: dict[int, SlidingWindowPath | FineTunedPath] = {}
        result_b: dict[int, SlidingWindowPath | FineTunedPath] = {}
        for window_size in window_sizes:
            if with_fine_tuning:
                dynamic_avoider.logger.info(
                    f"START PATHFINDING WITH {type_pathfinding.upper()} "
                    f"FINE-TUNING FOR {current_date}\n"
                )
            else:
                dynamic_avoider.logger.info(f"START PATHFINDING FOR {current_date}\n")
            dynamic_avoider.logger.info(f"====== Window size {window_size} ======")

            # Sliding window pathfinding A -> B
            dynamic_avoider.logger.info(" " * 7 + "(A)  ->  (B)")
            result_a[window_size] = dynamic_avoider.sliding_window_pathfinding(
                start=A,
                end=B,
                window_size=window_size,
                time_keys=time_keys,
                dict_obstacles=dict_obstacles,
                G_master=G_master,
                time_valid_edges=time_valid_edges,
                with_fine_tuning=with_fine_tuning,
            )
            dynamic_avoider.logger.info(
                f"Success: {result_a[window_size].success}, "
                f"Success Inter: {result_a[window_size].success_intermediate}\n",
            )

            # Sliding window pathfinding B -> A
            if with_backward_pathfinding:
                dynamic_avoider.logger.info(" " * 7 + "(B)  ->  (A)")
                result_b[window_size] = dynamic_avoider.sliding_window_pathfinding(
                    start=B,
                    end=A,
                    window_size=window_size,
                    time_keys=time_keys,
                    dict_obstacles=dict_obstacles,
                    G_master=G_master,
                    time_valid_edges=time_valid_edges,
                    with_fine_tuning=with_fine_tuning,
                )
                dynamic_avoider.logger.info(
                    f"Success: {result_b[window_size].success}, "
                    f"Success Inter: {result_b[window_size].success_intermediate}\n",
                )

        result = {
            "A": result_a,
            "B": result_b,
        }
        with open(dir_path_result / f"{file_name}.pkl", "wb") as file_out:
            pickle.dump(result, file_out)
        dynamic_avoider.logger.info(
            f"{i + 1:<{length}}/{len(timestamps)}: {current_date} saved to "
            f"'dynamic_avoider_{type_pathfinding}/{file_name}.pkl'\n\n\n",
        )

    # Save data to CSV
    df_result = pd.concat(
        [
            pickle_to_df(dir_path_result / file_name)
            for file_name in os.listdir(dir_path_result)
        ],
        ignore_index=True,
    )
    df_result.to_csv(RESULT_PATH / f"dynamic_avoider_{type_pathfinding}.csv", index=None)
    dynamic_avoider.logger.info(
        f"Final DataFrame saved successfully to 'results/dynamic_avoider_{type_pathfinding}.csv'"
    )


if __name__ == "__main__":
    with open(TIMESTAMPS_PATH, "rb") as file_in:
        timestamps = pickle.load(file_in)
    with open(AB_POINTS_PATH, "rb") as file_in:
        ab_points = pickle.load(file_in)

    dynamic_avoider = DynamicAvoider(
        crs=PROJECTED_CRS,
        velocity_kmh=VELOCITY_KMH,
        delta_minutes=DELTA_MINUTES,
        buffer=BUFFER,
        tolerance=TOLERANCE,
        k_neighbors=K_NEIGHBORS,
        max_distance=MAX_DISTANCE,
        simplification_tolerance=SIMPLIFICATION_TOLERANCE,
        smooth_tolerance=SMOOTH_TOLERANCE,
        max_iter=MAX_ITER,
        delta_length=DELTA_LENGTH,
        strategy="concave",
        tuning_strategy="greedy",
    )

    # Initial pathfinding
    process_data(
        dynamic_avoider=dynamic_avoider,
        timestamps=timestamps,
        ab_points=ab_points,
        window_sizes=WINDOW_SIZE_INITIAL,
        with_fine_tuning=False,
        with_backward_pathfinding=True,
    )

    # With greedy tuning
    process_data(
        dynamic_avoider=dynamic_avoider,
        timestamps=timestamps,
        ab_points=ab_points,
        window_sizes=WINDOW_SIZE_INITIAL,
        with_fine_tuning=True,
        with_backward_pathfinding=True,
    )

    # With smooth tuning
    dynamic_avoider.tuning_strategy = "smooth"
    process_data(
        dynamic_avoider=dynamic_avoider,
        timestamps=timestamps,
        ab_points=ab_points,
        window_sizes=WINDOW_SIZE_INITIAL,
        with_fine_tuning=True,
        with_backward_pathfinding=True,
    )
