import itertools as it
import logging
import math
import os
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import CRS
from scipy.spatial import Delaunay, KDTree
from shapely import STRtree
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import substring, unary_union
from shapely.prepared import prep

from thund_avoider.schemas.dynamic_avoider import SlidingWindowPath, FineTunedPath
from thund_avoider.settings import DynamicAvoiderConfig


class DynamicAvoider:
    def __init__(self, config: DynamicAvoiderConfig) -> None:
        """
        Initialize `DynamicAvoider` class

        Args:
            config (DynamicAvoiderConfig): Avoider Configuration:
                - crs (CRS): Coordinate Reference System
                - velocity_kmh (float): Velocity in km/h
                - delta_minutes (float): Forecast frequency in minutes
                - buffer (float): Buffer distance for geometry simplification
                - tolerance (float): Simplification tolerance for geometry
                - k_neighbors (int): Number of neighbors for master graph
                - max_distance (float): Split each segment into several subsegments for greedy fine-tuning
                - simplification_tolerance (float): Tolerance to simplify paths after densifying
                - smooth_tolerance (float): Tolerance for smoothing fine-tuning
                - max_iter (int): Maximum number of iterations for smooth fine-tuning
                - delta_length (float): Smooth fine-tuning length sensitivity
                - strategy (Literal["concave", "convex"]): Path-finding strategy to apply
                - tuning_strategy (Literal["greedy", "smooth"]): Fine-tuning strategy to apply
        """
        self.crs = CRS(config.crs)
        self.velocity_mpm = config.velocity_kmh * 1000 / 60  # Velocity in meters/minute
        self.delta_minutes = config.delta_minutes
        self.buffer = config.buffer
        self.tolerance = config.tolerance
        self.k_neighbors = config.k_neighbors
        self.max_distance = config.max_distance
        self.simplification_tolerance = config.simplification_tolerance
        self.smooth_tolerance = config.smooth_tolerance
        self.max_iter = config.max_iter
        self.delta_length = config.delta_length
        self.strategy = config.strategy
        self.tuning_strategy = config.tuning_strategy

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
            FileNotFoundError: If the directory does not exist or is empty
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
        dict_obstacles: dict[str, gpd.GeoDataFrame] | dict[str, dict[str, list[Polygon]]],
    ) -> tuple[nx.Graph, dict[str, list[tuple[int, int, float]]]]:
        """
        Build master graph for given vertices from all available obstacles

        Args:
            vertices (list[Point]): List of all available vertex Point objects
            time_keys (list[str]): List of master time keys
            dict_obstacles (dict[str, gpd.GeoDataFrame] | dict[str, dict[str, list[Polygon]]]): Obstacle geometries for each time key

        Returns:
            tuple:
                - nx.Graph: Master graph with all possible valid edges
                - dict[str, list[tuple[int, int, float]]]: List of valid edges for each time key
        """
        self.logger.info("Building master graph:")
        strtrees = {
            time_key: STRtree(list(dict_obstacles[time_key][self.strategy]))
            for time_key in time_keys
        }
        G_master = nx.Graph()
        candidates = set()
        centroids = self._simplify_vertices(vertices)

        if centroids:
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
        G = G_sub.copy()
        vertices = [data["point"] for _, data in G_master.nodes(data=True)]
        coords = [(p.x, p.y) for p in vertices]
        strtree = STRtree(obstacles)
        kd_tree = None

        has_vertices = len(coords) > 0
        if has_vertices:
            kd_tree = KDTree(coords)

        for point_name, point in points:
            G.add_node(point_name, pos=(point.x, point.y), point=point)
            if has_vertices:
                _, idxs = kd_tree.query([point.x, point.y], k=min(self.k_neighbors, len(coords)))
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

    def _pathfinding_iter(
        self,
        current_pos: Point,
        end: Point,
        *,
        sliding_window_result: SlidingWindowPath,
        current_time_index: int,
        window_size: int,
        time_keys: list[str],
        available_obstacles_dict: dict[str, gpd.GeoDataFrame] | dict[str, dict[str, list[Polygon]]],
        G_master: nx.Graph,
        master_vertices: list[Point],
        time_valid_edges: dict[str, list[tuple[int, int, float]]],
    ) -> tuple[Point, int, bool]:
        """
        Perform one step of the sliding window pathfinding

        Args:
            current_pos (Point): Aircraft current position
            end (Point): Final point
            sliding_window_result (SlidingWindowPath): Object to update results
            current_time_index (int): Index of current timestamp
            window_size (int): Size of pathfinding window
            time_keys (list[str]): All available time keys sorted chronologically
            available_obstacles_dict (dict[str, gpd.GeoDataFrame] | dict[str, dict[str, list[Polygon]]]): Known obstacles
            G_master (nx.Graph): Master graph with all possible vertices
            master_vertices (list[Point]): List of master graph's vertices
            time_valid_edges (dict[str, list[tuple[int, int, float]]]): List of valid edges for each time key

        Returns:
            Point: Updated position
            int: Updated time index
            bool: Whether to continue the loop
        """
        # Extract time keys for the current window
        window_time_keys = time_keys[current_time_index:current_time_index + window_size]
        if not window_time_keys:
            self.logger.info("No more available time keys")
            return current_pos, current_time_index, False

        # Collect obstacles for current window
        try:
            window_obstacles = [
                available_obstacles_dict[time_key][self.strategy]
                for time_key in window_time_keys
            ]
            window_obstacles_flat = list(it.chain(*window_obstacles))
        except KeyError:
            return current_pos, current_time_index, False

        # Check start and end points
        current_pos_updated, success_start = self._check_start_end_point(
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
        sliding_window_result.success_intermediate &= success_start
        if current_pos != current_pos_updated:
            sliding_window_result.moved_original_points_mapping[current_pos_updated] = current_pos

        # Build subgraph and add start and end points
        G_sub = self._subgraph_for_time_keys(
            G_master=G_master,
            time_keys=window_time_keys,
            time_valid_edges=time_valid_edges,
        )
        G_sub = self._add_points_to_subgraph(
            points=[("start", current_pos_updated), ("end", current_end)],
            G_master=G_master,
            G_sub=G_sub,
            obstacles=window_obstacles_flat,
        )

        # Find the shortest path
        shortest_path = self._find_shortest_path(G_sub)
        if shortest_path is None:
            sliding_window_result.success = False
            self.logger.info(
                f"No path found for time window at {time_keys[current_time_index]}"
            )
            return current_pos_updated, current_time_index, False
        _, path_idx = shortest_path

        # Tune shortest path
        shortest_path_tuned = self._tune_path(
            G_sub, path_idx, window_obstacles_flat
            )
        if shortest_path_tuned is None:
            sliding_window_result.success = False
            self.logger.info(
                f"Unable to tune path for time window at {time_keys[current_time_index]}"
            )
            return current_pos_updated, current_time_index, False
        path_tuned, _ = shortest_path_tuned

        # Add data
        sliding_window_result.all_paths.append(path_tuned)
        sliding_window_result.all_graphs.append(G_sub)

        # Update current position
        path_within_window = substring(LineString(path_tuned), 0,self.velocity_mpm * self.delta_minutes)
        points_within_window = [Point(coord) for coord in path_within_window.coords]
        new_pos = points_within_window[-1]
        new_time_index = current_time_index + 1

        # Extend the valid portion of the path
        sliding_window_result.path.append(points_within_window)
        length = len(str(len(time_keys)))
        self.logger.info(
            f"\t{current_time_index + 1:<{length}}/{len(time_keys) - window_size}: Path created, "
            f"Num edges: {len(G_sub.edges)}",
        )
        return new_pos, new_time_index, True

    def _perform_pathfinding(
        self,
        current_pos: Point,
        end: Point,
        *,
        current_time_index: int,
        window_size: int,
        time_keys: list[str],
        dict_obstacles: dict[str, gpd.GeoDataFrame] | dict[str, dict[str, list[Polygon]]],
        G_master: nx.Graph,
        master_vertices: list[Point],
        time_valid_edges: dict[str, list[tuple[int, int, float]]],
    ) -> SlidingWindowPath:
        result = SlidingWindowPath(strategy=self.strategy)
        self.logger.info("Start pathfinding:")

        while current_pos != end and current_time_index < len(time_keys) - window_size:
            current_pos, current_time_index, should_continue = self._pathfinding_iter(
                current_pos=current_pos,
                end=end,
                sliding_window_result=result,
                current_time_index=current_time_index,
                window_size=window_size,
                time_keys=time_keys,
                available_obstacles_dict=dict_obstacles,
                G_master=G_master,
                master_vertices=master_vertices,
                time_valid_edges=time_valid_edges,
            )
            if not should_continue:
                break

        # Add the end point if not reached yet
        result.num_segments = current_time_index - 1
        if result.path and list(it.chain(*result.path))[-1] != end:
            result.path.append([end])

        return result

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
            # Skip validation if no obstacle data available for this time_key
            if time_key not in strtrees:
                continue
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

    def _pathfinding_with_finetuning_iter(
        self,
        current_pos: Point,
        end: Point,
        *,
        fine_tuning_result: FineTunedPath,
        current_time_index: int,
        window_size: int,
        time_keys: list[str],
        available_obstacles_dict: dict[str, gpd.GeoDataFrame] | dict[str, dict[str, list[Polygon]]],
        G_master: nx.Graph,
        master_vertices: list[Point],
        time_valid_edges: dict[str, list[tuple[int, int, float]]],
        strtrees: dict[str, STRtree],
    ) -> tuple[Point, int, bool]:
        path_initial = self._perform_pathfinding(
            current_pos=current_pos,
            end=end,
            current_time_index=current_time_index,
            window_size=window_size,
            time_keys=time_keys,
            dict_obstacles=available_obstacles_dict,
            G_master=G_master,
            master_vertices=master_vertices,
            time_valid_edges=time_valid_edges,
        )
        fine_tuning_result.all_paths_initial_raw.append(path_initial)
        fine_tuning_result.success_intermediate &= path_initial.success_intermediate
        if not path_initial.success:
            fine_tuning_result.success = False
            self.logger.info("Unable to fine-tune non-success path")
            return current_pos, current_time_index, False

        # Fine-tune initial path according to selected strategy
        path_flat = list(it.chain(*path_initial.path))
        path_flat = [key for key, group in it.groupby(path_flat)]
        path_fine_tuned, num_iters, fine_tuning_time = self.fine_tuning_function(
            path_flat=path_flat,
            time_keys=time_keys[current_time_index:],
            strtrees=strtrees,
        )

        # Add data
        fine_tuning_result.all_paths_initial.append(path_flat)
        fine_tuning_result.all_paths_fine_tuned.append(path_fine_tuned)
        fine_tuning_result.fine_tuning_iters.append(num_iters)
        fine_tuning_result.fine_tuning_times.append(fine_tuning_time)

        # Update current position
        path_within_window = substring(LineString(path_fine_tuned), 0, self.velocity_mpm * self.delta_minutes)
        points_within_window = [Point(coord) for coord in path_within_window.coords]
        new_pos = points_within_window[-1]
        new_time_index = current_time_index + 1

        # Extend the valid portion of the path
        fine_tuning_result.path.append(points_within_window)
        length = len(str(len(time_keys)))
        self.logger.info(
            f"\t{current_time_index + 1:<{length}}/{len(time_keys) - window_size}: Path fine-tuned"
        )
        return new_pos, new_time_index, True

    def _perform_pathfinding_with_finetuning(
        self,
        current_pos: Point,
        end: Point,
        *,
        current_time_index: int,
        window_size: int,
        time_keys: list[str],
        dict_obstacles: dict[str, gpd.GeoDataFrame] | dict[str, dict[str, list[Polygon]]],
        G_master: nx.Graph,
        master_vertices: list[Point],
        time_valid_edges: dict[str, list[tuple[int, int, float]]],
    ) -> FineTunedPath:
        result = FineTunedPath(strategy=self.strategy)
        strtrees = {
            time_key: STRtree(list(dict_obstacles[time_key][self.strategy]))
            for time_key in time_keys
        }
        self.logger.info("Start pathfinding with fine-tuning:")

        while current_pos != end and current_time_index < len(time_keys) - window_size:
            current_pos, current_time_index, should_continue = self._pathfinding_with_finetuning_iter(
                current_pos=current_pos,
                end=end,
                fine_tuning_result=result,
                current_time_index=current_time_index,
                window_size=window_size,
                time_keys=time_keys,
                available_obstacles_dict=dict_obstacles,
                G_master=G_master,
                master_vertices=master_vertices,
                time_valid_edges=time_valid_edges,
                strtrees=strtrees,
            )
            if not should_continue:
                break

        # Add the end point if not reached yet
        result.num_segments = current_time_index - 1
        if result.path and list(it.chain(*result.path))[-1] != end:
            result.path.append([end])

        return result

    def create_master_graph(
        self,
        time_keys: list[str],
        dict_obstacles: dict[str, gpd.GeoDataFrame] | dict[str, dict[str, list[Polygon]]],
        prohibited_zone: Polygon = Polygon(),
        previous_path: list[Point] = [],
    ) -> tuple[nx.Graph, dict[str, list[tuple[int, int, float]]]]:
        """
        Create a master graph from `dict_obstacles`

        Args:
            time_keys (list[str]): All available time keys sorted chronologically
            dict_obstacles (dict[str, gpd.GeoDataFrame] | dict[str, dict[str, list[Polygon]]]): Dictionary of obstacles GeoDataFrames
            prohibited_zone (Polygon): Prohibited zone where graph's nodes are not allowed

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
        raw_vertices = set(
            it.chain(
                *[self._collect_vertices([obstacle]) for obstacle in master_obstacles_flat]
            )
        )
        all_vertices = list(raw_vertices)
        if not prohibited_zone.is_empty:
            prepared_zone = prep(prohibited_zone)
            all_vertices = [p for p in raw_vertices if not prepared_zone.intersects(p)]
        all_vertices.extend(previous_path)
        all_vertices.sort(key=lambda p: (p.x, p.y))
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
        dict_obstacles: dict[str, gpd.GeoDataFrame] | dict[str, dict[str, list[Polygon]]],
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
            dict_obstacles (dict[str, gpd.GeoDataFrame] | dict[str, dict[str, list[Polygon]]]): Dictionary of obstacles GeoDataFrames
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
