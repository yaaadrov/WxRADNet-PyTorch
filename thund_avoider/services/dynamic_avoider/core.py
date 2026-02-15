import itertools as it
import logging
from typing import Literal

import geopandas as gpd
import networkx as nx
from shapely import STRtree
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import substring

from thund_avoider.schemas.dynamic_avoider import SlidingWindowPath, FineTunedPath
from thund_avoider.services.dynamic_avoider.data_loader import DataLoader
from thund_avoider.services.dynamic_avoider.fine_tuner import FineTuner
from thund_avoider.services.utils import subgraph_for_time_keys
from thund_avoider.services.dynamic_avoider.graph_builder import GraphBuilder
from thund_avoider.settings import DynamicAvoiderConfig


class DynamicAvoider:
    """
    Main orchestrator for dynamic obstacle avoidance pathfinding.

    This class coordinates the graph building, pathfinding, and fine-tuning
    components to compute obstacle-avoiding paths through dynamic environments.
    """

    def __init__(self, config: DynamicAvoiderConfig) -> None:
        # Initialize components with their configs
        self._graph_builder = GraphBuilder(config.graph_builder_config)
        self._fine_tuner = FineTuner(config.fine_tuner_config)
        self._data_loader = DataLoader(config.data_loader_config)

        # Store commonly used values for convenience
        self.strategy = config.graph_builder_config.strategy
        self.velocity_mpm = config.fine_tuner_config.velocity_kmh * 1000 / 60
        self.delta_minutes = config.fine_tuner_config.delta_minutes

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def extract_time_keys(dir_path) -> list[str]:
        """Extract sorted time keys from CSV filenames in a directory."""
        return DataLoader.extract_time_keys(dir_path)

    def collect_obstacles(
        self,
        directory_path,
        time_keys: list[str],
    ) -> dict[str, gpd.GeoDataFrame]:
        """Collect GeoDataFrames with obstacles data into a dictionary."""
        return self._data_loader.collect_obstacles(directory_path, time_keys)

    def _find_shortest_path(
        self,
        G: nx.Graph,
        source_node: str | int = "start",
        target_node: str | int = "end",
    ) -> tuple[list[Point], list[int | str]] | None:
        """Find the shortest path in graph."""
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

    def _tune_path(
        self,
        G_sub: nx.Graph,
        path_idx: list[int | str],
        window_obstacles: list[Polygon],
    ) -> tuple[list[Point], list[int | str]] | None:
        """
        Tune initially found path using visibility graph.

        Args:
            G_sub (nx.Graph): Subgraph for a given window.
            path_idx (list[int | str]): List of initial shortest path indices.
            window_obstacles (list[Point]): List of obstacle geometries for a given window.

        Returns:
            tuple: Tuned path points and indices, or None if no path exists.
        """
        G_tune = nx.Graph()
        G_tune.add_nodes_from((node, G_sub.nodes[node]) for node in path_idx)
        G_tune = self._graph_builder.build_visibility_graph(G_tune, window_obstacles)
        return self._find_shortest_path(G_tune)

    def _check_start_end_point(
        self,
        point: Point,
        point_type: Literal["start", "end"],
        flat_obstacles: list[Polygon],
        vertices: list[Point],
    ):
        """
        Check if start/end point is inside an obstacle and find the closest valid vertex.

        Args:
            point (Point): Point to check.
            point_type (Literal["start", "end"]): Type of point.
            flat_obstacles (list[Polygon]): List of obstacle polygons.
            vertices (list[Point]): List of available vertices.

        Returns:
            tuple: Updated point and success flag.
        """
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

    def _densify_path(self, path: list[Point]) -> list[Point]:
        """Densify the path by interpolating points to ensure no segment exceeds max_distance."""
        return self._fine_tuner.densify_path(path)

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
        Perform one step of the sliding window pathfinding.

        Required by MaskedDynamicAvoider for inheritance compatibility.

        Args:
            current_pos (Point): Aircraft current position.
            end (Point): Final point.
            sliding_window_result (SlidingWindowPath): Object to update results.
            current_time_index (int): Index of current timestamp.
            window_size (int): Size of pathfinding window.
            time_keys (list[str]): All available time keys sorted chronologically.
            available_obstacles_dict: Known obstacles.
            G_master (nx.Graph): Master graph with all possible vertices.
            master_vertices (list[Point]): List of master graph's vertices.
            time_valid_edges: List of valid edges for each time key.

        Returns:
            tuple: Updated position, time index, and whether to continue.
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
        G_sub = subgraph_for_time_keys(
            G_master=G_master,
            time_keys=window_time_keys,
            time_valid_edges=time_valid_edges,
        )
        G_sub = self._graph_builder.add_points_to_subgraph(
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
        shortest_path_tuned = self._tune_path(G_sub, path_idx, window_obstacles_flat)
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
        path_within_window = substring(LineString(path_tuned), 0, self.velocity_mpm * self.delta_minutes)
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
        """
        Perform one step of the pathfinding with fine-tuning.

        Required by MaskedDynamicAvoider for inheritance compatibility.

        Args:
            current_pos (Point): Aircraft current position.
            end (Point): Final point.
            fine_tuning_result (FineTunedPath): Object to update results.
            current_time_index (int): Index of current timestamp.
            window_size (int): Size of pathfinding window.
            time_keys (list[str]): All available time keys sorted chronologically.
            available_obstacles_dict: Known obstacles.
            G_master (nx.Graph): Master graph with all possible vertices.
            master_vertices (list[Point]): List of master graph's vertices.
            time_valid_edges: List of valid edges for each time key.
            strtrees (dict[str, STRtree]): STRTrees for each time key.

        Returns:
            tuple: Updated position, time index, and whether to continue.
        """
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
        path_fine_tuned, num_iters, fine_tuning_time = self._fine_tuner.fine_tuning_function(
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
        """
        Perform sliding window pathfinding.

        Args:
            current_pos (Point): Starting position.
            end (Point): End position.
            current_time_index (int): Index of current timestamp.
            window_size (int): Size of pathfinding window.
            time_keys (list[str]): All available time keys.
            dict_obstacles: Obstacle geometries for each time key.
            G_master (nx.Graph): Master graph.
            master_vertices (list[Point]): Master graph vertices.
            time_valid_edges: Valid edges for each time key.

        Returns:
            SlidingWindowPath: Pathfinding result.
        """
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
        """
        Perform pathfinding with fine-tuning.

        Args:
            current_pos (Point): Starting position.
            end (Point): End position.
            current_time_index (int): Index of current timestamp.
            window_size (int): Size of pathfinding window.
            time_keys (list[str]): All available time keys.
            dict_obstacles: Obstacle geometries for each time key.
            G_master (nx.Graph): Master graph.
            master_vertices (list[Point]): Master graph vertices.
            time_valid_edges: Valid edges for each time key.

        Returns:
            FineTunedPath: Pathfinding result with fine-tuning.
        """
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
        Create a master graph from `dict_obstacles`.

        Args:
            time_keys (list[str]): All available time keys sorted chronologically.
            dict_obstacles: Dictionary of obstacles GeoDataFrames.
            prohibited_zone (Polygon): Prohibited zone where graph's nodes are not allowed.
            previous_path (list[Point]): Previous path points to include in the graph.

        Returns:
            tuple: Master graph with all possible valid edges and valid edges for each time key.
        """
        return self._graph_builder.create_master_graph(
            time_keys=time_keys,
            dict_obstacles=dict_obstacles,
            prohibited_zone=prohibited_zone,
            previous_path=previous_path,
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
        Create a path from start to end with dynamic obstacles.

        Args:
            start (Point): Starting point.
            end (Point): End point.
            window_size (int): Sliding window size.
            time_keys (list[str]): All available time keys sorted chronologically.
            dict_obstacles: Dictionary of obstacles GeoDataFrames.
            G_master (nx.Graph): Master graph with all possible valid edges.
            time_valid_edges: List of valid edges for each time key.
            with_fine_tuning (bool): Whether to apply global fine-tuning.

        Returns:
            SlidingWindowPath or FineTunedPath: Pathfinding result.
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
