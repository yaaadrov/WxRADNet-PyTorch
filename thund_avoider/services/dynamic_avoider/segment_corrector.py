import logging
from typing import Literal

import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, Polygon

from thund_avoider.services.dynamic_avoider.graph_builder import GraphBuilder
from thund_avoider.services.utils import subgraph_for_time_keys
from thund_avoider.settings import GraphBuilderConfig


class SegmentCorrector:
    """
    Validates and corrects path segments that end inside obstacles.

    This class provides functionality to detect when a path segment's endpoint
    falls inside an obstacle for the next time key, and corrects the segment
    by finding an escape route through a local visibility graph.
    """

    def __init__(
        self,
        strategy: Literal["convex", "concave"],
        buffer_distance: float = 10_000,
    ) -> None:
        """
        Initialize the SegmentCorrector.

        Args:
            strategy: Path-finding strategy ("concave" or "convex").
            buffer_distance: Buffer distance for collecting vertices around problem area.
        """
        self._strategy = strategy
        self._buffer_distance = buffer_distance
        self._graph_builder = GraphBuilder(GraphBuilderConfig(strategy=strategy))
        self.logger = logging.getLogger(__name__)

    # ==========================================================================
    # Point Validation Utilities
    # ==========================================================================

    @staticmethod
    def is_point_inside_obstacles(
        point: Point,
        obstacles: list[Polygon],
    ) -> bool:
        """
        Check if a point is inside any of the obstacles.

        Args:
            point: Point to check.
            obstacles: List of obstacle polygons.

        Returns:
            True if point is inside any obstacle.
        """
        return any(poly.contains(point) for poly in obstacles)

    @staticmethod
    def find_containing_obstacle(
        point: Point,
        obstacles: list[Polygon],
    ) -> Polygon | None:
        """
        Find the obstacle that contains the given point.

        Args:
            point: Point to check.
            obstacles: List of obstacle polygons.

        Returns:
            The containing polygon or None.
        """
        for poly in obstacles:
            if poly.contains(point):
                return poly
        return None

    @staticmethod
    def find_closest_valid_point(
        point: Point,
        obstacles: list[Polygon],
        vertices: list[Point],
    ) -> Point | None:
        """
        Find the closest valid vertex outside obstacles.

        Args:
            point: Point inside obstacle.
            obstacles: List of obstacle polygons.
            vertices: Available vertices.

        Returns:
            Closest valid point or None.
        """
        min_dist = float("inf")
        closest_point = None

        for vertex in vertices:
            if SegmentCorrector.is_point_inside_obstacles(vertex, obstacles):
                continue
            dist = point.distance(vertex)
            if dist < min_dist:
                min_dist = dist
                closest_point = vertex

        return closest_point

    # ==========================================================================
    # Graph Building
    # ==========================================================================

    def build_local_escape_graph(
        self,
        entry_point: Point,
        exit_target: Point,
        obstacles_current: list[Polygon],
        obstacles_next: list[Polygon],
    ) -> nx.Graph:
        """
        Build a local graph for escaping an obstacle using GraphBuilder.

        Uses GraphBuilder's build_master_graph method for optimal vertex
        collection and graph construction using Delaunay triangulation and KNN.

        Args:
            entry_point: Point where path enters obstacle.
            exit_target: Target point to reach after escaping.
            obstacles_current: Obstacles for current time key.
            obstacles_next: Obstacles for next time key.

        Returns:
            Local graph for pathfinding.
        """
        time_keys = ["current", "next"]
        dict_obstacles = {
            "current": {self._strategy: obstacles_current},
            "next": {self._strategy: obstacles_next},
        }
        G_master, time_valid_edges = self._graph_builder.create_master_graph(
            time_keys=time_keys,
            dict_obstacles=dict_obstacles,
        )
        G_sub = subgraph_for_time_keys(
            G_master=G_master,
            time_keys=time_keys,
            time_valid_edges=time_valid_edges,
        )
        return self._graph_builder.add_points_to_subgraph(
            points=[("start", entry_point), ("end", exit_target)],
            G_master=G_master,
            G_sub=G_sub,
            obstacles=obstacles_current + obstacles_next,
        )

    # ==========================================================================
    # Path Correction
    # ==========================================================================

    def correct_path_segment(
        self,
        segment: list[Point],
        obstacles_current: list[Polygon],
        obstacles_next: list[Polygon],
        full_path: list[Point],
        current_time_index: int,
    ) -> list[Point] | None:
        """
        Correct a path segment that ends inside an obstacle.

        Args:
            segment: The problematic path segment.
            obstacles_current: Obstacles for current time key.
            obstacles_next: Obstacles for next time key.
            full_path: The complete path found so far.
            current_time_index: Current time index.

        Returns:
            Corrected path segment or None if correction fails.
        """
        if not segment:
            return None

        final_point = segment[-1]
        combined_obstacles = obstacles_current + obstacles_next

        # Find which obstacle contains the final point
        containing_obstacle = self.find_containing_obstacle(final_point, combined_obstacles)
        if containing_obstacle is None:
            self.logger.warning(
                f"Point not found inside any obstacle at time_index={current_time_index}"
            )
            return segment

        # Find the last valid point before entering the obstacle
        entry_point = None
        entry_idx = len(segment) - 1
        for i in range(len(segment) - 1, -1, -1):
            if not self.is_point_inside_obstacles(segment[i], combined_obstacles):
                entry_point = segment[i]
                entry_idx = i
                break

        if entry_point is None:
            # Collect vertices using GraphBuilder for finding closest point
            all_vertices = set()
            for obstacle in combined_obstacles:
                vertices = self._graph_builder.collect_vertices([obstacle])
                all_vertices.update(vertices)

            closest = self.find_closest_valid_point(
                final_point, combined_obstacles, list(all_vertices)
            )
            if closest is None:
                self.logger.warning(
                    f"No valid vertex found for escape at time_index={current_time_index}"
                )
                return None
            entry_point = closest
            entry_idx = 0

        # Find a target point outside the obstacle
        target_point = None
        if full_path and len(full_path) > len(segment):
            # Try to find a valid point from the remaining path
            for pt in full_path[len(segment):]:
                if not self.is_point_inside_obstacles(pt, combined_obstacles):
                    target_point = pt
                    break

        if target_point is None:
            # Collect vertices using GraphBuilder for finding target
            all_vertices = set()
            for obstacle in combined_obstacles:
                vertices = self._graph_builder.collect_vertices([obstacle])
                all_vertices.update(vertices)

            target_point = self.find_closest_valid_point(
                final_point, combined_obstacles, list(all_vertices)
            )

        if target_point is None:
            self.logger.warning(
                f"No target point found for escape at time_index={current_time_index}"
            )
            return segment

        # Build local escape graph and find path
        g_escape = self.build_local_escape_graph(
            entry_point=entry_point,
            exit_target=target_point,
            obstacles_current=obstacles_current,
            obstacles_next=obstacles_next,
        )

        # Find the shortest path through the escape graph
        try:
            escape_path_idx = nx.shortest_path(
                g_escape,
                source="start",
                target="end",
                weight="weight",
            )
            escape_path = [g_escape.nodes[i]["point"] for i in escape_path_idx]

            # Construct corrected segment
            valid_prefix = segment[:entry_idx + 1] if entry_idx > 0 else [entry_point]
            corrected_segment = valid_prefix + escape_path[1:]

            self.logger.info(
                f"Path segment corrected at time_index={current_time_index}: "
                f"entry={entry_point}, target={target_point}, "
                f"original_len={len(segment)}, corrected_len={len(corrected_segment)}"
            )

            return corrected_segment

        except nx.NetworkXNoPath:
            self.logger.warning(
                f"No escape path found at time_index={current_time_index}"
            )
            return segment

    # ==========================================================================
    # Main Validation Method
    # ==========================================================================

    def validate_and_correct_segment(
        self,
        segment: list[Point],
        time_keys: list[str],
        current_time_index: int,
        dict_obstacles: dict[str, gpd.GeoDataFrame] | dict[str, dict[str, list[Polygon]]],
        full_path: list[Point],
    ) -> tuple[list[Point], bool]:
        """
        Validate that segment endpoint is not inside obstacle for next time key.
        If it is, correct the segment.

        Args:
            segment: Path segment for current time step.
            time_keys: All available time keys.
            current_time_index: Current time index.
            dict_obstacles: Ground truth obstacles (not predictions).
            full_path: The complete path found so far.

        Returns:
            Validated/corrected path segment.
        """
        if not segment or current_time_index + 1 >= len(time_keys):
            return segment, True

        next_time_key = time_keys[current_time_index + 1]
        current_time_key = time_keys[current_time_index]

        if next_time_key not in dict_obstacles or current_time_key not in dict_obstacles:
            return segment, True

        # Extract obstacles for current and next time keys
        try:
            obstacles_current = dict_obstacles[current_time_key][self._strategy]
            obstacles_next = dict_obstacles[next_time_key][self._strategy]

            # Convert GeoSeries to list if needed
            if isinstance(obstacles_current, gpd.GeoSeries):
                obstacles_current = obstacles_current.tolist()
            if isinstance(obstacles_next, gpd.GeoSeries):
                obstacles_next = obstacles_next.tolist()
        except KeyError:
            return segment, True

        final_point = segment[-1]

        # Check if final point is inside obstacles for next time key
        if not self.is_point_inside_obstacles(final_point, obstacles_next):
            return segment, True

        self.logger.info(
            f"Segment endpoint inside obstacle detected at time_index={current_time_index}, "
            f"attempting correction..."
        )

        # Correct the path segment
        corrected = self.correct_path_segment(
            segment=segment,
            obstacles_current=obstacles_current,
            obstacles_next=obstacles_next,
            full_path=full_path,
            current_time_index=current_time_index,
        )

        # Ensure we always return a valid tuple
        if corrected is None:
            self.logger.warning(
                f"Path correction failed at time_index={current_time_index}, returning original segment"
            )
            return segment, True

        return corrected, False
