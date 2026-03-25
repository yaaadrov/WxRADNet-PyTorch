from typing import Any

import geopandas as gpd
from shapely import Point, Polygon
from shapely.geometry import LineString

from app.config import settings
from app.schemas.responses import ValidationResult


class ValidationService:
    """Service for path validation against obstacles."""

    def __init__(self) -> None:
        """Initialize validation service."""
        self._segment_length_m = settings.segment_length_m

    @staticmethod
    def is_line_valid(line: LineString, obstacles: list[Polygon]) -> bool:
        """
        Check if a line intersects any obstacle.

        Reuses logic from thund_avoider.services.utils.is_line_valid().
        """
        return not any(line.intersects(obstacle) for obstacle in obstacles)

    @staticmethod
    def split_path_into_segments(
        path: list[Point],
        max_segment_length: float,
    ) -> list[LineString]:
        """
        Split a path into segments no longer than max_segment_length.

        Reuses logic from FineTuner.split_path_into_segments().
        """
        if len(path) < 2:
            return []

        line = LineString(path)
        segments = []
        start_dist = 0.0
        total_length = line.length

        while start_dist < total_length:
            end_dist = min(start_dist + max_segment_length, total_length)
            from shapely.ops import substring
            segments.append(substring(line, start_dist, end_dist))
            start_dist = end_dist

        return segments

    def validate_segment(
        self,
        segment: LineString,
        obstacles: list[Polygon],
    ) -> bool:
        """Validate a single segment against obstacles."""
        return self.is_line_valid(segment, obstacles)

    def validate_path(
        self,
        path: list[tuple[float, float]],
        time_keys: list[str],
        dict_obstacles: dict[str, gpd.GeoDataFrame] | dict[str, dict[str, list[Polygon]]],
        strategy: str = "concave",
    ) -> tuple[bool, list[ValidationResult]]:
        """
        Validate path against obstacles for each time step.

        Reuses logic from MaskedDynamicAvoider._validate_path_against_initial_obstacles().
        """
        if not path or len(path) < 2:
            return True, []

        points = [Point(p) for p in path]
        segments = self.split_path_into_segments(points, self._segment_length_m)

        results = []
        all_valid = True

        for i, segment in enumerate(segments):
            if i >= len(time_keys):
                break

            time_key = time_keys[i]
            if time_key not in dict_obstacles:
                continue

            obstacles_data = dict_obstacles[time_key]

            # Handle both GeoDataFrame and dict format
            if isinstance(obstacles_data, gpd.GeoDataFrame):
                obstacles = obstacles_data[strategy].tolist()
            else:
                obstacles = obstacles_data.get(strategy, [])

            is_valid = self.is_line_valid(segment, obstacles)

            results.append(ValidationResult(
                is_valid=is_valid,
                segment_index=i,
            ))

            if not is_valid:
                all_valid = False

        return all_valid, results

    def validate_single_segment(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        obstacles: list[Polygon],
    ) -> bool:
        """Validate a single line segment against obstacles."""
        line = LineString([start, end])
        return self.is_line_valid(line, obstacles)

    def calculate_path_distance(self, path: list[tuple[float, float]]) -> float:
        """Calculate total path distance in meters."""
        if len(path) < 2:
            return 0.0

        total = 0.0
        for i in range(len(path) - 1):
            p1 = Point(path[i])
            p2 = Point(path[i + 1])
            total += p1.distance(p2)

        return total

    def is_at_point(
        self,
        position: tuple[float, float],
        target: tuple[float, float],
        tolerance_m: float = 10_000.0,
    ) -> bool:
        """Check if position is within tolerance of target point."""
        p1 = Point(position)
        p2 = Point(target)
        return p1.distance(p2) <= tolerance_m
