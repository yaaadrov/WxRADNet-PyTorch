import logging
import time
from collections.abc import Callable

import numpy as np
from shapely import STRtree
from shapely.geometry import LineString, Point
from shapely.ops import substring

from thund_avoider.services.utils import (
    compute_angle,
    get_unit_direction,
    is_line_valid,
    linestring_to_points,
    shift_point,
)
from thund_avoider.settings import FineTunerConfig


class FineTuner:
    """Path fine-tuning with greedy shortcut and smooth angle optimization strategies."""

    def __init__(self, config: FineTunerConfig) -> None:
        """Initialize FineTuner with configuration."""
        self._config = config
        self._velocity_mpm = config.velocity_kmh * 1000 / 60
        self._logger = logging.getLogger(__name__)

    def interpolate_points(self, p1: Point, p2: Point) -> list[Point]:
        """Interpolates points between p1 and p2 if the distance exceeds `max_distance`."""
        distance = p1.distance(p2)
        if distance <= self._config.max_distance:
            return [p1, p2]

        num_segments = int(np.ceil(distance / self._config.max_distance))
        return [
            Point(
                p1.x + i * (p2.x - p1.x) / num_segments,
                p1.y + i * (p2.y - p1.y) / num_segments,
            )
            for i in range(num_segments + 1)
        ]

    def densify_path(self, path: list[Point]) -> list[Point]:
        """Densify the path by interpolating points to ensure no segment exceeds `max_distance`."""
        dense_path = []
        for i in range(len(path) - 1):
            segment_points = self.interpolate_points(path[i], path[i + 1])
            dense_path.extend(segment_points[:-1])
        dense_path.append(path[-1])
        return dense_path

    @staticmethod
    def split_path_into_segments(path: list[Point], max_segment_length: float) -> list[LineString]:
        """Split a LineString into segments no longer than `max_segment_length`."""
        line = LineString(path)
        if len(path) < 2:
            return [line]

        segments = []
        start_dist = 0.0
        total_length = line.length

        while start_dist < total_length:
            end_dist = min(start_dist + max_segment_length, total_length)
            segments.append(substring(line, start_dist, end_dist))
            start_dist = end_dist
        return segments

    def simplify_path(self, path: list[Point]) -> list[Point]:
        """Simplify path using configured tolerance."""
        return linestring_to_points(
            LineString(path).simplify(
                tolerance=self._config.simplification_tolerance,
                preserve_topology=True,
            )
        )

    # ==========================================================================
    # Validation Utilities
    # ==========================================================================

    @staticmethod
    def validate_segments(
        segments: list[LineString],
        time_keys: list[str],
        strtrees: dict[str, STRtree],
    ) -> bool:
        """Validate that all segments are obstacle-free."""
        for i, segment in enumerate(segments):
            time_key = time_keys[i]
            if time_key not in strtrees:
                continue

            tree = strtrees[time_key]
            possible_idxs = tree.query(segment)
            possible_obs = [tree.geometries[i] for i in possible_idxs]

            if not is_line_valid(segment, possible_obs):
                return False
        return True

    def is_path_valid(
        self,
        path: list[Point],
        time_keys: list[str],
        strtrees: dict[str, STRtree],
    ) -> bool:
        """Check if path is valid against obstacles."""
        segments = self.split_path_into_segments(
            path=path,
            max_segment_length=self._velocity_mpm * self._config.delta_minutes,
        )
        return self.validate_segments(segments, time_keys, strtrees)

    # ==========================================================================
    # Greedy Fine-Tuning
    # ==========================================================================

    def find_shortcut(
        self,
        dense_path: list[Point],
        start_idx: int,
        time_keys: list[str],
        strtrees: dict[str, STRtree],
        max_segments: int,
    ) -> list[Point]:
        """
        Find a valid shortcut from start_idx to a later point in the path.

        Args:
            dense_path: Dense path points.
            start_idx: Starting index for shortcut.
            time_keys: Time keys for validation.
            strtrees: STRTrees for each time key.
            max_segments: Maximum allowed number of segments.

        Returns:
            Path with shortcut if found, otherwise original path.
        """
        for j in range(len(dense_path) - 1, start_idx, -1):
            shortcut = [dense_path[start_idx], dense_path[j]]
            candidate_path = dense_path[:start_idx] + shortcut + dense_path[j + 1:]

            if not self._is_shortcut_valid(candidate_path, max_segments, time_keys, strtrees):
                continue

            shortcut_dense = self.densify_path(shortcut)
            return dense_path[:start_idx] + shortcut_dense + dense_path[j + 1:]

        return dense_path

    def _is_shortcut_valid(
        self,
        candidate_path: list[Point],
        max_segments: int,
        time_keys: list[str],
        strtrees: dict[str, STRtree],
    ) -> bool:
        """Check if a shortcut produces a valid path within segment limit."""
        segments = self.split_path_into_segments(
            path=candidate_path,
            max_segment_length=self._velocity_mpm * self._config.delta_minutes,
        )
        if len(segments) > max_segments:
            return False
        return self.validate_segments(segments, time_keys, strtrees)

    def _greedy_fine_tuning_iter(
        self,
        dense_path: list[Point],
        current_idx: int,
        time_keys: list[str],
        strtrees: dict[str, STRtree],
    ) -> tuple[list[Point], int]:
        """
        Perform one iteration of greedy fine-tuning.

        Returns:
            Updated path and new index.
        """
        num_segments = len(
            self.split_path_into_segments(
                path=dense_path,
                max_segment_length=self._velocity_mpm * self._config.delta_minutes,
            )
        )
        new_path = self.find_shortcut(
            dense_path=dense_path,
            start_idx=current_idx,
            time_keys=time_keys,
            strtrees=strtrees,
            max_segments=num_segments,
        )
        return new_path, current_idx + 1

    def _should_stop_greedy(
        self,
        path: list[Point],
        current_idx: int,
        quick: bool,
    ) -> bool:
        """Check if greedy fine-tuning should stop early."""
        if not quick:
            return False
        if current_idx + 1 >= len(path):
            return True
        traversed_length = LineString(path[:current_idx + 1]).length
        max_length = self._velocity_mpm * self._config.delta_minutes
        return traversed_length > max_length

    def greedy_fine_tuning(
        self,
        path_flat: list[Point],
        time_keys: list[str],
        strtrees: dict[str, STRtree],
        quick: bool = True,
    ) -> tuple[list[Point], int, float]:
        """
        Perform greedy fine-tuning on the path.

        Args:
            path_flat: Flattened path points.
            time_keys: Time keys for validation.
            strtrees: STRTrees for each time key.
            quick: Whether to use quick mode (stop after first time window).

        Returns:
            tuple: Fine-tuned path, iteration count, and time elapsed.
        """
        tic = time.perf_counter()

        dense_path = self.densify_path(path_flat)
        current_idx = 0

        while current_idx < len(dense_path) - 1:
            dense_path, current_idx = self._greedy_fine_tuning_iter(
                dense_path=dense_path,
                current_idx=current_idx,
                time_keys=time_keys,
                strtrees=strtrees,
            )

            if self._should_stop_greedy(dense_path, current_idx, quick):
                break

        simplified_path = self.simplify_path(dense_path)
        return simplified_path, current_idx, time.perf_counter() - tic

    # ==========================================================================
    # Smooth Fine-Tuning
    # ==========================================================================

    @staticmethod
    def compute_angles(path: list[Point]) -> list[tuple[int, float]]:
        """
        Compute all angles for a given path.

        Returns:
            List of (index, angle) tuples sorted by angle ascending.
        """
        angles = [
            (i, compute_angle(path[i - 1], path[i], path[i + 1]))
            for i in range(1, len(path) - 1)
        ]
        return sorted(angles, key=lambda x: x[1])

    @staticmethod
    def replace_point_with_shifted(
        points: list[Point],
        index: int,
        tolerance: float,
    ) -> list[Point]:
        """
        Replace the point at `index` with shifted points.

        Args:
            points: List of points.
            index: Index of the point to replace.
            tolerance: Distance threshold for inserting shifted points.

        Returns:
            Updated list of points with the point at `index` replaced.

        Raises:
            ValueError: If `index` is the first or last element in the list.
        """
        if index <= 0 or index >= len(points) - 1:
            raise ValueError("Point must not be first or last in the list to compute direction")

        prev_point = points[index - 1]
        next_point = points[index + 1]
        curr_point = points[index]
        new_points = []

        if curr_point.distance(prev_point) > tolerance:
            direction = get_unit_direction(prev_point, curr_point)
            new_points.append(shift_point(curr_point, direction, -tolerance))

        if curr_point.distance(next_point) > tolerance:
            direction = get_unit_direction(curr_point, next_point)
            new_points.append(shift_point(curr_point, direction, tolerance))

        return points[:index] + new_points + points[index + 1:]

    def _try_smooth_point(
        self,
        path: list[Point],
        point_idx: int,
        time_keys: list[str],
        strtrees: dict[str, STRtree],
    ) -> list[Point] | None:
        """
        Try to smooth a single point in the path.

        Returns:
            Smoothed path if valid, None otherwise.
        """
        smoothed = self.replace_point_with_shifted(
            points=path,
            index=point_idx,
            tolerance=self._config.smooth_tolerance,
        )

        if not self.is_path_valid(smoothed, time_keys, strtrees):
            return None

        return self.simplify_path(smoothed)

    def _check_smoothing_progress(
        self,
        original_path: list[Point],
        smoothed_path: list[Point],
        num_iter: int,
    ) -> tuple[bool, bool]:
        """
        Check if smoothing is making progress.

        Returns:
            Tuple of (should_continue, should_restart).
        """
        original_length = LineString(original_path).length
        new_length = LineString(smoothed_path).length
        length_change = original_length - new_length

        # Check if improvement is too small
        if length_change < self._config.delta_length:
            self._logger.info(
                f"Smooth fine-tuning: Length change ({length_change:.2f}m) "
                f"is less than threshold ({self._config.delta_length}m)"
            )
            return False, False

        # Check if max iterations reached
        if num_iter >= self._config.max_iter:
            self._logger.info(
                f"Smooth fine-tuning: Max iterations ({num_iter}) completed"
            )
            return False, False

        return True, True

    def _smooth_fine_tuning_iter(
        self,
        path: list[Point],
        time_keys: list[str],
        strtrees: dict[str, STRtree],
        num_iter: int,
    ) -> tuple[list[Point], bool, bool]:
        """
        Perform one iteration of smooth fine-tuning.

        Returns:
            Tuple of (updated_path, modified, should_continue).
        """
        angles = self.compute_angles(path)

        for point_idx, _ in angles:
            smoothed = self._try_smooth_point(path, point_idx, time_keys, strtrees)

            if smoothed is None:
                continue

            should_continue, should_restart = self._check_smoothing_progress(
                path, smoothed, num_iter + 1
            )

            return smoothed, True, should_continue and should_restart

        return path, False, False

    def smooth_fine_tuning(
        self,
        path_flat: list[Point],
        time_keys: list[str],
        strtrees: dict[str, STRtree],
    ) -> tuple[list[Point], int, float]:
        """
        Perform smooth fine-tuning on the path.

        Args:
            path_flat: Flattened path points.
            time_keys: Time keys for validation.
            strtrees: STRTrees for each time key.

        Returns:
            tuple: Fine-tuned path, iteration count, and time elapsed.
        """
        tic = time.perf_counter()
        path = path_flat
        num_iter = 0

        while True:
            path, modified, should_continue = self._smooth_fine_tuning_iter(
                path=path,
                time_keys=time_keys,
                strtrees=strtrees,
                num_iter=num_iter,
            )

            if modified:
                num_iter += 1

            if not modified or not should_continue:
                break

        self._logger.info(f"{num_iter} iterations completed")
        return path, num_iter, time.perf_counter() - tic

    # ==========================================================================
    # Strategy Selection
    # ==========================================================================

    @property
    def fine_tuning_function(self) -> Callable:
        """Get the fine-tuning function based on the configured strategy."""
        match self._config.tuning_strategy:
            case "greedy":
                return self.greedy_fine_tuning
            case "smooth":
                return self.smooth_fine_tuning
