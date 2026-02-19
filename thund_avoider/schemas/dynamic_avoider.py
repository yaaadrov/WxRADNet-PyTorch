from datetime import datetime
from typing import Any

import networkx as nx
from pydantic import BaseModel, ConfigDict
from shapely import Point
from shapely.geometry import LineString


class SlidingWindowPath(BaseModel):
    """Result of sliding window pathfinding without fine-tuning."""

    strategy: str
    path: list[list[Point]] = []
    all_paths: list[list[Point]] = []
    all_graphs: list[nx.Graph] = []
    success: bool = True
    success_intermediate: bool = True
    moved_original_points_mapping: dict[Point, Point] = {}
    num_segments: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)


class FineTunedPath(BaseModel):
    """Result of sliding window pathfinding with fine-tuning."""

    strategy: str
    path: list[list[Point]] = []
    all_paths_initial: list[list[Point]] = []
    all_paths_fine_tuned: list[list[Point]] = []
    all_paths_initial_raw: list[SlidingWindowPath] = []
    fine_tuning_iters: list[int] = []
    fine_tuning_times: list[float] = []
    success: bool = True
    success_intermediate: bool = True
    num_segments: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PathfindingResult(BaseModel):
    """Single pathfinding result for one window size."""

    strategy: str
    timestamp: datetime
    window_size: int
    direction: str
    path: LineString
    length: float
    success: bool
    success_intermediate: bool

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame construction."""
        return {
            "strategy": self.strategy,
            "timestamp": self.timestamp,
            "window_size": self.window_size,
            "direction": self.direction,
            "path": self.path,
            "length": self.length,
            "success": self.success,
            "success_intermediate": self.success_intermediate,
        }


class TimestampResult(BaseModel):
    """Collection of pathfinding results for a single timestamp."""

    timestamp: datetime
    results: list[PathfindingResult] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_result(
        self,
        path_result: SlidingWindowPath | FineTunedPath,
        window_size: int,
        direction: str,
    ) -> None:
        """Add a pathfinding result to the collection."""
        flat_path = [pt for segment in path_result.path for pt in segment]
        self.results.append(
            PathfindingResult(
                strategy=path_result.strategy,
                timestamp=self.timestamp,
                window_size=window_size,
                direction=direction,
                path=LineString(flat_path) if flat_path else LineString(),
                length=LineString(flat_path).length if flat_path else 0.0,
                success=path_result.success,
                success_intermediate=path_result.success_intermediate,
            )
        )

    def to_records(self) -> list[dict[str, Any]]:
        """Convert all results to list of dictionaries for DataFrame."""
        return [r.to_dict() for r in self.results]
