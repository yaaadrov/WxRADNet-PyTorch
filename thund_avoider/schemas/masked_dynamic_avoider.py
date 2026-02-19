from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict
from shapely import Polygon
from shapely.geometry import LineString

from thund_avoider.schemas.dynamic_avoider import SlidingWindowPath, FineTunedPath


class DirectionVector(BaseModel):
    dx: float
    dy: float


class SlidingWindowPathMasked(SlidingWindowPath):
    available_obstacles_dicts: list[dict[str, dict[str, list[Polygon]]]] = []
    is_pred_path_valid: bool = True


class FineTunedPathMasked(FineTunedPath):
    available_obstacles_dicts: list[dict[str, dict[str, list[Polygon]]]] = []
    is_pred_path_valid: bool = True


class MaskedPathfindingResult(BaseModel):
    """Single pathfinding result for one window size with masked obstacles."""

    strategy: str
    timestamp: datetime
    window_size: int
    direction: str
    prediction_mode: str
    path: LineString
    length: float
    success: bool
    success_intermediate: bool
    is_pred_path_valid: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame construction."""
        return {
            "strategy": self.strategy,
            "timestamp": self.timestamp,
            "window_size": self.window_size,
            "direction": self.direction,
            "prediction_mode": self.prediction_mode,
            "path": self.path,
            "length": self.length,
            "success": self.success,
            "success_intermediate": self.success_intermediate,
            "is_pred_path_valid": self.is_pred_path_valid,
        }


class MaskedTimestampResult(BaseModel):
    """Collection of masked pathfinding results for a single timestamp."""

    timestamp: datetime
    prediction_mode: str
    results: list[MaskedPathfindingResult] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_result(
        self,
        path_result: SlidingWindowPathMasked | FineTunedPathMasked,
        window_size: int,
        direction: str,
    ) -> None:
        """Add a pathfinding result to the collection."""
        flat_path = [pt for segment in path_result.path for pt in segment]
        self.results.append(
            MaskedPathfindingResult(
                strategy=path_result.strategy,
                timestamp=self.timestamp,
                window_size=window_size,
                direction=direction,
                prediction_mode=self.prediction_mode,
                path=LineString(flat_path) if flat_path else LineString(),
                length=LineString(flat_path).length if flat_path else 0.0,
                success=path_result.success,
                success_intermediate=path_result.success_intermediate,
                is_pred_path_valid=path_result.is_pred_path_valid,
            )
        )

    def to_records(self) -> list[dict[str, Any]]:
        """Convert all results to list of dictionaries for DataFrame."""
        return [r.to_dict() for r in self.results]
