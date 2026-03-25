from datetime import datetime
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field


class ExperimentResult(BaseModel):
    """Complete experiment result model."""

    timestamp: str = Field(..., description="Timestamp directory name")
    obstacle_mode: Literal["obstacles", "hull"] = Field(
        ...,
        description="Obstacle mode used in experiment",
    )
    window_size: Literal[1, 7] = Field(..., description="Window size used")
    strategy: str = Field(default="concave", description="Hull strategy")
    prediction_mode: str = Field(
        default="deterministic",
        description="Prediction mode",
    )
    all_paths: list[list[tuple[float, float]]] = Field(
        ...,
        description="All raw waypoints as coordinate pairs",
    )
    path: list[list[tuple[float, float]]] = Field(
        ...,
        description="Path segmented by delta_minutes",
    )
    path_valid: bool = Field(..., description="Whether path is valid")
    experiment_duration_seconds: float = Field(
        ...,
        description="Total experiment duration",
    )
    timestamp_start: datetime = Field(
        ...,
        description="Timestamp of experiment start",
    )
    success: bool = Field(..., description="Whether experiment completed successfully")
    total_waypoints: int = Field(..., description="Total number of waypoints")
    total_distance_m: float = Field(..., description="Total path distance in meters")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame construction."""
        return {
            "timestamp": self.timestamp,
            "obstacle_mode": self.obstacle_mode,
            "window_size": self.window_size,
            "strategy": self.strategy,
            "prediction_mode": self.prediction_mode,
            "all_paths": self.all_paths,
            "path": self.path,
            "path_valid": self.path_valid,
            "experiment_duration_seconds": self.experiment_duration_seconds,
            "timestamp_start": self.timestamp_start,
            "success": self.success,
            "total_waypoints": self.total_waypoints,
            "total_distance_m": self.total_distance_m,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame([self.to_dict()])
