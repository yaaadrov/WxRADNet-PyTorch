from typing import Literal

from pydantic import BaseModel, Field


class ObstaclesRequest(BaseModel):
    """Request model for fetching obstacles."""

    timestamp: str = Field(..., description="Timestamp directory name")
    time_index: int = Field(..., ge=0, description="Current time index")
    strategy: Literal["concave", "convex"] = Field(
        default="concave",
        description="Hull strategy for obstacles",
    )
    mode: Literal["obstacles", "hull"] = Field(
        default="obstacles",
        description="Obstacle mode: raw obstacles only or with hull",
    )
    current_position: list[float] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Current position [x, y] in EPSG:3067",
    )
    direction_vector: list[float] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Direction vector [dx, dy]",
    )
    window_size: Literal[1, 7] = Field(
        default=1,
        description="Number of time keys to load",
    )


class ValidatePathRequest(BaseModel):
    """Request model for validating a path against obstacles."""

    timestamp: str = Field(..., description="Timestamp directory name")
    strategy: Literal["concave", "convex"] = Field(
        default="concave",
        description="Hull strategy for obstacles",
    )
    all_paths: list[list[tuple[float, float]]] = Field(
        ...,
        description="All path segments as lists of coordinate pairs",
    )


class ResultsRequest(BaseModel):
    """Request model for saving experiment results."""

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
    timestamp_start: str = Field(..., description="ISO timestamp of experiment start")
    success: bool = Field(..., description="Whether experiment completed successfully")
    total_waypoints: int = Field(..., description="Total number of waypoints")
    total_distance_m: float = Field(..., description="Total path distance in meters")
