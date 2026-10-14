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


class LoginRequest(BaseModel):
    """Request model for participant login (participant is created on the fly)."""

    surname: str = Field(..., min_length=1, max_length=128, description="Participant surname")
    name: str = Field(..., min_length=1, max_length=128, description="Participant name")
    group: str = Field(..., min_length=1, max_length=128, description="Participant study group")


class SubmitTrialRequest(BaseModel):
    """Request model for submitting a trial: raw route data and interaction counters only."""

    all_paths: list[list[tuple[float, float]]] = Field(
        ...,
        description="Raw per-stage drawings (each stage starts fresh from its start position)",
    )
    path: list[list[tuple[float, float]]] = Field(
        ...,
        description="Final flown route split into exact 5-minute intervals",
    )
    num_zoom_actions: int = Field(default=0, ge=0, description="ZOOM_IN/ZOOM_OUT actions count")
    num_forecast_layers_viewed: int = Field(default=0, ge=0, description="Extra forecast layers viewed")


class QuestionnaireRequest(BaseModel):
    """Request model for the final session questionnaire (§1.5)."""

    ease_rating_borders: int = Field(..., ge=1, le=5, description="Ease of C0: borders only")
    ease_rating_zones: int = Field(..., ge=1, le=5, description="Ease of C1: borders + forbidden zones")
    ease_rating_forecast: int = Field(..., ge=1, le=5, description="Ease of C2: borders + zones + forecast")
    awareness_zones_helps: bool = Field(..., description="Do forbidden zones improve situational awareness?")
    awareness_forecast_helps: bool = Field(..., description="Does the forecast improve situational awareness?")
