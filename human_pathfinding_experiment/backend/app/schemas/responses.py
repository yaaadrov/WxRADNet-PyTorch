from typing import Any, Literal

from pydantic import BaseModel, Field


class BBoxRequest(BaseModel):
    """Request model for pre-calculating bounding box for all time keys."""

    timestamp: str = Field(..., description="Timestamp directory name")
    strategy: Literal["concave", "convex"] = Field(
        default="concave",
        description="Hull strategy for obstacles",
    )
    mode: Literal["obstacles", "hull"] = Field(
        default="obstacles",
        description="Obstacle mode: raw obstacles only or with hull",
    )


class TimestampsResponse(BaseModel):
    """Response model for available timestamps."""

    timestamps: list[str] = Field(..., description="List of available timestamp directories")


class PixelTransform(BaseModel):
    """Pixel transform for canvas rendering."""

    scale: float = Field(..., description="Scale factor for coordinate conversion")
    offset_x: float = Field(..., description="X offset in pixels")
    offset_y: float = Field(..., description="Y offset in pixels (Y-flipped)")
    bounds: dict[str, float] = Field(
        ...,
        description="Geographic bounds {minX, maxX, minY, maxY}",
    )


class BBoxResponse(BaseModel):
    """Response model for pre-calculated bounding box."""

    pixel_transform: PixelTransform = Field(
        ...,
        description="Transform for coordinate to pixel conversion (stable for entire experiment)",
    )
    all_time_keys: list[str] = Field(
        ...,
        description="All available time keys for the timestamp",
    )


class ABPointsResponse(BaseModel):
    """Response model for A and B points."""

    a_point: list[float] = Field(..., description="Point A coordinates [x, y]")
    b_point: list[float] = Field(..., description="Point B coordinates [x, y]")


class ObstacleLayer(BaseModel):
    """Single obstacle layer for one time key."""

    time_key: str = Field(..., description="Time key for this layer")
    geojson: dict[str, Any] = Field(..., description="GeoJSON FeatureCollection")
    color_index: int = Field(..., ge=0, description="Color palette index")


class ObstaclesResponse(BaseModel):
    """Response model for obstacles endpoint."""

    obstacles: list[ObstacleLayer] = Field(
        ...,
        description="List of obstacle layers (one per time key)",
    )
    pixel_transform: PixelTransform = Field(
        ...,
        description="Transform for coordinate to pixel conversion",
    )
    available_time_keys: list[str] = Field(
        ...,
        description="All available time keys for current position",
    )
    current_time_index: int = Field(
        ...,
        ge=0,
        description="Current time index in the full time keys list",
    )


class ValidationResult(BaseModel):
    """Result of path validation."""

    is_valid: bool = Field(..., description="Whether the segment is valid")
    segment_index: int = Field(..., description="Index of the segment in the path")


class ValidationResponse(BaseModel):
    """Response model for path validation."""

    is_valid: bool = Field(..., description="Whether the entire path is valid")
    segments: list[ValidationResult] = Field(
        ...,
        description="Validation result for each segment",
    )
    invalid_segments: list[int] = Field(
        ...,
        description="Indices of invalid segments",
    )
    validation_obstacles: list[ObstacleLayer] = Field(
        ...,
        description="Obstacle layers for invalid segments (concave hulls)",
    )


class ResultsResponse(BaseModel):
    """Response model for saved results."""

    filename: str = Field(..., description="Saved parquet filename")
    message: str = Field(..., description="Success message")
