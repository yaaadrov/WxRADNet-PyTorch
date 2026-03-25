from pathlib import Path
from typing import Final

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Data paths (mounted from host)
    data_path: Path = Path("/app/data")
    config_path: Path = Path("/app/config")
    results_path: Path = Path("/app/results")

    # Canvas settings (vertical orientation)
    canvas_width: int = 800
    canvas_height: int = 800
    canvas_padding: int = 50

    # Pathfinding constants (from thund_avoider settings)
    velocity_kmh: int = 900
    delta_minutes: int = 5
    snap_distance_m: float = 10_000.0  # 10 km snap to B point

    # Preprocessor settings
    square_side_length_m: int = 250_000  # 250 km radar range
    bbox_buffer_m: int = 10_000  # 10 km buffer

    @property
    def velocity_mpm(self) -> float:
        """Velocity in meters per minute."""
        return self.velocity_kmh * 1000 / 60

    @property
    def segment_length_m(self) -> float:
        """Maximum segment length per time step."""
        return self.velocity_mpm * self.delta_minutes

    @property
    def ab_points_path(self) -> Path:
        """Path to AB points pickle file."""
        return self.config_path / "ab_points.pkl"

    @property
    def timestamps_path(self) -> Path:
        """Path to timestamps pickle file."""
        return self.config_path / "timestamps.pkl"


settings = Settings()

# Constants
MASKING_STRATEGY: Final = "wide"
