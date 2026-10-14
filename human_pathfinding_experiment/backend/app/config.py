import re
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

    # Database (Postgres; see docker-compose db service for local development).
    # Managed providers (Supabase/Neon) hand out postgres:// or postgresql:// URLs —
    # use sqlalchemy_database_url for the psycopg2-driver-normalized form.
    database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/human_pathfinding"

    # Participant session tokens (opaque JWT; set JWT_SECRET via env in production)
    jwt_secret: str = "dev-secret-change-me"
    jwt_ttl_minutes: int = 240  # covers a full experiment session with headroom
    demo_ttl_minutes: int = 60  # anonymous demo session (never stored in the database)

    # Admin endpoints (/api/results*, /api/admin/*): requests must carry X-Admin-Token.
    # Empty token disables the endpoints entirely (local dev convenience).
    admin_token: str = ""

    # Comma-separated list of allowed CORS origins (e.g. the Vercel frontend URL).
    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    # tar.gz with the dataset (data/<timestamp>/*.csv + config/*.pkl) downloaded at container
    # start when the local copies are missing; empty disables the fetch (docker-compose mounts).
    data_archive_url: str = ""

    # Session structure
    trials_per_participant: int = 3

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

    @property
    def cors_origin_list(self) -> list[str]:
        """Allowed CORS origins parsed from the comma-separated env string."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def sqlalchemy_database_url(self) -> str:
        """Database URL normalized for the psycopg2 driver.

        Accepts provider-specific forms (postgres://, postgresql://) and rewrites them
        to postgresql+psycopg2://, which is the driver shipped with the backend.
        """
        return re.sub(r"^postgres(ql)?://", "postgresql+psycopg2://", self.database_url)


settings = Settings()

# Constants
MASKING_STRATEGY: Final = "wide"
