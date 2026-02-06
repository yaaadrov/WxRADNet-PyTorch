from pathlib import Path

from pydantic import BaseModel

ROOT_PATH: Path = Path(__file__).resolve().parents[1]
DATA_PATH: Path = ROOT_PATH / "data"
RESULT_PATH: Path = ROOT_PATH / "results"
TIMESTAMPS_PATH: Path = ROOT_PATH / "config" / "timestamps.pkl"
AB_POINTS_PATH: Path = ROOT_PATH / "config" / "ab_points.pkl"


class Settings(BaseModel):
    # Thund Avoider
    projected_crs: int = 3067  # ETRS89 / TM35FIN(E,N)
    velocity_kmh: int = 900  # Velocity in km/hour
    delta_minutes: int = 5  # Forecast frequency
    buffer: int = 5_000  # Additional 5 km buffer
    tolerance: int = 5_000  # Simplification tolerance
    k_neighbors: int = 10  # Number of neighbors for master graph
    max_distance: int = 20_000  # Split each segment into several subsegments for greedy fine-tuning
    simplification_tolerance: float = 1e-9  # Tolerance to simplify paths after densifying
    smooth_tolerance: int = 5_000 * 5  # Tolerance for smoothing fine-tuning (tolerance * 5)
    max_iter: int = 300  # Maximum number of iterations for smooth fine-tuning
    delta_length: float = 1.0  # Smooth fine-tuning length sensitivity
    bbox_buffer: int = 50_000  # Additional 50 km buffer for A and B points (Static Avoider)

    # Preprocessor / Parser
    base_url: str = "http://s3-eu-west-1.amazonaws.com/fmi-opendata-radar-geotiff/{year}/{month}/{day}/FIN-DBZ-3067-250M/{year}{month}{day}{hour}{minute}_FIN-DBZ-3067-250M.tif"
    intensity_threshold_low: int = 100
    intensity_threshold_high: int = 255
    distance_between: int = 25_000  # Minimum 50 km between two thunderstorms to proceed between
    distance_avoid: int = 15_000  # Minimum 15 km to thunderstorm to circumnavigate

    # Masked logic
    square_side_length_m: int = 250_000  # Radar range 250 km (equals to 1000 pixels)


settings = Settings()
