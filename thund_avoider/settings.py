from pydantic import BaseModel
from pyproj import CRS


class Settings(BaseModel):
    # Thund Avoider
    projected_crs: CRS = CRS(3067)  # ETRS89 / TM35FIN(E,N)
    velocity_kmh: float = 900  # Velocity in km/hour
    delta_minutes: float = 5  # Forecast frequency
    buffer: float = 5_000  # Additional 5 km buffer
    tolerance: float = 5_000  # Simplification tolerance
    k_neighbors: float = 10  # Number of neighbors for master graph
    max_distance: float = 20_000  # Split each segment into several subsegments for greedy fine-tuning
    simplification_tolerance: float = 1e-9  # Tolerance to simplify paths after densifying
    smooth_tolerance: float = 5_000 * 5  # Tolerance for smoothing fine-tuning (tolerance * 5)
    max_iter: float = 300  # Maximum number of iterations for smooth fine-tuning
    delta_length: float = 1.0  # Smooth fine-tuning length sensitivity
    bbox_buffer: float = 50_000  # Additional 50 km buffer for A and B points (Static Avoider)

    # Preprocessor / Parser
    base_url: str = "http://s3-eu-west-1.amazonaws.com/fmi-opendata-radar-geotiff/{year}/{month}/{day}/FIN-DBZ-3067-250M/{year}{month}{day}{hour}{minute}_FIN-DBZ-3067-250M.tif"
    intensity_threshold_low: float = 100
    intensity_threshold_high: float = 255
    distance_between: float = 25_000  # Minimum 50 km between two thunderstorms to proceed between
    distance_avoid: float = 15_000  # Minimum 15 km to thunderstorm to circumnavigate


settings = Settings()
