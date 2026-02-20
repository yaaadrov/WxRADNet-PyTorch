from datetime import datetime, timedelta
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel
from torch import device

from thund_avoider.schemas.predictor import ModelType

ROOT_PATH: Final = Path(__file__).resolve().parents[1]
THUND_AVOIDER_PATH: Final = Path(__file__).resolve().parent
PARSER_DATA_PATH: Final = ROOT_PATH / "parser_data"
PARSER_RESULT_PATH: Final = PARSER_DATA_PATH / "thunderstorm_data.npy"
DATA_PATH: Final = ROOT_PATH / "data"
IMAGES_PATH: Final = ROOT_PATH / "images"
RESULT_PATH: Final = ROOT_PATH / "results"
TIMESTAMPS_PATH: Final = ROOT_PATH / "config" / "timestamps.pkl"
AB_POINTS_PATH: Final = ROOT_PATH / "config" / "ab_points.pkl"
MODELS_PATH: Final = THUND_AVOIDER_PATH / "models"


# =============================================================================
# Parser Configuration
# =============================================================================

class CropBorderConfig(BaseModel):
    """Configuration for a single crop border region."""
    y_start: int
    y_end: int
    x_start: int
    x_end: int


class ParserConfig(BaseModel):
    """Configuration for Parser data collection."""
    base_url: str = "http://s3-eu-west-1.amazonaws.com/fmi-opendata-radar-geotiff/{year}/{month}/{day}/FIN-DBZ-3067-250M/{year}{month}{day}{hour}{minute}_FIN-DBZ-3067-250M.tif"
    progress_file: Path = Path("progress.json")  # Path to progress file
    first_date: datetime = datetime(2020, 12, 31, 23, 55)
    last_date: datetime = datetime(2024, 7, 31, 23, 55)
    delta_minutes: int = 5  # Time step in minutes
    intensity_threshold: int = 100  # Minimum pixel intensity for thundercloud
    min_pixels: int = 9000  # Minimum pixels for informative segment
    image_size: int = 256  # Output image size
    # Angle ranges for data augmentation (degrees)
    angles: list[tuple[int, int]] = [(60, 120), (150, 210), (240, 300)]
    # Crop border regions
    crop_borders: list[CropBorderConfig] = [
        CropBorderConfig(y_start=3000, y_end=4000, x_start=2500, x_end=3500),
        CropBorderConfig(y_start=4000, y_end=5000, x_start=1500, x_end=2500),
        CropBorderConfig(y_start=4000, y_end=5000, x_start=2500, x_end=3500),
        CropBorderConfig(y_start=5000, y_end=6000, x_start=1600, x_end=2600),
    ]
    gtiff_srs: str = "EPSG"

    @property
    def delta(self) -> timedelta:
        """Get time delta."""
        return timedelta(minutes=self.delta_minutes)

    @property
    def delta_hour(self) -> timedelta:
        """Get one hour delta."""
        return timedelta(hours=1)


# =============================================================================
# Preprocessor Configuration
# =============================================================================

class PreprocessorConfig(BaseModel):
    base_url: str = "http://s3-eu-west-1.amazonaws.com/fmi-opendata-radar-geotiff/{year}/{month}/{day}/FIN-DBZ-3067-250M/{year}{month}{day}{hour}{minute}_FIN-DBZ-3067-250M.tif"
    intensity_threshold_low: int = 100
    intensity_threshold_high: int = 255
    distance_between: int = 25_000  # Minimum 50 km between two thunderstorms to proceed between
    distance_avoid: int = 15_000  # Minimum 15 km to thunderstorm to circumnavigate


class MaskedPreprocessorConfig(BaseModel):
    preprocessor_config: PreprocessorConfig = PreprocessorConfig()
    square_side_length_m: int = 250_000  # Radar range 250 km (equals to 1000 pixels)
    bbox_buffer_m: int = 10_000  # 10 km buffer to remove G_master nodes from around the bounding box


# =============================================================================
# Avoider Configuration
# =============================================================================

class StaticAvoiderConfig(BaseModel):
    buffer: int = 5_000  # Additional 5 km buffer
    tolerance: int = 5_000  # Simplification tolerance
    bbox_buffer: int = 50_000  # Additional 50 km buffer for A and B points


class GraphBuilderConfig(BaseModel):
    """Configuration for GraphBuilder component."""
    crs: int = 3067  # ETRS89 / TM35FIN(E,N)
    buffer: int = 5_000  # Additional 5 km buffer
    tolerance: int = 5_000  # Simplification tolerance
    k_neighbors: int = 10  # Number of neighbors for master graph
    strategy: Literal["concave", "convex"] = "concave"  # Path-finding strategy to apply


class FineTunerConfig(BaseModel):
    """Configuration for FineTuner component."""
    max_distance: int = 20_000  # Split each segment into several subsegments for greedy fine-tuning
    simplification_tolerance: float = 1e-9  # Tolerance to simplify paths after densifying
    smooth_tolerance: int = 5_000 * 5  # Tolerance for smoothing fine-tuning (tolerance * 5)
    max_iter: int = 300  # Maximum number of iterations for smooth fine-tuning
    delta_length: float = 1.0  # Smooth fine-tuning length sensitivity
    velocity_kmh: int = 900  # Velocity in km/hour
    delta_minutes: int = 5  # Forecast frequency
    tuning_strategy: Literal["greedy", "smooth"] = "greedy"  # Fine-tuning strategy to apply


class DataLoaderConfig(BaseModel):
    """Configuration for DataLoader component."""
    crs: int = 3067  # ETRS89 / TM35FIN(E,N)


class DynamicAvoiderConfig(BaseModel):
    """Configuration for DynamicAvoider - aggregates component configs."""
    graph_builder_config: GraphBuilderConfig = GraphBuilderConfig()
    fine_tuner_config: FineTunerConfig = FineTunerConfig()
    data_loader_config: DataLoaderConfig = DataLoaderConfig()


# =============================================================================
# Predictor Configuration
# =============================================================================

class PredictorConfig(BaseModel):
    """Configuration for ThunderstormPredictor component."""

    model_type: ModelType = ModelType.CONV_GRU
    checkpoints_dir: Path = MODELS_PATH / "checkpoints"
    input_channels: int = 1
    hidden_channels: int = 64
    output_channels: int = 1
    kernel_size: int = 3
    num_layers: int = 2
    image_size: int = 256
    input_frames: int = 6
    output_frames: int = 6
    delta_minutes: int = 5

    @property
    def device(self) -> device:
        """Get the device for model inference."""
        import torch

        return device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Settings
# =============================================================================

class Settings(BaseModel):
    parser_config: ParserConfig = ParserConfig()
    preprocessor_config: PreprocessorConfig = PreprocessorConfig()
    masked_preprocessor_config: MaskedPreprocessorConfig = MaskedPreprocessorConfig()
    static_avoider_config: StaticAvoiderConfig = StaticAvoiderConfig()
    dynamic_avoider_config: DynamicAvoiderConfig = DynamicAvoiderConfig()
    predictor_config: PredictorConfig = PredictorConfig()


settings = Settings()
