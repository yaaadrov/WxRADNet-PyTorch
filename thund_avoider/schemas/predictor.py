from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict
from shapely import Polygon


class ModelType(StrEnum):
    """Enumeration of available thunderstorm prediction model types."""

    RNN = "RNN"
    LSTM = "LSTM"
    GRU = "GRU"
    CONV_RNN = "ConvRNN"
    CONV_LSTM = "ConvLSTM"
    CONV_GRU = "ConvGRU"

    @property
    def is_convolutional(self) -> bool:
        """Check if the model is a convolutional variant."""
        match self:
            case ModelType.CONV_RNN | ModelType.CONV_LSTM | ModelType.CONV_GRU:
                return True
            case ModelType.RNN | ModelType.LSTM | ModelType.GRU:
                return False

    @property
    def checkpoint_filename(self) -> str:
        """Get the checkpoint filename for this model type."""
        return f"{self.value}.pt"


class PredictionResult(BaseModel):
    """Container for prediction results with polygons for each time key."""

    time_keys: list[str]
    obstacles_dict: dict[str, dict[str, list[Polygon]]]
    model_type: ModelType
    strategy: Literal["concave", "convex"]

    model_config = ConfigDict(arbitrary_types_allowed=True)
