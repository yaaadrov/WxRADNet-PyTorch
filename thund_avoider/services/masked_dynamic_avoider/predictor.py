import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import rasterio
import torch
from affine import Affine
from shapely import Point, Polygon

from thund_avoider.schemas.masked_dynamic_avoider import DirectionVector
from thund_avoider.schemas.predictor import ModelType, PredictionResult
from thund_avoider.services.masked_dynamic_avoider.masked_preprocessor import MaskedPreprocessor
from thund_avoider.settings import IMAGES_PATH, PredictorConfig


class ThunderstormPredictor:
    """
    Thunderstorm prediction using Seq2Seq deep learning models.

    Predicts future thunderstorm positions based on historical radar imagery
    and converts predictions to polygon obstacles for pathfinding.
    """

    def __init__(self, config: PredictorConfig, preprocessor: MaskedPreprocessor) -> None:
        """
        Initialize ThunderstormPredictor with configuration and preprocessor.

        Args:
            config (PredictorConfig): Predictor configuration.
            preprocessor (MaskedPreprocessor): Preprocessor for image cropping and polygon extraction.
        """
        self._config = config
        self._preprocessor = preprocessor
        self._model = self._load_model()

    # ==========================================================================
    # Model Loading and Initialization
    # ==========================================================================

    def _load_model(self) -> torch.nn.Module:
        """
        Load the Seq2Seq model based on configuration.

        Returns:
            torch.nn.Module: Loaded model in evaluation mode.
        """
        checkpoint_path = self._config.checkpoints_dir / self._config.model_type.checkpoint_filename

        model = self._create_model_instance(self._config.model_type)
        model.load_state_dict(torch.load(checkpoint_path, map_location=self._config.device, weights_only=True))
        model.eval()
        return model

    def _create_model_instance(self, model_type: ModelType) -> torch.nn.Module:
        """
        Create model instance based on model type.

        Args:
            model_type (ModelType): Type of model to create.

        Returns:
            torch.nn.Module: Uninitialized model instance.
        """
        if model_type.is_convolutional:
            return self._create_conv_model(model_type)
        return self._create_rnn_model(model_type)

    def _create_conv_model(self, model_type: ModelType) -> torch.nn.Module:
        """Create convolutional Seq2Seq model."""
        # Import here to avoid circular imports and allow lazy loading
        from thund_avoider.models.conv_gru import ConvGRUSeq2Seq
        from thund_avoider.models.conv_lstm import ConvLSTMSeq2Seq
        from thund_avoider.models.conv_rnn import ConvRNNSeq2Seq

        match model_type:
            case ModelType.CONV_LSTM:
                return ConvLSTMSeq2Seq(
                    input_channels=self._config.input_channels,
                    hidden_channels=self._config.hidden_channels,
                    output_channels=self._config.output_channels,
                    kernel_size=self._config.kernel_size,
                    num_layers=self._config.num_layers,
                    device=self._config.device,
                )
            case ModelType.CONV_GRU:
                return ConvGRUSeq2Seq(
                    input_channels=self._config.input_channels,
                    hidden_channels=self._config.hidden_channels,
                    output_channels=self._config.output_channels,
                    kernel_size=self._config.kernel_size,
                    num_layers=self._config.num_layers,
                    device=self._config.device,
                )
            case ModelType.CONV_RNN:
                return ConvRNNSeq2Seq(
                    input_channels=self._config.input_channels,
                    hidden_channels=self._config.hidden_channels,
                    output_channels=self._config.output_channels,
                    kernel_size=self._config.kernel_size,
                    num_layers=self._config.num_layers,
                    device=self._config.device,
                )
            case _:
                raise ValueError(f"Unknown convolutional model type: {model_type}")

    def _create_rnn_model(self, model_type: ModelType) -> torch.nn.Module:
        """Create RNN-based Seq2Seq model."""
        from thund_avoider.models.rnn import GRUDecoder, GRUEncoder, LSTMDecoder, LSTMEncoder
        from thund_avoider.models.rnn import RNNEncoder, RNNDecoder, Seq2SeqModel

        input_size = self._config.image_size * self._config.image_size
        hidden_size = self._config.hidden_channels
        output_size = input_size

        match model_type:
            case ModelType.LSTM:
                encoder = LSTMEncoder(input_size, hidden_size, self._config.num_layers)
                decoder = LSTMDecoder(hidden_size, output_size, self._config.num_layers)
            case ModelType.GRU:
                encoder = GRUEncoder(input_size, hidden_size, self._config.num_layers)
                decoder = GRUDecoder(hidden_size, output_size, self._config.num_layers)
            case ModelType.RNN:
                encoder = RNNEncoder(input_size, hidden_size, self._config.num_layers)
                decoder = RNNDecoder(hidden_size, output_size, self._config.num_layers)
            case _:
                raise ValueError(f"Unknown RNN model type: {model_type}")

        return Seq2SeqModel(encoder, decoder)

    # ==========================================================================
    # Time Key Utilities
    # ==========================================================================

    @staticmethod
    def _parse_time_key(time_key: str) -> datetime:
        """Parse time key string to datetime."""
        return datetime.strptime(time_key, "%Y_%m_%d_%H_%M_%S")

    @staticmethod
    def _format_time_key(dt: datetime) -> str:
        """Format datetime to time key string."""
        return dt.strftime("%Y_%m_%d_%H_%M_%S")

    def _get_previous_time_keys(self, current_time_key: str) -> list[str]:
        """Get time keys for previous frames with delta_minutes interval."""
        current_dt = self._parse_time_key(current_time_key)
        delta = timedelta(minutes=self._config.delta_minutes)

        previous_keys = []
        for i in range(self._config.input_frames -1, -1, -1):
            prev_dt = current_dt - (delta * i)
            previous_keys.append(self._format_time_key(prev_dt))

        return previous_keys

    def _get_future_time_keys(self, current_time_key: str,) -> list[str]:
        """Get time keys for future predictions with delta_minutes interval."""
        current_dt = self._parse_time_key(current_time_key)
        delta = timedelta(minutes=self._config.delta_minutes)

        future_keys = []
        for i in range(1, self._config.output_frames + 1):
            future_dt = current_dt + (delta * i)
            future_keys.append(self._format_time_key(future_dt))

        return future_keys

    # ==========================================================================
    # Image Loading and Preprocessing
    # ==========================================================================

    def _get_image_path(self, time_key: str, base_time_key: str) -> Path:
        """
        Get the path to a GeoTIFF image.

        Args:
            time_key (str): Time key for the image.
            base_time_key (str): Base time key for directory.

        Returns:
            Path: Path to the GeoTIFF file.
        """
        return IMAGES_PATH / base_time_key / f"{time_key}.tif"

    def _load_and_crop_image(
        self,
        time_key: str,
        base_time_key: str,
        current_position: Point,
        direction_vector: DirectionVector,
        strategy: Literal["center", "left", "right", "wide"],
        by_url: bool = False,
    ) -> tuple[np.ndarray, Affine, rasterio.crs.CRS]:
        """
        Load and crop a single image using MaskedPreprocessor.

        Args:
            time_key (str): Time key for the image.
            base_time_key (str): Base time key for directory.
            current_position (Point): Aircraft position.
            direction_vector (DirectionVector): Direction vector for cropping.
            strategy (Literal["left", "right"]): Cropping strategy.

        Returns:
            tuple: Cropped data, transform, and CRS.
        """
        image_path = self._get_image_path(time_key, base_time_key)
        if by_url:
            image_path = self._preprocessor.generate_url(self._parse_time_key(time_key))
        return self._preprocessor.fetch_and_crop_raster_data(
            url=image_path,
            current_position=current_position,
            direction_vector=direction_vector,
            strategy=strategy,
        )

    @staticmethod
    def _resize_with_max_pooling(data: np.ndarray, target_size: int) -> np.ndarray:
        current_h, current_w = data.shape[:2]
        lcm_h, lcm_w = (
           (current_h * target_size) // math.gcd(current_h, target_size),
           (current_w * target_size) // math.gcd(current_w, target_size)
        )
        intermediate_img = cv2.resize(data, (lcm_w, lcm_h), interpolation=cv2.INTER_NEAREST)
        block_h, block_w = lcm_h // target_size, lcm_w // target_size
        reshaped = intermediate_img.reshape(target_size, block_h, target_size, block_w)
        pooled = reshaped.max(axis=(1, 3))
        return pooled

        # h, w = data.shape[:2]
        # pad_h = math.ceil(h / target_size) * target_size - h
        # pad_w = math.ceil(w / target_size) * target_size - w
        # padded = np.pad(data, ((0, pad_h), (0, pad_w)), mode="constant")
        #
        # block_h = padded.shape[0] // target_size
        # block_w = padded.shape[1] // target_size
        # return padded.reshape(target_size, block_h, target_size, block_w).max(axis=(1, 3))

    def _preprocess_image_for_model(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess image data for model input.

        Resize to model image size and apply threshold.

        Args:
            data (np.ndarray): Raw image data.

        Returns:
            np.ndarray: Preprocessed image.
        """
        resized = self._resize_with_max_pooling(data, self._config.image_size)
        thresholded = np.where(resized > self._preprocessor.intensity_threshold_low, resized, 0)
        return thresholded.reshape(self._config.image_size, self._config.image_size, 1)

    def _load_images_for_strategy(
        self,
        time_keys: list[str],
        base_time_key: str,
        current_position: Point,
        direction_vector: DirectionVector,
        strategy: Literal["left", "right"],
        by_url: bool = False,
    ) -> tuple[list[np.ndarray], Affine, rasterio.crs.CRS]:
        """
        Load and preprocess images for a cropping strategy.

        Args:
            time_keys: List of time keys to load.
            base_time_key: Base time key for directory.
            current_position: Aircraft position.
            direction_vector: Direction vector for cropping.
            strategy: Cropping strategy ("left" or "right").

        Returns:
            tuple: List of preprocessed images, transform, and CRS.
        """
        images = []
        transform = None
        crs = None

        for time_key in time_keys:
            data, transform, crs = self._load_and_crop_image(
                time_key=time_key,
                base_time_key=base_time_key,
                current_position=current_position,
                direction_vector=direction_vector,
                strategy=strategy,
                by_url=by_url,
            )
            images.append(self._preprocess_image_for_model(data))

        return images, transform, crs

    # ==========================================================================
    # Prediction Pipeline
    # ==========================================================================

    def _prepare_input_tensor(
        self,
        images: list[np.ndarray],
    ) -> torch.Tensor:
        """
        Prepare input tensor from list of preprocessed images.

        Args:
            images (list[np.ndarray]): List of preprocessed images.

        Returns:
            torch.Tensor: Input tensor of shape [1, seq_len, channels, height, width].
        """
        stacked = np.stack(images, axis=0)
        transposed = np.transpose(stacked, (0, 3, 1, 2))
        tensor = torch.from_numpy(transposed.astype(np.float32))
        return tensor.unsqueeze(0).to(self._config.device)

    def _run_inference(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Run model inference and return predictions.

        Args:
            input_tensor (torch.Tensor): Input tensor.

        Returns:
            np.ndarray: Predicted frames as numpy array.
        """
        with torch.no_grad():
            predictions = self._model(input_tensor, self._config.output_frames)
        return predictions.squeeze(0).cpu().numpy()

    @staticmethod
    def _combine_left_right_predictions(
        left_predictions: np.ndarray,
        right_predictions: np.ndarray,
    ) -> np.ndarray:
        """
        Combine left and right predictions into wide format.

        Args:
            left_predictions (np.ndarray): Left predictions [frames, channels, h, w].
            right_predictions (np.ndarray): Right predictions [frames, channels, h, w].

        Returns:
            np.ndarray: Combined predictions [frames, channels, h, w*2].
        """
        return np.concatenate([left_predictions, right_predictions], axis=3)

    # ==========================================================================
    # Main Public Methods
    # ==========================================================================

    def predict(
        self,
        time_keys: list[str],
        current_time_index: int,
        current_position: Point,
        direction_vector: DirectionVector,
        strategy: Literal["concave", "convex"] = "concave",
        by_url: bool = False,
    ) -> PredictionResult:
        """
        Generate thunderstorm predictions for future time steps.

        Args:
            time_keys (list[str]): List of all available time keys.
            current_time_index (int): Index of current time in time_keys.
            current_position (Point): Current aircraft position.
            direction_vector (DirectionVector): Direction vector for cropping.
            strategy (Literal["concave", "convex"]): Hull strategy for polygons.

        Returns:
            PredictionResult: Container with predicted obstacles for each time key.
        """
        current_time_key = time_keys[current_time_index]
        base_time_key = time_keys[0]

        # Get time keys
        previous_keys = self._get_previous_time_keys(current_time_key)
        future_keys = self._get_future_time_keys(current_time_key)

        # Load images for both strategies
        left_images, left_transform, left_crs = self._load_images_for_strategy(
            time_keys=previous_keys,
            base_time_key=base_time_key,
            current_position=current_position,
            direction_vector=direction_vector,
            strategy="left",
            by_url=by_url,
        )
        right_images, _, _ = self._load_images_for_strategy(
            time_keys=previous_keys,
            base_time_key=base_time_key,
            current_position=current_position,
            direction_vector=direction_vector,
            strategy="right",
            by_url=by_url,
        )

        # Run inference
        left_predictions = self._run_inference(self._prepare_input_tensor(left_images))
        right_predictions = self._run_inference(self._prepare_input_tensor(right_images))
        combined_predictions = self._combine_left_right_predictions(left_predictions, right_predictions)

        # Build wide transform using preprocessor
        # Use image_size as pixel_width to match the resized prediction dimensions
        wide_transform, _ = self._preprocessor.build_transform(
            current_position=current_position,
            direction_vector=direction_vector,
            strategy="wide",
            res=self._config.image_size,  # Placeholder, overridden by pixel_width
            pixel_width=self._config.image_size,
        )

        # Convert predictions to polygons
        obstacles_dict: dict[str, dict[str, list[Polygon]]] = {
            future_key: {
                strategy: self._preprocessor.prediction_to_polygons(
                    prediction=combined_predictions[i],
                    transform=wide_transform,
                    crs=left_crs,
                    strategy=strategy,
                )
            }
            for i, future_key in enumerate(future_keys)
        }

        return PredictionResult(
            time_keys=future_keys,
            obstacles_dict=obstacles_dict,
            model_type=self._config.model_type,
            strategy=strategy,
        )
