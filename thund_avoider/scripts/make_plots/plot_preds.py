import math
from pathlib import Path
from typing import Literal

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pydantic import BaseModel
from shapely import Point, Polygon

from thund_avoider.schemas.masked_dynamic_avoider import DirectionVector
from thund_avoider.schemas.predictor import ModelType, PredictionResult
from thund_avoider.scripts.run_masked_dynamic_avoider import collect_obstacles_for_timestamp
from thund_avoider.services.masked_dynamic_avoider.masked_preprocessor import MaskedPreprocessor
from thund_avoider.services.masked_dynamic_avoider.predictor import ThunderstormPredictor
from thund_avoider.settings import (
    settings,
    DATA_PATH,
    IMAGES_PATH,
    RESULT_PATH,
    PredictorConfig,
)


class PlotPredsConfig(BaseModel):
    """Configuration for prediction visualization plotting."""

    timestamp: str = "2023_09_20_03_00_00"
    point_coords: tuple[float, float] = (400000.0, 6600000.0)
    num_plots: int = 4
    figsize_per_plot: tuple[float, float] = (6, 6)
    output_dir: Path = RESULT_PATH / "plots" / "dyn_masked"
    output_filename: str = "predictions_comparison"
    legend_fontsize: int = 8

    # Colors
    geometry_color: str = "#E2F700"
    concave_color: str = "#89A1AE"
    convgru_color: str = "#4042EE"
    convlstm_color: str = "#00DA72"


def load_data_and_images(
    timestamp: str,
) -> tuple[list[str], dict, Path]:
    """
    Load obstacles and find corresponding images directory.

    Args:
        timestamp (str): Timestamp string for data directory.

    Returns:
        tuple: time_keys, dict_obstacles, images_dir.
    """
    time_keys, dict_obstacles = collect_obstacles_for_timestamp(DATA_PATH / timestamp)

    # Find corresponding images directory (5 images before data start)
    # The first data time key corresponds to images from 25 minutes earlier
    images_dir = IMAGES_PATH / timestamp

    return time_keys, dict_obstacles, images_dir


def create_bounding_box(
    point_coords: tuple[float, float],
    masked_preprocessor: MaskedPreprocessor,
) -> Polygon:
    """
    Create oriented bounding box for visualization.

    Args:
        point_coords (tuple[float, float]): (x, y) center coordinates.
        masked_preprocessor (MaskedPreprocessor): Preprocessor instance.

    Returns:
        Polygon: Oriented bounding box.
    """
    bbox = masked_preprocessor.get_oriented_bbox(
        current_position=Point(point_coords),
        direction_vector=DirectionVector(dx=0, dy=1),
        strategy="center",
    )
    return bbox


def create_predictor(
    model_type: ModelType,
    masked_preprocessor: MaskedPreprocessor,
) -> ThunderstormPredictor:
    """
    Create ThunderstormPredictor with specified model type.

    Args:
        model_type (ModelType): Type of model (ConvGRU or ConvLSTM).
        masked_preprocessor (MaskedPreprocessor): Preprocessor instance.

    Returns:
        ThunderstormPredictor: Configured predictor instance.
    """
    config = PredictorConfig(model_type=model_type)
    return ThunderstormPredictor(config=config, preprocessor=masked_preprocessor)


def run_predictions(
    time_keys: list[str],
    point_coords: tuple[float, float],
    masked_preprocessor: MaskedPreprocessor,
    strategy: Literal["concave", "convex"] = "concave",
    output_frames: int = 6,
) -> tuple[PredictionResult, PredictionResult]:
    """
    Run predictions for both ConvGRU and ConvLSTM models.

    Args:
        time_keys (list[str]): List of time keys.
        point_coords (tuple[float, float]): Center point coordinates.
        masked_preprocessor (MaskedPreprocessor): Preprocessor instance.
        strategy (Literal["concave", "convex"]): Hull strategy.
        output_frames (int): Number of output frames.

    Returns:
        tuple: (result_gru, result_lstm) prediction results.
    """
    point = Point(point_coords)
    direction_vector = DirectionVector(dx=0, dy=1)

    # ConvGRU prediction
    predictor_gru = create_predictor(ModelType.CONV_GRU, masked_preprocessor)
    result_gru = predictor_gru.predict(
        time_keys=time_keys,
        current_time_index=0,
        current_position=point,
        direction_vector=direction_vector,
        strategy=strategy,
        by_url=False,
        output_frames=output_frames,
    )

    # ConvLSTM prediction
    predictor_lstm = create_predictor(ModelType.CONV_LSTM, masked_preprocessor)
    result_lstm = predictor_lstm.predict(
        time_keys=time_keys,
        current_time_index=0,
        current_position=point,
        direction_vector=direction_vector,
        strategy=strategy,
        by_url=False,
        output_frames=output_frames,
    )

    return result_gru, result_lstm


def plot_predictions_on_axes(
    ax: plt.Axes,
    gdf: gpd.GeoDataFrame,
    predicted_polygons: list[Polygon],
    prediction_color: str,
    config: PlotPredsConfig,
) -> None:
    """
    Plot GeoDataFrame and predictions on axes.

    Args:
        ax (plt.Axes): Matplotlib axes.
        gdf (gpd.GeoDataFrame): GeoDataFrame with geometry and concave columns.
        predicted_polygons (list[Polygon]): Predicted polygon obstacles.
        prediction_color (str): Color for predicted polygons.
        config (PlotPredsConfig): Configuration.
    """
    legend_handles: list[Patch | Line2D] = []

    # Plot geometry column
    if "geometry" in gdf.columns:
        gdf.plot(ax=ax, facecolor=config.geometry_color, alpha=0.1, edgecolor="#1D2336", linewidth=1)
    legend_handles.append(
        Patch(facecolor=config.geometry_color, alpha=0.1, label="Зона грозы (фактические данные)")
    )

    # Plot concave column
    if "concave" in gdf.columns:
        gdf.set_geometry("concave").plot(
            ax=ax, facecolor=config.concave_color, alpha=0.3, edgecolor="#1D2336", linewidth=1
        )
    legend_handles.append(
        Patch(facecolor=config.concave_color, alpha=0.3, label="Вогнутые оболочки")
    )

    # Plot predicted polygons
    if predicted_polygons:
        for polygon in predicted_polygons:
            x, y = polygon.exterior.xy
            ax.fill(x, y, facecolor=prediction_color, alpha=0.5, edgecolor=prediction_color, linewidth=1)
    legend_handles.append(
        Patch(facecolor=prediction_color, alpha=0.5, label="Предсказанные препятствия")
    )

    ax.legend(handles=legend_handles, loc="upper left", fontsize=config.legend_fontsize)
    ax.set_aspect("equal")

    # Remove ticks and spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#1D2336")
        spine.set_linewidth(1)
    ax.set_xticks([])
    ax.set_yticks([])


def create_combined_plot(
    dict_obstacles: dict,
    time_keys: list[str],
    result_gru: PredictionResult,
    result_lstm: PredictionResult,
    config: PlotPredsConfig,
) -> plt.Figure:
    """
    Create combined plot with 2 rows (ConvGRU, ConvLSTM) × N columns.

    Args:
        dict_obstacles (dict): Dictionary of obstacles for each time key.
        time_keys (list[str]): List of time keys.
        result_gru (PredictionResult): ConvGRU prediction result.
        result_lstm (PredictionResult): ConvLSTM prediction result.
        config (PlotPredsConfig): Configuration.

    Returns:
        plt.Figure: Combined figure.
    """
    n_cols = config.num_plots
    n_rows = 2

    fig_width = config.figsize_per_plot[0] * n_cols
    fig_height = config.figsize_per_plot[1] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    # Ensure axes is always 2D
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for col in range(n_cols):
        # time_keys[0] is used for prediction input, so actual data starts from [1]
        data_time_key = time_keys[col + 1]
        gdf = dict_obstacles[data_time_key]

        # Prediction for time_keys[col+1] is at result.time_keys[col]
        pred_time_key = result_gru.time_keys[col]
        pred_polygons_gru = result_gru.obstacles_dict[pred_time_key]["concave"]
        pred_polygons_lstm = result_lstm.obstacles_dict[pred_time_key]["concave"]

        # Row 0: ConvGRU
        plot_predictions_on_axes(
            ax=axes[0, col],
            gdf=gdf,
            predicted_polygons=pred_polygons_gru,
            prediction_color=config.convgru_color,
            config=config,
        )

        # Row 1: ConvLSTM
        plot_predictions_on_axes(
            ax=axes[1, col],
            gdf=gdf,
            predicted_polygons=pred_polygons_lstm,
            prediction_color=config.convlstm_color,
            config=config,
        )

        # Add time labels below each column
        time_label = f"$t+{col + 1}$"
        axes[0, col].text(
            0.5, -0.02, time_label, transform=axes[0, col].transAxes,
            ha="center", va="top", fontsize=14
        )

    # Add row labels
    axes[0, 0].set_ylabel("ConvGRU", fontsize=14)
    axes[1, 0].set_ylabel("ConvLSTM", fontsize=14)

    plt.tight_layout()
    return fig


def save_plot(fig: plt.Figure, config: PlotPredsConfig) -> None:
    """
    Save figure as SVG and PDF.

    Args:
        fig (plt.Figure): Figure to save.
        config (PlotPredsConfig): Configuration with output paths.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = config.output_dir / f"{config.output_filename}.svg"
    pdf_path = config.output_dir / f"{config.output_filename}.pdf"
    fig.savefig(svg_path, bbox_inches="tight", format="svg")
    fig.savefig(pdf_path, bbox_inches="tight", format="pdf")
    print(f"Saved: {svg_path}")
    print(f"Saved: {pdf_path}")


def main() -> None:
    """Run prediction visualization and create combined plot."""
    config = PlotPredsConfig()

    # Create masked preprocessor
    masked_preprocessor = MaskedPreprocessor(config=settings.masked_preprocessor_config)

    # Load data
    print(f"Loading data for timestamp: {config.timestamp}")
    time_keys, dict_obstacles, images_dir = load_data_and_images(config.timestamp)
    print(f"Found {len(time_keys)} time keys")
    print(f"Images directory: {images_dir}")

    # Create bounding box for visualization
    bbox = create_bounding_box(config.point_coords, masked_preprocessor)
    print(f"Bounding box created around point: {config.point_coords}")

    # Run predictions
    print("Running predictions with ConvGRU and ConvLSTM...")
    result_gru, result_lstm = run_predictions(
        time_keys=time_keys,
        point_coords=config.point_coords,
        masked_preprocessor=masked_preprocessor,
        strategy="concave",
        output_frames=config.num_plots,
    )
    print(f"ConvGRU prediction keys: {result_gru.time_keys}")
    print(f"ConvLSTM prediction keys: {result_lstm.time_keys}")

    # Create and save plot
    print("Creating combined plot...")
    fig = create_combined_plot(
        dict_obstacles=dict_obstacles,
        time_keys=time_keys,
        result_gru=result_gru,
        result_lstm=result_lstm,
        config=config,
    )
    save_plot(fig, config)
    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    main()
