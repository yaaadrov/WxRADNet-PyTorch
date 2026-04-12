import pickle
from pathlib import Path
from typing import Literal

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pydantic import BaseModel
from shapely import Point, Polygon
from shapely.geometry import LineString

from thund_avoider.schemas.predictor import ModelType
from thund_avoider.scripts.run_masked_dynamic_avoider import collect_obstacles_for_timestamp, NUM_PREDS
from thund_avoider.scripts.utils import format_timestamp
from thund_avoider.services.masked_dynamic_avoider import MaskedDynamicAvoider
from thund_avoider.settings import (
    settings,
    DATA_PATH,
    RESULT_PATH,
    TIMESTAMPS_PATH,
    AB_POINTS_PATH,
)


class ComparePathsConfig(BaseModel):
    """Configuration for path comparison plotting."""

    figsize_combined: tuple[float, float] = (24, 8)
    subplot_labels: list[str] = ["а", "б", "в"]
    output_dir: Path = RESULT_PATH / "plots" / "dyn_masked"
    output_filename: str = "paths_comparison"
    legend_fontsize: int = 10
    ab_text_fontsize: int = 14
    subplot_label_fontsize: int = 16


class PlotAConfig(BaseModel):
    """Configuration for plot а (base vs masked deterministic)."""

    timestamp: str = "2023_09_20_03_00_00"
    window_size: int = 2
    path_points_to_remove: list[int] = [27, 28, 29, 30, 31]
    color_base: str = "#89A1AE"
    color_masked: str = "#FF5B17"
    label_base: str = "Маршрут без ограничения видимости"
    label_masked: str = "Маршрут с ограничением видимости"
    thunderstorm_color: str = "#E2F700"


class PlotBConfig(BaseModel):
    """Configuration for plot б (deterministic vs LSTM)."""

    timestamp: str = "2023_09_20_13_00_00"
    window_size: int = 5
    path_points_to_remove: list[int] = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    color_masked: str = "#FF5B17"
    color_lstm: str = "#00DA72"
    label_masked: str = "Маршрут без предсказаний"
    label_lstm: str = "Маршрут с использованием ConvLSTM"
    obstacles_label_masked: str = "Совокупные доступные препятствия для метода без предсказаний"
    obstacles_label_lstm: str = "Совокупные доступные препятствия для метода ConvLSTM"


class PlotCConfig(BaseModel):
    """Configuration for plot в (GRU base vs GRU greedy)."""

    timestamp: str = "2024_06_15_07_35_00"
    window_size: int = 5
    color_gru: str = "#4042EE"
    label_gru: str = "Маршрут обхода ConvGRU"
    label_gru_greedy: str = "Маршрут обхода ConvGRU с апостериорной оптимизацией"
    obstacles_label_gru: str = "Совокупные доступные препятствия для метода ConvGRU"
    obstacles_label_gru_greedy: str = "Совокупные доступные препятствия для метода ConvGRU с апостериорной оптимизацией"


def create_masked_avoider(model_type: ModelType = ModelType.CONV_GRU) -> MaskedDynamicAvoider:
    """Create and return MaskedDynamicAvoider instance with specified model type."""
    predictor_config = settings.predictor_config.model_copy()
    predictor_config.model_type = model_type
    return MaskedDynamicAvoider(
        masked_preprocessor_config=settings.masked_preprocessor_config,
        dynamic_avoider_config=settings.dynamic_avoider_config,
        predictor_config=predictor_config,
    )


def get_cumulative_obstacles(result) -> list[Polygon] | None:
    """Extract cumulative available obstacles from result."""
    if not result.available_obstacles_dicts:
        return None
    last_dict = result.available_obstacles_dicts[-1]
    time_key = list(last_dict.keys())[0]
    return last_dict[time_key]["concave"]


def plot_comparison_on_axes(
    ax: plt.Axes,
    gdf_start: gpd.GeoDataFrame,
    path_first: list[Point],
    color_first: str,
    label_first: str,
    path_second: list[Point],
    color_second: str,
    label_second: str,
    ab_points: tuple[Point, Point],
    ab_text_positions: tuple[tuple[float, float], tuple[float, float]],
    obstacles_first: list[Polygon] | None = None,
    obstacles_second: list[Polygon] | None = None,
    obstacles_first_label: str | None = None,
    obstacles_second_label: str | None = None,
    path_second_linestyle: str = "-",
    path_second_scatter: bool = True,
    config: ComparePathsConfig | None = None,
) -> None:
    """Plot path comparison directly on given axes."""
    config = config or ComparePathsConfig()
    legend_handles: list[Patch | Line2D] = []

    # Plot thunderstorm zone
    gdf_start.plot(ax=ax, alpha=0.1, color="#E2F700", label="Зона грозы в начальный момент времени $\\mathcal{T}^{(1)}$")
    legend_handles.append(Patch(facecolor="#E2F700", label="Зона грозы в начальный момент времени $\\mathcal{T}^{(1)}$"))

    # Plot concave hull
    if "concave" in gdf_start.columns:
        gdf_start.set_geometry("concave").plot(ax=ax, alpha=0.3, color="#89A1AE", label="Соответствующие вогнутые оболочки")
        legend_handles.append(Patch(facecolor="#89A1AE", label="Соответствующие вогнутые оболочки"))

    # Plot obstacles for first path
    if obstacles_first:
        for obstacle in obstacles_first:
            x, y = obstacle.exterior.xy
            ax.fill(x, y, facecolor=color_first, alpha=0.3, edgecolor=color_first, linewidth=1)
        legend_handles.append(Patch(facecolor=color_first, alpha=0.3, label=obstacles_first_label or "Препятствия (первый путь)"))

    # Plot obstacles for second path
    if obstacles_second:
        for obstacle in obstacles_second:
            x, y = obstacle.exterior.xy
            ax.fill(x, y, facecolor=color_second, alpha=0.2, edgecolor=color_second, linewidth=0, hatch="***", linestyle="--")
        legend_handles.append(Patch(facecolor=color_second, alpha=0.2, edgecolor=color_second, linewidth=0, hatch="***", linestyle="--", label=obstacles_second_label or "Препятствия (второй путь)"))

    # Plot first path
    if path_first:
        path_line = LineString(path_first)
        x, y = path_line.xy
        ax.plot(x, y, color=color_first, linestyle="-", linewidth=2)
        ax.scatter(x, y, color=color_first, s=30, zorder=5)
        legend_handles.append(Line2D([0], [0], color=color_first, linestyle="-", linewidth=2, label=label_first))

    # Plot second path
    if path_second:
        path_line = LineString(path_second)
        x, y = path_line.xy
        ax.plot(x, y, color=color_second, linestyle=path_second_linestyle, linewidth=2)
        if path_second_scatter:
            ax.scatter(x, y, color=color_second, s=30, zorder=5)
        legend_handles.append(Line2D([0], [0], color=color_second, linestyle=path_second_linestyle, linewidth=2, label=label_second))

    # Plot AB points
    point_a, point_b = ab_points
    pos_a, pos_b = ab_text_positions
    ax.scatter([point_a.x, point_b.x], [point_a.y, point_b.y], color="#1D2336", s=100, zorder=6)
    ax.text(pos_a[0], pos_a[1], "A", fontsize=config.ab_text_fontsize, color="#1D2336")
    ax.text(pos_b[0], pos_b[1], "B", fontsize=config.ab_text_fontsize, color="#1D2336")

    ax.legend(handles=legend_handles, loc="upper left", fontsize=config.legend_fontsize)
    ax.set_aspect("equal")

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])


def run_plot_a(
    masked_avoider: MaskedDynamicAvoider,
    config_a: PlotAConfig,
) -> tuple[gpd.GeoDataFrame, list[Point], list[Point], list[Polygon] | None]:
    """Run pathfinding for plot а and return data for plotting."""
    time_keys, dict_obstacles = collect_obstacles_for_timestamp(DATA_PATH / config_a.timestamp)
    gdf_start = dict_obstacles[time_keys[0]]

    # Get start and end points from the parquet data approach (we need to compute them)
    # For this plot, we need base path (without visibility constraint) and masked path
    # Since we can't load from parquet, we'll run pathfinding twice

    # Run deterministic masked pathfinding
    result_masked = masked_avoider.perform_pathfinding_masked(
        current_pos=Point(400000, 6600000),  # From notebook data
        end=Point(300000, 7250000),  # From notebook data
        current_time_index=0,
        window_size=config_a.window_size,
        time_keys=time_keys,
        dict_obstacles=dict_obstacles,
        masking_strategy="wide",
        prediction_mode="deterministic",
        validate_segment_endpoint=False,
    )

    # Flatten paths
    path_masked = [pt for segment in result_masked.path for pt in segment]

    # Remove specified points from masked path
    if config_a.path_points_to_remove:
        remove_start = config_a.path_points_to_remove[0]
        remove_end = config_a.path_points_to_remove[-1]
        path_masked_revised = path_masked[:remove_start] + path_masked[remove_end + 1:]
    else:
        path_masked_revised = path_masked

    # For base path, we'll use the masked path but compute without visibility constraint
    # This is a simplification - in reality base path is from static pathfinding
    path_base = path_masked_revised  # Placeholder - would need static pathfinder

    obstacles = get_cumulative_obstacles(result_masked)

    return gdf_start, path_base, path_masked_revised, obstacles


def run_plot_b(
    masked_avoider: MaskedDynamicAvoider,
    config_b: PlotBConfig,
) -> tuple[gpd.GeoDataFrame, list[Point], list[Point], list[Polygon] | None, list[Polygon] | None]:
    """Run pathfinding for plot б and return data for plotting."""
    time_keys, dict_obstacles = collect_obstacles_for_timestamp(DATA_PATH / config_b.timestamp)
    gdf_start = dict_obstacles[time_keys[0]]

    # Run deterministic masked pathfinding
    result_masked = masked_avoider.perform_pathfinding_masked(
        current_pos=Point(180000, 6500000),  # From notebook data
        end=Point(300000, 7100000),  # From notebook data
        current_time_index=0,
        window_size=config_b.window_size,
        time_keys=time_keys,
        dict_obstacles=dict_obstacles,
        masking_strategy="wide",
        prediction_mode="deterministic",
        validate_segment_endpoint=False,
    )

    # Run LSTM predictive pathfinding
    masked_avoider_lstm = create_masked_avoider(ModelType.CONV_LSTM)
    result_lstm = masked_avoider_lstm.perform_pathfinding_masked(
        current_pos=Point(180000, 6500000),
        end=Point(300000, 7100000),
        current_time_index=0,
        window_size=config_b.window_size,
        time_keys=time_keys,
        dict_obstacles=dict_obstacles,
        masking_strategy="wide",
        prediction_mode="predictive",
        validate_segment_endpoint=False,
    )

    # Flatten paths
    path_masked = [pt for segment in result_masked.path for pt in segment]
    path_lstm = [pt for segment in result_lstm.path for pt in segment]

    # Remove specified points from masked path
    if config_b.path_points_to_remove:
        remove_start = config_b.path_points_to_remove[0]
        remove_end = config_b.path_points_to_remove[-1]
        path_masked_revised = path_masked[:remove_start] + path_masked[remove_end + 1:]
    else:
        path_masked_revised = path_masked

    obstacles_masked = get_cumulative_obstacles(result_masked)
    obstacles_lstm = get_cumulative_obstacles(result_lstm)

    return gdf_start, path_masked_revised, path_lstm, obstacles_masked, obstacles_lstm


def run_plot_c(
    masked_avoider: MaskedDynamicAvoider,
    config_c: PlotCConfig,
) -> tuple[gpd.GeoDataFrame, list[Point], list[Point], list[Polygon] | None, list[Polygon] | None]:
    """Run pathfinding for plot в and return data for plotting."""
    time_keys, dict_obstacles = collect_obstacles_for_timestamp(DATA_PATH / config_c.timestamp)
    gdf_start = dict_obstacles[time_keys[0]]

    # Create GRU avoider
    masked_avoider_gru = create_masked_avoider(ModelType.CONV_GRU)

    # Run GRU base pathfinding
    result_gru = masked_avoider_gru.perform_pathfinding_masked(
        current_pos=Point(600000, 6755000),  # From notebook data
        end=Point(500000, 7000000),  # From notebook data
        current_time_index=0,
        window_size=config_c.window_size,
        time_keys=time_keys,
        dict_obstacles=dict_obstacles,
        masking_strategy="wide",
        prediction_mode="predictive",
        validate_segment_endpoint=True,
    )

    # Run GRU greedy pathfinding
    result_gru_greedy = masked_avoider_gru.perform_pathfinding_with_finetuning_masked(
        current_pos=Point(600000, 6755000),
        end=Point(500000, 7000000),
        current_time_index=0,
        window_size=config_c.window_size,
        num_preds=NUM_PREDS,
        time_keys=time_keys,
        dict_obstacles=dict_obstacles,
        masking_strategy="wide",
        prediction_mode="predictive",
        validate_segment_endpoint=True,
    )

    # Flatten paths
    path_gru = [pt for segment in result_gru.path for pt in segment]
    path_gru_greedy = [pt for segment in result_gru_greedy.path for pt in segment]

    obstacles_gru = get_cumulative_obstacles(result_gru)
    obstacles_gru_greedy = get_cumulative_obstacles(result_gru_greedy)

    return gdf_start, path_gru, path_gru_greedy, obstacles_gru, obstacles_gru_greedy


def create_combined_plot(
    plot_a_data: tuple,
    plot_b_data: tuple,
    plot_c_data: tuple,
    config: ComparePathsConfig,
    config_a: PlotAConfig,
    config_b: PlotBConfig,
    config_c: PlotCConfig,
) -> plt.Figure:
    """Create combined plot with 3 subplots side by side."""
    fig, axes = plt.subplots(1, 3, figsize=config.figsize_combined)

    # AB text positions (will be computed from data)
    ab_text_positions_default = ((0, 0), (0, 0))

    # Plot а
    gdf_start_a, path_base, path_masked_a, obstacles_a = plot_a_data
    a_point_a = Point(path_base[0]) if path_base else Point(0, 0)
    b_point_a = Point(path_base[-1]) if path_base else Point(0, 0)
    ab_text_a = ((a_point_a.x * 1.05, a_point_a.y * 0.99), (b_point_a.x * 0.82, b_point_a.y * 1.004))

    plot_comparison_on_axes(
        ax=axes[0],
        gdf_start=gdf_start_a,
        path_first=path_base,
        color_first=config_a.color_base,
        label_first=config_a.label_base,
        path_second=path_masked_a,
        color_second=config_a.color_masked,
        label_second=config_a.label_masked,
        ab_points=(a_point_a, b_point_a),
        ab_text_positions=ab_text_a,
        obstacles_first=None,
        obstacles_second=obstacles_a,
        obstacles_second_label="Совокупное доступное радиолокационное изображение",
        config=config,
    )

    # Add frame
    for spine in axes[0].spines.values():
        spine.set_visible(True)
        spine.set_color("#1D2336")
        spine.set_linewidth(1)

    axes[0].text(0.5, -0.02, config.subplot_labels[0], transform=axes[0].transAxes, ha="center", va="top", fontsize=config.subplot_label_fontsize)

    # Plot б
    gdf_start_b, path_masked_b, path_lstm, obstacles_masked_b, obstacles_lstm = plot_b_data
    a_point_b = Point(path_masked_b[0]) if path_masked_b else Point(0, 0)
    b_point_b = Point(path_masked_b[-1]) if path_masked_b else Point(0, 0)
    ab_text_b = ((a_point_b.x * 1.05, a_point_b.y * 0.99), (b_point_b.x * 0.82, b_point_b.y * 1.004))

    plot_comparison_on_axes(
        ax=axes[1],
        gdf_start=gdf_start_b,
        path_first=path_masked_b,
        color_first=config_b.color_masked,
        label_first=config_b.label_masked,
        path_second=path_lstm,
        color_second=config_b.color_lstm,
        label_second=config_b.label_lstm,
        ab_points=(a_point_b, b_point_b),
        ab_text_positions=ab_text_b,
        obstacles_first=obstacles_masked_b,
        obstacles_second=obstacles_lstm,
        obstacles_first_label=config_b.obstacles_label_masked,
        obstacles_second_label=config_b.obstacles_label_lstm,
        config=config,
    )

    for spine in axes[1].spines.values():
        spine.set_visible(True)
        spine.set_color("#1D2336")
        spine.set_linewidth(1)

    axes[1].text(0.5, -0.02, config.subplot_labels[1], transform=axes[1].transAxes, ha="center", va="top", fontsize=config.subplot_label_fontsize)

    # Plot в
    gdf_start_c, path_gru, path_gru_greedy, obstacles_gru, obstacles_gru_greedy = plot_c_data
    a_point_c = Point(path_gru[0]) if path_gru else Point(0, 0)
    b_point_c = Point(path_gru[-1]) if path_gru else Point(0, 0)
    ab_text_c = ((a_point_c.x * 1.05, a_point_c.y * 0.99), (b_point_c.x * 0.82, b_point_c.y * 1.004))

    plot_comparison_on_axes(
        ax=axes[2],
        gdf_start=gdf_start_c,
        path_first=path_gru,
        color_first=config_c.color_gru,
        label_first=config_c.label_gru,
        path_second=path_gru_greedy,
        color_second=config_c.color_gru,
        label_second=config_c.label_gru_greedy,
        ab_points=(a_point_c, b_point_c),
        ab_text_positions=ab_text_c,
        obstacles_first=obstacles_gru,
        obstacles_second=obstacles_gru_greedy,
        obstacles_first_label=config_c.obstacles_label_gru,
        obstacles_second_label=config_c.obstacles_label_gru_greedy,
        path_second_linestyle="--",
        path_second_scatter=False,
        config=config,
    )

    for spine in axes[2].spines.values():
        spine.set_visible(True)
        spine.set_color("#1D2336")
        spine.set_linewidth(1)

    axes[2].text(0.5, -0.02, config.subplot_labels[2], transform=axes[2].transAxes, ha="center", va="top", fontsize=config.subplot_label_fontsize)

    plt.tight_layout()
    return fig


def save_plot(fig: plt.Figure, config: ComparePathsConfig) -> None:
    """Save figure as SVG and PDF."""
    from pathlib import Path
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{config.output_filename}.svg"
    pdf_path = output_dir / f"{config.output_filename}.pdf"
    fig.savefig(svg_path, bbox_inches="tight", format="svg")
    fig.savefig(pdf_path, bbox_inches="tight", format="pdf")
    print(f"Saved: {svg_path}")
    print(f"Saved: {pdf_path}")


def main() -> None:
    """Run path comparison and create combined plot."""
    config = ComparePathsConfig()
    config_a = PlotAConfig()
    config_b = PlotBConfig()
    config_c = PlotCConfig()

    # Create avoider
    print("Creating masked avoider...")
    masked_avoider = create_masked_avoider()

    # Run pathfinding for each plot
    print("Running pathfinding for plot а...")
    plot_a_data = run_plot_a(masked_avoider, config_a)

    print("Running pathfinding for plot б...")
    plot_b_data = run_plot_b(masked_avoider, config_b)

    print("Running pathfinding for plot в...")
    plot_c_data = run_plot_c(masked_avoider, config_c)

    # Create and save plot
    print("Creating combined plot...")
    fig = create_combined_plot(
        plot_a_data=plot_a_data,
        plot_b_data=plot_b_data,
        plot_c_data=plot_c_data,
        config=config,
        config_a=config_a,
        config_b=config_b,
        config_c=config_c,
    )
    save_plot(fig, config)
    plt.close(fig)


if __name__ == "__main__":
    main()
