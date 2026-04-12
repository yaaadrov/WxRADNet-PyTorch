import pickle
from pathlib import Path
from typing import Literal

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pydantic import BaseModel
from shapely import Point
from shapely.geometry import LineString

from thund_avoider.scripts.run_masked_dynamic_avoider import collect_obstacles_for_timestamp
from thund_avoider.scripts.utils import format_timestamp
from thund_avoider.services.masked_dynamic_avoider import MaskedDynamicAvoider
from thund_avoider.settings import (
    settings,
    DATA_PATH,
    RESULT_PATH,
    TIMESTAMPS_PATH,
    AB_POINTS_PATH,
)


class PlotSimulationConfig(BaseModel):
    """Configuration for path simulation plotting."""

    time_index: int = -1
    window_size: int = 1
    prediction_mode: Literal["deterministic", "predictive"] = "deterministic"
    with_fine_tuning: bool = False
    figsize_combined: tuple[float, float] = (16.5, 7)
    subplot_labels: list[str] = ["$t=1$", "$t=2$", "$t=3$"]
    timestamps_to_plot: list[int] = [0, 1, 2]
    path_color: str = "#4042EE"
    path_label: str = "Маршрут обхода"
    output_dir: Path = RESULT_PATH / "plots" / "dyn_masked"
    output_filename: str = "masked_path_simulation"
    legend_fontsize: int = 10
    ab_text_fontsize: int = 24
    subplot_label_fontsize: int = 16


def load_timestamps_and_points() -> tuple[list, list[tuple[Point, Point]]]:
    """Load timestamps and AB points from pickle files."""
    with open(TIMESTAMPS_PATH, "rb") as f:
        timestamps = pickle.load(f)
    with open(AB_POINTS_PATH, "rb") as f:
        ab_points = pickle.load(f)
    return timestamps, ab_points


def create_masked_avoider() -> MaskedDynamicAvoider:
    """Create and return MaskedDynamicAvoider instance."""
    return MaskedDynamicAvoider(
        masked_preprocessor_config=settings.masked_preprocessor_config,
        dynamic_avoider_config=settings.dynamic_avoider_config,
        predictor_config=settings.predictor_config,
    )


def run_pathfinding(
    masked_avoider: MaskedDynamicAvoider,
    a_point: Point,
    b_point: Point,
    time_keys: list[str],
    dict_obstacles: dict,
    config: PlotSimulationConfig,
):
    """Run pathfinding and return result."""
    if config.with_fine_tuning:
        return masked_avoider.perform_pathfinding_with_finetuning_masked(
            current_pos=a_point,
            end=b_point,
            current_time_index=0,
            window_size=config.window_size,
            num_preds=7,
            time_keys=time_keys,
            dict_obstacles=dict_obstacles,
            masking_strategy="wide",
            prediction_mode=config.prediction_mode,
            validate_segment_endpoint=config.prediction_mode == "predictive",
        )
    return masked_avoider.perform_pathfinding_masked(
        current_pos=a_point,
        end=b_point,
        current_time_index=0,
        window_size=config.window_size,
        time_keys=time_keys,
        dict_obstacles=dict_obstacles,
        masking_strategy="wide",
        prediction_mode=config.prediction_mode,
        validate_segment_endpoint=config.prediction_mode == "predictive",
    )


def get_available_obstacles(result, timestamp: int) -> list | None:
    """Extract available obstacles for a given timestamp from result."""
    if timestamp >= len(result.available_obstacles_dicts):
        return None
    time_key = list(result.available_obstacles_dicts[timestamp].keys())[-1]
    return result.available_obstacles_dicts[timestamp][time_key]["concave"]


def plot_on_axes(
    ax: plt.Axes,
    gdf_start: gpd.GeoDataFrame,
    gdf_finish: gpd.GeoDataFrame | None,
    available_obstacles: list | None,
    timestamp: int,
    color: str,
    label: str,
    path,
    ab_points: tuple[Point, Point],
    ab_text_positions: tuple[tuple[float, float], tuple[float, float]],
    config: PlotSimulationConfig,
) -> None:
    """Plot pathfinding visualization directly on given axes."""
    legend_handles: list[Patch | Line2D] = []

    # Plot gdf_start & gdf_finish geometry
    for gdf in [gdf_start, gdf_finish]:
        if gdf is not None and "geometry" in gdf.columns:
            gdf.plot(ax=ax, facecolor="#E2F700", alpha=0.1, edgecolor="#1D2336", linewidth=1)
    legend_handles.append(
        Patch(facecolor="#E2F700", label=f"Зона грозы в текущий момент времени $\mathcal{{T}}^{{({timestamp + 1})}}$")
    )

    # Plot gdf_start & gdf_finish concave
    for gdf in [gdf_start, gdf_finish]:
        if gdf is not None and "concave" in gdf.columns:
            gdf.set_geometry("concave").plot(
                ax=ax, facecolor="#89A1AE", edgecolor="#1D2336", alpha=0.2, linewidth=1
            )
    legend_handles.append(
        Patch(facecolor="#89A1AE", label="Соответствующие вогнутые оболочки $H_\ell$")
    )

    # Plot available_obstacles if provided
    if available_obstacles:
        for obstacle in available_obstacles:
            x, y = obstacle.exterior.xy
            ax.fill(x, y, facecolor="#FF5B17", alpha=0.5, edgecolor="#FF5B17", linewidth=1)
        legend_handles.append(
            Patch(facecolor="#FF5B17", label="Доступное радиолокационное изображение")
        )

    # Plot clipping_bboxes border
    if hasattr(path, "clipping_bboxes") and path.clipping_bboxes and timestamp < len(path.clipping_bboxes):
        bbox = path.clipping_bboxes[timestamp]
        x, y = bbox.exterior.xy
        ax.plot(x, y, color="#FF5B17", linestyle="--", linewidth=1.5)
        legend_handles.append(
            Line2D([0], [0], color="#FF5B17", linestyle="--", linewidth=1.5, label="Граница радиолокационной видимости")
        )

    # Plot path until timestamp (flatten segments to points)
    flat_path = [pt for segment in path.path[: timestamp + 1] for pt in segment]
    if flat_path:
        path_line = LineString(flat_path)
        x, y = path_line.xy
        ax.plot(x, y, color=color, linestyle="-", linewidth=2, label=label)
        ax.scatter(x, y, color=color, s=30, zorder=5)
        legend_handles.append(Line2D([0], [0], color=color, linestyle="-", linewidth=2, label=label))

    # Plot path at current timestamp (dashed)
    if timestamp < len(path.all_paths) and path.all_paths[timestamp]:
        current_segment = LineString(path.all_paths[timestamp])
        x, y = current_segment.xy
        ax.plot(x, y, color=color, linestyle="--", linewidth=1.5, alpha=1)

    # Plot graph at timestamp
    if timestamp < len(path.all_graphs) and hasattr(path, "master_vertices") and path.master_vertices:
        graph = path.all_graphs[timestamp]
        vertices = path.master_vertices[timestamp]
        point_a, point_b = ab_points

        def get_point(idx: int | str) -> Point:
            """Get Point from index or special string ('start'/'end')."""
            match idx:
                case "start":
                    return point_a
                case "end":
                    return point_b
                case _:
                    return vertices[idx]

        # Plot edges
        for u_idx, v_idx in graph.edges():
            pt_u, pt_v = get_point(u_idx), get_point(v_idx)
            ax.plot([pt_u.x, pt_v.x], [pt_u.y, pt_v.y], color="#1D2336", alpha=0.3, linewidth=0.5)
        legend_handles.append(Line2D([0], [0], color="#1D2336", linewidth=0.5, label="Граф видимости"))
        # Plot nodes
        node_indices = list(graph.nodes())
        node_x = [get_point(i).x for i in node_indices]
        node_y = [get_point(i).y for i in node_indices]
        ax.scatter(node_x, node_y, color="#1D2336", alpha=0.5, s=10, zorder=4)

    # Plot AB points
    point_a, point_b = ab_points
    pos_a, pos_b = ab_text_positions
    ax.scatter([point_a.x, point_b.x], [point_a.y, point_b.y], color="#1D2336", s=100, zorder=6)
    ax.text(pos_a[0], pos_a[1], "A", fontsize=config.ab_text_fontsize, color="#1D2336")
    ax.text(pos_b[0], pos_b[1], "B", fontsize=config.ab_text_fontsize, color="#1D2336")

    ax.legend(handles=legend_handles, loc="upper left", fontsize=config.legend_fontsize)
    ax.set_aspect("equal")

    # Remove ticks and spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_single_timestamp(
    ax: plt.Axes,
    result,
    dict_obstacles: dict,
    time_keys: list[str],
    timestamp: int,
    a_point: Point,
    b_point: Point,
    config: PlotSimulationConfig,
) -> None:
    """Plot a single timestamp on given axes with frame."""
    gdf_start = dict_obstacles[time_keys[timestamp]]
    gdf_finish = None  # dict_obstacles.get(time_keys[len(result.all_paths)]) if len(result.all_paths) < len(time_keys) else None
    available_obstacles = get_available_obstacles(result, timestamp)

    ab_text_positions = (
        (a_point.x * 1.05, a_point.y * 0.99),
        (b_point.x * 0.82, b_point.y * 1.004),
    )

    plot_on_axes(
        ax=ax,
        gdf_start=gdf_start,
        gdf_finish=gdf_finish,
        available_obstacles=available_obstacles,
        timestamp=timestamp,
        color=config.path_color,
        label=config.path_label,
        path=result,
        ab_points=(a_point, b_point),
        ab_text_positions=ab_text_positions,
        config=config,
    )

    # Add frame
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#1D2336")
        spine.set_linewidth(1)


def create_combined_plot(
    result,
    dict_obstacles: dict,
    time_keys: list[str],
    a_point: Point,
    b_point: Point,
    config: PlotSimulationConfig,
) -> plt.Figure:
    """Create combined plot with subplots side by side."""
    n_plots = len(config.timestamps_to_plot)
    fig, axes = plt.subplots(1, n_plots, figsize=config.figsize_combined)

    # Ensure axes is always iterable
    if n_plots == 1:
        axes = [axes]

    for i, timestamp in enumerate(config.timestamps_to_plot):
        ax = axes[i]
        plot_single_timestamp(
            ax=ax,
            result=result,
            dict_obstacles=dict_obstacles,
            time_keys=time_keys,
            timestamp=timestamp,
            a_point=a_point,
            b_point=b_point,
            config=config,
        )
        ax.text(
            0.5,
            -0.02,
            config.subplot_labels[i],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=config.subplot_label_fontsize,
        )

    plt.tight_layout()
    return fig


def save_plot(fig: plt.Figure, config: PlotSimulationConfig) -> None:
    """Save figure as SVG and PDF."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = config.output_dir / f"{config.output_filename}.svg"
    pdf_path = config.output_dir / f"{config.output_filename}.pdf"
    fig.savefig(svg_path, bbox_inches="tight", format="svg")
    fig.savefig(pdf_path, bbox_inches="tight", format="pdf")
    print(f"Saved: {svg_path}")
    print(f"Saved: {pdf_path}")


def main() -> None:
    """Run path simulation and create combined plot."""
    config = PlotSimulationConfig()

    # Load data
    timestamps, ab_points = load_timestamps_and_points()

    # Get specific timestamp and points
    timestamp = timestamps[config.time_index]
    a_point, b_point = ab_points[config.time_index]

    print(f"Processing timestamp: {timestamp}")
    print(f"A: {a_point}, B: {b_point}")

    # Create avoider and collect obstacles
    masked_avoider = create_masked_avoider()
    file_name = format_timestamp(timestamp)
    time_keys, dict_obstacles = collect_obstacles_for_timestamp(DATA_PATH / file_name)

    # Run pathfinding
    print("Running pathfinding...")
    result = run_pathfinding(
        masked_avoider=masked_avoider,
        a_point=a_point,
        b_point=b_point,
        time_keys=time_keys,
        dict_obstacles=dict_obstacles,
        config=config,
    )

    # Create and save plot
    print("Creating combined plot...")
    fig = create_combined_plot(
        result=result,
        dict_obstacles=dict_obstacles,
        time_keys=time_keys,
        a_point=a_point,
        b_point=b_point,
        config=config,
    )
    save_plot(fig, config)
    plt.close(fig)


if __name__ == "__main__":
    main()
