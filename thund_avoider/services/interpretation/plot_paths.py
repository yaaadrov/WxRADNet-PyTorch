import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from shapely import Point, Polygon
from shapely.geometry import LineString

from thund_avoider.schemas.dynamic_avoider import SlidingWindowPath
from thund_avoider.schemas.masked_dynamic_avoider import SlidingWindowPathMasked


def _setup_style(dpi: int = 150) -> None:
    """Setup matplotlib style consistent with plot_generator.py."""
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "axes.grid": False,
    })


def _clean_axes(ax: plt.Axes) -> None:
    """Remove spines, ticks, and tick labels from axes."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_path_until_timestamp(
    figsize: tuple[float, float],
    gdf_start: gpd.GeoDataFrame,
    gdf_finish: gpd.GeoDataFrame | None,
    available_obstacles: list[Polygon] | None,
    timestamp: int,
    color: str,
    label: str,
    path: SlidingWindowPath | SlidingWindowPathMasked,
    ab_points: tuple[Point, Point],
    ab_text_positions: tuple[tuple[float, float], tuple[float, float]],
    dpi: int = 150,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot path evolution until a specific timestamp.

    Visualizes the complete pathfinding context including obstacles,
    the cumulative path, current segment, and graph structure.

    Args:
        figsize: Figure dimensions (width, height).
        gdf_start: GeoDataFrame with start zone geometry.
        gdf_finish: GeoDataFrame with finish zone geometry.
        available_obstacles: List of obstacle polygons (can be None).
        timestamp: Current timestamp index to visualize.
        color: Color for the path visualization.
        label: Label for the path in the legend.
        path: SlidingWindowPath or SlidingWindowPathMasked object.
        ab_points: Tuple of (start_point, end_point) for A and B markers.
        ab_text_positions: Tuple of text positions for A and B labels.
        dpi: Resolution for the figure.

    Returns:
        Tuple of (Figure, Axes) for further customization.
    """
    _setup_style(dpi)
    fig, ax = plt.subplots(figsize=figsize)

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
        legend_handles.append(Line2D([0], [0], color="#FF5B17", linestyle="--", linewidth=1.5, label=f"Граница радиолокационной видимости"))

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
            if idx == "start":
                return point_a
            elif idx == "end":
                return point_b
            return vertices[idx]

        # Plot edges
        for u_idx, v_idx in graph.edges():
            pt_u, pt_v = get_point(u_idx), get_point(v_idx)
            ax.plot([pt_u.x, pt_v.x], [pt_u.y, pt_v.y], color="#1D2336", alpha=0.3, linewidth=0.5)
        legend_handles.append(Line2D([0], [0], color="#1D2336", linewidth=0.5, label=f"Граф видимости"))
        # Plot nodes
        node_indices = list(graph.nodes())
        node_x = [get_point(i).x for i in node_indices]
        node_y = [get_point(i).y for i in node_indices]
        ax.scatter(node_x, node_y, color="#1D2336", alpha=0.5, s=10, zorder=4)

    # Plot AB points
    point_a, point_b = ab_points
    pos_a, pos_b = ab_text_positions
    ax.scatter([point_a.x, point_b.x], [point_a.y, point_b.y], color="#1D2336", s=100, zorder=6)
    ax.text(pos_a[0], pos_a[1], "A", fontsize=16, color="#1D2336")
    ax.text(pos_b[0], pos_b[1], "B", fontsize=16, color="#1D2336")

    ax.legend(handles=legend_handles, loc="upper left")
    ax.set_aspect("equal")
    _clean_axes(ax)
    plt.tight_layout()

    return fig, ax


def compare_two_paths(
    figsize: tuple[float, float],
    gdf_start: gpd.GeoDataFrame,
    gdf_finish: gpd.GeoDataFrame | None,
    path_first: list[Point],
    color_first: str,
    label_first: str,
    path_second: list[Point],
    color_second: str,
    label_second: str,
    ab_points: tuple[Point, Point] | None,
    ab_text_positions: tuple[tuple[float, float], tuple[float, float]] | None,
    obstacles_first: list[Polygon] | None = None,
    obstacles_second: list[Polygon] | None = None,
    obstacles_first_label: str | None = None,
    obstacles_second_label: str | None = None,
    path_second_linestyle: str = "-",
    path_second_scatter: bool = True,
    dpi: int = 150,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Compare two paths side by side on the same plot.

    Visualizes two alternative paths with different colors and labels
    for direct comparison of pathfinding strategies.

    Args:
        figsize: Figure dimensions (width, height).
        gdf_start: GeoDataFrame with start zone geometry.
        gdf_finish: GeoDataFrame with finish zone geometry.
        path_first: First path as list of Points.
        color_first: Color for the first path.
        label_first: Label for the first path in the legend.
        path_second: Second path as list of Points.
        color_second: Color for the second path.
        label_second: Label for the second path in the legend.
        ab_points: Tuple of (start_point, end_point) for A and B markers.
        ab_text_positions: Tuple of text positions for A and B labels.
        obstacles_first: Optional list of obstacle polygons for first path.
        obstacles_second: Optional list of obstacle polygons for second path.
        obstacles_first_label: Label for first obstacles in legend.
        obstacles_second_label: Label for second obstacles in legend.
        path_second_linestyle: Line style for second path (default: "-").
        path_second_scatter: Whether to add scatter points for second path.
        dpi: Resolution for the figure.

    Returns:
        Tuple of (Figure, Axes) for further customization.
    """
    _setup_style(dpi)
    fig, ax = plt.subplots(figsize=figsize)

    legend_handles: list[Patch | Line2D] = []

    # Plot gdf_start & gdf_finish geometry
    for gdf in [gdf_start, gdf_finish]:
        if gdf is not None and "geometry" in gdf.columns:
            gdf.plot(ax=ax, facecolor="#E2F700", alpha=0.1, edgecolor="#1D2336", linewidth=1)
    legend_handles.append(
        Patch(facecolor="#E2F700", label="Зона грозы в начальный момент времени $\\mathcal{T}^{(1)}$")
    )

    # Plot gdf_start & gdf_finish concave
    for gdf in [gdf_start, gdf_finish]:
        if gdf is not None and "concave" in gdf.columns:
            gdf.set_geometry("concave").plot(
                ax=ax, facecolor="#89A1AE", edgecolor="#1D2336", alpha=0.3, linewidth=1
            )
    legend_handles.append(
        Patch(facecolor="#89A1AE", label="Соответствующие вогнутые оболочки")
    )

    # Plot obstacles for first path
    if obstacles_first:
        for obstacle in obstacles_first:
            x, y = obstacle.exterior.xy
            ax.fill(x, y, facecolor=color_first, alpha=0.3, edgecolor=color_first, linewidth=1)
        legend_handles.append(
            Patch(facecolor=color_first, alpha=0.3, label=obstacles_first_label or "Препятствия (первый путь)")
        )

    # Plot obstacles for second path
    if obstacles_second:
        for obstacle in obstacles_second:
            x, y = obstacle.exterior.xy
            ax.fill(x, y, facecolor=color_second, alpha=0.2, edgecolor=color_second, linewidth=0, hatch="***")
        legend_handles.append(
            Patch(facecolor=color_second, alpha=0.2, hatch="***", label=obstacles_second_label or "Препятствия (второй путь)")
        )

    # Plot first path
    if path_first:
        path_line = LineString(path_first)
        x, y = path_line.xy
        ax.plot(x, y, color=color_first, linestyle="-", linewidth=2, label=label_first)
        ax.scatter(x, y, color=color_first, s=30, zorder=5)
        legend_handles.append(Line2D([0], [0], color=color_first, linestyle="-", linewidth=2, label=label_first))

    # Plot second path
    if path_second:
        path_line = LineString(path_second)
        x, y = path_line.xy
        ax.plot(x, y, color=color_second, linestyle=path_second_linestyle, linewidth=2, label=label_second)
        if path_second_scatter:
            ax.scatter(x, y, color=color_second, s=30, zorder=5)
        legend_handles.append(
            Line2D([0], [0], color=color_second, linestyle=path_second_linestyle, linewidth=2, label=label_second)
        )

    # Plot AB points
    if ab_points and ab_text_positions:
        point_a, point_b = ab_points
        pos_a, pos_b = ab_text_positions
        ax.scatter([point_a.x, point_b.x], [point_a.y, point_b.y], color="#1D2336", s=100, zorder=6)
        ax.text(pos_a[0], pos_a[1], "A", fontsize=14, fontweight="bold", color="#1D2336")
        ax.text(pos_b[0], pos_b[1], "B", fontsize=14, fontweight="bold", color="#1D2336")

    ax.legend(handles=legend_handles, loc="upper left")
    ax.set_aspect("equal")
    _clean_axes(ax)
    plt.tight_layout()

    return fig, ax
