import pickle
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import networkx as nx
from shapely import Point

from thund_avoider.schemas.dynamic_avoider import TimestampResult
from thund_avoider.services.dynamic_avoider import DynamicAvoider
from thund_avoider.settings import settings, DATA_PATH, RESULT_PATH, TIMESTAMPS_PATH, AB_POINTS_PATH
from thund_avoider.scripts.utils import format_timestamp, save_combined_results, WINDOW_SIZES


def build_master_graph(
    dynamic_avoider: DynamicAvoider,
    data_dir: Path,
) -> tuple[list[str], dict, nx.Graph, dict]:
    """
    Build master graph from obstacles data.

    Args:
        dynamic_avoider: DynamicAvoider instance.
        data_dir: Path to data directory.

    Returns:
        Tuple of time_keys, dict_obstacles, graph_master, time_valid_edges.
    """
    time_keys = dynamic_avoider.extract_time_keys(data_dir)
    dict_obstacles = dynamic_avoider.collect_obstacles(data_dir, time_keys)
    graph_master, time_valid_edges = dynamic_avoider.create_master_graph(
        time_keys=time_keys,
        dict_obstacles=dict_obstacles,
    )
    return time_keys, dict_obstacles, graph_master, time_valid_edges


def run_pathfinding_for_direction(
    dynamic_avoider: DynamicAvoider,
    start: Point,
    end: Point,
    window_size: int,
    time_keys: list[str],
    dict_obstacles: dict,
    graph_master: nx.Graph,
    time_valid_edges: dict,
    with_fine_tuning: bool,
    direction: str,
):
    """
    Run pathfinding for a single direction and log results.

    Args:
        dynamic_avoider: DynamicAvoider instance.
        start: Starting point.
        end: Ending point.
        window_size: Sliding window size.
        time_keys: List of time keys.
        dict_obstacles: Dictionary of obstacles.
        graph_master: Master graph.
        time_valid_edges: Valid edges for each time key.
        with_fine_tuning: Whether to apply fine-tuning.
        direction: Direction label for logging (e.g., "A -> B").

    Returns:
        Pathfinding result.
    """
    dynamic_avoider.logger.info(f"{' ' * 7}{direction}")
    result = dynamic_avoider.sliding_window_pathfinding(
        start=start,
        end=end,
        window_size=window_size,
        time_keys=time_keys,
        dict_obstacles=dict_obstacles,
        G_master=graph_master,
        time_valid_edges=time_valid_edges,
        with_fine_tuning=with_fine_tuning,
    )
    dynamic_avoider.logger.info(
        f"Success: {result.success}, Success Inter: {result.success_intermediate}\n"
    )
    return result


def process_timestamp(
    dynamic_avoider: DynamicAvoider,
    timestamp: datetime,
    a_point: Point,
    b_point: Point,
    with_fine_tuning: bool,
    with_backward_pathfinding: bool,
) -> TimestampResult:
    """
    Process a single timestamp with pathfinding for all window sizes.

    Args:
        dynamic_avoider: DynamicAvoider instance.
        timestamp: Timestamp to process.
        a_point: Start point A.
        b_point: End point B.
        with_fine_tuning: Whether to apply fine-tuning.
        with_backward_pathfinding: Whether to also compute B -> A paths.

    Returns:
        TimestampResult with all pathfinding results.
    """
    file_name = format_timestamp(timestamp)
    tuning_type = dynamic_avoider.tuning_strategy if with_fine_tuning else "master"
    dynamic_avoider.logger.info(f"START BUILDING MASTER GRAPH FOR {timestamp}\n")

    time_keys, dict_obstacles, graph_master, time_valid_edges = build_master_graph(
        dynamic_avoider, DATA_PATH / file_name
    )
    dynamic_avoider.logger.info(f"\nMASTER GRAPH FOR {timestamp} READY\n")

    result = TimestampResult(timestamp=timestamp)
    tuning_label = f"{tuning_type.upper()} FINE-TUNING" if with_fine_tuning else "PATHFINDING"

    for window_size in WINDOW_SIZES:
        dynamic_avoider.logger.info(f"START {tuning_label} FOR {timestamp}\n")
        dynamic_avoider.logger.info(f"====== Window size {window_size} ======")

        result_a = run_pathfinding_for_direction(
            dynamic_avoider=dynamic_avoider,
            start=a_point,
            end=b_point,
            window_size=window_size,
            time_keys=time_keys,
            dict_obstacles=dict_obstacles,
            graph_master=graph_master,
            time_valid_edges=time_valid_edges,
            with_fine_tuning=with_fine_tuning,
            direction="(A)  ->  (B)",
        )
        result.add_result(result_a, window_size, "A->B")

        if with_backward_pathfinding:
            result_b = run_pathfinding_for_direction(
                dynamic_avoider=dynamic_avoider,
                start=b_point,
                end=a_point,
                window_size=window_size,
                time_keys=time_keys,
                dict_obstacles=dict_obstacles,
                graph_master=graph_master,
                time_valid_edges=time_valid_edges,
                with_fine_tuning=with_fine_tuning,
                direction="(B)  ->  (A)",
            )
            result.add_result(result_b, window_size, "B->A")

    return result


def process_data(
    dynamic_avoider: DynamicAvoider,
    *,
    timestamps: list[datetime],
    ab_points: list[tuple[Point, Point]],
    with_fine_tuning: bool = True,
    with_backward_pathfinding: bool = False,
) -> None:
    """
    Run pathfinding for all timestamps and save results to parquet.

    Args:
        dynamic_avoider: DynamicAvoider instance.
        timestamps: List of timestamps to process.
        ab_points: List of (start, end) point tuples.
        with_fine_tuning: Whether to apply fine-tuning.
        with_backward_pathfinding: Whether to compute B -> A paths.
    """
    tuning_type = dynamic_avoider.tuning_strategy if with_fine_tuning else "master"
    result_name = f"dynamic_avoider_{tuning_type}"
    result_dir = RESULT_PATH / result_name
    result_dir.mkdir(parents=True, exist_ok=True)

    existing_files = {f.stem for f in result_dir.glob("*.parquet")}
    total_width = len(str(len(timestamps)))

    for i, timestamp in enumerate(timestamps):
        file_name = format_timestamp(timestamp)

        if file_name in existing_files:
            dynamic_avoider.logger.info(
                f"{i + 1:<{total_width}}/{len(timestamps)}: "
                f"File {file_name}.parquet exists. Skipping {timestamp}"
            )
            continue

        a_point, b_point = ab_points[i]
        timestamp_result = process_timestamp(
            dynamic_avoider=dynamic_avoider,
            timestamp=timestamp,
            a_point=a_point,
            b_point=b_point,
            with_fine_tuning=with_fine_tuning,
            with_backward_pathfinding=with_backward_pathfinding,
        )

        df = gpd.GeoDataFrame(timestamp_result.to_records(), crs="EPSG:3067")
        df.to_parquet(result_dir / f"{file_name}.parquet", index=False)
        dynamic_avoider.logger.info(
            f"{i + 1:<{total_width}}/{len(timestamps)}: {timestamp} saved to "
            f"'{result_name}/{file_name}.parquet'\n\n\n"
        )

    save_combined_results(result_dir, result_name, dynamic_avoider.logger)


def main() -> None:
    """Run pathfinding experiments with different configurations."""
    with open(TIMESTAMPS_PATH, "rb") as f:
        timestamps = pickle.load(f)
    with open(AB_POINTS_PATH, "rb") as f:
        ab_points = pickle.load(f)

    dynamic_avoider = DynamicAvoider(settings.dynamic_avoider_config)

    # Initial pathfinding (no fine-tuning)
    process_data(
        dynamic_avoider=dynamic_avoider,
        timestamps=timestamps,
        ab_points=ab_points,
        with_fine_tuning=False,
        with_backward_pathfinding=True,
    )

    # With greedy fine-tuning
    process_data(
        dynamic_avoider=dynamic_avoider,
        timestamps=timestamps,
        ab_points=ab_points,
        with_fine_tuning=True,
        with_backward_pathfinding=True,
    )

    # With smooth fine-tuning
    dynamic_avoider.tuning_strategy = "smooth"
    process_data(
        dynamic_avoider=dynamic_avoider,
        timestamps=timestamps,
        ab_points=ab_points,
        with_fine_tuning=True,
        with_backward_pathfinding=True,
    )


if __name__ == "__main__":
    main()
