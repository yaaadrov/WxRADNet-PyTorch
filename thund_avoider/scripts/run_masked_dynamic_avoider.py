import pickle
from datetime import datetime
from pathlib import Path
from typing import Final, Literal

import geopandas as gpd
from shapely import Point

from thund_avoider.schemas.masked_dynamic_avoider import MaskedTimestampResult
from thund_avoider.services.masked_dynamic_avoider import MaskedDynamicAvoider
from thund_avoider.services.dynamic_avoider.data_loader import DataLoader
from thund_avoider.settings import (
    settings,
    DATA_PATH,
    RESULT_PATH,
    TIMESTAMPS_PATH,
    AB_POINTS_PATH,
)
from thund_avoider.scripts.utils import format_timestamp, save_combined_results, WINDOW_SIZES

NUM_PREDS: Final = 7
MASKING_STRATEGY: Final = "wide"


def collect_obstacles_for_timestamp(
    data_dir: Path,
) -> tuple[list[str], dict]:
    """
    Load obstacles for a timestamp using DataLoader.

    Args:
        data_dir: Path to data directory.

    Returns:
        Tuple of time_keys and dict_obstacles.
    """
    data_loader_config = settings.dynamic_avoider_config.data_loader_config
    time_keys = DataLoader.extract_time_keys(data_dir)
    dict_obstacles = DataLoader(data_loader_config).collect_obstacles(data_dir, time_keys)
    return time_keys, dict_obstacles


def run_pathfinding_masked(
    masked_avoider: MaskedDynamicAvoider,
    start: Point,
    end: Point,
    window_size: int,
    time_keys: list[str],
    dict_obstacles: dict,
    prediction_mode: Literal["deterministic", "predictive"],
    with_fine_tuning: bool,
    direction: str,
):
    """
    Run pathfinding for a single direction and log results.

    Args:
        masked_avoider: MaskedDynamicAvoider instance.
        start: Starting point.
        end: Ending point.
        window_size: Sliding window size.
        time_keys: List of time keys.
        dict_obstacles: Dictionary of obstacles.
        prediction_mode: Prediction mode (deterministic or predictive).
        with_fine_tuning: Whether to apply fine-tuning.
        direction: Direction label for logging (e.g., "A -> B").

    Returns:
        Pathfinding result.
    """
    masked_avoider.logger.info(f"{' ' * 7}{direction}")

    if with_fine_tuning:
        result = masked_avoider.perform_pathfinding_with_finetuning_masked(
            current_pos=start,
            end=end,
            current_time_index=0,
            window_size=window_size,
            num_preds=NUM_PREDS,
            time_keys=time_keys,
            dict_obstacles=dict_obstacles,
            masking_strategy=MASKING_STRATEGY,
            prediction_mode=prediction_mode,
        )
    else:
        result = masked_avoider.perform_pathfinding_masked(
            current_pos=start,
            end=end,
            current_time_index=0,
            window_size=window_size,
            time_keys=time_keys,
            dict_obstacles=dict_obstacles,
            masking_strategy=MASKING_STRATEGY,
            prediction_mode=prediction_mode,
        )

    masked_avoider.logger.info(
        f"Success: {result.success}, Success Inter: {result.success_intermediate}, "
        f"Pred Valid: {result.is_pred_path_valid}\n"
    )
    return result


def process_timestamp_masked(
    masked_avoider: MaskedDynamicAvoider,
    timestamp: datetime,
    a_point: Point,
    b_point: Point,
    prediction_mode: Literal["deterministic", "predictive"],
    with_fine_tuning: bool,
    with_backward_pathfinding: bool,
) -> MaskedTimestampResult:
    """
    Process a single timestamp with pathfinding for all window sizes.

    Args:
        masked_avoider: MaskedDynamicAvoider instance.
        timestamp: Timestamp to process.
        a_point: Start point A.
        b_point: End point B.
        prediction_mode: Prediction mode (deterministic or predictive).
        with_fine_tuning: Whether to apply fine-tuning.
        with_backward_pathfinding: Whether to also compute B -> A paths.

    Returns:
        MaskedTimestampResult with all pathfinding results.
    """
    file_name = format_timestamp(timestamp)
    mode_label = f"{prediction_mode.upper()} {'GREEDY' if with_fine_tuning else 'BASE'}"
    masked_avoider.logger.info(f"START LOADING OBSTACLES FOR {timestamp}\n")

    time_keys, dict_obstacles = collect_obstacles_for_timestamp(DATA_PATH / file_name)
    masked_avoider.logger.info(f"\nOBSTACLES FOR {timestamp} READY\n")

    result = MaskedTimestampResult(timestamp=file_name, prediction_mode=prediction_mode)

    for window_size in WINDOW_SIZES:
        masked_avoider.logger.info(f"START {mode_label} PATHFINDING FOR {timestamp}\n")
        masked_avoider.logger.info(f"====== Window size {window_size} ======")

        result_a = run_pathfinding_masked(
            masked_avoider=masked_avoider,
            start=a_point,
            end=b_point,
            window_size=window_size,
            time_keys=time_keys,
            dict_obstacles=dict_obstacles,
            prediction_mode=prediction_mode,
            with_fine_tuning=with_fine_tuning,
            direction="(A)  ->  (B)",
        )
        result.add_result(result_a, window_size, "A->B")

        if with_backward_pathfinding:
            result_b = run_pathfinding_masked(
                masked_avoider=masked_avoider,
                start=b_point,
                end=a_point,
                window_size=window_size,
                time_keys=time_keys,
                dict_obstacles=dict_obstacles,
                prediction_mode=prediction_mode,
                with_fine_tuning=with_fine_tuning,
                direction="(B)  ->  (A)",
            )
            result.add_result(result_b, window_size, "B->A")

    return result


def get_result_name(
    prediction_mode: Literal["deterministic", "predictive"],
    with_fine_tuning: bool,
) -> str:
    """Generate result directory name based on configuration."""
    tuning_suffix = "greedy" if with_fine_tuning else "base"
    return f"dyn_masked_{prediction_mode}_{tuning_suffix}"


def process_data_masked(
    masked_avoider: MaskedDynamicAvoider,
    *,
    timestamps: list[datetime],
    ab_points: list[tuple[Point, Point]],
    prediction_mode: Literal["deterministic", "predictive"],
    with_fine_tuning: bool,
    with_backward_pathfinding: bool = True,
) -> None:
    """
    Run pathfinding for all timestamps and save results to parquet.

    Args:
        masked_avoider: MaskedDynamicAvoider instance.
        timestamps: List of timestamps to process.
        ab_points: List of (start, end) point tuples.
        prediction_mode: Prediction mode (deterministic or predictive).
        with_fine_tuning: Whether to apply fine-tuning.
        with_backward_pathfinding: Whether to compute B -> A paths.
    """
    result_name = get_result_name(prediction_mode, with_fine_tuning)
    result_dir = RESULT_PATH / result_name
    result_dir.mkdir(parents=True, exist_ok=True)

    existing_files = {f.stem for f in result_dir.glob("*.parquet")}
    total_width = len(str(len(timestamps)))

    for i, timestamp in enumerate(timestamps):
        file_name = format_timestamp(timestamp)

        if file_name in existing_files:
            masked_avoider.logger.info(
                f"{i + 1:<{total_width}}/{len(timestamps)}: "
                f"File {file_name}.parquet exists. Skipping {timestamp}"
            )
            continue

        a_point, b_point = ab_points[i]
        timestamp_result = process_timestamp_masked(
            masked_avoider=masked_avoider,
            timestamp=timestamp,
            a_point=a_point,
            b_point=b_point,
            prediction_mode=prediction_mode,
            with_fine_tuning=with_fine_tuning,
            with_backward_pathfinding=with_backward_pathfinding,
        )

        df = gpd.GeoDataFrame(timestamp_result.to_records(), geometry="path", crs="EPSG:3067")
        df.to_parquet(result_dir / f"{file_name}.parquet", index=False)
        masked_avoider.logger.info(
            f"{i + 1:<{total_width}}/{len(timestamps)}: {timestamp} saved to "
            f"'{result_name}/{file_name}.parquet'\n\n\n"
        )

    save_combined_results(result_dir, result_name, masked_avoider.logger)


def main() -> None:
    """Run all 4 pathfinding experiments with MaskedDynamicAvoider."""
    with open(TIMESTAMPS_PATH, "rb") as f:
        timestamps = pickle.load(f)
    with open(AB_POINTS_PATH, "rb") as f:
        ab_points = pickle.load(f)

    masked_avoider = MaskedDynamicAvoider(
        masked_preprocessor_config=settings.masked_preprocessor_config,
        dynamic_avoider_config=settings.dynamic_avoider_config,
        predictor_config=settings.predictor_config,
    )

    # Experiment 1: Deterministic base (no fine-tuning)
    masked_avoider.logger.info("=" * 60)
    masked_avoider.logger.info("EXPERIMENT 1: DETERMINISTIC BASE")
    masked_avoider.logger.info("=" * 60)
    process_data_masked(
        masked_avoider=masked_avoider,
        timestamps=timestamps,
        ab_points=ab_points,
        prediction_mode="deterministic",
        with_fine_tuning=False,
        with_backward_pathfinding=True,
    )

    # Experiment 2: Deterministic with greedy fine-tuning
    masked_avoider.tuning_strategy = "greedy"
    masked_avoider.logger.info("=" * 60)
    masked_avoider.logger.info("EXPERIMENT 2: DETERMINISTIC GREEDY")
    masked_avoider.logger.info("=" * 60)
    process_data_masked(
        masked_avoider=masked_avoider,
        timestamps=timestamps,
        ab_points=ab_points,
        prediction_mode="deterministic",
        with_fine_tuning=True,
        with_backward_pathfinding=True,
    )

    # Experiment 3: Predictive base (no fine-tuning)
    masked_avoider.logger.info("=" * 60)
    masked_avoider.logger.info("EXPERIMENT 3: PREDICTIVE BASE")
    masked_avoider.logger.info("=" * 60)
    process_data_masked(
        masked_avoider=masked_avoider,
        timestamps=timestamps,
        ab_points=ab_points,
        prediction_mode="predictive",
        with_fine_tuning=False,
        with_backward_pathfinding=True,
    )

    # Experiment 4: Predictive with greedy fine-tuning
    masked_avoider.tuning_strategy = "greedy"
    masked_avoider.logger.info("=" * 60)
    masked_avoider.logger.info("EXPERIMENT 4: PREDICTIVE GREEDY")
    masked_avoider.logger.info("=" * 60)
    process_data_masked(
        masked_avoider=masked_avoider,
        timestamps=timestamps,
        ab_points=ab_points,
        prediction_mode="predictive",
        with_fine_tuning=True,
        with_backward_pathfinding=True,
    )


if __name__ == "__main__":
    main()
