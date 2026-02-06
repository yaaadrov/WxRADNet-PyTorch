import os
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Final

import pandas as pd
from pyproj import CRS
from shapely import Point, LineString
import itertools as it

from thund_avoider.services.dynamic_avoider import DynamicAvoider
from thund_avoider.schemas.dynamic_avoider import SlidingWindowPath, FineTunedPath
from thund_avoider.settings import settings, DATA_PATH, RESULT_PATH, TIMESTAMPS_PATH, AB_POINTS_PATH


WINDOW_SIZES: Final = [1, 2, 3, 4, 5, 6, 7]


def pickle_to_df(pkl_file_path: Path) -> pd.DataFrame:
    """
    Load `result_dict` pickle data and create a result DataFrame

    Args:
        pkl_file_path (Path): Path to `result_dict` pickle file

    Returns:
        pd.DataFrame: Result data for further analysis
    """
    with open(pkl_file_path, "rb") as file_in:
        result_dicts: dict[str, dict[int, SlidingWindowPath | FineTunedPath]] = pickle.load(file_in)
    current_date = str(pkl_file_path).split("/")[-1].split(".")[0]
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "strategy": [
                        result_dicts[start_point][w_size].strategy
                        for w_size in result_dicts[start_point].keys()
                    ],
                    "timestamp": [current_date] * len(result_dicts[start_point]),
                    "window_size": list(result_dicts[start_point].keys()),
                    "start_point": [start_point] * len(result_dicts[start_point]),
                    "path": [
                        LineString(list(it.chain(*result_dicts[start_point][w_size].path)))
                        for w_size in result_dicts[start_point].keys()
                    ],
                    "length": [
                        LineString(
                            list(it.chain(*result_dicts[start_point][w_size].path))
                        ).length
                        for w_size in result_dicts[start_point].keys()
                    ],
                    "success": [
                        result_dicts[start_point][w_size].success
                        for w_size in result_dicts[start_point].keys()
                    ],
                    "success_intermediate": [
                        result_dicts[start_point][w_size].success_intermediate
                        for w_size in result_dicts[start_point].keys()
                    ],
                },
            )
            for start_point in result_dicts.keys()
        ],
        ignore_index=True,
    )


def process_data(
    dynamic_avoider: DynamicAvoider,
    *,
    timestamps: list[datetime],
    ab_points: list[tuple[Point, Point]],
    window_sizes: list[int],
    with_fine_tuning: bool = True,
    with_backward_pathfinding: bool = False,
) -> None:
    """
    Initial / fine-tuning (with or w/o backward) pathfinding for given `timestamps`, `ab_points` and `window_sizes`
    with saving intermediate (pickle) results to "result/dynamic_avoider_master/" or "result/dynamic_avoider_{tuning_strategy}/"
    and final results to "result/dynamic_avoider_master.csv" or "result/dynamic_avoider_{tuning_strategy}.csv"

    Args:
        dynamic_avoider (DynamicAvoider): DynamicAvoider object tp use for pathfinding
        timestamps (list[datetime]): Initial timestampt to process
        ab_points (list[tuple[Point, Point]]): Start and end points
        window_sizes (list[int]): Sliding window sizes
        with_fine_tuning (bool): Fine-tuning (True) or initial (False) pathfinding
        with_backward_pathfinding (bool): Perform B -> A pathfinding (True) or not (False)
    """
    length = len(str(len(timestamps)))
    type_pathfinding = "master"
    if with_fine_tuning:
        type_pathfinding = dynamic_avoider.tuning_strategy
    dir_path_result = RESULT_PATH / f"dynamic_avoider_{type_pathfinding}"
    if not dir_path_result.exists():
        dir_path_result.mkdir(parents=True, exist_ok=True)

    for i, current_date in enumerate(timestamps):
        A, B = ab_points[i]
        file_name = "_".join(re.split(r"[- :]", str(current_date)))
        if f"{file_name}.pkl" in os.listdir(dir_path_result):
            dynamic_avoider.logger.info(
                f"{i + 1:<{length}}/{len(timestamps)}: "
                f"File {file_name}.pkl exists. Skipping {current_date}"
            )
            continue

        # Get time keys and corresponding obstacles, build master graph
        dynamic_avoider.logger.info(
            f"START BUILDING MASTER GRAPH FOR {current_date}\n"
        )
        time_keys = dynamic_avoider.extract_time_keys(DATA_PATH / file_name)
        dict_obstacles = dynamic_avoider.collect_obstacles(DATA_PATH / file_name, time_keys)
        G_master, time_valid_edges = dynamic_avoider.create_master_graph(
            time_keys=time_keys,
            dict_obstacles=dict_obstacles,
        )
        dynamic_avoider.logger.info(
            f"\nMASTER GRAPH FOR {current_date} READY\n"
        )

        result_a: dict[int, SlidingWindowPath | FineTunedPath] = {}
        result_b: dict[int, SlidingWindowPath | FineTunedPath] = {}
        for window_size in window_sizes:
            if with_fine_tuning:
                dynamic_avoider.logger.info(
                    f"START PATHFINDING WITH {type_pathfinding.upper()} "
                    f"FINE-TUNING FOR {current_date}\n"
                )
            else:
                dynamic_avoider.logger.info(f"START PATHFINDING FOR {current_date}\n")
            dynamic_avoider.logger.info(f"====== Window size {window_size} ======")

            # Sliding window pathfinding A -> B
            dynamic_avoider.logger.info(" " * 7 + "(A)  ->  (B)")
            result_a[window_size] = dynamic_avoider.sliding_window_pathfinding(
                start=A,
                end=B,
                window_size=window_size,
                time_keys=time_keys,
                dict_obstacles=dict_obstacles,
                G_master=G_master,
                time_valid_edges=time_valid_edges,
                with_fine_tuning=with_fine_tuning,
            )
            dynamic_avoider.logger.info(
                f"Success: {result_a[window_size].success}, "
                f"Success Inter: {result_a[window_size].success_intermediate}\n",
            )

            # Sliding window pathfinding B -> A
            if with_backward_pathfinding:
                dynamic_avoider.logger.info(" " * 7 + "(B)  ->  (A)")
                result_b[window_size] = dynamic_avoider.sliding_window_pathfinding(
                    start=B,
                    end=A,
                    window_size=window_size,
                    time_keys=time_keys,
                    dict_obstacles=dict_obstacles,
                    G_master=G_master,
                    time_valid_edges=time_valid_edges,
                    with_fine_tuning=with_fine_tuning,
                )
                dynamic_avoider.logger.info(
                    f"Success: {result_b[window_size].success}, "
                    f"Success Inter: {result_b[window_size].success_intermediate}\n",
                )

        result = {
            "A": result_a,
            "B": result_b,
        }
        with open(dir_path_result / f"{file_name}.pkl", "wb") as file_out:
            pickle.dump(result, file_out)
        dynamic_avoider.logger.info(
            f"{i + 1:<{length}}/{len(timestamps)}: {current_date} saved to "
            f"'dynamic_avoider_{type_pathfinding}/{file_name}.pkl'\n\n\n",
        )

    # Save data to CSV
    df_result = pd.concat(
        [
            pickle_to_df(dir_path_result / file_name)
            for file_name in os.listdir(dir_path_result)
        ],
        ignore_index=True,
    )
    df_result.to_csv(RESULT_PATH / f"dynamic_avoider_{type_pathfinding}.csv", index=None)
    dynamic_avoider.logger.info(
        f"Final DataFrame saved successfully to 'results/dynamic_avoider_{type_pathfinding}.csv'"
    )


if __name__ == "__main__":
    with open(TIMESTAMPS_PATH, "rb") as file_in:
        timestamps = pickle.load(file_in)
    with open(AB_POINTS_PATH, "rb") as file_in:
        ab_points = pickle.load(file_in)

    dynamic_avoider = DynamicAvoider(
        crs=CRS(settings.projected_crs),
        velocity_kmh=settings.velocity_kmh,
        delta_minutes=settings.delta_minutes,
        buffer=settings.buffer,
        tolerance=settings.tolerance,
        k_neighbors=settings.k_neighbors,
        max_distance=settings.max_distance,
        simplification_tolerance=settings.simplification_tolerance,
        smooth_tolerance=settings.smooth_tolerance,
        max_iter=settings.max_iter,
        delta_length=settings.delta_length,
        strategy="concave",
        tuning_strategy="greedy",
    )

    # Initial pathfinding
    process_data(
        dynamic_avoider=dynamic_avoider,
        timestamps=timestamps,
        ab_points=ab_points,
        window_sizes=WINDOW_SIZES,
        with_fine_tuning=False,
        with_backward_pathfinding=True,
    )

    # With greedy tuning
    process_data(
        dynamic_avoider=dynamic_avoider,
        timestamps=timestamps,
        ab_points=ab_points,
        window_sizes=WINDOW_SIZES,
        with_fine_tuning=True,
        with_backward_pathfinding=True,
    )

    # With smooth tuning
    dynamic_avoider.tuning_strategy = "smooth"
    process_data(
        dynamic_avoider=dynamic_avoider,
        timestamps=timestamps,
        ab_points=ab_points,
        window_sizes=WINDOW_SIZES,
        with_fine_tuning=True,
        with_backward_pathfinding=True,
    )
