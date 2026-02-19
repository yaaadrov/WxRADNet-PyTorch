import pickle
import re
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from rasterio import RasterioIOError

from thund_avoider.services.preprocessor import Preprocessor
from thund_avoider.settings import settings, TIMESTAMPS_PATH, DATA_PATH, IMAGES_PATH

warnings.simplefilter(action="ignore", category=UserWarning)

NUM_ITERATIONS_PLUS: int = 30
NUM_ITERATIONS_MINUS: int = 5
DELTA_MINUTES: int = 5


def process_csv(
    preprocessor: Preprocessor,
    current_date: datetime,
    delta_minutes: int,
    path_to_csv: Path,
    data_directory_path: Path,
    i: int,
    length: int,
    iterations: int,
    depth: int = 0
) -> None:
    if depth > 10:
        print(f"{i + 1:<{length}}/{iterations}: Error – could not find any data for {current_date}")
        return
    try:
        gdf_union = preprocessor.get_gdf_for_current_date(current_date)
        preprocessor.save_geodataframe_to_csv(gdf_union, path_to_csv)
        status = "ready" if depth == 0 else f"ready (fallback from {depth} intervals ago)"
        print(f"{i + 1:<{length}}/{iterations}: {current_date} DataFrame {status}")
    except (RasterioIOError, FileNotFoundError):
        previous_date = current_date - timedelta(minutes=delta_minutes)
        previous_file_name = "_".join(re.split("[- :]", str(previous_date)))
        path_to_previous_csv = data_directory_path / f"{previous_file_name}.csv"
        if path_to_previous_csv.exists():
            gdf_union_previous = preprocessor.load_geodataframe_from_csv(path_to_previous_csv)
            preprocessor.save_geodataframe_to_csv(gdf_union_previous, path_to_csv)
            print(f"{i + 1:<{length}}/{iterations}: {current_date} not available; used previous date {previous_date} instead")
        else:
            process_csv(
                preprocessor=preprocessor,
                current_date=previous_date,
                delta_minutes=delta_minutes,
                path_to_csv=path_to_csv,
                data_directory_path=data_directory_path,
                i=i,
                length=length,
                iterations=iterations,
                depth=depth + 1,
            )


def process_tif(
    preprocessor: Preprocessor,
    current_date: datetime,
    delta_minutes: int,
    path_to_tif: Path,
    i: int,
    length: int,
    iterations: int,
    depth: int = 0,
) -> None:
    if depth > 10:
        print(f"{i + 1:<{length}}/{iterations}: Error – could not find any data for {current_date}")
        return
    try:
        preprocessor.save_raster_from_url(current_date, path_to_tif)
        status = "ready" if depth == 0 else f"ready (fallback from {depth} intervals ago)"
        print(f"{i + 1:<{length}}/{iterations}: {current_date} GeoTIFF {status}")
    except (RasterioIOError, FileNotFoundError):
        previous_date = current_date - timedelta(minutes=delta_minutes)
        process_tif(
            preprocessor=preprocessor,
            current_date=previous_date,
            delta_minutes=delta_minutes,
            path_to_tif=path_to_tif,
            i=i,
            length=length,
            iterations=iterations,
            depth=depth + 1,
        )


def run_iteration_loop(
    preprocessor: Preprocessor,
    start_date: datetime,
    iterations: int,
    delta_minutes: int,
    data_dir: Path,
    images_dir: Path,
    direction: Literal[-1, 1],
):
    length = len(str(iterations))
    current_date = start_date
    for i in range(iterations):
        file_name = "_".join(re.split("[- :]", str(current_date)))
        p_csv = data_dir / f"{file_name}.csv"
        p_tif = images_dir / f"{file_name}.tif"

        # If forward (1), we do both. If backward (-1), we only do TIF.
        if direction == 1 and not p_csv.exists():
            process_csv(preprocessor, current_date, delta_minutes, p_csv, data_dir, i, length, iterations)

        if not p_tif.exists():
            process_tif(preprocessor, current_date, delta_minutes, p_tif, i, length, iterations)

        current_date += (timedelta(minutes=delta_minutes) * direction)


def process_dates(
    preprocessor: Preprocessor,
    timestamps: list[datetime],
    iterations: int,
    iterations_minus: int,
    delta_minutes: int,
    data_path: Path,
    images_path: Path,
) -> None:
    """Save GeoDataFrames for given timestamps"""
    for j, main_date in enumerate(timestamps):
        file_name_main = "_".join(re.split("[- :]", str(main_date)))
        data_directory_path = data_path / file_name_main
        images_directory_path = images_path / file_name_main

        if data_directory_path.exists() and images_directory_path.exists():
            print(f"\n({j + 1}/{len(timestamps)}) Skipping {main_date} – already exists")
            continue

        data_directory_path.mkdir(parents=True, exist_ok=True)
        images_directory_path.mkdir(parents=True, exist_ok=True)
        print(
            f"\n({j + 1}/{len(timestamps)}) Directories {data_directory_path} and "
            f"{images_directory_path} created"
        )

        run_iteration_loop(
            preprocessor=preprocessor,
            start_date=main_date,
            iterations=iterations,
            delta_minutes=delta_minutes,
            data_dir=data_directory_path,
            images_dir=images_directory_path,
            direction=1,
        )

        run_iteration_loop(
            preprocessor=preprocessor,
            start_date=main_date - timedelta(minutes=delta_minutes),
            iterations=iterations_minus,
            delta_minutes=delta_minutes,
            data_dir=data_directory_path,
            images_dir=images_directory_path,
            direction=-1,
        )


def main():
    preprocessor = Preprocessor(settings.preprocessor_config)
    with open(TIMESTAMPS_PATH, "rb") as file_in:
        timestamps_to_process = pickle.load(file_in)
    process_dates(
        preprocessor=preprocessor,
        timestamps=timestamps_to_process,
        iterations=NUM_ITERATIONS_PLUS,
        iterations_minus=NUM_ITERATIONS_MINUS,
        delta_minutes=DELTA_MINUTES,
        data_path=DATA_PATH,
        images_path=IMAGES_PATH,
    )


if __name__ == "__main__":
    main()
