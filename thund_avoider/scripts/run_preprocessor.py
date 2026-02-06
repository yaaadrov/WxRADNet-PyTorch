import os
import pickle
import re
import warnings
from datetime import datetime, timedelta
from pathlib import Path

from pandas.errors import SettingWithCopyWarning
from rasterio import RasterioIOError

from thund_avoider.services.preprocessor import Preprocessor
from thund_avoider.settings import settings, TIMESTAMPS_PATH, DATA_PATH

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


NUM_ITERATIONS: int = 25
DELTA_MINUTES: int = 5


def process_dates(preprocessor: Preprocessor, timestamps: list[datetime], iterations: int, delta_minutes: int):
    """Save GeoDataFrames for given timestamps"""
    for j, current_date in enumerate(timestamps):
        directory_path = Path()
        for i in range(iterations):
            file_name = "_".join(re.split("[- :]", str(current_date)))
            length = len(str(iterations))

            # Create corresponding directory if necessary
            if i == 0:
                directory_path = DATA_PATH / file_name
                if directory_path.exists():
                    print(f"\n({j + 1}/{len(timestamps)}) Directory {directory_path} exists. Skipping {current_date}")
                    break
                os.makedirs(directory_path, exist_ok=False)
                print(f"\n({j + 1}/{len(timestamps)}) Directory {directory_path} created")

            try:
                gdf_union = preprocessor.get_gpd_for_current_date(current_date)
                preprocessor.save_geodataframe_to_csv(gdf_union, directory_path / f'{file_name}.csv')
                print(f"{i + 1:<{length}}/{iterations}: {current_date} ready")
            except RasterioIOError:
                previous_file_name = sorted(os.listdir(directory_path))[-1]
                gdf_union = preprocessor.load_geodataframe_from_csv(directory_path / previous_file_name)
                preprocessor.save_geodataframe_to_csv(gdf_union, directory_path / f'{file_name}.csv')
                print(f"{i + 1:<{length}}/{iterations}: {current_date} not available; used previous file instead")

            current_date += timedelta(minutes=delta_minutes)


if __name__ == "__main__":
    preprocessor = Preprocessor(
        base_url=settings.base_url,
        intensity_threshold_low=settings.intensity_threshold_low,
        intensity_threshold_high=settings.intensity_threshold_high,
        distance_between=settings.distance_between,
        distance_avoid=settings.distance_avoid,
    )
    with open(TIMESTAMPS_PATH, "rb") as file_in:
        timestamps_to_process = pickle.load(file_in)
    process_dates(preprocessor, timestamps_to_process, iterations=NUM_ITERATIONS, delta_minutes=DELTA_MINUTES)
