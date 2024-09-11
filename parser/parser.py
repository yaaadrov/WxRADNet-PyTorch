import os
import numpy as np
import rasterio
import cv2
import json
from datetime import datetime, timedelta
from typing import List, Tuple
from wakepy import keep

BASE_URL = 'http://s3-eu-west-1.amazonaws.com/fmi-opendata-radar-geotiff/{year}/{month}/{day}/FIN-DBZ-3067-250M/{year}{month}{day}{hour}{minute}_FIN-DBZ-3067-250M.tif'
PROGRESS_FILE = 'progress.json'
FIRST_DATE = datetime(2020, 12, 31, 23, 55)
LAST_DATE = datetime(2024, 7, 31, 23, 55)
DELTA = timedelta(minutes=5)
DELTA_HOUR = timedelta(hours=1)
INTENSITY_THRESHOLD = 100
MIN_PIXELS = 9000
ANGLES = [
    (60, 120),
    (150, 210),
    (240, 300)
]


def collect_data(input_folder: str) -> None:
    """
    Collect all separate `.npy` files
    Args:
        input_folder (str): Data folder
    """
    arrays_list = []

    for filename in os.listdir(input_folder):
        if filename.endswith('.npy'):
            file_path = os.path.join(input_folder, filename)
            array = np.load(file_path)
            arrays_list.append(array)

    count = len(arrays_list)
    concatenated_array = np.concatenate(arrays_list, axis=0)
    np.save('../thunderstorm_data.npy', concatenated_array)
    print(
        f'\n\n=====    Final array of shape {concatenated_array.shape} containing {count} files saved successfully    =====\n\n'
    )


def rotate_image(image: np.array, angle: float) -> np.array:
    """
    Rotate an image to a specified angle
    Args:
        image (np.array): Image to rotate
        angle (float): Angle to rotate to
    Returns:
        np.array: Rotated image
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated_image


def add_augmentation(hour_data: List) -> List:
    """
    Add data augmentation by rotating images to random angles within predefined ranges
    Args:
        hour_data (list): List of original images
    Returns:
        list: List of augmented images
    """
    augmented_data = []
    for i, border_data in enumerate(hour_data):
        if len(border_data) > 0:
            for angle_range in ANGLES:
                angle = np.random.uniform(angle_range[0], angle_range[1])
                rotated_data = [
                    rotate_image(image, angle).reshape(image.shape[0], image.shape[1], 1) for image in border_data
                ]
                augmented_data.append(rotated_data)
    return augmented_data


class Parser:
    def __init__(
            self,
            base_url: str,
            progress_file: str,
            first_date: datetime,
            last_date: datetime,
            delta: timedelta,
            delta_hour: timedelta,
            intensity_threshold: int,
            min_pixels: int,
            angles: List
    ) -> None:
        """
        Initialize `Parser` class
        Args:
            base_url (str): Base URL to get data from
            progress_file (str): Path to a `json` file to load progress
            first_date (datetime): Date to start parsing with
            last_date (datetime): Last date to parse
            delta (timedelta): 5-minutes step
            delta_hour (timedelta): One-hour step
            intensity_threshold (int): Minimum pixel intensity to consider it as a part of a thundercloud
            min_pixels (int): Minimum pixels count to consider segment informative
            angles (list): List of angle ranges for data augmentation
        """
        self.base_url = base_url
        self.progress_file = progress_file
        self.first_date = first_date
        self.last_date = last_date
        self.delta = delta
        self.delta_hour = delta_hour
        self.intensity_threshold = intensity_threshold
        self.min_pixels = min_pixels
        self.angles = angles

    def save_progress(self, date: datetime) -> None:
        """
        Save current date to `progress.json`
        Args:
            date (datetime): Current date to save
        """
        with open(self.progress_file, 'w') as file:
            dt = date.replace(hour=23, minute=55)
            json.dump({"last_date": dt.strftime('%Y-%m-%d %H:%M')}, file)

    def load_progress(self) -> datetime:
        """
        Get last saved date from `progress.json`
        Returns:
            datetime: Last saved date
        """
        try:
            with open(self.progress_file, 'r') as file:
                data = json.load(file)
                return datetime.strptime(data['last_date'], '%Y-%m-%d %H:%M')
        except (FileNotFoundError, KeyError, ValueError):
            return self.first_date

    def count_informative_segments(self, data: np.array, crop_borders: List) -> Tuple[int, List]:
        """
        Count segments which are considered informative
        Args:
            data (np.array): Raw radar image
            crop_borders (list): List of segments' borders
        Returns:
            tuple: Informative segments count and updated crop_borders
        """
        for border in crop_borders:
            data_slice = cv2.resize(
                data[border[1][0]:border[1][1], border[2][0]:border[2][1]],
                (256, 256),
                interpolation=cv2.INTER_AREA
            )
            border[0] = (data_slice > self.intensity_threshold).sum() > self.min_pixels
        informative_segments = np.sum([border[0] for border in crop_borders])
        return informative_segments, crop_borders

    def is_last_minute_of_month(self, date: datetime) -> bool:
        """
        Check whether date is 23:55 of last day of the month
        Args:
            date (datetime): Current date
        Returns:
            bool: Whether date is 23:55 of last day of the month
        """
        next_day = date + self.delta
        return next_day.month != date.month

    def is_last_hour_of_month(self, date: datetime) -> bool:
        """
        Check whether date is 00:00 of last day of the month
        Args:
            date (datetime): Current date
        Returns:
            bool: Whether date is 23:00 of last day of the month
        """
        next_day = date + self.delta_hour
        return next_day.month != date.month

    def save_to_numpy(self, data_folder: List, current_date: datetime) -> None:
        """
        Save images in `data_folder` to `data` folder as a `.npy` object
        Args:
            data_folder (list): List of images
            current_date (datetime): Current date
        """
        data_folder = [data for data in data_folder if data.ndim == 5]
        if data_folder:
            data_folder = np.concatenate(data_folder)
            np.save(f'data/{current_date.year}-{current_date.month:02d}.npy', data_folder)
            print(
                f'\n=====    Month {current_date.year}-{current_date.month:02d} is ready! Array of shape {data_folder.shape} saved to "data" folder    =====\n'
            )
        else:
            print(
                f'\n=====    Month {current_date.year}-{current_date.month:02d} is ready! Nothing to save    =====\n'
            )
        self.save_progress(current_date)

    def get_data(self) -> None:
        """
        Save each month's data as `.npy` objects
        """
        current_date = self.load_progress() + self.delta
        data_folder = []
        crop_borders = [
            [True, (3000, 4000), (2500, 3500)],
            [True, (4000, 5000), (1500, 2500)],
            [True, (4000, 5000), (2500, 3500)],
            [True, (5000, 6000), (1600, 2600)]
        ]

        while current_date <= self.last_date:

            ######
            print(current_date)
            ######

            url = self.base_url.format(
                year=current_date.year,
                month=f'{current_date.month:02d}',
                day=f'{current_date.day:02d}',
                hour=f'{current_date.hour:02d}',
                minute=f'{current_date.minute:02d}'
            )

            try:
                with rasterio.open(url) as geotiff:
                    data = geotiff.read(1)

                    # If beginning of the hour
                    if current_date.minute == 0:

                        # Arrays to fill with data
                        hour_data = [[] for _ in range(len(crop_borders))]

                        # Set all flags to True
                        for i in range(len(crop_borders)):
                            crop_borders[i][0] = True

                        # Check if particular segments are informative
                        informative_segments, crop_borders = self.count_informative_segments(data, crop_borders)
                        if informative_segments == 0:
                            print(f'-> SKIPPING hour {current_date.strftime("%Y-%m-%d %H")} as non-informative')

                            # Save progress if necessary
                            if self.is_last_hour_of_month(current_date):
                                self.save_to_numpy(data_folder, current_date)
                                data_folder = []

                            current_date += self.delta_hour
                            continue

                        print(
                            f'DOWNLOADING hour {current_date.strftime("%Y-%m-%d %H")} with {informative_segments} informative segments'
                        )

                    # Get and reshape each informative segment
                    for i, border in enumerate(crop_borders):
                        if border[0]:
                            data_slice = cv2.resize(
                                data[border[1][0]:border[1][1], border[2][0]:border[2][1]],
                                (256, 256),
                                interpolation=cv2.INTER_AREA
                            )
                            data_slice = np.where(data_slice > self.intensity_threshold, data_slice, 0)
                            data_slice = data_slice.reshape(data_slice.shape[0], data_slice.shape[1], 1)
                            hour_data[i].append(data_slice)

            except BaseException:
                if current_date.minute == 0:
                    hour_data = [[] for _ in range(len(crop_borders))]
                    print(f'!!    ERROR hour {current_date.strftime("%Y-%m-%d %H")}, first value unavailable')
                    current_date += timedelta(minutes=55)
                else:
                    print(f'!     ERROR with {current_date.strftime("%Y-%m-%d %H:%M")}, using previous data instead')

                    # Add last value
                    for i in range(len(hour_data)):
                        if hour_data[i]:
                            hour_data[i].append(hour_data[i][-1])

            if current_date.minute == 55 and not all(len(inner) == 0 for inner in hour_data):
                # Add augmentation and gather everything in `data_folder`
                # augmented_data = add_augmentation(hour_data)
                # hour_data.extend(augmented_data)
                data_folder.append(
                    np.array([np.array(border_data) for border_data in hour_data if len(border_data) > 0])
                )

            # Save array
            if self.is_last_minute_of_month(current_date):
                self.save_to_numpy(data_folder, current_date)
                data_folder = []

            current_date += self.delta


if __name__ == "__main__":
    parser = Parser(
        base_url=BASE_URL,
        progress_file=PROGRESS_FILE,
        first_date=FIRST_DATE,
        last_date=LAST_DATE,
        delta=DELTA,
        delta_hour=DELTA_HOUR,
        intensity_threshold=INTENSITY_THRESHOLD,
        min_pixels=MIN_PIXELS,
        angles=ANGLES
    )
    with keep.presenting():
        parser.get_data()
        collect_data('data')
