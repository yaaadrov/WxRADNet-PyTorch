import json
import logging
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rasterio

from thund_avoider.settings import CropBorderConfig, ParserConfig, PARSER_DATA_PATH


class Parser:
    """
    Radar data parser for collecting and processing thunderstorm imagery.

    Downloads radar data from FMI OpenData, segments it into regions of interest,
    applies data augmentation, and saves monthly numpy arrays.
    """

    def __init__(self, config: ParserConfig) -> None:
        """
        Initialize Parser with configuration.

        Args:
            config (ParserConfig): Parser configuration.
        """
        self._config = config
        self._logger = logging.getLogger(__name__)

        # Initialize crop borders with active flags
        # Format: [is_active, (y_start, y_end), (x_start, x_end)]
        self._crop_borders: list[list] = [
            [True, (border.y_start, border.y_end), (border.x_start, border.x_end)]
            for border in config.crop_borders
        ]

    # ==========================================================================
    # Static Utility Methods
    # ==========================================================================

    @staticmethod
    def collect_data(
        input_folder: str | Path,
        output_path: str | Path = Path("thunderstorm_data.npy"),
    ) -> None:
        """
        Collect all separate `.npy` files into a single array.

        Args:
            input_folder: Data folder containing .npy files.
            output_path: Output path for concatenated array.
        """
        input_folder = Path(input_folder)
        output_path = Path(output_path)
        arrays_list = []

        for file_path in input_folder.glob("*.npy"):
            array = np.load(file_path)
            arrays_list.append(array)

        if not arrays_list:
            print("No .npy files found in input folder")
            return

        count = len(arrays_list)
        concatenated_array = np.concatenate(arrays_list, axis=0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, concatenated_array)
        print(
            f"\n=====    Final array of shape {concatenated_array.shape} containing "
            f"{count} files saved to {output_path}    =====\n"
        )

    @staticmethod
    def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate an image to a specified angle.

        Args:
            image (np.ndarray): Image to rotate.
            angle (float): Angle to rotate to (degrees).

        Returns:
            np.ndarray: Rotated image.
        """
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        return cv2.warpAffine(image, rotation_matrix, (w, h))

    @staticmethod
    def add_augmentation(
        hour_data: list[list[np.ndarray]],
        angles: list[tuple[int, int]],
    ) -> list[list[np.ndarray]]:
        """
        Add data augmentation by rotating images to random angles.

        Args:
            hour_data: List of original image arrays for each border.
            angles: List of angle ranges for rotation.

        Returns:
            List of augmented image arrays.
        """
        augmented_data = []
        for border_data in hour_data:
            if not border_data:
                continue
            for angle_range in angles:
                angle = np.random.uniform(angle_range[0], angle_range[1])
                rotated_data = [
                    Parser.rotate_image(image, angle).reshape(image.shape[0], image.shape[1], 1)
                    for image in border_data
                ]
                augmented_data.append(rotated_data)
        return augmented_data

    # ==========================================================================
    # Progress Management
    # ==========================================================================

    def save_progress(self, date: datetime) -> None:
        """Save current date to progress file."""
        self._config.progress_file.parent.mkdir(parents=True, exist_ok=True)

        dt = date.replace(hour=23, minute=55)
        with open(self._config.progress_file, 'w') as file:
            json.dump({"last_date": dt.strftime('%Y-%m-%d %H:%M')}, file)

    def load_progress(self) -> datetime:
        """
        Get last saved date from progress file.

        Returns:
            datetime: Last saved date, or first_date if not found.
        """
        try:
            with open(self._config.progress_file, 'r') as file:
                data = json.load(file)
                return datetime.strptime(data['last_date'], '%Y-%m-%d %H:%M')
        except (FileNotFoundError, KeyError, ValueError):
            return self._config.first_date

    # ==========================================================================
    # Date Utilities
    # ==========================================================================

    def is_last_minute_of_month(self, date: datetime) -> bool:
        """Check whether date is 23:55 of last day of the month."""
        next_time = date + self._config.delta
        return next_time.month != date.month

    def is_last_hour_of_month(self, date: datetime) -> bool:
        """Check whether date is last hour of the month."""
        next_time = date + self._config.delta_hour
        return next_time.month != date.month

    # ==========================================================================
    # Data Processing Methods
    # ==========================================================================

    def _build_url(self, date: datetime) -> str:
        """Build URL for a specific datetime."""
        return self._config.base_url.format(
            year=date.year,
            month=f"{date.month:02d}",
            day=f"{date.day:02d}",
            hour=f"{date.hour:02d}",
            minute=f"{date.minute:02d}",
        )

    def _extract_segment(self, data: np.ndarray, border: list) -> np.ndarray:
        """
        Extract and resize a segment from radar data.

        Args:
            data: Raw radar image.
            border: Border definition [active, (y_start, y_end), (x_start, x_end)].

        Returns:
            Processed segment image.
        """
        segment = cv2.resize(
            data[border[1][0]:border[1][1], border[2][0]:border[2][1]],
            (self._config.image_size, self._config.image_size),
            interpolation=cv2.INTER_AREA,
        )
        # Apply threshold
        segment = np.where(segment > self._config.intensity_threshold, segment, 0)
        return segment.reshape(segment.shape[0], segment.shape[1], 1)

    def _count_informative_segments(self, data: np.ndarray) -> int:
        """
        Count segments which are considered informative.

        Args:
            data: Raw radar image.

        Returns:
            Number of informative segments.
        """
        count = 0
        for border in self._crop_borders:
            segment = cv2.resize(
                data[border[1][0]:border[1][1], border[2][0]:border[2][1]],
                (self._config.image_size, self._config.image_size),
                interpolation=cv2.INTER_AREA,
            )
            is_informative = (segment > self._config.intensity_threshold).sum() > self._config.min_pixels
            border[0] = is_informative
            if is_informative:
                count += 1
        return count

    def _initialize_hour_data(self) -> list[list[np.ndarray]]:
        """Initialize empty hour data for each border."""
        for border in self._crop_borders:
            border[0] = True
        return [[] for _ in range(len(self._crop_borders))]

    @staticmethod
    def _download_raster(url: str) -> np.ndarray | None:
        """
        Download and read raster data from URL.

        Args:
            url: URL to download from.

        Returns:
            Raster data or None if download failed.
        """
        try:
            with rasterio.open(url) as geotiff:
                return geotiff.read(1)
        except (rasterio.errors.RasterioIOError, Exception):
            return None

    def _process_hour_start(self, data: np.ndarray) -> tuple[list[list[np.ndarray]], int] | None:
        """
        Process the start of an hour (minute 0).

        Args:
            data: First raster data of the hour.

        Returns:
            Tuple of (hour_data, informative_count) or None if hour should be skipped.
        """
        hour_data = self._initialize_hour_data()
        informative_count = self._count_informative_segments(data)

        if informative_count == 0:
            return None

        # Add first segment to hour_data
        for i, border in enumerate(self._crop_borders):
            if border[0]:
                segment = self._extract_segment(data, border)
                hour_data[i].append(segment)

        return hour_data, informative_count

    def _process_minute(self, data: np.ndarray, hour_data: list[list[np.ndarray]]) -> None:
        """
        Process a single minute of data.

        Args:
            data: Raster data for this minute.
            hour_data: Current hour data to append to.
        """
        for i, border in enumerate(self._crop_borders):
            if border[0]:
                segment = self._extract_segment(data, border)
                hour_data[i].append(segment)

    def _handle_download_error(self, date: datetime, hour_data: list[list[np.ndarray]]) -> list[list[np.ndarray]]:
        """
        Handle download errors by using previous data.

        Args:
            date: Current date.
            hour_data: Current hour data.

        Returns:
            Updated hour data.
        """
        if date.minute == 0:
            self._logger.warning(
                f"ERROR: First value unavailable for hour {date.strftime('%Y-%m-%d %H')}"
            )
            return self._initialize_hour_data()
        else:
            self._logger.warning(
                f"ERROR: Using previous data for {date.strftime('%Y-%m-%d %H:%M')}"
            )
            # Append last available value
            for i in range(len(hour_data)):
                if hour_data[i]:
                    hour_data[i].append(hour_data[i][-1])
            return hour_data

    def _finalize_hour(
        self,
        hour_data: list[list[np.ndarray]],
        data_folder: list[np.ndarray],
    ) -> list[np.ndarray]:
        """
        Finalize hour data with augmentation and add to data folder.

        Args:
            hour_data: Hour data to finalize.
            data_folder: Current data folder.

        Returns:
            Updated data folder.
        """
        if all(len(inner) == 0 for inner in hour_data):
            return data_folder

        # Add augmentation
        augmented = self.add_augmentation(hour_data, self._config.angles)
        hour_data.extend(augmented)

        # Convert to numpy array
        hour_array = np.array([
            np.array(border_data)
            for border_data in hour_data
            if len(border_data) > 0
        ])

        if hour_array.size > 0:
            data_folder.append(hour_array)

        return data_folder

    def _save_month_data(
        self,
        output_dir: Path,
        data_folder: list[np.ndarray],
        current_date: datetime,
    ) -> list[np.ndarray]:
        """
        Save month data to numpy file.

        Args:
            data_folder: Data to save.
            current_date: Current date for filename.

        Returns:
            Empty data folder.
        """
        valid_data = [data for data in data_folder if data.ndim == 5]

        if valid_data:
            month_data = np.concatenate(valid_data)
            output_path = output_dir / f"{current_date.year}-{current_date.month:02d}.npy"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, month_data)
            print(
                f"\n=====    Month {current_date.year}-{current_date.month:02d} ready! "
                f"Array of shape {month_data.shape} saved    =====\n"
            )
        else:
            print(
                f"\n=====    Month {current_date.year}-{current_date.month:02d} ready! "
                f"Nothing to save    =====\n"
            )

        self.save_progress(current_date)
        return []

    # ==========================================================================
    # Main Data Collection
    # ==========================================================================

    def get_data(self, output_dir: Path) -> None:
        """
        Download and process radar data.

        Iterates through dates from progress/first_date to last_date,
        downloads radar imagery, processes segments, and saves monthly data.
        """
        current_date = self.load_progress() + self._config.delta
        data_folder: list[np.ndarray] = []
        hour_data: list[list[np.ndarray]] = []

        print(f"Starting data collection from {current_date} to {self._config.last_date}")

        while current_date <= self._config.last_date:
            url = self._build_url(current_date)
            data = self._download_raster(url)

            if data is None:
                # Handle download error
                if current_date.minute == 0:
                    hour_data = self._initialize_hour_data()
                else:
                    hour_data = self._handle_download_error(current_date, hour_data)
            elif current_date.minute == 0:
                # Process start of hour
                result = self._process_hour_start(data)
                if result is None:
                    print(f"-> SKIPPING hour {current_date.strftime('%Y-%m-%d %H')} as non-informative")
                    if self.is_last_hour_of_month(current_date):
                        data_folder = self._save_month_data(output_dir, data_folder, current_date)
                    current_date += self._config.delta_hour
                    continue

                hour_data, informative_count = result
                print(
                    f"DOWNLOADING hour {current_date.strftime('%Y-%m-%d %H')} "
                    f"with {informative_count} informative segments"
                )
            else:
                # Process minute within hour
                self._process_minute(data, hour_data)

            # Finalize hour at minute 55
            if current_date.minute == 55 and not all(len(h) == 0 for h in hour_data):
                data_folder = self._finalize_hour(hour_data, data_folder)

            # Save month data
            if self.is_last_minute_of_month(current_date):
                data_folder = self._save_month_data(output_dir, data_folder, current_date)

            current_date += self._config.delta

        print("\n=====    Data collection completed!    =====\n")
