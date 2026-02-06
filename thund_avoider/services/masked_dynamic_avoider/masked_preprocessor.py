import math
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from pydantic import BaseModel
from rasterio.warp import reproject, Resampling
from shapely import Point, Polygon, STRtree, MultiPolygon
from shapely.affinity import rotate, translate

from thund_avoider.services.preprocessor import Preprocessor


class DirectionVector(BaseModel):
    dx: float
    dy: float


class PreprocessorConfig(BaseModel):
    base_url: str
    intensity_threshold_low: int
    intensity_threshold_high: int
    distance_between: int
    distance_avoid: int
    square_side_length_m: int


class MaskedPreprocessor(Preprocessor):
    def __init__(
        self,
        base_url: str,
        intensity_threshold_low: int,
        intensity_threshold_high: int,
        distance_between: int,
        distance_avoid: int,
        square_side_length_m: int,
    ) -> None:
        """
        Initialize `Preprocessor` class

        Args:
            base_url (str): Base URL to get data from
            intensity_threshold_low (int): Threshold for minimum intensity value
            intensity_threshold_high (int): Threshold for maximum intensity value
            distance_between (int): Half distance between two thunderstorms to proceed between
            distance_avoid (int): Minimum buffer to avoid thunderstorm with
            square_side_length_m (int): Size of a square to crop raster image in meters
        """
        super().__init__(
            base_url=base_url,
            intensity_threshold_low=intensity_threshold_low,
            intensity_threshold_high=intensity_threshold_high,
            distance_between=distance_between,
            distance_avoid=distance_avoid,
        )
        self.square_side_length_m = square_side_length_m

    @staticmethod
    def _normalize_direction_vector(direction_vector: DirectionVector) -> tuple[float, float]:
        mag = math.sqrt(direction_vector.dx ** 2 + direction_vector.dy ** 2)
        if np.isclose(mag, 0, atol=1e-6):
            raise ValueError("Direction vector magnitude cannot be zero")
        return direction_vector.dx / mag, direction_vector.dy / mag

    def _fetch_and_crop_raster_data(
        self,
        url: str | Path,
        current_position: Point,
        direction_vector: DirectionVector,
    ) -> tuple[np.ndarray, Affine, rasterio.crs.CRS]:
        """
        Fetch data from geotiff image and crop to a square of a given size based on given direction

        Args:
            url (str | Path): URL or path to fetch image from
            current_position (Point): Aircraft position (center of the bottom edge)
            direction_vector: Direction vector perpendicular to the bottom edge
        """
        with rasterio.open(url) as dataset:
            res = dataset.res[0]  # Pixel size
            pixel_width = self.square_side_length_m // res  # Square size in pixels
            ux, uy = self._normalize_direction_vector(direction_vector)

            # Calculate the center of the square
            dist_to_center = self.square_side_length_m // 2
            center_x = current_position.x + (ux * dist_to_center)
            center_y = current_position.y + (uy * dist_to_center)

            # Calculate rotation angle for the Affine transform
            angle_rad = math.atan2(uy, ux)
            rotation_deg = math.degrees(angle_rad) - 90

            # Construct the Destination Transform
            dst_transform = (
                Affine.translation(center_x, center_y) *
                Affine.rotation(rotation_deg) *
                Affine.translation(-500 * res, 500 * res) *
                Affine.scale(res, -res)
            )

            # Reproject the data
            dest_data = np.empty((pixel_width, pixel_width), dtype=dataset.dtypes[0])
            reproject(
                source=rasterio.band(dataset, 1),
                destination=dest_data,
                src_transform=dataset.transform,
                src_crs=dataset.crs,
                dst_transform=dst_transform,
                dst_crs=dataset.crs,
                resampling=Resampling.bilinear
            )

        return dest_data, dst_transform, dataset.crs

    def _crop_polygons(
        self,
        geometry: list[Polygon],
        current_position: Point,
        direction_vector: DirectionVector,
    ) -> list[Polygon]:
        """Crop a list of polygons to a square of a given size based on given direction"""
        ux, uy = self._normalize_direction_vector(direction_vector)
        half_side = self.square_side_length_m // 2
        center_x = current_position.x + (ux * half_side)
        center_y = current_position.y + (uy * half_side)

        # Construct the oriented cropping box, rotate and move it to the calculated center
        bbox = Polygon(
            [
                (-half_side, -half_side),
                (half_side, -half_side),
                (half_side, half_side),
                (-half_side, half_side),
            ]
        )
        angle_deg = math.degrees(math.atan2(uy, ux)) - 90
        oriented_bbox = rotate(bbox, angle_deg, origin=(0, 0))
        oriented_bbox = translate(oriented_bbox, xoff=center_x, yoff=center_y)

        # Clipping
        tree = STRtree(geometry)
        indices = tree.query(oriented_bbox, predicate="intersects")
        cropped_result = []
        for idx in indices:
            poly = geometry[idx]
            clipped = poly.intersection(oriented_bbox)
            if not clipped.is_empty:
                if isinstance(clipped, Polygon):
                    cropped_result.append(clipped)
                elif isinstance(clipped, MultiPolygon):
                    cropped_result.extend(list(clipped.geoms))

        return cropped_result
