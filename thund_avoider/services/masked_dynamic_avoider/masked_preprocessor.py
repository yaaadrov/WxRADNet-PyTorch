import math
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import reproject, Resampling
from shapely import Point, Polygon, STRtree, MultiPolygon, MultiPoint
from shapely.ops import unary_union
from shapely.affinity import rotate, translate

from thund_avoider.schemas.masked_dynamic_avoider import DirectionVector
from thund_avoider.services.preprocessor import Preprocessor
from thund_avoider.settings import MaskedPreprocessorConfig


class MaskedPreprocessor(Preprocessor):
    def __init__(self, config: MaskedPreprocessorConfig = MaskedPreprocessorConfig()) -> None:
        """
        Initialize `Preprocessor` class

        Args:
            config (MaskedPreprocessorConfig): Masked Preprocessor configuration:
                - base_url (str): Base URL to get data from
                - intensity_threshold_low (int): Threshold for minimum intensity value
                - intensity_threshold_high (int): Threshold for maximum intensity value
                - distance_between (int): Half distance between two thunderstorms to proceed between
                - distance_avoid (int): Minimum buffer to avoid thunderstorm with
                - square_side_length_m (int): Size of a square to crop raster image in meters
        """
        super().__init__(config.preprocessor_config)
        self.square_side_length_m = config.square_side_length_m
        self.bbox_buffer_m = config.bbox_buffer_m

    @staticmethod
    def _normalize_direction_vector(direction_vector: DirectionVector) -> tuple[float, float]:
        mag = math.sqrt(direction_vector.dx ** 2 + direction_vector.dy ** 2)
        if np.isclose(mag, 0, atol=1e-6):
            raise ValueError("Direction vector magnitude cannot be zero")
        return direction_vector.dx / mag, direction_vector.dy / mag

    @staticmethod
    def _get_center_offset_raster(
        strategy: Literal["center", "left", "right", "wide"],
        current_position: Point,
        pixel_width: float,
        half_side: float,
        ux: float,
        uy: float,
    ) -> tuple[float, float, tuple[float, float]]:
        lx, ly = -uy, ux  # Perpendicular vector (Points to the LEFT of the direction)
        match strategy:
            case "left":  # Move half-side Forward, and half-side Left
                center_x = current_position.x + (ux * half_side) + (lx * half_side)
                center_y = current_position.y + (uy * half_side) + (ly * half_side)
                out_shape = (pixel_width, pixel_width)
            case "right":  # Move half-side Forward, and half-side Right (negative Left)
                center_x = current_position.x + (ux * half_side) - (lx * half_side)
                center_y = current_position.y + (uy * half_side) - (ly * half_side)
                out_shape = (pixel_width, pixel_width)
            case "wide":  # Move half-side Forward and double the width
                center_x = current_position.x + (ux * half_side)
                center_y = current_position.y + (uy * half_side)
                out_shape = (pixel_width, pixel_width * 2)
            case "center":  # Default behavior
                center_x = current_position.x + (ux * half_side)
                center_y = current_position.y + (uy * half_side)
                out_shape = (pixel_width, pixel_width)
        return center_x, center_y, out_shape

    @staticmethod
    def _get_center_offset_poly(
        strategy: Literal["center", "left", "right", "wide"],
        side: float,
        half_side: float,
    ) -> tuple[float, float, float]:
        match strategy:
            case "left":  # Center is half-side forward, half-side to the LEFT
                width = side
                offset_fwd = half_side
                offset_side = half_side
            case "right":  # Center is half-side forward, half-side to the RIGHT (negative left)
                width = side
                offset_fwd = half_side
                offset_side = -half_side
            case "wide":  # Center is half-side forward, 0 offset sideways
                width = side * 2
                offset_fwd = half_side
                offset_side = 0
            case "center":  # Default behavior
                width = side
                offset_fwd = half_side
                offset_side = 0
        return width, offset_fwd, offset_side

    def fetch_and_crop_raster_data(
        self,
        url: str | Path,
        current_position: Point,
        direction_vector: DirectionVector,
        strategy: Literal["center", "left", "right", "wide"] = "center",
    ) -> tuple[np.ndarray, Affine, rasterio.crs.CRS]:
        """
        Fetch data from geotiff image and crop to a square of a given size based on given direction

        Args:
            url (str | Path): URL or path to fetch image from
            current_position (Point): Aircraft position (center of the bottom edge)
            direction_vector (DirectionVector): Direction vector perpendicular to the bottom edge
            strategy (Literal["center", "left", "right", "wide"]): Strategy to crop
        """
        with rasterio.open(url) as dataset:
            res = dataset.res[0]  # Pixel size
            pixel_width = int(self.square_side_length_m // res)  # Square size in pixels
            half_side = self.square_side_length_m / 2
            ux, uy = self._normalize_direction_vector(direction_vector)

            # Calculate offset
            center_x, center_y, (rows, cols) = self._get_center_offset_raster(
                strategy=strategy,
                current_position=current_position,
                pixel_width=pixel_width,
                half_side=half_side,
                ux=ux,
                uy=uy,
            )

            # Calculate rotation angle for the Affine transform
            angle_rad = math.atan2(uy, ux)
            rotation_deg = math.degrees(angle_rad) - 90

            # Construct the Destination Transform
            dst_transform = (
                Affine.translation(center_x, center_y) *
                Affine.rotation(rotation_deg) *
                Affine.translation(-cols / 2 * res, rows / 2 * res) *
                Affine.scale(res, -res)
            )

            # Reproject the data
            dest_data = np.empty((rows, cols), dtype=dataset.dtypes[0])
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

    def get_oriented_bbox(
        self,
        current_position: Point,
        direction_vector: DirectionVector,
        strategy: Literal["center", "left", "right", "wide"] = "center",
    ) -> Polygon:
        ux, uy = self._normalize_direction_vector(direction_vector)
        lx, ly = -uy, ux
        half_side = self.square_side_length_m / 2
        width, offset_fwd, offset_side = self._get_center_offset_poly(
            strategy=strategy,
            side=self.square_side_length_m,
            half_side=half_side
        )

        # Construct the local Bounding Box (unrotated, centered at 0, 0)
        half_width = width / 2
        bbox = Polygon(
            [
                (-half_width, -half_side),
                (half_width, -half_side),
                (half_width, half_side),
                (-half_width, half_side),
            ]
        )

        # Calculate world coordinates for the bbox center
        center_x = current_position.x + (ux * offset_fwd) + (lx * offset_side)
        center_y = current_position.y + (uy * offset_fwd) + (ly * offset_side)

        # Transform the bbox to the world position
        angle_deg = math.degrees(math.atan2(uy, ux)) - 90
        oriented_bbox = rotate(bbox, angle_deg, origin=(0, 0))
        return translate(oriented_bbox, xoff=center_x, yoff=center_y)

    @staticmethod
    def clip_polygons(
        geometry: list[Polygon],
        bbox: Polygon,
    ) -> list[Polygon]:
        """Crop a list of polygons to a square of a given size based on given direction"""
        tree = STRtree(geometry)
        indices = tree.query(bbox, predicate="intersects")
        cropped_result = []
        for idx in indices:
            poly = geometry[idx]
            clipped = poly.intersection(bbox)
            if not clipped.is_empty:
                if isinstance(clipped, Polygon):
                    cropped_result.append(clipped)
                elif isinstance(clipped, MultiPolygon):
                    cropped_result.extend(list(clipped.geoms))
        return cropped_result

    def get_prohibited_boundary_zone(
        self,
        geometry: list[Polygon],
        bbox: Polygon,
    ) -> Polygon:
        """Identify parts of the bbox edges that are inside obstacles and create a buffer around"""
        merged_obstacles = unary_union(geometry)
        coords = list(bbox.exterior.coords)[:-1]
        corners = [Point(c) for c in coords]
        intersected_corners = [p for p in corners if p.intersects(merged_obstacles)]
        if not intersected_corners:
            return Polygon()
        corner_collection = MultiPoint(intersected_corners)
        prohibited_zone = corner_collection.buffer(self.bbox_buffer_m)
        prohibited_outside = prohibited_zone.difference(bbox)
        return prohibited_outside

    # def get_prohibited_boundary_zone(  # Вариант 1 - без углов
    #     self,
    #     geometry: list[Polygon],
    #     bbox: Polygon,
    # ) -> Polygon:
    #     """Identify parts of the bbox edges that are inside obstacles and create a buffer around"""
    #     merged_obstacles = unary_union(geometry)
    #     boundary_line = bbox.boundary
    #     intersected_edges = boundary_line.intersection(merged_obstacles)
    #     if intersected_edges.is_empty:
    #         return Polygon()
    #     prohibited_zone = intersected_edges.buffer(self.bbox_buffer_m)
    #     prohibited_outside = prohibited_zone.difference(bbox)
    #     return prohibited_outside

    # def get_prohibited_boundary_zone(  # Вариант 2 - угловые составляющие bbox
    #     self,
    #     geometry: list[Polygon],
    #     bbox: Polygon,
    # ) -> Polygon:
    #     """Identify parts of the bbox edges that are inside obstacles and create a buffer around"""
    #     merged_obstacles = unary_union(geometry)
    #
    #     # Identify corners inside obstacles
    #     coords = list(bbox.exterior.coords)[:-1]
    #     corners = [Point(c) for c in coords]
    #     intersected_corners = [p for p in corners if p.intersects(merged_obstacles)]
    #     if not intersected_corners:
    #         return Polygon()
    #
    #     # Get all edge segments that are inside obstacles
    #     boundary_line = bbox.boundary
    #     all_intersected_edges = boundary_line.intersection(merged_obstacles)
    #     if all_intersected_edges.is_empty:
    #         return Polygon()
    #
    #     # Filter segments: keep only those that touch an intersected corner
    #     if isinstance(all_intersected_edges, LineString):
    #         segments = [all_intersected_edges]
    #     else:
    #         segments = list(all_intersected_edges.geoms)
    #     valid_segments = []
    #     for seg in segments:
    #         if any(seg.intersects(corner) for corner in intersected_corners):
    #             valid_segments.append(seg)
    #     if not valid_segments:
    #         return Polygon()
    #
    #     # Buffer the valid segments and return the area outside the bbox
    #     intersected_edges_valid = unary_union(valid_segments)
    #     prohibited_zone = intersected_edges_valid.buffer(self.bbox_buffer_m)
    #     prohibited_outside = prohibited_zone.difference(bbox)
    #
    #     return prohibited_outside
