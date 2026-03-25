import math
from typing import Any, Literal

import geopandas as gpd
from shapely import Point, Polygon, STRtree
from shapely.geometry import mapping
from shapely.ops import unary_union

from app.config import settings
from app.schemas.responses import ObstacleLayer, PixelTransform


class GeometryService:
    """Service for geometry operations and coordinate transformations."""

    def __init__(self) -> None:
        """Initialize geometry service."""
        self._square_side_length_m = settings.square_side_length_m
        self._bbox_buffer_m = settings.bbox_buffer_m

    @staticmethod
    def _normalize_direction_vector(dx: float, dy: float) -> tuple[float, float]:
        """Normalize direction vector to unit length."""
        mag = math.sqrt(dx ** 2 + dy ** 2)
        if mag == 0:
            raise ValueError("Direction vector magnitude cannot be zero")
        return dx / mag, dy / mag

    def get_oriented_bbox(
        self,
        current_position: tuple[float, float],
        direction_vector: tuple[float, float],
        strategy: Literal["center", "left", "right", "wide"] = "wide",
    ) -> Polygon:
        """
        Get an oriented bounding box polygon based on position and direction.

        Reuses logic from MaskedPreprocessor.get_oriented_bbox().
        """
        ux, uy = self._normalize_direction_vector(direction_vector[0], direction_vector[1])
        lx, ly = -uy, ux  # Perpendicular vector (points to the LEFT of direction)

        half_side = self._square_side_length_m / 2

        # Get width and offsets based on strategy
        match strategy:
            case "left":
                width = self._square_side_length_m
                offset_fwd = half_side
                offset_side = half_side
            case "right":
                width = self._square_side_length_m
                offset_fwd = half_side
                offset_side = -half_side
            case "wide":
                width = self._square_side_length_m * 2
                offset_fwd = half_side
                offset_side = 0
            case "center":
                width = self._square_side_length_m
                offset_fwd = half_side
                offset_side = 0
            case _:
                width = self._square_side_length_m * 2
                offset_fwd = half_side
                offset_side = 0

        # Construct local bounding box (unrotated, centered at 0, 0)
        half_width = width / 2
        bbox = Polygon([
            (-half_width, -half_side),
            (half_width, -half_side),
            (half_width, half_side),
            (-half_width, half_side),
        ])

        # Calculate world coordinates for the bbox center
        current_pos = Point(current_position)
        center_x = current_pos.x + (ux * offset_fwd) + (lx * offset_side)
        center_y = current_pos.y + (uy * offset_fwd) + (ly * offset_side)

        # Transform the bbox to world position
        from shapely.affinity import rotate, translate
        angle_deg = math.degrees(math.atan2(uy, ux)) - 90
        oriented_bbox = rotate(bbox, angle_deg, origin=(0, 0))
        return translate(oriented_bbox, xoff=center_x, yoff=center_y)

    @staticmethod
    def clip_polygons(
        geometry: list[Polygon],
        bbox: Polygon,
    ) -> list[Polygon]:
        """
        Clip polygons to bounding box, returning only the intersecting portions.

        Reuses logic from MaskedPreprocessor.clip_polygons().
        """
        tree = STRtree(geometry)
        indices = tree.query(bbox, predicate="intersects")
        cropped_result = []

        for idx in indices:
            poly = geometry[idx]
            clipped = poly.intersection(bbox)
            if not clipped.is_empty:
                if isinstance(clipped, Polygon):
                    cropped_result.append(clipped)
                else:
                    # Handle MultiPolygon
                    cropped_result.extend(list(clipped.geoms))

        return cropped_result

    def compute_pixel_transform(
        self,
        bbox: Polygon,
        canvas_width: int | None = None,
        canvas_height: int | None = None,
        padding: int | None = None,
    ) -> PixelTransform:
        """Compute pixel transform for canvas rendering."""
        canvas_width = canvas_width or settings.canvas_width
        canvas_height = canvas_height or settings.canvas_height
        padding = padding or settings.canvas_padding

        minx, miny, maxx, maxy = bbox.bounds

        scale = min(
            (canvas_width - 2 * padding) / (maxx - minx),
            (canvas_height - 2 * padding) / (maxy - miny),
        )

        return PixelTransform(
            scale=scale,
            offset_x=padding - minx * scale,
            offset_y=canvas_height - padding + miny * scale,  # Y flipped
            bounds={
                "minX": minx,
                "maxX": maxx,
                "minY": miny,
                "maxY": maxy,
            },
        )

    @staticmethod
    def polygon_to_geojson(polygon: Polygon) -> dict[str, Any]:
        """Convert a Shapely polygon to GeoJSON format."""
        return mapping(polygon)

    @staticmethod
    def polygons_to_feature_collection(
        polygons: list[Polygon],
        properties: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convert a list of polygons to a GeoJSON FeatureCollection."""
        features = []
        for i, poly in enumerate(polygons):
            feature = {
                "type": "Feature",
                "geometry": mapping(poly),
                "properties": {**(properties or {}), "id": i},
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features,
        }

    def create_obstacle_layer(
        self,
        time_key: str,
        polygons: list[Polygon],
        color_index: int,
    ) -> ObstacleLayer:
        """Create an obstacle layer with GeoJSON and metadata."""
        return ObstacleLayer(
            time_key=time_key,
            geojson=self.polygons_to_feature_collection(polygons),
            color_index=color_index,
        )

    def get_total_bounds(
        self,
        obstacles_layers: list[list[Polygon]],
        a_point: tuple[float, float],
        b_point: tuple[float, float],
        buffer_percent: float = 0.1,
    ) -> Polygon:
        """Get total bounding polygon encompassing all obstacles and A/B points."""
        # Start with A and B points as the primary bounds
        a_pt = Point(a_point)
        b_pt = Point(b_point)

        # Calculate initial bounds from A and B
        minx = min(a_pt.x, b_pt.x)
        maxx = max(a_pt.x, b_pt.x)
        miny = min(a_pt.y, b_pt.y)
        maxy = max(a_pt.y, b_pt.y)

        # Add buffer around A-B corridor (50% of A-B distance)
        ab_distance = math.sqrt((b_pt.x - a_pt.x) ** 2 + (b_pt.y - a_pt.y) ** 2)
        corridor_buffer = ab_distance * 0.5

        minx -= corridor_buffer
        maxx += corridor_buffer
        miny -= corridor_buffer
        maxy += corridor_buffer

        # Now include obstacles, but only those within the A-B corridor buffer
        for layer in obstacles_layers:
            for poly in layer:
                if poly.is_empty:
                    continue
                px_min, py_min, px_max, py_max = poly.bounds
                # Only include if polygon intersects or is near the A-B corridor
                if (px_min <= maxx and px_max >= minx and
                    py_min <= maxy and py_max >= miny):
                    minx = min(minx, px_min)
                    maxx = max(maxx, px_max)
                    miny = min(miny, py_min)
                    maxy = max(maxy, py_max)

        # Add final buffer
        dx = (maxx - minx) * buffer_percent
        dy = (maxy - miny) * buffer_percent

        return Polygon([
            (minx - dx, miny - dy),
            (maxx + dx, miny - dy),
            (maxx + dx, maxy + dy),
            (minx - dx, maxy + dy),
        ])

    def is_path_segment_valid(
        self,
        path_segment: list[tuple[float, float]],
        obstacles: list[Any],
    ) -> bool:
        """
        Check if a path segment intersects any obstacle.

        Args:
            path_segment: List of (x, y) coordinate pairs.
            obstacles: List of Shapely Polygon obstacles.

        Returns:
            True if the segment does not intersect any obstacle.
        """
        from shapely.geometry import LineString

        if len(path_segment) < 2:
            return True

        # Create LineString from path segment
        line = LineString(path_segment)

        # Check intersection with each obstacle
        for obstacle in obstacles:
            if line.intersects(obstacle):
                return False

        return True
