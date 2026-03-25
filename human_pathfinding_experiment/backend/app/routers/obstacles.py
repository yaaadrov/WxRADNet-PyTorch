from fastapi import APIRouter, HTTPException

from app.config import MASKING_STRATEGY
from app.schemas.requests import ObstaclesRequest, ValidatePathRequest
from app.schemas.responses import BBoxRequest, BBoxResponse, ObstaclesResponse, ValidationResponse, ValidationResult
from app.services.data_service import DataService
from app.services.geometry_service import GeometryService

router = APIRouter()

# Shared service instances
_data_service = DataService()
_geometry_service = GeometryService()


@router.post("/obstacles/bbox", response_model=BBoxResponse)
async def get_bbox_for_experiment(request: BBoxRequest):
    """
    Pre-calculate bounding box for the entire experiment.

    Returns a pixel transform that encompasses all obstacles for all time_keys,
    plus A and B points. This ensures a stable coordinate system throughout the experiment.
    """
    try:
        # Load all obstacles for the timestamp
        time_keys, dict_obstacles = _data_service.get_obstacles_for_timestamp(request.timestamp)

        # Get AB points
        a_point, b_point = _data_service.get_ab_points_for_timestamp(request.timestamp)

        # Collect all obstacle polygons across all time keys
        all_obstacle_polygons: list[list[tuple[float, float]]] = []

        for time_key in time_keys:
            gdf = dict_obstacles[time_key]

            # Get raw geometry obstacles
            raw_obstacles = gdf["geometry"].tolist()

            if request.mode == "hull":
                # Also get hull obstacles
                hull_obstacles = gdf[request.strategy].tolist()
                all_polygons = raw_obstacles + hull_obstacles
            else:
                all_polygons = raw_obstacles

            all_obstacle_polygons.append(all_polygons)

        # Compute total bounds for the entire experiment
        total_bounds = _geometry_service.get_total_bounds(
            obstacles_layers=all_obstacle_polygons,
            a_point=a_point,
            b_point=b_point,
        )

        pixel_transform = _geometry_service.compute_pixel_transform(total_bounds)

        return BBoxResponse(
            pixel_transform=pixel_transform,
            all_time_keys=time_keys,
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating bounding box: {str(e)}",
        )


@router.post("/obstacles", response_model=ObstaclesResponse)
async def get_obstacles(request: ObstaclesRequest):
    """
    Get obstacles for the current position and direction.

    Returns GeoJSON obstacles clipped to oriented bbox,
    pixel transform for canvas rendering, and available time keys.
    """
    try:
        # Load obstacles for the timestamp
        time_keys, dict_obstacles = _data_service.get_obstacles_for_timestamp(request.timestamp)

        # Determine how many time keys to load
        num_keys = request.window_size
        start_index = request.time_index
        end_index = min(start_index + num_keys, len(time_keys))

        available_time_keys = time_keys[start_index:end_index]

        if not available_time_keys:
            raise HTTPException(
                status_code=404,
                detail="No time keys available for current position",
            )

        # Get oriented bounding box for clipping
        current_position = tuple(request.current_position)
        direction_vector = tuple(request.direction_vector)

        bbox = _geometry_service.get_oriented_bbox(
            current_position=current_position,
            direction_vector=direction_vector,
            strategy=MASKING_STRATEGY,
        )

        # Collect all obstacle polygons for computing total bounds
        all_obstacle_polygons: list[list[tuple[float, float]]] = []

        # Build obstacle layers
        obstacle_layers = []

        for i, time_key in enumerate(available_time_keys):
            gdf = dict_obstacles[time_key]

            # Get raw geometry obstacles
            raw_obstacles = gdf["geometry"].tolist()

            # Clip to bbox
            clipped_raw = _geometry_service.clip_polygons(raw_obstacles, bbox)

            if request.mode == "hull":
                # Also get hull obstacles
                hull_obstacles = gdf[request.strategy].tolist()
                clipped_hull = _geometry_service.clip_polygons(hull_obstacles, bbox)

                # Combine raw and hull
                all_polygons = clipped_raw + clipped_hull
            else:
                all_polygons = clipped_raw

            all_obstacle_polygons.append(all_polygons)

            # Create obstacle layer
            layer = _geometry_service.create_obstacle_layer(
                time_key=time_key,
                polygons=all_polygons,
                color_index=i,
            )
            obstacle_layers.append(layer)

        # Get AB points for bounds calculation
        a_point, b_point = _data_service.get_ab_points_for_timestamp(request.timestamp)

        # Compute total bounds for pixel transform
        total_bounds = _geometry_service.get_total_bounds(
            obstacles_layers=all_obstacle_polygons,
            a_point=a_point,
            b_point=b_point,
        )

        pixel_transform = _geometry_service.compute_pixel_transform(total_bounds)

        return ObstaclesResponse(
            obstacles=obstacle_layers,
            pixel_transform=pixel_transform,
            available_time_keys=available_time_keys,
            current_time_index=request.time_index,
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading obstacles: {str(e)}",
        )


@router.post("/obstacles/validate", response_model=ValidationResponse)
async def validate_path(request: ValidatePathRequest):
    """
    Validate path segments against concave hull obstacles.

    Returns validation results for each segment and obstacles for invalid segments.
    """
    try:
        # Load obstacles for the timestamp
        time_keys, dict_obstacles = _data_service.get_obstacles_for_timestamp(request.timestamp)

        segments: list[ValidationResult] = []
        invalid_segments: list[int] = []
        validation_obstacles: list[dict] = []

        for i, path_segment in enumerate(request.all_paths):
            if i >= len(time_keys):
                break

            time_key = time_keys[i]
            gdf = dict_obstacles[time_key]

            # Get concave hull obstacles for validation
            hull_obstacles = gdf[request.strategy].tolist()

            # Check if segment intersects any obstacle
            is_valid = _geometry_service.is_path_segment_valid(path_segment, hull_obstacles)

            segments.append(ValidationResult(
                is_valid=is_valid,
                segment_index=i,
            ))

            if not is_valid:
                invalid_segments.append(i)
                # Add obstacle layer for this invalid segment
                layer = _geometry_service.create_obstacle_layer(
                    time_key=time_key,
                    polygons=hull_obstacles,
                    color_index=i,
                )
                validation_obstacles.append(layer)

        return ValidationResponse(
            is_valid=len(invalid_segments) == 0,
            segments=segments,
            invalid_segments=invalid_segments,
            validation_obstacles=validation_obstacles,
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error validating path: {str(e)}",
        )
