import itertools as it
import math
from typing import Literal

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely import Point, Polygon, STRtree
from shapely.geometry import LineString
from shapely.ops import unary_union

from thund_avoider.schemas.dynamic_avoider import SlidingWindowPath, FineTunedPath
from thund_avoider.schemas.masked_dynamic_avoider import (
    DirectionVector,
    SlidingWindowPathMasked,
    FineTunedPathMasked,
)
from thund_avoider.services.dynamic_avoider import DynamicAvoider
from thund_avoider.services.dynamic_avoider.fine_tuner import FineTuner
from thund_avoider.services.masked_dynamic_avoider.masked_preprocessor import MaskedPreprocessor
from thund_avoider.services.masked_dynamic_avoider.predictor import ThunderstormPredictor
from thund_avoider.services.utils import is_line_valid
from thund_avoider.settings import MaskedPreprocessorConfig, DynamicAvoiderConfig, PredictorConfig


class MaskedDynamicAvoider(DynamicAvoider):
    def __init__(
        self,
        masked_preprocessor_config: MaskedPreprocessorConfig,
        dynamic_avoider_config: DynamicAvoiderConfig,
        predictor_config: PredictorConfig,
    ) -> None:
        super().__init__(dynamic_avoider_config)
        self.preprocessor = MaskedPreprocessor(masked_preprocessor_config)
        self.predictor = ThunderstormPredictor(config=predictor_config, preprocessor=self.preprocessor)

    @staticmethod
    def _create_direction_vector(
        path: SlidingWindowPath | FineTunedPath | list[Point],
    ) -> DirectionVector:
        if isinstance(path, (SlidingWindowPath, FineTunedPath)):
            path = list(it.chain(*path.path))
        unique_path = [key for key, _ in it.groupby(path)]
        # Iterate backwards to find two points with non-zero direction
        for i in range(len(unique_path) - 1, 0, -1):
            point_a, point_b = unique_path[i - 1], unique_path[i]
            dx, dy = point_b.x - point_a.x, point_b.y - point_a.y
            magnitude = math.sqrt(dx ** 2 + dy ** 2)
            if not np.isclose(magnitude, 0, atol=1e-6):
                return DirectionVector(dx=dx, dy=dy)
        raise ValueError(f"Unable to create Direction Vector: all points in path are identical")

    def _add_previous_obstacles(
        self,
        current_result: SlidingWindowPathMasked | FineTunedPathMasked,
        available_obstacles_dict: dict[str, dict[str, list[Polygon]]],
        time_keys: list[str],
        current_time_index: int,
        clipping_bbox: Polygon,
    ) -> dict[str, dict[str, list[Polygon]]]:
        previous_obstacles = (
            current_result.available_obstacles_dicts[-1][time_keys[current_time_index - 1]][self.strategy]
            if current_result.available_obstacles_dicts else []
        )
        # obstacle_to_add = Polygon()
        # if previous_obstacles:
        #     remaining_prev = unary_union(previous_obstacles).difference(clipping_bbox)
        #     if isinstance(remaining_prev, Polygon):
        #         obstacle_to_add = remaining_prev
        #     if isinstance(remaining_prev, MultiPolygon):
        #         polys = [geom for geom in remaining_prev.geoms if isinstance(geom, Polygon)]
        #         obstacle_to_add = max(polys, key=lambda p: p.area) if polys else Polygon()
        if previous_obstacles:
            remaining_prev = unary_union(previous_obstacles).difference(clipping_bbox)
            extracted_list = [remaining_prev] if isinstance(remaining_prev, Polygon) else list(remaining_prev.geoms)
            available_obstacles_dict[time_keys[current_time_index]][self.strategy].extend(extracted_list)
        current_result.available_obstacles_dicts.append(available_obstacles_dict)
        return available_obstacles_dict

    def _get_available_obstacles(
        self,
        current_pos: Point,
        time_keys: list[str],
        current_time_index: int,
        num_preds: int,
        dict_obstacles: dict[str, gpd.GeoDataFrame] | dict[str, dict[str, list[Polygon]]],
        current_direction_vector: DirectionVector,
        clipping_bbox: Polygon,
        prediction_mode: Literal["deterministic", "predictive"] = "deterministic",
    ) -> tuple[list[str], dict[str, dict[str, list[Polygon]]]]:
        match prediction_mode:
            case "deterministic":
                available_time_keys = time_keys[
                    current_time_index : current_time_index + num_preds
                ]
                available_obstacles_dict: dict[str, dict[str, list[Polygon]]] = {
                    time_key: {
                        self.strategy: self.preprocessor.clip_polygons(
                            dict_obstacles[time_key][self.strategy].tolist(),
                            bbox=clipping_bbox,
                        )
                    }
                    for time_key in available_time_keys
                }
            case "predictive":
                current_time_key = time_keys[current_time_index]
                prediction_result = self.predictor.predict(
                    time_keys=time_keys,
                    current_time_index=current_time_index,
                    current_position=current_pos,
                    direction_vector=current_direction_vector,
                    strategy=self.strategy,
                )
                available_time_keys = [current_time_key] + prediction_result.time_keys
                available_obstacles_dict: dict[str, dict[str, list[Polygon]]] = {
                    current_time_key: {
                        self.strategy: self.preprocessor.clip_polygons(
                            dict_obstacles[current_time_key][self.strategy].tolist(),
                            bbox=clipping_bbox,
                        )
                    }
                }
                available_obstacles_dict.update(prediction_result.obstacles_dict)
        return available_time_keys, available_obstacles_dict

    def _prepare_obstacles(
        self,
        current_result: SlidingWindowPathMasked | FineTunedPathMasked,
        current_pos: Point,
        time_keys: list[str],
        current_time_index: int,
        num_preds: int,
        dict_obstacles: dict[str, gpd.GeoDataFrame] | dict[str, dict[str, list[Polygon]]],
        current_direction_vector: DirectionVector,
        masking_strategy: Literal["center", "left", "right", "wide"] = "wide",
        prediction_mode: Literal["deterministic", "predictive"] = "deterministic",
    ) -> tuple[list[str], dict[str, dict[str, list[Polygon]]], Polygon]:
        clipping_bbox = self.preprocessor.get_oriented_bbox(
            current_position=current_pos,
            direction_vector=current_direction_vector,
            strategy=masking_strategy,
        )
        available_time_keys, available_obstacles_dict = self._get_available_obstacles(
            current_pos=current_pos,
            time_keys=time_keys,
            current_time_index=current_time_index,
            num_preds=num_preds,
            dict_obstacles=dict_obstacles,
            current_direction_vector=current_direction_vector,
            clipping_bbox=clipping_bbox,
            prediction_mode=prediction_mode,
        )
        available_obstacles_dict_with_previous = self._add_previous_obstacles(
            current_result=current_result,
            available_obstacles_dict=available_obstacles_dict,
            time_keys=time_keys,
            current_time_index=current_time_index,
            clipping_bbox=clipping_bbox,
        )
        prohibited_boundary_zone = Polygon()  # TODO: Remove prohibited_boundary_zone logic if unnecessary
        return available_time_keys, available_obstacles_dict_with_previous, prohibited_boundary_zone


    def _prepare_master_graph_local(
        self,
        available_time_keys: list[str],
        available_obstacles_dict: dict[str, gpd.GeoDataFrame] | dict[str, dict[str, list[Polygon]]],
        prohibited_zone: Polygon = Polygon(),
        with_str_trees: bool = False,
        previous_path: list[Point] = [],
    ) -> tuple[
        nx.Graph,
        dict[str, list[tuple[int, int, float]]],
        list[Point],
        dict[str, STRtree] | None,
    ]:
        G_master_local, time_valid_edges_local = self.create_master_graph(
            time_keys=available_time_keys,
            dict_obstacles=available_obstacles_dict,
            prohibited_zone=prohibited_zone,
            previous_path=previous_path,
        )
        master_vertices_local = [data["point"] for _, data in G_master_local.nodes(data=True)]
        strtrees = {
            time_key: STRtree(available_obstacles_dict[time_key][self.strategy])
            for time_key in available_time_keys
        } if with_str_trees else None
        return G_master_local, time_valid_edges_local, master_vertices_local, strtrees

    def _validate_path_against_initial_obstacles(
        self,
        path: list[Point],
        time_keys: list[str],
        dict_obstacles: dict[str, gpd.GeoDataFrame] | dict[str, dict[str, list[Polygon]]],
    ) -> bool:
        """
        Validate path against initial obstacles for each time step.

        Used in predictive mode to verify the computed path is valid
        against actual (not predicted) obstacles.

        Args:
            path: Flattened path points.
            time_keys: Time keys corresponding to each path segment.
            dict_obstacles: Initial obstacles dictionary.

        Returns:
            True if path is valid against all initial obstacles.
        """
        max_segment_length = self.velocity_mpm * self.delta_minutes
        segments = FineTuner.split_path_into_segments(path, max_segment_length)
        for i, segment in enumerate(segments):
            if i >= len(time_keys):
                break
            time_key = time_keys[i]
            if time_key not in dict_obstacles:
                continue
            obstacles = dict_obstacles[time_key][self.strategy]
            obstacle_list = obstacles.tolist() if isinstance(obstacles, gpd.GeoSeries) else obstacles
            if not is_line_valid(segment, obstacle_list):
                self.logger.info(
                    f"Predicted path validation failed at time_key={time_key}, segment={i}"
                )
                return False
        return True

    def perform_pathfinding_masked(
        self,
        current_pos: Point,
        end: Point,
        *,
        current_time_index: int,
        window_size: int,
        time_keys: list[str],
        dict_obstacles: dict[str, gpd.GeoDataFrame] | dict[str, dict[str, list[Polygon]]],
        masking_strategy: Literal["center", "left", "right", "wide"] = "wide",
        prediction_mode: Literal["deterministic", "predictive"] = "deterministic",
    ) -> SlidingWindowPathMasked:
        result = SlidingWindowPathMasked(strategy=self.strategy)
        current_direction_vector = self._create_direction_vector([current_pos, end])
        self.logger.info("Start masked pathfinding:")

        while (
            current_pos != end and
            current_time_index < len(time_keys) - window_size
        ):
            available_time_keys, available_obstacles_dict, prohibited_boundary_zone = self._prepare_obstacles(
                current_result=result,
                current_pos=current_pos,
                time_keys=time_keys,
                current_time_index=current_time_index,
                num_preds=window_size,
                dict_obstacles=dict_obstacles,
                current_direction_vector=current_direction_vector,
                masking_strategy=masking_strategy,
                prediction_mode=prediction_mode,
            )
            G_master_local, time_valid_edges_local, master_vertices_local, _ = self._prepare_master_graph_local(
                available_time_keys=available_time_keys,  # Use only available time_keys
                available_obstacles_dict=available_obstacles_dict,
                prohibited_zone=prohibited_boundary_zone,
                previous_path=self._densify_path(result.all_paths[-1]) if result.all_paths else [],
            )
            current_pos, current_time_index, should_continue = self._pathfinding_iter(
                current_pos=current_pos,
                end=end,
                sliding_window_result=result,
                current_time_index=current_time_index,
                window_size=window_size,
                time_keys=time_keys,  # Pass all time_keys, required window will be extracted inside
                available_obstacles_dict=available_obstacles_dict,
                G_master=G_master_local,
                master_vertices=master_vertices_local,
                time_valid_edges=time_valid_edges_local,
            )
            if not should_continue:
                break
            current_direction_vector = self._create_direction_vector(result)

        # Add the end point if not reached yet
        result.num_segments = current_time_index - 1
        if result.path and list(it.chain(*result.path))[-1] != end:
            result.path.append([end])

        # Validate path against initial obstacles in predictive mode
        if prediction_mode == "predictive" and result.path:
            flat_path = [pt for segment in result.path for pt in segment]
            result.is_pred_path_valid = self._validate_path_against_initial_obstacles(
                path=flat_path,
                time_keys=time_keys,
                dict_obstacles=dict_obstacles,
            )

        return result

    def perform_pathfinding_with_finetuning_masked(
        self,
        current_pos: Point,
        end: Point,
        *,
        current_time_index: int,
        window_size: int,
        num_preds: int,
        time_keys: list[str],
        dict_obstacles: dict[str, gpd.GeoDataFrame] | dict[str, dict[str, list[Polygon]]],
        masking_strategy: Literal["center", "left", "right", "wide"] = "wide",
        prediction_mode: Literal["deterministic", "predictive"] = "deterministic",
    ) -> FineTunedPathMasked:
        result = FineTunedPathMasked(strategy=self.strategy)
        current_direction_vector = self._create_direction_vector([current_pos, end])
        if window_size > num_preds:
            raise RuntimeError("Window size cannot be greater than number of available predictions")
        self.logger.info("Start pathfinding with fine-tuning:")

        while (
            current_pos != end and
            current_time_index < len(time_keys) - num_preds
        ):
            available_time_keys, available_obstacles_dict, prohibited_boundary_zone = self._prepare_obstacles(
                current_result=result,
                current_pos=current_pos,
                time_keys=time_keys,
                current_time_index=current_time_index,
                num_preds=num_preds,
                dict_obstacles=dict_obstacles,
                current_direction_vector=current_direction_vector,
                masking_strategy=masking_strategy,
                prediction_mode=prediction_mode,
            )
            G_master_local, time_valid_edges_local, master_vertices_local, strtrees_local = self._prepare_master_graph_local(
                available_time_keys=available_time_keys,  # Use only available time_keys
                available_obstacles_dict=available_obstacles_dict,
                prohibited_zone=prohibited_boundary_zone,
                with_str_trees=True,
                previous_path=self._densify_path(result.all_paths_fine_tuned[-1]) if result.all_paths_fine_tuned else [],
            )
            current_pos, current_time_index, should_continue = self._pathfinding_with_finetuning_iter(
                current_pos=current_pos,
                end=end,
                fine_tuning_result=result,
                current_time_index=current_time_index,
                window_size=window_size,
                time_keys=time_keys,  # Pass all time_keys, required window will be extracted inside
                available_obstacles_dict=available_obstacles_dict,
                G_master=G_master_local,
                master_vertices=master_vertices_local,
                time_valid_edges=time_valid_edges_local,
                strtrees=strtrees_local,
            )
            if not should_continue:
                break
            current_direction_vector = self._create_direction_vector(result)

        # Add the end point if not reached yet
        result.num_segments = current_time_index - 1
        if result.path and list(it.chain(*result.path))[-1] != end:
            result.path.append([end])

        # Validate path against initial obstacles in predictive mode
        if prediction_mode == "predictive" and result.path:
            flat_path = [pt for segment in result.path for pt in segment]
            result.is_pred_path_valid = self._validate_path_against_initial_obstacles(
                path=flat_path,
                time_keys=time_keys,
                dict_obstacles=dict_obstacles,
            )

        return result
