from pyproj import CRS

from thund_avoider.services.dynamic_avoider import DynamicAvoider
from thund_avoider.schemas.dynamic_avoider import DynamicAvoiderConfig
from thund_avoider.services.masked_dynamic_avoider.masked_preprocessor import MaskedPreprocessor, PreprocessorConfig


class MaskedDynamicAvoider(DynamicAvoider):
    def __init__(
        self,
        preprocessor_config: PreprocessorConfig,
        dynamic_avoider_config: DynamicAvoiderConfig,
    ) -> None:
        super().__init__(
            crs=CRS(dynamic_avoider_config.crs),
            velocity_kmh=dynamic_avoider_config.velocity_kmh,
            delta_minutes=dynamic_avoider_config.delta_minutes,
            buffer=dynamic_avoider_config.buffer,
            tolerance=dynamic_avoider_config.tolerance,
            k_neighbors=dynamic_avoider_config.k_neighbors,
            max_distance=dynamic_avoider_config.max_distance,
            simplification_tolerance=dynamic_avoider_config.simplification_tolerance,
            smooth_tolerance=dynamic_avoider_config.smooth_tolerance,
            max_iter=dynamic_avoider_config.max_iter,
            delta_length=dynamic_avoider_config.delta_length,
            strategy=dynamic_avoider_config.strategy,
            tuning_strategy=dynamic_avoider_config.tuning_strategy,
        )
        self.preprocessor = MaskedPreprocessor(
            base_url=preprocessor_config.base_url,
            intensity_threshold_low=preprocessor_config.intensity_threshold_low,
            intensity_threshold_high=preprocessor_config.intensity_threshold_high,
            distance_between=preprocessor_config.distance_between,
            distance_avoid=preprocessor_config.distance_avoid,
            square_side_length_m=preprocessor_config.square_side_length_m,
        )

