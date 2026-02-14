from pydantic import BaseModel
from shapely import Polygon

from thund_avoider.schemas.dynamic_avoider import SlidingWindowPath, FineTunedPath


class DirectionVector(BaseModel):
    dx: float
    dy: float


class SlidingWindowPathMasked(SlidingWindowPath):
    available_obstacles_dicts: list[dict[str, dict[str, list[Polygon]]]] = []


class FineTunedPathMasked(FineTunedPath):
    available_obstacles_dicts: list[dict[str, dict[str, list[Polygon]]]] = []
