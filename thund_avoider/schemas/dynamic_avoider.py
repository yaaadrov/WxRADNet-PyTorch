import networkx as nx
from pydantic import BaseModel, ConfigDict
from shapely import Point


class SlidingWindowPath(BaseModel):
    strategy: str
    path: list[list[Point]]
    all_paths: list[list[Point]]
    all_graphs: list[nx.Graph]
    success: bool
    success_intermediate: bool
    num_segments: int

    model_config = ConfigDict(arbitrary_types_allowed=True)


class FineTunedPath(BaseModel):
    strategy: str
    path: list[list[Point]]
    all_paths_initial: list[list[Point]]
    all_paths_fine_tuned: list[list[Point]]
    all_paths_initial_raw: list[SlidingWindowPath]
    fine_tuning_iters: list[int]
    fine_tuning_times: list[float]
    success: bool
    success_intermediate: bool
    num_segments: int

    model_config = ConfigDict(arbitrary_types_allowed=True)
