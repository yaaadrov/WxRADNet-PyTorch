import itertools as it
import logging
from collections import defaultdict
from typing import Literal

import geopandas as gpd
import networkx as nx
from pyproj import CRS
from scipy.spatial import Delaunay, KDTree
from shapely import STRtree
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from thund_avoider.services.utils import is_line_valid
from thund_avoider.settings import GraphBuilderConfig


class GraphBuilder:
    """Builds master graphs for pathfinding from obstacle polygons."""

    def __init__(self, config: GraphBuilderConfig) -> None:
        """Initialize GraphBuilder with configuration."""
        self._config = config
        self._crs = CRS(config.crs)
        self._logger = logging.getLogger(__name__)

    @property
    def strategy(self) -> Literal["concave", "convex"]:
        """Get the current strategy."""
        return self._config.strategy

    def collect_vertices(self, flat_obstacles: list[Polygon]) -> list[Point]:
        """Collect vertices from the current obstacle set."""
        vertices = []
        obstacles_simplified = (
            unary_union(gpd.GeoDataFrame(geometry=flat_obstacles, crs=self._crs))
            .buffer(self._config.buffer)
            .simplify(self._config.tolerance, preserve_topology=True)
        )
        if isinstance(obstacles_simplified, Polygon):
            vertices.extend(obstacles_simplified.exterior.coords)
            return list(set(Point(coord) for coord in vertices))
        if isinstance(obstacles_simplified, Polygon.__mro__[1]):  # MultiPolygon
            for polygon in obstacles_simplified.geoms:
                vertices.extend(polygon.exterior.coords)
            return sorted({Point(coord) for coord in vertices}, key=lambda p: (p.x, p.y))
        raise TypeError(f"Unexpected type {type(obstacles_simplified)}")

    def simplify_vertices(self, vertices: list[Point]) -> list[Point]:
        """Simplify vertices under given tolerance."""
        visited = set()
        clusters = []
        tree = STRtree(vertices)
        for i, point in enumerate(vertices):
            if i in visited:
                continue
            buffer = point.buffer(self._config.tolerance)
            idxs = tree.query(buffer)
            neighbors = [vertices[j] for j in idxs if vertices[j].distance(point) <= self._config.tolerance]
            for j in idxs:
                visited.add(j)
            clusters.append(neighbors)
        return [unary_union(cluster).centroid for cluster in clusters]

    def validate_candidates(
        self,
        G: nx.Graph,
        centroids: list[Point],
        candidates: set[tuple[int, int]],
        strtrees: dict[str, STRtree],
    ) -> tuple[nx.Graph, dict[str, list[tuple[int, int, float]]]]:
        """
        Validate candidate edges of the master graph for each time keys.

        Args:
            G (nx.Graph): Master graph to validate candidate edges.
            centroids (list[Point]): List of master centroids.
            candidates (set[tuple[int, int]]): List of candidate edges.
            strtrees (dict[str, STRtree]): Dictionary of strtrees for each time key.

        Returns:
            tuple:
                - dict[str, list[tuple[int, int, float]]]: List of valid edges for each time key.
                - nx.Graph: Master graph with all possible valid edges.
        """
        G_master = G.copy()
        length = len(str(len(strtrees)))
        time_valid_edges: defaultdict[str, list[tuple[int, int, float]]] = defaultdict(list)
        for num, (time_key, tree) in enumerate(strtrees.items()):
            for i, j in candidates:
                p_i, p_j = centroids[i], centroids[j]
                line = LineString([p_i, p_j])
                possible_idxs = tree.query(line)
                possible_obs = [tree.geometries[i] for i in possible_idxs]
                if is_line_valid(line, possible_obs):
                    w = p_i.distance(p_j)
                    time_valid_edges[time_key].append((i, j, w))
                    if not G_master.has_edge(i, j):
                        G_master.add_edge(i, j, weight=w)
            self._logger.info(
                f"\t{num + 1:<{length}}/{len(strtrees)}: Time key {time_key} processed"
            )
        return G_master, time_valid_edges

    def build_master_graph(
        self,
        vertices: list[Point],
        time_keys: list[str],
        dict_obstacles: dict[str, gpd.GeoDataFrame] | dict[str, dict[str, list[Polygon]]],
    ) -> tuple[nx.Graph, dict[str, list[tuple[int, int, float]]]]:
        """
        Build master graph for given vertices from all available obstacles.

        Args:
            vertices (list[Point]): List of all available vertex Point objects.
            time_keys (list[str]): List of master time keys.
            dict_obstacles (dict[str, gpd.GeoDataFrame] | dict[str, dict[str, list[Polygon]]]): Obstacle geometries for each time key.

        Returns:
            tuple:
                - nx.Graph: Master graph with all possible valid edges.
                - dict[str, list[tuple[int, int, float]]]: List of valid edges for each time key.
        """
        self._logger.info("Building master graph:")
        strtrees = {
            time_key: STRtree(list(dict_obstacles[time_key][self._config.strategy]))
            for time_key in time_keys
        }
        G_master = nx.Graph()
        candidates = set()
        centroids = self.simplify_vertices(vertices)

        if centroids:
            for i, p in enumerate(centroids):
                G_master.add_node(i, pos=(p.x, p.y), point=p)
            coords = [(p.x, p.y) for p in centroids]
            tri = Delaunay(coords)
            delaunay_edges = {
                tuple(sorted((i, j)))
                for simplex in tri.simplices
                for i, j in it.combinations(simplex, 2)
            }
            kd = KDTree(coords)
            knn_edges = {
                tuple(sorted((i, j)))
                for i, coord in enumerate(coords)
                for j in kd.query(coord, k=self._config.k_neighbors + 1)[1]
                if i != j
            }
            candidates = delaunay_edges | knn_edges

        return self.validate_candidates(
            G=G_master,
            centroids=centroids,
            candidates=candidates,
            strtrees=strtrees,
        )

    def add_points_to_subgraph(
        self,
        points: list[tuple[Literal["start", "end"], Point]],
        G_master: nx.Graph,
        G_sub: nx.Graph,
        obstacles: list[Polygon],
    ) -> nx.Graph:
        """
        Adds start and end points to the graph and connect them to visible nearby nodes.

        Args:
            points (list[tuple[Literal["start", "end"], Point]]): Start and end points.
            G_master (nx.Graph): Master graph.
            G_sub (nx.Graph): Subgraph for a given window.
            obstacles (list[Polygon]): List of obstacle polygons for a given window.

        Returns:
            nx.Graph: Subgraph with points added to the graph.
        """
        G = G_sub.copy()
        vertices = [data["point"] for _, data in G_master.nodes(data=True)]
        coords = [(p.x, p.y) for p in vertices]
        strtree = STRtree(obstacles)
        kd_tree = None

        has_vertices = len(coords) > 0
        if has_vertices:
            kd_tree = KDTree(coords)

        for point_name, point in points:
            G.add_node(point_name, pos=(point.x, point.y), point=point)
            if has_vertices:
                _, idxs = kd_tree.query([point.x, point.y], k=min(self._config.k_neighbors, len(coords)))
                for i in idxs:
                    target_point = vertices[i]
                    line = LineString([point, target_point])
                    possible_idxs = strtree.query(line)
                    possible_obs = [strtree.geometries[i] for i in possible_idxs]
                    if is_line_valid(line, possible_obs):
                        w = point.distance(target_point)
                        G.add_edge(point_name, i, weight=w)
        return G

    @staticmethod
    def build_visibility_graph(
        G: nx.Graph,
        obstacles: list[Polygon],
    ) -> nx.Graph:
        """
        Build a visibility graph for the given vertices and obstacles.

        Args:
            G (nx.Graph): Graph with vertices.
            obstacles (list[Point]): List of obstacle geometries.

        Returns:
            nx.Graph: Visibility graph with weighted edges.
        """
        for i, node_i in enumerate(G.nodes):
            for j, node_j in enumerate(G.nodes):
                p1, p2 = G.nodes[node_i]["point"], G.nodes[node_j]["point"]
                line = LineString([p1, p2])
                if is_line_valid(line, obstacles):
                    w = p1.distance(p2)
                    G.add_edge(node_i, node_j, weight=w)
        return G

    def create_master_graph(
        self,
        time_keys: list[str],
        dict_obstacles: dict[str, gpd.GeoDataFrame] | dict[str, dict[str, list[Polygon]]],
        prohibited_zone: Polygon = Polygon(),
        previous_path: list[Point] = [],
    ) -> tuple[nx.Graph, dict[str, list[tuple[int, int, float]]]]:
        """
        Create a master graph from `dict_obstacles`.

        Args:
            time_keys (list[str]): All available time keys sorted chronologically.
            dict_obstacles (dict[str, gpd.GeoDataFrame] | dict[str, dict[str, list[Polygon]]]): Dictionary of obstacles GeoDataFrames.
            prohibited_zone (Polygon): Prohibited zone where graph's nodes are not allowed.
            previous_path (list[Point]): Previous path points to include in the graph.

        Returns:
            tuple:
                - nx.Graph: Master graph with all possible valid edges.
                - dict[str, list[tuple[int, int, float]]]: List of valid edges for each time key.
        """
        master_obstacles = [
            dict_obstacles[time_key][self._config.strategy]
            for time_key in time_keys
        ]
        master_obstacles_flat = list(it.chain(*master_obstacles))
        raw_vertices = set(
            it.chain(
                *[self.collect_vertices([obstacle]) for obstacle in master_obstacles_flat]
            )
        )
        all_vertices = list(raw_vertices)
        if not prohibited_zone.is_empty:
            from shapely.prepared import prep
            prepared_zone = prep(prohibited_zone)
            all_vertices = [p for p in raw_vertices if not prepared_zone.intersects(p)]
        all_vertices.extend(previous_path)
        all_vertices.sort(key=lambda p: (p.x, p.y))
        return self.build_master_graph(
            vertices=all_vertices,
            time_keys=time_keys,
            dict_obstacles=dict_obstacles,
        )
