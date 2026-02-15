import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point, Polygon

from thund_avoider.services.utils import is_line_valid
from thund_avoider.settings import StaticAvoiderConfig


class StaticAvoider:
    def __init__(self, config: StaticAvoiderConfig):
        """
        Initialize `StaticAvoider` class

        Args:
            config (StaticAvoiderConfig): Static Avoider configuration:
                - buffer (int): Buffer distance for geometry simplification
                - tolerance (int): Simplification tolerance for geometry
                - bbox_buffer (int): Buffer distance for bounding box when choosing A and B points
        """
        self.buffer = config.buffer
        self.tolerance = config.tolerance
        self.bbox_buffer = config.bbox_buffer

    def _process_obstacles(
        self,
        gdf: gpd.GeoDataFrame,
        strategy: str,
    ) -> tuple[list[Polygon], list[Polygon]]:
        """
        Simplify obstacles based on the chosen strategy

        Args:
            gdf: GeoDataFrame containing obstacle data
            strategy (str): Strategy to apply, either 'convex' or 'concave'

        Returns:
            tuple[list[Polygon], list[Polygon]]: List of obstacle geometries and list of simplified obstacle geometries in view of given strategy
        """
        obstacles = gdf[strategy]
        simplified_obstacles = obstacles.buffer(self.buffer).simplify(self.tolerance, preserve_topology=True)
        return obstacles, list(simplified_obstacles)

    def _extract_vertices(self, simplified_obstacles: list[Polygon], points: list[Point]) -> list[Point]:
        """
        Extract unique vertices from obstacles and additional points

        Args:
            obstacles (list[Polygon]): List of obstacle geometries
            points (list[Point]): A and B points to include

        Returns:
            list[Point]: Deduplicated list of Point objects
        """
        all_vertices = []
        for polygon in simplified_obstacles:
            all_vertices.extend(polygon.exterior.coords)

        all_vertices = list(set(all_vertices))  # Deduplicate
        all_vertices = [Point(p) for p in all_vertices]
        all_vertices.extend(points)  # Add A and B points
        return all_vertices

    def _build_visibility_graph(self, vertices: list[Point], obstacles: list[Polygon]) -> nx.Graph:
        """
        Build a visibility graph for the given vertices and obstacles

        Args:
            vertices (list[Point]): List of vertex Point objects
            obstacles (list[Point]): List of obstacle geometries

        Returns:
            nx.Graph: Visibility graph with weighted edges
        """
        G = nx.Graph()

        # Add nodes
        for i, point in enumerate(vertices):
            G.add_node(i, pos=(point.x, point.y), point=point)

        # Add edges if visible
        for i, p1 in enumerate(vertices):
            for j, p2 in enumerate(vertices):
                if i != j:  # Avoid self-loops
                    line = LineString([p1, p2])
                    if is_line_valid(line, obstacles):
                        distance = p1.distance(p2)
                        G.add_edge(i, j, weight=distance)

        return G

    def find_shortest_path(
        self,
        gdf: gpd.GeoDataFrame,
        strategy: str,
        A: Point,
        B: Point,
    ) -> list[Point] | ValueError:
        """
        Find the shortest path between two points avoiding obstacles

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame containing obstacle data
            strategy (str): Strategy to apply, either 'convex' or 'concave'
            A (Point): Starting point
            B (Point): Ending point

        Returns:
            list[Point]: List of Point objects representing the shortest path

        Raises:
            ValueError: If strategy is not 'convex' or 'concave'
        """
        if strategy not in ["convex", "concave"]:
            raise ValueError("Strategy must be 'convex' or 'concave'")

        obstacles, simplified_obstacles = self._process_obstacles(gdf, strategy=strategy)
        all_vertices = self._extract_vertices(simplified_obstacles, [A, B])
        G = self._build_visibility_graph(all_vertices, obstacles)
        source_node = len(all_vertices) - 2  # Index of start_point
        target_node = len(all_vertices) - 1  # Index of end_point
        shortest_path_nodes = nx.shortest_path(G, source=source_node, target=target_node, weight="weight")
        return [all_vertices[node] for node in shortest_path_nodes]
