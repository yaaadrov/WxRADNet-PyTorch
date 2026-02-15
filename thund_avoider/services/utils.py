import math

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point, Polygon


def is_line_valid(line: LineString, obstacles: list[Polygon]) -> bool:
    """
    Check if a line intersects any obstacle.

    Args:
        line (LineString): Line to check.
        obstacles (list[Polygon]): List of obstacle geometries.

    Returns:
        bool: True if the line is valid (does not intersect any obstacle), False otherwise.
    """
    return not any(line.intersects(obstacle) for obstacle in obstacles)


def compute_angle(A: Point, B: Point, C: Point) -> float:
    """
    Compute the angle (in degrees) between three shapely Points A, B, C with B as the vertex.

    Args:
        A (Point): Shapely Point.
        B (Point): Shapely Point - vertex.
        C (Point): Shapely Point.

    Returns:
        float: Angle in degrees between vectors BA and BC.
    """
    try:
        ba_x = A.x - B.x
        ba_y = A.y - B.y

        bc_x = C.x - B.x
        bc_y = C.y - B.y

        dot_product = ba_x * bc_x + ba_y * bc_y
        mag_ba = math.hypot(ba_x, ba_y)
        mag_bc = math.hypot(bc_x, bc_y)

        if mag_ba == 0 or mag_bc == 0:
            raise ValueError("Cannot compute angle with zero-length vector")

        cos_theta = dot_product / (mag_ba * mag_bc)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angle_rad = math.acos(cos_theta)

        return math.degrees(angle_rad)

    except Exception:
        return 180.0


def get_unit_direction(p1: Point, p2: Point) -> tuple[float, float]:
    """
    Compute the unit direction vector from point p1 to point p2.

    Args:
        p1 (Point): The starting point.
        p2 (Point): The ending point.

    Returns:
        tuple[float, float]: The unit vector (dx, dy) pointing from p1 to p2.
    """
    dx, dy = p2.x - p1.x, p2.y - p1.y
    length = np.hypot(dx, dy)
    if length == 0:
        return 0, 0
    return dx / length, dy / length


def shift_point(point: Point, direction: tuple[float, float], distance: float) -> Point:
    """
    Shift a point along a unit direction vector by a given distance.

    Args:
        point (Point): The original point to be shifted.
        direction (tuple[float, float]): A 2D unit direction vector (dx, dy).
        distance (float): Distance to shift along the direction vector.

    Returns:
        Point: A new point shifted from the original by the specified distance.
    """
    dx, dy = direction
    return Point(point.x + dx * distance, point.y + dy * distance)


def linestring_to_points(linestring: LineString) -> list[Point]:
    """
    Convert a LineString to a list of Points.

    Args:
        linestring (LineString): The LineString to convert.

    Returns:
        list[Point]: List of Point objects extracted from the LineString.
    """
    return [Point(coord) for coord in linestring.coords]


def subgraph_for_time_keys(
    G_master: nx.Graph,
    time_keys: list[str],
    time_valid_edges: dict[str, list[tuple[int, int, float]]],
) -> nx.Graph:
    """
    Create a subgraph with edges valid across specified time keys.

    Args:
        G_master (nx.Graph): Master graph.
        time_keys (list[str]): List of time keys (within a window) to validate edges for.
        time_valid_edges (dict[str, list[tuple[int, int, float]]]): Valid edges for time keys.

    Returns:
        nx.Graph: Subgraph with edges valid across specified time keys.
    """
    edge_sets = [
        set(
            (i, j, w)
            for i, j, w in time_valid_edges[time_key]
        )
        for time_key in time_keys
    ]
    common_edges = set.intersection(*edge_sets)
    G_sub = nx.Graph()
    G_sub.add_nodes_from(G_master.nodes(data=True))
    for i, j, w in common_edges:
        G_sub.add_edge(i, j, weight=w)
    return G_sub
