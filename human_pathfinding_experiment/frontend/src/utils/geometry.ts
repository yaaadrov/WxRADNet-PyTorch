import { Point2D, PixelTransform } from "../types";

/**
 * Convert geographic coordinates to canvas pixel coordinates.
 * Y-axis is flipped (geographic Y increases upward, canvas Y increases downward).
 */
export function geoToPixel(coord: Point2D, transform: PixelTransform): Point2D {
  return {
    x: coord.x * transform.scale + transform.offset_x,
    y: -coord.y * transform.scale + transform.offset_y,
  };
}

/**
 * Convert canvas pixel coordinates to geographic coordinates.
 */
export function pixelToGeo(pixel: Point2D, transform: PixelTransform): Point2D {
  return {
    x: (pixel.x - transform.offset_x) / transform.scale,
    y: -(pixel.y - transform.offset_y) / transform.scale,
  };
}

/**
 * Calculate distance between two points in meters.
 */
export function distanceMeters(p1: Point2D, p2: Point2D): number {
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Calculate the unit direction vector from p1 to p2.
 */
export function getDirection(p1: Point2D, p2: Point2D): Point2D {
  const dist = distanceMeters(p1, p2);
  if (dist === 0) {
    return { x: 0, y: 0 };
  }
  return {
    x: (p2.x - p1.x) / dist,
    y: (p2.y - p1.y) / dist,
  };
}

/**
 * Move a point along a direction by a given distance.
 */
export function moveAlongDirection(point: Point2D, direction: Point2D, distance: number): Point2D {
  return {
    x: point.x + direction.x * distance,
    y: point.y + direction.y * distance,
  };
}

/**
 * Calculate the angle between two points in radians.
 */
export function angleBetweenPoints(p1: Point2D, p2: Point2D): number {
  return Math.atan2(p2.y - p1.y, p2.x - p1.x);
}

/**
 * Calculate the angle between two points in degrees.
 */
export function angleBetweenPointsDeg(p1: Point2D, p2: Point2D): number {
  return (angleBetweenPoints(p1, p2) * 180) / Math.PI;
}

/**
 * Check if a point is within a given distance of a target.
 */
export function isWithinDistance(p1: Point2D, p2: Point2D, distance: number): boolean {
  return distanceMeters(p1, p2) <= distance;
}

/**
 * Calculate total path length in meters.
 */
export function totalPathLength(path: Point2D[]): number {
  let total = 0;
  for (let i = 1; i < path.length; i++) {
    total += distanceMeters(path[i - 1], path[i]);
  }
  return total;
}

/**
 * Simplify a path by removing points that are too close together.
 */
export function simplifyPath(path: Point2D[], minDistance: number): Point2D[] {
  if (path.length < 2) return path;

  const result: Point2D[] = [path[0]];
  for (let i = 1; i < path.length; i++) {
    if (distanceMeters(result[result.length - 1], path[i]) >= minDistance) {
      result.push(path[i]);
    }
  }
  return result;
}
