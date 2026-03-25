import { useRef, useEffect, useCallback, useState } from "react";
import { Point2D, ObstacleLayer, PixelTransform, GeoJSONFeature } from "../../types";
import { geoToPixel, pixelToGeo } from "../../utils/geometry";
import { getFlareColor, getCrestColor } from "../../utils/colors";
import { SNAP_DISTANCE_M, CANVAS_WIDTH, CANVAS_HEIGHT, RADII_KM } from "../../utils/constants";
import styles from "./PathfindingCanvas.module.css";

interface PathfindingCanvasProps {
  aPoint: Point2D | null;
  bPoint: Point2D | null;
  currentPosition: Point2D | null;
  directionVector: Point2D | null;
  obstacles: ObstacleLayer[];
  loadedTimeKeys: Set<string>;
  allPaths: Point2D[][];
  currentPath: Point2D[];
  pixelTransform: PixelTransform | null;
  onWaypointAdd: (point: Point2D) => void;
  obstacleMode: "obstacles" | "hull";
  zoomLevel: number;
  isComplete: boolean;
  invalidSegments: number[]; // Indices of invalid segments
  validationObstacles: ObstacleLayer[]; // Obstacles for invalid segments
}

// Device pixel ratio for high-DPI rendering
const DPR = window.devicePixelRatio || 1;

export function PathfindingCanvas({
  aPoint,
  bPoint,
  currentPosition,
  directionVector,
  obstacles,
  loadedTimeKeys,
  allPaths,
  currentPath,
  pixelTransform,
  onWaypointAdd,
  obstacleMode,
  zoomLevel,
  isComplete,
  invalidSegments,
  validationObstacles,
}: PathfindingCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [mousePos, setMousePos] = useState<Point2D | null>(null);
  const [isSnapped, setIsSnapped] = useState(false);

  // Compute zoomed transform centered on current position
  const getZoomedTransform = useCallback((): PixelTransform | null => {
    if (!pixelTransform || !currentPosition) return pixelTransform;

    // Canvas center
    const canvasCenterX = CANVAS_WIDTH / 2;
    const canvasCenterY = CANVAS_HEIGHT / 2;

    // Scale factor with zoom
    const zoomedScale = pixelTransform.scale * zoomLevel;

    // Calculate offset so current position is at canvas center
    const offsetX = canvasCenterX - currentPosition.x * zoomedScale;
    const offsetY = canvasCenterY + currentPosition.y * zoomedScale; // Y is flipped

    return {
      scale: zoomedScale,
      offset_x: offsetX,
      offset_y: offsetY,
      bounds: pixelTransform.bounds,
    };
  }, [pixelTransform, currentPosition, zoomLevel]);

  const toPixel = useCallback(
    (coord: Point2D): Point2D | null => {
      const transform = getZoomedTransform();
      if (!transform) return null;
      return geoToPixel(coord, transform);
    },
    [getZoomedTransform]
  );

  const toGeo = useCallback(
    (pixel: Point2D): Point2D | null => {
      const transform = getZoomedTransform();
      if (!transform) return null;
      return pixelToGeo(pixel, transform);
    },
    [getZoomedTransform]
  );

  // Draw on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !pixelTransform) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set up high-DPI canvas
    canvas.width = CANVAS_WIDTH * DPR;
    canvas.height = CANVAS_HEIGHT * DPR;
    canvas.style.width = `${CANVAS_WIDTH}px`;
    canvas.style.height = `${CANVAS_HEIGHT}px`;
    ctx.scale(DPR, DPR);

    // Clear canvas (black background)
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    // Enable anti-aliasing for smoother rendering
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";

    // Draw concentric circles at current position
    if (currentPosition) {
      const center = toPixel(currentPosition);
      if (center) {
        ctx.strokeStyle = "#333333";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 5]);

        RADII_KM.forEach((radiusKm) => {
          const radiusPx = (radiusKm * 1000) * pixelTransform.scale * zoomLevel;
          ctx.beginPath();
          ctx.arc(center.x, center.y, radiusPx, 0, 2 * Math.PI);
          ctx.stroke();
        });

        ctx.setLineDash([]);

        // Draw radius labels
        ctx.fillStyle = "#666666";
        ctx.font = "12px monospace";
        RADII_KM.forEach((radiusKm) => {
          const radiusPx = (radiusKm * 1000) * pixelTransform.scale * zoomLevel;
          ctx.fillText(`${radiusKm}km`, center.x + radiusPx + 5, center.y);
        });
      }
    }

    // Draw obstacles for loaded time keys
    const sortedLayers = [...obstacles]
      .filter((layer) => loadedTimeKeys.has(layer.time_key))
      .sort((a, b) => a.color_index - b.color_index);

    sortedLayers.forEach((layer) => {
      const color = obstacleMode === "hull"
        ? getCrestColor(layer.color_index)
        : getFlareColor(layer.color_index, 0.6);

      ctx.fillStyle = color;
      ctx.strokeStyle = obstacleMode === "hull"
        ? getCrestColor(layer.color_index, 0.8)
        : getFlareColor(layer.color_index, 0.9);
      ctx.lineWidth = 1.5;

      layer.geojson.features.forEach((feature: GeoJSONFeature) => {
        if (feature.geometry.type === "Polygon") {
          drawPolygon(ctx, feature.geometry.coordinates[0], toPixel);
        } else if (feature.geometry.type === "MultiPolygon") {
          feature.geometry.coordinates.forEach((polygon: number[][][]) => {
            drawPolygon(ctx, polygon[0], toPixel);
          });
        }
      });
    });

    // Draw validation obstacles (concave hull) for invalid segments when complete
    if (isComplete && validationObstacles.length > 0) {
      ctx.globalAlpha = 0.4;
      validationObstacles.forEach((layer) => {
        ctx.fillStyle = "#ff0000";
        ctx.strokeStyle = "#ff0000";
        ctx.lineWidth = 2;

        layer.geojson.features.forEach((feature: GeoJSONFeature) => {
          if (feature.geometry.type === "Polygon") {
            drawPolygon(ctx, feature.geometry.coordinates[0], toPixel);
          } else if (feature.geometry.type === "MultiPolygon") {
            feature.geometry.coordinates.forEach((polygon: number[][][]) => {
              drawPolygon(ctx, polygon[0], toPixel);
            });
          }
        });
      });
      ctx.globalAlpha = 1.0;
    }

    // Draw traveled paths (solid white or colored based on validation)
    ctx.lineWidth = 3;
    ctx.setLineDash([]);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    allPaths.forEach((path, index) => {
      if (isComplete) {
        // When complete, show green for valid segments, red for invalid
        ctx.strokeStyle = invalidSegments.includes(index) ? "#ff0000" : "#00ff00";
      } else {
        ctx.strokeStyle = "#ffffff";
      }
      drawPath(ctx, path, toPixel);
    });

    // Draw current path being built (dashed white) - only if not complete
    // Current position is implicitly the first point
    if (!isComplete && currentPath.length > 0) {
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2.5;
      ctx.setLineDash([8, 4]);
      // Draw from current position to first waypoint, then through all waypoints
      const fullCurrentPath = [currentPosition, ...currentPath].filter(Boolean) as Point2D[];
      drawPath(ctx, fullCurrentPath, toPixel);
      ctx.setLineDash([]);
    }

    // Draw dashed line from last point to cursor - only if not complete
    if (!isComplete) {
      const lastPathPoint = currentPath.length > 0 ? currentPath[currentPath.length - 1] : currentPosition;
      if (lastPathPoint && mousePos && !isSnapped) {
        const lastPixel = toPixel(lastPathPoint);
        if (lastPixel) {
          ctx.strokeStyle = "#888888";
          ctx.lineWidth = 1.5;
          ctx.setLineDash([5, 5]);
          ctx.beginPath();
          ctx.moveTo(lastPixel.x, lastPixel.y);
          ctx.lineTo(mousePos.x, mousePos.y);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
    }

    // Draw A point (large white circle)
    if (aPoint) {
      const aPixel = toPixel(aPoint);
      if (aPixel) {
        ctx.fillStyle = "#ffffff";
        ctx.beginPath();
        ctx.arc(aPixel.x, aPixel.y, 12, 0, 2 * Math.PI);
        ctx.fill();

        ctx.fillStyle = "#000000";
        ctx.font = "bold 14px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("A", aPixel.x, aPixel.y);
      }
    }

    // Draw B point (large white circle)
    if (bPoint) {
      const bPixel = toPixel(bPoint);
      if (bPixel) {
        ctx.fillStyle = isSnapped ? "#00ff00" : "#ffffff";
        ctx.beginPath();
        ctx.arc(bPixel.x, bPixel.y, 12, 0, 2 * Math.PI);
        ctx.fill();

        ctx.fillStyle = "#000000";
        ctx.font = "bold 14px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("B", bPixel.x, bPixel.y);
      }
    }

    // Draw aircraft at current position
    if (currentPosition && directionVector) {
      const posPixel = toPixel(currentPosition);
      if (posPixel) {
        drawAircraft(ctx, posPixel, directionVector);
      }
    }

    // Draw waypoints on current path
    currentPath.forEach((point) => {
      const pixel = toPixel(point);
      if (pixel) {
        ctx.fillStyle = "#ffffff";
        ctx.beginPath();
        ctx.arc(pixel.x, pixel.y, 5, 0, 2 * Math.PI);
        ctx.fill();
      }
    });
  }, [pixelTransform, aPoint, bPoint, currentPosition, directionVector, obstacles, loadedTimeKeys, allPaths, currentPath, mousePos, isSnapped, toPixel, obstacleMode, zoomLevel, isComplete, invalidSegments, validationObstacles]);

  const handleMouseMove = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      setMousePos({ x, y });

      // Check if snapped to B
      if (bPoint && pixelTransform) {
        const bPixel = toPixel(bPoint);
        if (bPixel) {
          const dist = Math.sqrt((x - bPixel.x) ** 2 + (y - bPixel.y) ** 2);
          const snapDistPx = SNAP_DISTANCE_M * pixelTransform.scale * zoomLevel;
          setIsSnapped(dist <= snapDistPx);
        }
      }
    },
    [bPoint, pixelTransform, toPixel, zoomLevel]
  );

  const handleClick = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      // Don't allow adding waypoints when experiment is complete
      if (isComplete || !pixelTransform) return;

      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      const geoPoint = toGeo({ x, y });
      if (!geoPoint) return;

      // If snapped to B, use B point
      if (isSnapped && bPoint) {
        onWaypointAdd(bPoint);
      } else {
        onWaypointAdd(geoPoint);
      }
    },
    [isComplete, pixelTransform, isSnapped, bPoint, onWaypointAdd, toGeo]
  );

  return (
    <div className={styles.container}>
      <canvas
        ref={canvasRef}
        width={CANVAS_WIDTH * DPR}
        height={CANVAS_HEIGHT * DPR}
        className={styles.canvas}
        onMouseMove={handleMouseMove}
        onClick={handleClick}
      />
    </div>
  );
}

function drawPolygon(
  ctx: CanvasRenderingContext2D,
  coordinates: number[][],
  toPixel: (coord: Point2D) => Point2D | null
) {
  if (coordinates.length < 3) return;

  const firstPixel = toPixel({ x: coordinates[0][0], y: coordinates[0][1] });
  if (!firstPixel) return;

  ctx.beginPath();
  ctx.moveTo(firstPixel.x, firstPixel.y);

  for (let i = 1; i < coordinates.length; i++) {
    const pixel = toPixel({ x: coordinates[i][0], y: coordinates[i][1] });
    if (pixel) {
      ctx.lineTo(pixel.x, pixel.y);
    }
  }

  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}

function drawPath(
  ctx: CanvasRenderingContext2D,
  path: Point2D[],
  toPixel: (coord: Point2D) => Point2D | null
) {
  if (path.length < 2) return;

  const firstPixel = toPixel(path[0]);
  if (!firstPixel) return;

  ctx.beginPath();
  ctx.moveTo(firstPixel.x, firstPixel.y);

  for (let i = 1; i < path.length; i++) {
    const pixel = toPixel(path[i]);
    if (pixel) {
      ctx.lineTo(pixel.x, pixel.y);
    }
  }

  ctx.stroke();
}

function drawAircraft(
  ctx: CanvasRenderingContext2D,
  pos: Point2D,
  direction: Point2D
) {
  const size = 15;
  // Negate Y to account for Canvas Y-axis being flipped (Y increases downward)
  const angle = Math.atan2(-direction.y, direction.x);

  ctx.save();
  ctx.translate(pos.x, pos.y);
  ctx.rotate(angle);

  ctx.fillStyle = "#00ff00";
  ctx.beginPath();
  ctx.moveTo(size, 0);
  ctx.lineTo(-size / 2, -size / 2);
  ctx.lineTo(-size / 4, 0);
  ctx.lineTo(-size / 2, size / 2);
  ctx.closePath();
  ctx.fill();

  ctx.restore();
}