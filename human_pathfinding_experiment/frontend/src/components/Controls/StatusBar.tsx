import { Point2D } from "../../types";
import { SNAP_DISTANCE_M } from "../../utils/constants";
import { distanceMeters } from "../../utils/geometry";
import styles from "./StatusBar.module.css";

interface StatusBarProps {
  isRunning: boolean;
  isComplete: boolean;
  timeIndex: number;
  currentPosition: Point2D | null;
  bPoint: Point2D | null;
  startTime: Date | null;
  availableTimeKeys: string[];
  loadedTimeKeys: Set<string>;
  currentPath: Point2D[];
  onConfirmSegment: () => void;
  zoomLevel: number;
  pathValid: boolean | null;
  invalidSegments: number[];
}

export function StatusBar({
  isRunning,
  isComplete,
  timeIndex,
  currentPosition,
  bPoint,
  startTime,
  availableTimeKeys,
  loadedTimeKeys,
  currentPath,
  onConfirmSegment,
  zoomLevel,
  pathValid,
  invalidSegments,
}: StatusBarProps) {
  if (!isRunning) {
    return <div className={styles.container}>Configure and start the experiment</div>;
  }

  if (isComplete) {
    const duration = startTime
      ? Math.round((new Date().getTime() - startTime.getTime()) / 1000)
      : 0;

    const validationStatus = pathValid === null
      ? "Validating..."
      : pathValid
        ? "Path valid"
        : `Invalid segments: ${invalidSegments.join(", ")}`;

    return (
      <div className={styles.container}>
        <span className={pathValid === true ? styles.success : pathValid === false ? styles.error : styles.pending}>
          Experiment complete!
        </span>
        <span>Duration: {duration}s</span>
        <span className={pathValid === true ? styles.success : pathValid === false ? styles.error : styles.pending}>
          {validationStatus}
        </span>
      </div>
    );
  }

  const currentTimeKey = availableTimeKeys[timeIndex] || "N/A";
  const distanceToB = currentPosition && bPoint
    ? Math.round(Math.sqrt(
        (bPoint.x - currentPosition.x) ** 2 + (bPoint.y - currentPosition.y) ** 2
      ) / 1000)
    : "N/A";

  // Calculate path length (from current position through all waypoints)
  const currentPathLength = (() => {
    if (!currentPosition || currentPath.length === 0) return 0;
    let total = distanceMeters(currentPosition, currentPath[0]);
    for (let i = 1; i < currentPath.length; i++) {
      total += distanceMeters(currentPath[i - 1], currentPath[i]);
    }
    return total;
  })();

  // Check if path ends at B
  const pathEndsAtB = currentPath.length > 0 && bPoint
    ? distanceMeters(currentPath[currentPath.length - 1], bPoint) <= SNAP_DISTANCE_M
    : false;

  const segmentLengthKm = Math.round(currentPathLength / 1000);
  const zoomPercent = Math.round(zoomLevel * 100);

  // Can confirm if: path >= 75km OR path ends at B
  const canConfirm = currentPathLength >= 75000 || pathEndsAtB;

  return (
    <div className={styles.container}>
      <div className={styles.item}>
        <span className={styles.label}>Time Index:</span>
        <span className={styles.value}>{timeIndex}</span>
      </div>
      <div className={styles.item}>
        <span className={styles.label}>Current Time:</span>
        <span className={styles.value}>{currentTimeKey}</span>
      </div>
      <div className={styles.item}>
        <span className={styles.label}>Distance to B:</span>
        <span className={styles.value}>{distanceToB} km</span>
      </div>
      <div className={styles.item}>
        <span className={styles.label}>Segment:</span>
        <span className={styles.value}>{segmentLengthKm} km</span>
      </div>
      <div className={styles.item}>
        <span className={styles.label}>Zoom:</span>
        <span className={styles.value}>{zoomPercent}%</span>
      </div>
      <div className={styles.item}>
        <span className={styles.label}>Loaded Layers:</span>
        <span className={styles.value}>{loadedTimeKeys.size} / {availableTimeKeys.length}</span>
      </div>
      <button
        className={`${styles.confirmButton} ${canConfirm ? styles.enabled : styles.disabled}`}
        onClick={onConfirmSegment}
        disabled={!canConfirm}
      >
        Confirm Segment (Enter)
      </button>
      <div className={styles.hint}>
        <span>Arrow Up: Zoom in | Arrow Down: Zoom out | Enter: Confirm segment</span>
      </div>
    </div>
  );
}
