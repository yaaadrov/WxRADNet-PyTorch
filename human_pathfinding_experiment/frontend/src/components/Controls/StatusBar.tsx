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
  availableTimeKeys,
  loadedTimeKeys,
  currentPath,
  onConfirmSegment,
  zoomLevel,
  pathValid,
  invalidSegments,
}: StatusBarProps) {
  if (!isRunning) {
    return <div className={styles.container}>Нажмите «Начать», чтобы приступить к попытке</div>;
  }

  if (isComplete) {
    const validationStatus = pathValid === null
      ? "Проверка маршрута…"
      : pathValid
        ? "Маршрут корректен"
        : `Некорректные участки: ${invalidSegments.join(", ")}`;

    return (
      <div className={styles.container}>
        <span>Маршрут построен!</span>
        <span className={pathValid === true ? styles.success : pathValid === false ? styles.error : styles.pending}>
          {validationStatus}
        </span>
      </div>
    );
  }

  const currentTimeKey = availableTimeKeys[timeIndex] || "—";
  const distanceToB = currentPosition && bPoint
    ? Math.round(Math.sqrt(
        (bPoint.x - currentPosition.x) ** 2 + (bPoint.y - currentPosition.y) ** 2
      ) / 1000)
    : "—";

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
        <span className={styles.label}>Номер шага:</span>
        <span className={styles.value}>{timeIndex}</span>
      </div>
      <div className={styles.item}>
        <span className={styles.label}>Текущее время:</span>
        <span className={styles.value}>{currentTimeKey}</span>
      </div>
      <div className={styles.item}>
        <span className={styles.label}>Расстояние до B:</span>
        <span className={styles.value}>{distanceToB} км</span>
      </div>
      <div className={styles.item}>
        <span className={styles.label}>Сегмент:</span>
        <span className={styles.value}>{segmentLengthKm} км</span>
      </div>
      <div className={styles.item}>
        <span className={styles.label}>Масштаб:</span>
        <span className={styles.value}>{zoomPercent}%</span>
      </div>
      <div className={styles.item}>
        <span className={styles.label}>Загружено слоёв:</span>
        <span className={styles.value}>{loadedTimeKeys.size} / {availableTimeKeys.length}</span>
      </div>
      <button
        className={`${styles.confirmButton} ${canConfirm ? styles.enabled : styles.disabled}`}
        onClick={onConfirmSegment}
        disabled={!canConfirm}
      >
        Подтвердить сегмент (Enter)
      </button>
    </div>
  );
}
