import { useEffect, useState } from "react";
import { api } from "../../services/api";
import { ObstacleMode, WindowSize, Strategy } from "../../types";
import styles from "./ExperimentSetup.module.css";

interface ExperimentSetupProps {
  timestamp: string | null;
  obstacleMode: ObstacleMode;
  windowSize: WindowSize;
  strategy: Strategy;
  isRunning: boolean;
  onTimestampChange: (timestamp: string) => void;
  onObstacleModeChange: (mode: ObstacleMode) => void;
  onWindowSizeChange: (size: WindowSize) => void;
  onStrategyChange: (strategy: Strategy) => void;
  onStart: () => void;
}

export function ExperimentSetup({
  timestamp,
  obstacleMode,
  windowSize,
  strategy,
  isRunning,
  onTimestampChange,
  onObstacleModeChange,
  onWindowSizeChange,
  onStrategyChange,
  onStart,
}: ExperimentSetupProps) {
  const [timestamps, setTimestamps] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getTimestamps()
      .then((data) => setTimestamps(data.timestamps))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <div className={styles.container}>Loading timestamps...</div>;
  }

  return (
    <div className={styles.container}>
      <div className={styles.row}>
        <label className={styles.label}>Timestamp:</label>
        <select
          className={styles.select}
          value={timestamp || ""}
          onChange={(e) => onTimestampChange(e.target.value)}
          disabled={isRunning}
        >
          <option value="">Select timestamp</option>
          {timestamps.map((ts) => (
            <option key={ts} value={ts}>
              {ts}
            </option>
          ))}
        </select>
      </div>

      <div className={styles.row}>
        <label className={styles.label}>Obstacle Mode:</label>
        <div className={styles.radioGroup}>
          <label className={styles.radioLabel}>
            <input
              type="radio"
              name="obstacleMode"
              value="obstacles"
              checked={obstacleMode === "obstacles"}
              onChange={() => onObstacleModeChange("obstacles")}
              disabled={isRunning}
            />
            Obstacles (raw only)
          </label>
          <label className={styles.radioLabel}>
            <input
              type="radio"
              name="obstacleMode"
              value="hull"
              checked={obstacleMode === "hull"}
              onChange={() => onObstacleModeChange("hull")}
              disabled={isRunning}
            />
            Hull (raw + concave)
          </label>
        </div>
      </div>

      <div className={styles.row}>
        <label className={styles.label}>Window Size:</label>
        <div className={styles.radioGroup}>
          <label className={styles.radioLabel}>
            <input
              type="radio"
              name="windowSize"
              value="1"
              checked={windowSize === 1}
              onChange={() => onWindowSizeChange(1)}
              disabled={isRunning}
            />
            1 (current only)
          </label>
          <label className={styles.radioLabel}>
            <input
              type="radio"
              name="windowSize"
              value="7"
              checked={windowSize === 7}
              onChange={() => onWindowSizeChange(7)}
              disabled={isRunning}
            />
            7 (with predictions)
          </label>
        </div>
      </div>

      <div className={styles.row}>
        <label className={styles.label}>Hull Strategy:</label>
        <div className={styles.radioGroup}>
          <label className={styles.radioLabel}>
            <input
              type="radio"
              name="strategy"
              value="concave"
              checked={strategy === "concave"}
              onChange={() => onStrategyChange("concave")}
              disabled={isRunning}
            />
            Concave
          </label>
          <label className={styles.radioLabel}>
            <input
              type="radio"
              name="strategy"
              value="convex"
              checked={strategy === "convex"}
              onChange={() => onStrategyChange("convex")}
              disabled={isRunning}
            />
            Convex
          </label>
        </div>
      </div>

      {obstacleMode === "obstacles" && (
        <div className={styles.warning}>
          15 km - minimum distance to obstacle; 50 km - minimum distance between obstacles
        </div>
      )}

      <button
        className={styles.startButton}
        onClick={onStart}
        disabled={!timestamp || isRunning}
      >
        {isRunning ? "Running..." : "Start Experiment"}
      </button>
    </div>
  );
}
