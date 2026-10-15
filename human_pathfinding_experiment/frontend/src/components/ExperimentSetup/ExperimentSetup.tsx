import { conditionDescription } from "../../utils/conditions";
import styles from "./ExperimentSetup.module.css";

interface ExperimentSetupProps {
  trialIndex: number;
  trialsTotal: number;
  condition: "C0" | "C1" | "C2" | null;
  timestamp: string | null;
  isDemo: boolean;
  isSandbox: boolean;
  onStart: () => void;
}

export function ExperimentSetup({ trialIndex, trialsTotal, condition, timestamp, isDemo, isSandbox, onStart }: ExperimentSetupProps) {
  return (
    <div className={styles.container}>
      {isSandbox && <div className={styles.demoBadge}>Свободный тест — результаты не сохраняются</div>}
      {isDemo && !isSandbox && <div className={styles.demoBadge}>ДЕМО-режим — результаты не сохраняются</div>}
      <div className={styles.row}>
        <span className={styles.attempt}>Сценарий {trialIndex} из {trialsTotal}</span>
      </div>
      <div className={styles.row}>
        <span className={styles.label}>Ситуация:</span>
        <span className={styles.value}>{timestamp ?? "—"}</span>
      </div>
      <div className={styles.row}>
        <span className={styles.label}>Постановка:</span>
        <span className={styles.value}>{condition ? conditionDescription(condition) : "—"}</span>
      </div>
      <button className={styles.startButton} onClick={onStart}>
        Начать
      </button>
    </div>
  );
}
