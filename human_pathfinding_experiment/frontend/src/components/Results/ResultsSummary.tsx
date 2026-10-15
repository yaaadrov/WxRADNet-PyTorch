import styles from "./Results.module.css";

interface ResultsSummaryProps {
  isComplete: boolean;
  pathValid: boolean | null;
  invalidSegments: number[];
  trialIndex: number;
  trialsTotal: number;
  isDemo: boolean;
  isSandbox: boolean;
  submitting: boolean;
  error: string | null;
  onNext: () => void;
}

export function ResultsSummary({
  isComplete,
  pathValid,
  invalidSegments,
  trialIndex,
  trialsTotal,
  isDemo,
  isSandbox,
  submitting,
  error,
  onNext,
}: ResultsSummaryProps) {
  if (!isComplete) return null;

  const validationStatus = pathValid === null
    ? "Проверка маршрута…"
    : pathValid
      ? "Маршрут корректен"
      : `Некорректные участки: ${invalidSegments.join(", ")}`;

  const isLastTrial = trialIndex >= trialsTotal;
  const nextLabel = isSandbox
    ? "Выбрать другой сценарий"
    : isDemo
      ? (isLastTrial ? "Вернуться ко входу" : "Следующий демо-сценарий")
      : (isLastTrial ? "Перейти к опросу" : "Перейти к следующему эксперименту");

  return (
    <div className={styles.container}>
      <div className={styles.actions}>
        <span className={pathValid === true ? styles.success : pathValid === false ? styles.error : styles.pending}>
          {validationStatus}
        </span>
        <button className={styles.saveButton} onClick={onNext} disabled={submitting || (!isDemo && pathValid === null)}>
          {submitting ? "Отправка…" : nextLabel}
        </button>
      </div>
      {error && <div className={styles.savedMessage}>{error}</div>}
    </div>
  );
}
