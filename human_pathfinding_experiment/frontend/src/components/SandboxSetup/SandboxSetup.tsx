import { useEffect, useState } from "react";
import { api } from "../../services/api";
import { conditionDescription } from "../../utils/conditions";
import styles from "./SandboxSetup.module.css";

const CONDITIONS = ["C0", "C1", "C2"] as const;

type Condition = (typeof CONDITIONS)[number];

const CONDITION_LABELS: Record<Condition, string> = {
  C0: "Только очаги",
  C1: "+ буферные зоны",
  C2: "+ прогноз",
};

interface SandboxSetupProps {
  onStart: (timestamp: string, condition: Condition) => Promise<void>;
  onBack: () => void;
}

/** Free sandbox picker: any available timestamp and condition, nothing is stored in the database. */
export function SandboxSetup({ onStart, onBack }: SandboxSetupProps) {
  const [timestamps, setTimestamps] = useState<string[]>([]);
  const [timestamp, setTimestamp] = useState<string>("");
  const [condition, setCondition] = useState<Condition>("C0");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.getTimestamps()
      .then((data) => {
        setTimestamps(data.timestamps);
        setTimestamp(data.timestamps[0] ?? "");
      })
      .catch(() => setError("Не удалось загрузить список timestamps"));
  }, []);

  const handleStart = async () => {
    setError(null);
    setLoading(true);
    try {
      await onStart(timestamp, condition);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Не удалось запустить тест");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.container}>
      <h2 className={styles.title}>Свободный тест</h2>
      <p className={styles.subtitle}>Выберите любую ситуацию и постановку. Результаты не сохраняются.</p>
      <div className={styles.form}>
        <label className={styles.field}>
          <span className={styles.label}>Ситуация (timestamp)</span>
          <select
            className={styles.select}
            value={timestamp}
            onChange={(e) => setTimestamp(e.target.value)}
          >
            {timestamps.map((ts) => (
              <option key={ts} value={ts}>{ts}</option>
            ))}
          </select>
        </label>
        <div className={styles.field}>
          <span className={styles.label}>Постановка</span>
          <div className={styles.options}>
            {CONDITIONS.map((c) => (
              <label
                key={c}
                className={`${styles.option} ${condition === c ? styles.optionChecked : ""}`}
              >
                <input
                  type="radio"
                  name="condition"
                  checked={condition === c}
                  onChange={() => setCondition(c)}
                />
                <span className={styles.optionCondition}>{CONDITION_LABELS[c]}</span>
                <span className={styles.optionDescription}>{conditionDescription(c)}</span>
              </label>
            ))}
          </div>
        </div>
        {error && <div className={styles.error}>{error}</div>}
        <button className={styles.submitButton} onClick={handleStart} disabled={loading || !timestamp}>
          {loading ? "…" : "Начать"}
        </button>
      </div>
      <button className={styles.backButton} onClick={onBack}>← Назад ко входу</button>
    </div>
  );
}
