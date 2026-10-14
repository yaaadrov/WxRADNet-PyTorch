import { useState } from "react";
import { QuestionnaireAnswers } from "../../types";
import styles from "./Questionnaire.module.css";

interface QuestionnaireProps {
  onSubmit: (answers: QuestionnaireAnswers) => Promise<void>;
}

const EASE_QUESTIONS: Array<{ key: keyof QuestionnaireAnswers; text: string }> = [
  { key: "ease_rating_borders", text: "Насколько легко было обходить грозы, видя только границы очагов?" },
  { key: "ease_rating_zones", text: "Насколько легко было обходить грозы, видя границы очагов и запретные зоны вокруг них?" },
  { key: "ease_rating_forecast", text: "Насколько легко было обходить грозы, видя границы, запретные зоны и прогноз?" },
];

const AWARENESS_QUESTIONS: Array<{ key: keyof QuestionnaireAnswers; text: string }> = [
  { key: "awareness_zones_helps", text: "По вашему мнению, отображение запретных зон вокруг очагов грозы способствует повышению ситуационной осведомлённости пилота при обходе грозы?" },
  { key: "awareness_forecast_helps", text: "По вашему мнению, дополнительное отображение прогноза развития грозы способствует повышению ситуационной осведомлённости пилота при обходе грозы?" },
];

export function Questionnaire({ onSubmit }: QuestionnaireProps) {
  const [ease, setEase] = useState<Record<string, number>>({});
  const [awareness, setAwareness] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const ready = EASE_QUESTIONS.every((q) => ease[q.key] !== undefined)
    && AWARENESS_QUESTIONS.every((q) => awareness[q.key] !== undefined);

  const handleSubmit = async () => {
    if (!ready) return;
    setError(null);
    setLoading(true);
    try {
      await onSubmit({
        ease_rating_borders: ease.ease_rating_borders,
        ease_rating_zones: ease.ease_rating_zones,
        ease_rating_forecast: ease.ease_rating_forecast,
        awareness_zones_helps: awareness.awareness_zones_helps,
        awareness_forecast_helps: awareness.awareness_forecast_helps,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Не удалось отправить ответы");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.container}>
      <h2 className={styles.title}>Итоговый опрос</h2>
      <p className={styles.subtitle}>Оцените весь эксперимент в целом: все три попытки уже пройдены.</p>

      <div className={styles.section}>
        {EASE_QUESTIONS.map((question) => (
          <div key={question.key} className={styles.question}>
            <div className={styles.questionText}>{question.text}</div>
            <div className={styles.scaleRow}>
              {[1, 2, 3, 4, 5].map((value) => (
                <label key={value} className={styles.scaleOption}>
                  <input
                    type="radio"
                    name={question.key}
                    checked={ease[question.key] === value}
                    onChange={() => setEase((prev) => ({ ...prev, [question.key]: value }))}
                  />
                  <span>{value}</span>
                </label>
              ))}
            </div>
          </div>
        ))}
        <div className={styles.scaleLegend}>1 — нерешаемо, 5 — очень просто</div>
      </div>

      <div className={styles.section}>
        {AWARENESS_QUESTIONS.map((question) => (
          <div key={question.key} className={styles.question}>
            <div className={styles.questionText}>{question.text}</div>
            <div className={styles.scaleRow}>
              {[true, false].map((value) => (
                <label key={String(value)} className={styles.scaleOption}>
                  <input
                    type="radio"
                    name={question.key}
                    checked={awareness[question.key] === value}
                    onChange={() => setAwareness((prev) => ({ ...prev, [question.key]: value }))}
                  />
                  <span>{value ? "Да" : "Нет"}</span>
                </label>
              ))}
            </div>
          </div>
        ))}
      </div>

      {error && <div className={styles.error}>{error}</div>}
      <button className={styles.submitButton} onClick={handleSubmit} disabled={!ready || loading}>
        {loading ? "Отправка…" : "Отправить ответы"}
      </button>
    </div>
  );
}
