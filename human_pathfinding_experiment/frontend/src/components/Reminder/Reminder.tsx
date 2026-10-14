import styles from "./Reminder.module.css";

/** Reference information displayed to the participant during the whole experiment. */
export function Reminder() {
  return (
    <div className={styles.container}>
      <span className={styles.title}>Памятка</span>
      <span>15 км — минимальное расстояние до границы грозы; 50 км — минимальное расстояние между грозами.</span>
      <span>→ — добавить слой прогноза; ← — убрать слой (если есть прогноз).</span>
      <span>↑ — увеличить масштаб; ↓ — уменьшить; Enter — подтвердить сегмент.</span>
    </div>
  );
}
