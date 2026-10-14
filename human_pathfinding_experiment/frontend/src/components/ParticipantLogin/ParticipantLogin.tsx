import { useState } from "react";
import styles from "./ParticipantLogin.module.css";

interface ParticipantLoginProps {
  onLogin: (surname: string, name: string, group: string) => Promise<void>;
  onDemo: () => Promise<void>;
}

export function ParticipantLogin({ onLogin, onDemo }: ParticipantLoginProps) {
  const [surname, setSurname] = useState("");
  const [name, setName] = useState("");
  const [group, setGroup] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [demoLoading, setDemoLoading] = useState(false);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await onLogin(surname.trim(), name.trim(), group.trim());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Не удалось войти");
    } finally {
      setLoading(false);
    }
  };

  const handleDemo = async () => {
    setError(null);
    setDemoLoading(true);
    try {
      await onDemo();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Не удалось запустить демо");
    } finally {
      setDemoLoading(false);
    }
  };

  return (
    <div className={styles.container}>
      <h2 className={styles.title}>Вход участника</h2>
      <p className={styles.subtitle}>
        Введите фамилию, имя и группу. Эксперимент можно пройти только один раз.
      </p>
      <form className={styles.form} onSubmit={handleSubmit}>
        <label className={styles.field}>
          <span className={styles.label}>Фамилия</span>
          <input
            className={styles.input}
            value={surname}
            onChange={(e) => setSurname(e.target.value)}
            required
            autoComplete="off"
          />
        </label>
        <label className={styles.field}>
          <span className={styles.label}>Имя</span>
          <input
            className={styles.input}
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
            autoComplete="off"
          />
        </label>
        <label className={styles.field}>
          <span className={styles.label}>Группа</span>
          <input
            className={styles.input}
            value={group}
            onChange={(e) => setGroup(e.target.value)}
            required
            autoComplete="off"
          />
        </label>
        {error && <div className={styles.error}>{error}</div>}
        <button className={styles.submitButton} type="submit" disabled={loading}>
          {loading ? "Вход…" : "Войти"}
        </button>
      </form>
      <button
        className={styles.demoButton}
        onClick={handleDemo}
        disabled={demoLoading}
        title="Демонстрационный режим: результаты не сохраняются"
      >
        {demoLoading ? "…" : "ДЕМО"}
      </button>
    </div>
  );
}
