import { useState, useEffect } from "react";
import { api } from "../../services/api";
import styles from "./Results.module.css";

interface ResultsSummaryProps {
  onSave: () => Promise<{ filename: string; message: string } | null>;
  isComplete: boolean;
  onReset: () => void;
}

export function ResultsSummary({ onSave, isComplete, onReset }: ResultsSummaryProps) {
  const [savedFiles, setSavedFiles] = useState<string[]>([]);
  const [saving, setSaving] = useState(false);
  const [lastSaved, setLastSaved] = useState<string | null>(null);

  useEffect(() => {
    api.listResults()
      .then((data) => setSavedFiles(data.results))
      .catch(console.error);
  }, []);

  const handleSave = async () => {
    setSaving(true);
    try {
      const result = await onSave();
      if (result) {
        setLastSaved(result.filename);
        // Refresh list
        const data = await api.listResults();
        setSavedFiles(data.results);
      }
    } finally {
      setSaving(false);
    }
  };

  const handleDownload = (filename: string) => {
    window.open(api.getResultsUrl(filename), "_blank");
  };

  return (
    <div className={styles.container}>
      <h3 className={styles.title}>Results</h3>

      {isComplete && (
        <div className={styles.actions}>
          <button
            className={styles.saveButton}
            onClick={handleSave}
            disabled={saving}
          >
            {saving ? "Saving..." : "Save Results"}
          </button>
          <button className={styles.resetButton} onClick={onReset}>
            New Experiment
          </button>
        </div>
      )}

      {lastSaved && (
        <div className={styles.savedMessage}>
          Saved: {lastSaved}
        </div>
      )}

      <div className={styles.fileList}>
        <h4 className={styles.subtitle}>Saved Files</h4>
        {savedFiles.length === 0 ? (
          <p className={styles.empty}>No saved results yet</p>
        ) : (
          <ul className={styles.list}>
            {savedFiles.map((file) => (
              <li key={file} className={styles.listItem}>
                <span className={styles.filename}>{file}</span>
                <button
                  className={styles.downloadButton}
                  onClick={() => handleDownload(file)}
                >
                  Download
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
