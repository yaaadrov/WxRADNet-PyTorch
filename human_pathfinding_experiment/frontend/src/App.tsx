import { useCallback, useState } from "react";
import { ExperimentSetup } from "./components/ExperimentSetup";
import { ParticipantLogin } from "./components/ParticipantLogin";
import { PathfindingCanvas } from "./components/Canvas";
import { Questionnaire } from "./components/Questionnaire";
import { Reminder } from "./components/Reminder";
import { StatusBar } from "./components/Controls";
import { ResultsSummary } from "./components/Results";
import { useExperiment } from "./hooks/useExperiment";
import { useKeyboardControls } from "./hooks/useKeyboardControls";
import "./App.css";

type Screen = "login" | "brief" | "experiment" | "questionnaire" | "end";

function App() {
  const [screen, setScreen] = useState<Screen>("login");
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const {
    state,
    login,
    startDemo,
    nextDemoScenario,
    loadNextTrial,
    startTrial,
    submitTrial,
    submitQuestionnaire,
    clearSession,
    addWaypoint,
    confirmSegment,
    addTimeKeyLayer,
    removeTimeKeyLayer,
    zoomIn,
    zoomOut,
  } = useExperiment();

  // Set up keyboard controls
  useKeyboardControls({
    isRunning: state.isRunning,
    isComplete: state.isComplete,
    windowSize: state.windowSize,
    availableTimeKeys: state.availableTimeKeys,
    loadedTimeKeys: state.loadedTimeKeys,
    onAddTimeKeyLayer: addTimeKeyLayer,
    onRemoveTimeKeyLayer: removeTimeKeyLayer,
    onConfirmSegment: confirmSegment,
    onZoomIn: zoomIn,
    onZoomOut: zoomOut,
  });

  const handleLogin = useCallback(async (surname: string, name: string, group: string) => {
    await login(surname, name, group);
    setScreen("brief");
  }, [login]);

  const handleDemo = useCallback(async () => {
    await startDemo();
    setScreen("brief");
  }, [startDemo]);

  const handleStart = useCallback(async () => {
    await startTrial();
    setScreen("experiment");
  }, [startTrial]);

  const handleNext = useCallback(async () => {
    if (state.isDemo) {
      // Demo session: nothing is submitted, just move to the next demo scenario
      if (nextDemoScenario()) {
        setScreen("brief");
      } else {
        clearSession();
        setScreen("login");
      }
      return;
    }
    setSubmitting(true);
    setSubmitError(null);
    try {
      const result = await submitTrial();
      if (result === null) return;
      if (result.session_finished) {
        setScreen("questionnaire");
      } else {
        await loadNextTrial();
        setScreen("brief");
      }
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Не удалось сохранить результат");
    } finally {
      setSubmitting(false);
    }
  }, [state.isDemo, submitTrial, loadNextTrial, nextDemoScenario, clearSession]);

  const handleQuestionnaire = useCallback(async (answers: Parameters<typeof submitQuestionnaire>[0]) => {
    await submitQuestionnaire(answers);
    setScreen("end");
  }, [submitQuestionnaire]);

  const handleFinish = useCallback(() => {
    clearSession();
    setScreen("login");
  }, [clearSession]);

  return (
    <div className="app">
      <header className="header">
        <h1>Эксперимент по обходу грозы</h1>
      </header>

      <main className="main">
        {screen === "login" && <ParticipantLogin onLogin={handleLogin} onDemo={handleDemo} />}

        {screen === "brief" && (
          <>
            <ExperimentSetup
              trialIndex={state.trialIndex}
              trialsTotal={state.trialsTotal}
              condition={state.condition}
              timestamp={state.timestamp}
              isDemo={state.isDemo}
              onStart={handleStart}
            />
            <Reminder />
          </>
        )}

        {screen === "experiment" && (
          <>
            <Reminder />

            <PathfindingCanvas
              aPoint={state.aPoint}
              bPoint={state.bPoint}
              currentPosition={state.currentPosition}
              directionVector={state.directionVector}
              obstacles={state.obstacles}
              loadedTimeKeys={state.loadedTimeKeys}
              routeSegments={state.routeSegments}
              currentPath={state.currentPath}
              visibleArea={state.visibleArea}
              pixelTransform={state.pixelTransform}
              onWaypointAdd={addWaypoint}
              obstacleMode={state.obstacleMode}
              zoomLevel={state.zoomLevel}
              isComplete={state.isComplete}
              invalidSegments={state.invalidSegments}
              validationObstacles={state.validationObstacles}
            />

            <StatusBar
              isRunning={state.isRunning}
              isComplete={state.isComplete}
              timeIndex={state.timeIndex}
              currentPosition={state.currentPosition}
              bPoint={state.bPoint}
              availableTimeKeys={state.availableTimeKeys}
              loadedTimeKeys={state.loadedTimeKeys}
              currentPath={state.currentPath}
              onConfirmSegment={confirmSegment}
              zoomLevel={state.zoomLevel}
              pathValid={state.pathValid}
              invalidSegments={state.invalidSegments}
            />

            <ResultsSummary
              isComplete={state.isComplete}
              pathValid={state.pathValid}
              invalidSegments={state.invalidSegments}
              trialIndex={state.trialIndex}
              trialsTotal={state.trialsTotal}
              isDemo={state.isDemo}
              submitting={submitting}
              error={submitError}
              onNext={handleNext}
            />
          </>
        )}

        {screen === "questionnaire" && (
          <Questionnaire onSubmit={handleQuestionnaire} />
        )}

        {screen === "end" && (
          <div className="endScreen">
            <h2>Спасибо за участие!</h2>
            <p>Ваши ответы сохранены.</p>
            <button className="endButton" onClick={handleFinish}>Завершить</button>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
