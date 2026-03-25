import { ExperimentSetup } from "./components/ExperimentSetup";
import { PathfindingCanvas } from "./components/Canvas";
import { StatusBar } from "./components/Controls";
import { ResultsSummary } from "./components/Results";
import { useExperiment } from "./hooks/useExperiment";
import { useKeyboardControls } from "./hooks/useKeyboardControls";
import "./App.css";

function App() {
  const {
    state,
    setTimestamp,
    setObstacleMode,
    setWindowSize,
    setStrategy,
    startExperiment,
    addWaypoint,
    confirmSegment,
    reset,
    addTimeKeyLayer,
    removeTimeKeyLayer,
    zoomIn,
    zoomOut,
    saveResults,
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
    onSetWindowSize: setWindowSize,
    onConfirmSegment: confirmSegment,
    onZoomIn: zoomIn,
    onZoomOut: zoomOut,
  });

  return (
    <div className="app">
      <header className="header">
        <h1>Human Pathfinding Experiment</h1>
      </header>

      <main className="main">
        <ExperimentSetup
          timestamp={state.timestamp}
          obstacleMode={state.obstacleMode}
          windowSize={state.windowSize}
          strategy={state.strategy}
          isRunning={state.isRunning}
          onTimestampChange={setTimestamp}
          onObstacleModeChange={setObstacleMode}
          onWindowSizeChange={setWindowSize}
          onStrategyChange={setStrategy}
          onStart={startExperiment}
        />

        <PathfindingCanvas
          aPoint={state.aPoint}
          bPoint={state.bPoint}
          currentPosition={state.currentPosition}
          directionVector={state.directionVector}
          obstacles={state.obstacles}
          loadedTimeKeys={state.loadedTimeKeys}
          allPaths={state.allPaths}
          currentPath={state.currentPath}
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
          startTime={state.startTime}
          availableTimeKeys={state.availableTimeKeys}
          loadedTimeKeys={state.loadedTimeKeys}
          currentPath={state.currentPath}
          onConfirmSegment={confirmSegment}
          zoomLevel={state.zoomLevel}
          pathValid={state.pathValid}
          invalidSegments={state.invalidSegments}
        />

        <ResultsSummary
          onSave={saveResults}
          isComplete={state.isComplete}
          onReset={reset}
        />
      </main>
    </div>
  );
}

export default App;
