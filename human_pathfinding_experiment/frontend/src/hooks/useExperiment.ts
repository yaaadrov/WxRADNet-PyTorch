import { useReducer, useCallback, useEffect } from "react";
import { api, setAuthToken } from "../services/api";
import {
  ExperimentState,
  ExperimentAction,
  Point2D,
  ObstaclesResponse,
  QuestionnaireAnswers,
} from "../types";
import { SEGMENT_LENGTH_M, SNAP_DISTANCE_M } from "../utils/constants";
import { distanceMeters, getDirection } from "../utils/geometry";

const initialState: ExperimentState = {
  trialId: null,
  trialIndex: 0,
  trialsTotal: 3,
  condition: null,
  isDemo: false,
  timestamp: null,
  obstacleMode: "obstacles",
  windowSize: 1,
  strategy: "concave",
  aPoint: null,
  bPoint: null,
  currentPosition: null,
  directionVector: null,
  timeIndex: 0,
  allPaths: [],
  routeSegments: [],
  currentPath: [],
  obstacles: [],
  availableTimeKeys: [],
  pixelTransform: null,
  visibleArea: null,
  isRunning: false,
  isComplete: false,
  startTime: null,
  loadedTimeKeys: new Set(),
  zoomLevel: 1,
  numZoomActions: 0,
  forecastLayersViewed: 0,
  pathValid: null,
  invalidSegments: [],
  validationObstacles: [],
};

/**
 * Find the point at exactly targetDistance along a path.
 * Returns the point and the remaining path after that point.
 */
function findPointAtDistance(
  path: Point2D[],
  targetDistance: number
): { point: Point2D; remainingPath: Point2D[]; exactDistance: number } | null {
  if (path.length < 2) return null;

  let accumulatedDistance = 0;

  for (let i = 1; i < path.length; i++) {
    const segmentDistance = distanceMeters(path[i - 1], path[i]);
    const distanceAfterSegment = accumulatedDistance + segmentDistance;

    if (distanceAfterSegment >= targetDistance) {
      // The target point is on this segment
      const remainingInSegment = targetDistance - accumulatedDistance;
      const ratio = remainingInSegment / segmentDistance;

      // Interpolate the point
      const point: Point2D = {
        x: path[i - 1].x + (path[i].x - path[i - 1].x) * ratio,
        y: path[i - 1].y + (path[i].y - path[i - 1].y) * ratio,
      };

      // Remaining path: interpolated point + rest of path
      const remainingPath = [point, ...path.slice(i)];

      return { point, remainingPath, exactDistance: targetDistance };
    }

    accumulatedDistance = distanceAfterSegment;
  }

  // Target distance is beyond the path
  return null;
}

/** Fixed demo scenarios shown by the secret ДЕМО button (nothing is stored in the database).
 *  Order: with forbidden zones -> without anything -> with forecast (same storm as the first one). */
const DEMO_SCENARIOS = [
  { condition: "C1" as const, obstacleMode: "hull" as const, windowSize: 1 as const, timestamp: "2023_08_05_13_10_00" },
  { condition: "C0" as const, obstacleMode: "obstacles" as const, windowSize: 1 as const, timestamp: "2023_09_20_13_00_00" },
  { condition: "C2" as const, obstacleMode: "hull" as const, windowSize: 7 as const, timestamp: "2023_08_05_13_10_00" },
];

function experimentReducer(state: ExperimentState, action: ExperimentAction): ExperimentState {
  switch (action.type) {
    case "SET_TRIAL":
      // A new trial starts from a completely clean slate (no memory of previous trials)
      return {
        ...initialState,
        trialId: action.payload.trialId,
        trialIndex: action.payload.trialIndex,
        trialsTotal: action.payload.trialsTotal,
        condition: action.payload.condition,
        isDemo: action.payload.isDemo,
        timestamp: action.payload.timestamp,
        obstacleMode: action.payload.obstacleMode,
        windowSize: action.payload.windowSize,
      };

    case "SET_AB_POINTS":
      return {
        ...state,
        aPoint: action.payload.aPoint,
        bPoint: action.payload.bPoint,
        currentPosition: action.payload.aPoint,
        directionVector: getDirection(action.payload.aPoint, action.payload.bPoint),
      };

    case "SET_BBOX":
      return {
        ...state,
        pixelTransform: action.payload.pixelTransform,
        availableTimeKeys: action.payload.allTimeKeys,
      };

    case "SET_OBSTACLES":
      return {
        ...state,
        obstacles: action.payload.obstacles,
        availableTimeKeys: action.payload.availableTimeKeys,
        visibleArea: action.payload.visibleArea,
        loadedTimeKeys: new Set([action.payload.availableTimeKeys[0]]),
      };

    case "ADD_WAYPOINT": {
      if (!state.currentPosition) return state;

      const newWaypoint = action.payload;

      // Check if reached B point - snap to B
      const finalWaypoint = state.bPoint && distanceMeters(newWaypoint, state.bPoint) <= SNAP_DISTANCE_M
        ? state.bPoint
        : newWaypoint;

      // Add waypoint to current path (don't complete yet - user must confirm segment)
      return {
        ...state,
        currentPath: [...state.currentPath, finalWaypoint],
      };
    }

    case "CONFIRM_SEGMENT": {
      if (!state.currentPosition) return state;
      if (state.currentPath.length < 1) return state;

      // Full path includes current position as starting point
      const fullPath = [state.currentPosition, ...state.currentPath];

      // Calculate total path length from current position through all waypoints
      const totalPathLength = fullPath.reduce((acc, _, i, arr) => {
        if (i === 0) return 0;
        return acc + distanceMeters(arr[i - 1], arr[i]);
      }, 0);

      // Check if last waypoint is B (experiment completion condition)
      const lastWaypoint = fullPath[fullPath.length - 1];
      const isAtB = state.bPoint && distanceMeters(lastWaypoint, state.bPoint) <= SNAP_DISTANCE_M;

      // If path reaches B and is shorter than segment length, complete the experiment
      if (isAtB && totalPathLength < SEGMENT_LENGTH_M) {
        return {
          ...state,
          allPaths: [...state.allPaths, fullPath],
          routeSegments: [...state.routeSegments, fullPath],
          currentPath: [],
          currentPosition: state.bPoint,
          isComplete: true,
        };
      }

      // If path is shorter than segment length (but not at B), don't advance
      if (totalPathLength < SEGMENT_LENGTH_M) return state;

      // Find point at exactly SEGMENT_LENGTH_M
      const result = findPointAtDistance(fullPath, SEGMENT_LENGTH_M);
      if (!result) return state;

      const { point } = result;

      // Build the confirmed segment (fullPath up to and including the 75km point)
      const confirmedPath: Point2D[] = [];
      let accumulatedDist = 0;
      for (let i = 1; i < fullPath.length; i++) {
        const segmentDist = distanceMeters(fullPath[i - 1], fullPath[i]);
        if (accumulatedDist + segmentDist >= SEGMENT_LENGTH_M) {
          // This segment contains the 75km point
          confirmedPath.push(point);
          break;
        }
        confirmedPath.push(fullPath[i]);
        accumulatedDist += segmentDist;
      }
      // Prepend current position if not already there
      if (confirmedPath[0] !== state.currentPosition) {
        confirmedPath.unshift(state.currentPosition);
      }

      // Get direction from last two points of confirmed segment
      const direction = confirmedPath.length >= 2
        ? getDirection(confirmedPath[confirmedPath.length - 2], confirmedPath[confirmedPath.length - 1])
        : state.directionVector || { x: 0, y: 0 };

      // Check if new position is at B
      const newPositionAtB = state.bPoint && distanceMeters(point, state.bPoint) <= SNAP_DISTANCE_M;

      return {
        ...state,
        currentPath: [], // Clear path - user draws fresh from new position
        allPaths: [...state.allPaths, fullPath], // Raw stage drawing (includes overshoot beyond the cut)
        routeSegments: [...state.routeSegments, confirmedPath], // Exact 75 km piece of the flown route
        currentPosition: newPositionAtB ? state.bPoint : point,
        directionVector: direction,
        timeIndex: state.timeIndex + 1,
        isComplete: !!newPositionAtB,
      };
    }

    case "START_EXPERIMENT":
      return {
        ...state,
        isRunning: true,
        startTime: new Date(),
        timeIndex: 0,
        allPaths: [],
        routeSegments: [],
        currentPath: [],
        isComplete: false,
        currentPosition: state.aPoint,
        directionVector: state.aPoint && state.bPoint
          ? getDirection(state.aPoint, state.bPoint)
          : null,
        // Reset validation state
        pathValid: null,
        invalidSegments: [],
        validationObstacles: [],
        // Reset loaded time keys and counters
        loadedTimeKeys: new Set(),
        numZoomActions: 0,
        forecastLayersViewed: 0,
      };

    case "ADD_TIME_KEY_LAYER": {
      const newLoaded = new Set(state.loadedTimeKeys);
      newLoaded.add(action.payload);
      return { ...state, loadedTimeKeys: newLoaded, forecastLayersViewed: state.forecastLayersViewed + 1 };
    }

    case "REMOVE_TIME_KEY_LAYER": {
      const keys = Array.from(state.loadedTimeKeys);
      if (keys.length <= 1) return state;
      const newLoaded = new Set(keys.slice(0, -1));
      return { ...state, loadedTimeKeys: newLoaded };
    }

    case "ZOOM_IN": {
      const newZoom = Math.min(state.zoomLevel * 1.5, 10); // Max 10x zoom
      return { ...state, zoomLevel: newZoom, numZoomActions: state.numZoomActions + 1 };
    }

    case "ZOOM_OUT": {
      const newZoom = Math.max(state.zoomLevel / 1.5, 1); // Min 1x zoom
      return { ...state, zoomLevel: newZoom, numZoomActions: state.numZoomActions + 1 };
    }

    case "SET_VALIDATION":
      return {
        ...state,
        pathValid: action.payload.pathValid,
        invalidSegments: action.payload.invalidSegments,
        validationObstacles: action.payload.validationObstacles,
      };

    case "RESET_SESSION":
      return initialState;

    default:
      return state;
  }
}

export function useExperiment() {
  const [state, dispatch] = useReducer(experimentReducer, initialState);

  // Load AB points when timestamp changes
  useEffect(() => {
    if (state.timestamp) {
      api.getABPoints(state.timestamp)
        .then((data) => {
          dispatch({
            type: "SET_AB_POINTS",
            payload: {
              aPoint: { x: data.a_point[0], y: data.a_point[1] },
              bPoint: { x: data.b_point[0], y: data.b_point[1] },
            },
          });
        })
        .catch(console.error);
    }
  }, [state.timestamp]);

  // Load bbox when experiment starts (pre-calculate for all time_keys)
  useEffect(() => {
    if (state.isRunning && state.timestamp && state.aPoint && state.bPoint && !state.pixelTransform) {
      api.getBBox({
        timestamp: state.timestamp,
        strategy: state.strategy,
        mode: state.obstacleMode,
      })
        .then((data) => {
          dispatch({
            type: "SET_BBOX",
            payload: {
              pixelTransform: data.pixel_transform,
              allTimeKeys: data.all_time_keys,
            },
          });
        })
        .catch(console.error);
    }
  }, [state.isRunning, state.timestamp, state.aPoint, state.bPoint, state.strategy, state.obstacleMode, state.pixelTransform]);

  // Load obstacles when position/time changes
  const loadObstacles = useCallback(async () => {
    if (!state.timestamp || !state.currentPosition || !state.directionVector) return;

    try {
      const data: ObstaclesResponse = await api.getObstacles({
        timestamp: state.timestamp,
        time_index: state.timeIndex,
        strategy: state.strategy,
        mode: state.obstacleMode,
        current_position: [state.currentPosition.x, state.currentPosition.y],
        direction_vector: [state.directionVector.x, state.directionVector.y],
        window_size: state.windowSize,
      });

      dispatch({
        type: "SET_OBSTACLES",
        payload: {
          obstacles: data.obstacles,
          availableTimeKeys: data.available_time_keys,
          visibleArea: data.visible_area.map(([x, y]) => ({ x, y })),
        },
      });
    } catch (error) {
      console.error("Failed to load obstacles:", error);
    }
  }, [state.timestamp, state.currentPosition, state.directionVector, state.timeIndex, state.strategy, state.obstacleMode, state.windowSize]);

  // Load obstacles when experiment starts or time advances
  useEffect(() => {
    if (state.isRunning && !state.isComplete && state.pixelTransform) {
      loadObstacles();
    }
  }, [state.isRunning, state.isComplete, state.timeIndex, state.pixelTransform, loadObstacles]);

  // ------------------------------------------------------------ participant session

  const loadNextTrial = useCallback(async () => {
    const trial = await api.getNextTrial();
    dispatch({
      type: "SET_TRIAL",
      payload: {
        trialId: trial.trial_id,
        trialIndex: trial.trial_index,
        trialsTotal: trial.trials_total,
        condition: trial.condition,
        timestamp: trial.timestamp,
        obstacleMode: trial.obstacle_mode,
        windowSize: trial.window_size,
        isDemo: false,
      },
    });
  }, []);

  const login = useCallback(async (surname: string, name: string, group: string) => {
    const data = await api.login(surname, name, group);
    setAuthToken(data.token);
    await loadNextTrial();
  }, [loadNextTrial]);

  const startDemo = useCallback(async () => {
    const data = await api.startDemoSession();
    setAuthToken(data.token);
    const scenario = DEMO_SCENARIOS[0];
    dispatch({ type: "SET_TRIAL", payload: { ...scenario, trialId: null, trialIndex: 1, trialsTotal: DEMO_SCENARIOS.length, isDemo: true } });
  }, []);

  const nextDemoScenario = useCallback(() => {
    const index = state.trialIndex; // 1-based within the demo
    if (index >= DEMO_SCENARIOS.length) return false;
    const scenario = DEMO_SCENARIOS[index];
    dispatch({ type: "SET_TRIAL", payload: { ...scenario, trialId: null, trialIndex: index + 1, trialsTotal: DEMO_SCENARIOS.length, isDemo: true } });
    return true;
  }, [state.trialIndex]);

  const startTrial = useCallback(async () => {
    if (state.isDemo) {
      // Demo session: no server-side trial, just start locally
      dispatch({ type: "START_EXPERIMENT" });
      return;
    }
    if (state.trialId === null) return;
    await api.startTrial(state.trialId);
    dispatch({ type: "START_EXPERIMENT" });
  }, [state.trialId, state.isDemo]);

  const submitTrial = useCallback(async () => {
    if (state.trialId === null) return null;

    // Raw per-stage drawings: confirmed stage paths plus the in-progress stage (if any)
    const stagePaths = state.allPaths.map(path =>
      path.map(p => [p.x, p.y] as [number, number])
    );
    if (state.currentPath.length > 0) {
      stagePaths.push(state.currentPath.map(p => [p.x, p.y] as [number, number]));
    }

    // Final flown route split into exact 5-minute intervals
    const routeSegments = state.routeSegments.map(path =>
      path.map(p => [p.x, p.y] as [number, number])
    );

    return api.submitTrial(state.trialId, {
      all_paths: stagePaths,
      path: routeSegments,
      num_zoom_actions: state.numZoomActions,
      num_forecast_layers_viewed: state.forecastLayersViewed,
    });
  }, [state]);

  const submitQuestionnaire = useCallback(async (answers: QuestionnaireAnswers) => {
    await api.submitQuestionnaire(answers);
  }, []);

  const clearSession = useCallback(() => {
    setAuthToken(null);
    dispatch({ type: "RESET_SESSION" });
  }, []);

  // ---------------------------------------------------------------- drawing actions

  const addWaypoint = useCallback((point: Point2D) => {
    dispatch({ type: "ADD_WAYPOINT", payload: point });
  }, []);

  const confirmSegment = useCallback(() => {
    dispatch({ type: "CONFIRM_SEGMENT" });
  }, []);

  const addTimeKeyLayer = useCallback((timeKey: string) => {
    dispatch({ type: "ADD_TIME_KEY_LAYER", payload: timeKey });
  }, []);

  const removeTimeKeyLayer = useCallback(() => {
    dispatch({ type: "REMOVE_TIME_KEY_LAYER" });
  }, []);

  const zoomIn = useCallback(() => {
    dispatch({ type: "ZOOM_IN" });
  }, []);

  const zoomOut = useCallback(() => {
    dispatch({ type: "ZOOM_OUT" });
  }, []);

  const validatePath = useCallback(async () => {
    if (!state.timestamp || state.routeSegments.length === 0) return;

    // Validate the final flown route segments (not the raw stage drawings)
    const allPaths = state.routeSegments.map(path =>
      path.map(p => [p.x, p.y] as [number, number])
    );

    try {
      const data = await api.validatePath({
        timestamp: state.timestamp,
        strategy: state.strategy,
        all_paths: allPaths,
      });

      dispatch({
        type: "SET_VALIDATION",
        payload: {
          pathValid: data.is_valid,
          invalidSegments: data.invalid_segments,
          validationObstacles: data.validation_obstacles,
        },
      });
    } catch (error) {
      console.error("Failed to validate path:", error);
    }
  }, [state.timestamp, state.routeSegments, state.strategy]);

  // Validate path when experiment is complete
  useEffect(() => {
    if (state.isComplete && state.routeSegments.length > 0 && state.pathValid === null) {
      validatePath();
    }
  }, [state.isComplete, state.routeSegments, state.pathValid, validatePath]);

  return {
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
    loadObstacles,
  };
}
