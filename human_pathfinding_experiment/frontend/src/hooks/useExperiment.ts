import { useReducer, useCallback, useEffect } from "react";
import { api } from "../services/api";
import {
  ExperimentState,
  ExperimentAction,
  Point2D,
  ObstacleMode,
  WindowSize,
  Strategy,
  ObstaclesResponse,
} from "../types";
import { SEGMENT_LENGTH_M, SNAP_DISTANCE_M } from "../utils/constants";
import { distanceMeters, getDirection } from "../utils/geometry";

const initialState: ExperimentState = {
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
  currentPath: [],
  obstacles: [],
  availableTimeKeys: [],
  pixelTransform: null,
  isRunning: false,
  isComplete: false,
  startTime: null,
  loadedTimeKeys: new Set(),
  zoomLevel: 1,
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

function experimentReducer(state: ExperimentState, action: ExperimentAction): ExperimentState {
  switch (action.type) {
    case "SET_TIMESTAMP":
      return { ...state, timestamp: action.payload };

    case "SET_OBSTACLE_MODE":
      return { ...state, obstacleMode: action.payload };

    case "SET_WINDOW_SIZE":
      return { ...state, windowSize: action.payload };

    case "SET_STRATEGY":
      return { ...state, strategy: action.payload };

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
        allPaths: [...state.allPaths, confirmedPath],
        currentPosition: newPositionAtB ? state.bPoint : point,
        directionVector: direction,
        timeIndex: state.timeIndex + 1,
        isComplete: !!newPositionAtB,
      };
    }

    case "MOVE_AIRCRAFT":
      return {
        ...state,
        currentPosition: action.payload.position,
        directionVector: action.payload.direction,
        timeIndex: action.payload.timeIndex,
      };

    case "COMPLETE_PATH":
      return {
        ...state,
        isComplete: true,
      };

    case "RESET":
      return {
        ...initialState,
        // Preserve configuration
        timestamp: state.timestamp,
        obstacleMode: state.obstacleMode,
        windowSize: state.windowSize,
        strategy: state.strategy,
        // Preserve loaded data for quick restart
        aPoint: state.aPoint,
        bPoint: state.bPoint,
        pixelTransform: state.pixelTransform,
        availableTimeKeys: state.availableTimeKeys,
      };

    case "START_EXPERIMENT":
      return {
        ...state,
        isRunning: true,
        startTime: new Date(),
        timeIndex: 0,
        allPaths: [],
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
        // Reset loaded time keys
        loadedTimeKeys: new Set(),
      };

    case "ADD_TIME_KEY_LAYER": {
      const newLoaded = new Set(state.loadedTimeKeys);
      newLoaded.add(action.payload);
      return { ...state, loadedTimeKeys: newLoaded };
    }

    case "REMOVE_TIME_KEY_LAYER": {
      const keys = Array.from(state.loadedTimeKeys);
      if (keys.length <= 1) return state;
      const newLoaded = new Set(keys.slice(0, -1));
      return { ...state, loadedTimeKeys: newLoaded };
    }

    case "ZOOM_IN": {
      const newZoom = Math.min(state.zoomLevel * 1.5, 10); // Max 10x zoom
      return { ...state, zoomLevel: newZoom };
    }

    case "ZOOM_OUT": {
      const newZoom = Math.max(state.zoomLevel / 1.5, 1); // Min 1x zoom
      return { ...state, zoomLevel: newZoom };
    }

    case "SET_VALIDATION":
      return {
        ...state,
        pathValid: action.payload.pathValid,
        invalidSegments: action.payload.invalidSegments,
        validationObstacles: action.payload.validationObstacles,
      };

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

  const setTimestamp = useCallback((timestamp: string) => {
    dispatch({ type: "SET_TIMESTAMP", payload: timestamp });
  }, []);

  const setObstacleMode = useCallback((mode: ObstacleMode) => {
    dispatch({ type: "SET_OBSTACLE_MODE", payload: mode });
  }, []);

  const setWindowSize = useCallback((size: WindowSize) => {
    dispatch({ type: "SET_WINDOW_SIZE", payload: size });
  }, []);

  const setStrategy = useCallback((strategy: Strategy) => {
    dispatch({ type: "SET_STRATEGY", payload: strategy });
  }, []);

  const startExperiment = useCallback(() => {
    dispatch({ type: "START_EXPERIMENT" });
  }, []);

  const addWaypoint = useCallback((point: Point2D) => {
    dispatch({ type: "ADD_WAYPOINT", payload: point });
  }, []);

  const confirmSegment = useCallback(() => {
    dispatch({ type: "CONFIRM_SEGMENT" });
  }, []);

  const reset = useCallback(() => {
    dispatch({ type: "RESET" });
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
    if (!state.timestamp || state.allPaths.length === 0) return;

    // Convert allPaths to the format expected by API
    const allPaths = state.allPaths.map(path =>
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
  }, [state.timestamp, state.allPaths, state.strategy]);

  // Validate path when experiment is complete
  useEffect(() => {
    if (state.isComplete && state.allPaths.length > 0 && state.pathValid === null) {
      validatePath();
    }
  }, [state.isComplete, state.allPaths, state.pathValid, validatePath]);

  const saveResults = useCallback(async () => {
    if (!state.timestamp || !state.startTime) return null;

    // Flatten all paths
    const allPaths = state.allPaths.map(path =>
      path.map(p => [p.x, p.y] as [number, number])
    );

    // Include current path if not empty
    if (state.currentPath.length > 0) {
      allPaths.push(state.currentPath.map(p => [p.x, p.y] as [number, number]));
    }

    const totalDistance = [...state.allPaths, state.currentPath].reduce((acc, path) => {
      return acc + path.reduce((pathAcc, _, i, arr) => {
        if (i === 0) return 0;
        return pathAcc + distanceMeters(arr[i - 1], arr[i]);
      }, 0);
    }, 0);

    const totalWaypoints = [...state.allPaths, state.currentPath].reduce((acc, path) => acc + path.length, 0);

    const request = {
      timestamp: state.timestamp,
      obstacle_mode: state.obstacleMode,
      window_size: state.windowSize,
      strategy: state.strategy,
      prediction_mode: "deterministic",
      all_paths: allPaths,
      path: allPaths, // Same as all_paths for now
      path_valid: state.pathValid ?? true, // Use validated value or true if not validated
      experiment_duration_seconds: (new Date().getTime() - state.startTime.getTime()) / 1000,
      timestamp_start: state.startTime.toISOString(),
      success: state.isComplete,
      total_waypoints: totalWaypoints,
      total_distance_m: totalDistance,
    };

    try {
      return await api.saveResults(request);
    } catch (error) {
      console.error("Failed to save results:", error);
      return null;
    }
  }, [state]);

  return {
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
    validatePath,
    saveResults,
    loadObstacles,
  };
}
