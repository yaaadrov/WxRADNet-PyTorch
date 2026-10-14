export type ObstacleMode = "obstacles" | "hull";
export type WindowSize = 1 | 7;
export type Strategy = "concave" | "convex";

export interface Point2D {
  x: number;
  y: number;
}

export interface PixelTransform {
  scale: number;
  offset_x: number;
  offset_y: number;
  bounds: {
    minX: number;
    maxX: number;
    minY: number;
    maxY: number;
  };
}

// Simple GeoJSON types
export interface GeoJSONPolygon {
  type: "Polygon";
  coordinates: number[][][];
}

export interface GeoJSONMultiPolygon {
  type: "MultiPolygon";
  coordinates: number[][][][];
}

export interface GeoJSONFeature {
  type: "Feature";
  geometry: GeoJSONPolygon | GeoJSONMultiPolygon;
  properties?: Record<string, unknown>;
}

export interface GeoJSONFeatureCollection {
  type: "FeatureCollection";
  features: GeoJSONFeature[];
}

export interface ObstacleLayer {
  time_key: string;
  geojson: GeoJSONFeatureCollection;
  color_index: number;
}

export interface ObstaclesResponse {
  obstacles: ObstacleLayer[];
  pixel_transform: PixelTransform;
  available_time_keys: string[];
  current_time_index: number;
  visible_area: Array<[number, number]>;
}

export interface ABPointsResponse {
  a_point: [number, number];
  b_point: [number, number];
}

export interface TimestampsResponse {
  timestamps: string[];
}

export interface BBoxResponse {
  pixel_transform: PixelTransform;
  all_time_keys: string[];
}

export interface ValidationResult {
  is_valid: boolean;
  segment_index: number;
}

export interface ValidationResponse {
  is_valid: boolean;
  segments: ValidationResult[];
  invalid_segments: number[];
  validation_obstacles: ObstacleLayer[];
}

export interface ResultsRequest {
  timestamp: string;
  obstacle_mode: ObstacleMode;
  window_size: WindowSize;
  strategy: Strategy;
  prediction_mode: string;
  all_paths: Array<Array<[number, number]>>;
  path: Array<Array<[number, number]>>;
  path_valid: boolean;
  experiment_duration_seconds: number;
  timestamp_start: string;
  success: boolean;
  total_waypoints: number;
  total_distance_m: number;
}

export interface ExperimentState {
  /** Assigned trial (null until login and trial loading; stays null in demo mode). */
  trialId: number | null;
  trialIndex: number;
  trialsTotal: number;
  condition: "C0" | "C1" | "C2" | null;
  /** Demo session (secret button on the login screen): nothing is stored in the database. */
  isDemo: boolean;
  timestamp: string | null;
  obstacleMode: ObstacleMode;
  windowSize: WindowSize;
  strategy: Strategy;
  aPoint: Point2D | null;
  bPoint: Point2D | null;
  currentPosition: Point2D | null;
  directionVector: Point2D | null;
  timeIndex: number;
  /** Raw per-stage drawings: everything the participant drew during each 5-minute stage,
   *  starting fresh from the stage's start position (includes overshoot beyond the 75 km cut). */
  allPaths: Array<Array<Point2D>>;
  /** Final flown route split into exact 5-minute (75 km) intervals; the last one is shorter (reaches B). */
  routeSegments: Array<Array<Point2D>>;
  currentPath: Array<Point2D>;
  obstacles: ObstacleLayer[];
  availableTimeKeys: string[];
  pixelTransform: PixelTransform | null;
  visibleArea: Array<Point2D> | null;
  isRunning: boolean;
  isComplete: boolean;
  startTime: Date | null;
  loadedTimeKeys: Set<string>;
  zoomLevel: number; // 1 = default, >1 = zoomed in
  /** Number of ZOOM_IN/ZOOM_OUT actions during the trial (secondary metric). */
  numZoomActions: number;
  /** Number of extra forecast layers viewed beyond the first one (secondary metric, C2 only). */
  forecastLayersViewed: number;
  pathValid: boolean | null; // null = not validated yet
  invalidSegments: number[];
  validationObstacles: ObstacleLayer[];
}

export type ExperimentAction =
  | { type: "SET_TRIAL"; payload: { trialId: number | null; trialIndex: number; trialsTotal: number; condition: "C0" | "C1" | "C2"; timestamp: string; obstacleMode: ObstacleMode; windowSize: WindowSize; isDemo: boolean } }
  | { type: "SET_AB_POINTS"; payload: { aPoint: Point2D; bPoint: Point2D } }
  | { type: "SET_BBOX"; payload: { pixelTransform: PixelTransform; allTimeKeys: string[] } }
  | { type: "SET_OBSTACLES"; payload: { obstacles: ObstacleLayer[]; availableTimeKeys: string[]; visibleArea: Array<Point2D> } }
  | { type: "ADD_WAYPOINT"; payload: Point2D }
  | { type: "CONFIRM_SEGMENT" }
  | { type: "START_EXPERIMENT" }
  | { type: "ADD_TIME_KEY_LAYER"; payload: string }
  | { type: "REMOVE_TIME_KEY_LAYER" }
  | { type: "ZOOM_IN" }
  | { type: "ZOOM_OUT" }
  | { type: "SET_VALIDATION"; payload: { pathValid: boolean; invalidSegments: number[]; validationObstacles: ObstacleLayer[] } }
  | { type: "RESET_SESSION" };

// ---------------------------------------------------------------- Participant session API

export interface LoginResponse {
  token: string;
  participant_id: number;
  is_completed: boolean;
}

export interface NextTrialResponse {
  trial_id: number;
  trial_index: number;
  trials_total: number;
  condition: "C0" | "C1" | "C2";
  obstacle_mode: ObstacleMode;
  window_size: WindowSize;
  timestamp: string;
}

export interface SubmitTrialResponse {
  trial_id: number;
  trials_submitted: number;
  session_finished: boolean;
}

export interface QuestionnaireAnswers {
  ease_rating_borders: number;
  ease_rating_zones: number;
  ease_rating_forecast: number;
  awareness_zones_helps: boolean;
  awareness_forecast_helps: boolean;
}