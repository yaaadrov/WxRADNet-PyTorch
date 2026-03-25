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
  timestamp: string | null;
  obstacleMode: ObstacleMode;
  windowSize: WindowSize;
  strategy: Strategy;
  aPoint: Point2D | null;
  bPoint: Point2D | null;
  currentPosition: Point2D | null;
  directionVector: Point2D | null;
  timeIndex: number;
  allPaths: Array<Array<Point2D>>;
  currentPath: Array<Point2D>;
  obstacles: ObstacleLayer[];
  availableTimeKeys: string[];
  pixelTransform: PixelTransform | null;
  isRunning: boolean;
  isComplete: boolean;
  startTime: Date | null;
  loadedTimeKeys: Set<string>;
  zoomLevel: number; // 1 = default, >1 = zoomed in
  pathValid: boolean | null; // null = not validated yet
  invalidSegments: number[];
  validationObstacles: ObstacleLayer[];
}

export type ExperimentAction =
  | { type: "SET_TIMESTAMP"; payload: string }
  | { type: "SET_OBSTACLE_MODE"; payload: ObstacleMode }
  | { type: "SET_WINDOW_SIZE"; payload: WindowSize }
  | { type: "SET_STRATEGY"; payload: Strategy }
  | { type: "SET_AB_POINTS"; payload: { aPoint: Point2D; bPoint: Point2D } }
  | { type: "SET_BBOX"; payload: { pixelTransform: PixelTransform; allTimeKeys: string[] } }
  | { type: "SET_OBSTACLES"; payload: { obstacles: ObstacleLayer[]; availableTimeKeys: string[] } }
  | { type: "ADD_WAYPOINT"; payload: Point2D }
  | { type: "CONFIRM_SEGMENT" }
  | { type: "MOVE_AIRCRAFT"; payload: { position: Point2D; direction: Point2D; timeIndex: number } }
  | { type: "COMPLETE_PATH" }
  | { type: "RESET" }
  | { type: "START_EXPERIMENT" }
  | { type: "ADD_TIME_KEY_LAYER"; payload: string }
  | { type: "REMOVE_TIME_KEY_LAYER" }
  | { type: "ZOOM_IN" }
  | { type: "ZOOM_OUT" }
  | { type: "SET_VALIDATION"; payload: { pathValid: boolean; invalidSegments: number[]; validationObstacles: ObstacleLayer[] } };