// Velocity and time constants (from backend settings)
export const VELOCITY_KMH = 900;
export const DELTA_MINUTES = 5;

// Computed constants
export const VELOCITY_MPM = (VELOCITY_KMH * 1000) / 60; // meters per minute
export const SEGMENT_LENGTH_M = VELOCITY_MPM * DELTA_MINUTES; // ~75km per time step

// UI constants
export const SNAP_DISTANCE_M = 10_000; // 10 km snap to B point
export const MIN_OBSTACLE_DISTANCE_M = 15_000; // 15 km minimum to obstacle
export const MIN_BETWEEN_OBSTACLES_M = 50_000; // 50 km minimum between obstacles

// Canvas constants (vertical orientation)
export const CANVAS_WIDTH = 800;
export const CANVAS_HEIGHT = 800;
export const CANVAS_PADDING = 50;

// Radii for display circles (in km)
export const RADII_KM = [20, 40, 60, 80, 100];
