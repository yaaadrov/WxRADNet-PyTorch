import {
  TimestampsResponse,
  ABPointsResponse,
  ObstaclesResponse,
  ResultsRequest,
  BBoxResponse,
  ValidationResponse,
} from "../types";

const API_BASE = "/api";

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

export const api = {
  async getTimestamps(): Promise<TimestampsResponse> {
    return fetchJson<TimestampsResponse>(`${API_BASE}/timestamps`);
  },

  async getABPoints(timestamp: string): Promise<ABPointsResponse> {
    return fetchJson<ABPointsResponse>(`${API_BASE}/ab-points?timestamp=${encodeURIComponent(timestamp)}`);
  },

  async getBBox(request: {
    timestamp: string;
    strategy: "concave" | "convex";
    mode: "obstacles" | "hull";
  }): Promise<BBoxResponse> {
    return fetchJson<BBoxResponse>(`${API_BASE}/obstacles/bbox`, {
      method: "POST",
      body: JSON.stringify(request),
    });
  },

  async getObstacles(request: {
    timestamp: string;
    time_index: number;
    strategy: "concave" | "convex";
    mode: "obstacles" | "hull";
    current_position: [number, number];
    direction_vector: [number, number];
    window_size: 1 | 7;
  }): Promise<ObstaclesResponse> {
    return fetchJson<ObstaclesResponse>(`${API_BASE}/obstacles`, {
      method: "POST",
      body: JSON.stringify(request),
    });
  },

  async validatePath(request: {
    timestamp: string;
    strategy: "concave" | "convex";
    all_paths: Array<Array<[number, number]>>;
  }): Promise<ValidationResponse> {
    return fetchJson<ValidationResponse>(`${API_BASE}/obstacles/validate`, {
      method: "POST",
      body: JSON.stringify(request),
    });
  },

  async saveResults(request: ResultsRequest): Promise<{ filename: string; message: string }> {
    return fetchJson(`${API_BASE}/results`, {
      method: "POST",
      body: JSON.stringify(request),
    });
  },

  async listResults(): Promise<{ results: string[] }> {
    return fetchJson(`${API_BASE}/results`);
  },

  getResultsUrl(filename: string): string {
    return `${API_BASE}/results/${filename}`;
  },
};
