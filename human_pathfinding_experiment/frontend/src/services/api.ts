import {
  ABPointsResponse,
  BBoxResponse,
  LoginResponse,
  NextTrialResponse,
  ObstaclesResponse,
  QuestionnaireAnswers,
  SubmitTrialResponse,
  ValidationResponse,
} from "../types";

const API_BASE = "/api";

// Session token is kept in memory only (a page reload requires a fresh login by design)
let authToken: string | null = null;

export function setAuthToken(token: string | null) {
  authToken = token;
}

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
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
  async login(surname: string, name: string, group: string): Promise<LoginResponse> {
    return fetchJson<LoginResponse>(`${API_BASE}/participants/login`, {
      method: "POST",
      body: JSON.stringify({ surname, name, group }),
    });
  },

  async startDemoSession(): Promise<{ token: string }> {
    return fetchJson<{ token: string }>(`${API_BASE}/participants/demo`, { method: "POST" });
  },

  async getNextTrial(): Promise<NextTrialResponse> {
    return fetchJson<NextTrialResponse>(`${API_BASE}/trials/next`);
  },

  async startTrial(trialId: number): Promise<NextTrialResponse> {
    return fetchJson<NextTrialResponse>(`${API_BASE}/trials/${trialId}/start`, { method: "POST" });
  },

  async submitTrial(trialId: number, request: {
    all_paths: Array<Array<[number, number]>>;
    path: Array<Array<[number, number]>>;
    num_zoom_actions: number;
    num_forecast_layers_viewed: number;
  }): Promise<SubmitTrialResponse> {
    return fetchJson<SubmitTrialResponse>(`${API_BASE}/trials/${trialId}/submit`, {
      method: "POST",
      body: JSON.stringify(request),
    });
  },

  async submitQuestionnaire(answers: QuestionnaireAnswers): Promise<{ is_completed: boolean }> {
    return fetchJson(`${API_BASE}/participants/me/questionnaire`, {
      method: "POST",
      body: JSON.stringify(answers),
    });
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
};
