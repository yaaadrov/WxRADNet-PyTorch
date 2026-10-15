import type { ObstacleMode, WindowSize } from "../types";

/** Human-readable description of an experimental condition (shown to the participant). */
export function conditionDescription(condition: "C0" | "C1" | "C2"): string {
  switch (condition) {
    case "C0":
      return "Только текущая радиолокационная картина (границы очагов гроз)";
    case "C1":
      return "Дополнительно отображаются запретные буферные зоны вокруг очагов";
    case "C2":
      return "Дополнительно отображаются запретные зоны и прогноз развития грозы";
  }
}

/** Runtime parameters of an experimental condition (mirrors backend ExperimentCondition). */
export function conditionParams(condition: "C0" | "C1" | "C2"): { obstacleMode: ObstacleMode; windowSize: WindowSize } {
  switch (condition) {
    case "C0":
      return { obstacleMode: "obstacles", windowSize: 1 };
    case "C1":
      return { obstacleMode: "hull", windowSize: 1 };
    case "C2":
      return { obstacleMode: "hull", windowSize: 7 };
  }
}
