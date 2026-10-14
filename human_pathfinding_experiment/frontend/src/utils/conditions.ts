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
