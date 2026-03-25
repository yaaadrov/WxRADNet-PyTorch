// Flare palette (dark orange to light yellow) for obstacles
export const FLARE_PALETTE = [
  "#E8590C", // dark orange (first prediction)
  "#F76707", // deeper orange
  "#FF922B", // dark orange
  "#FFA94D", // orange
  "#FFD43B", // golden yellow
  "#FFEC99", // yellow
  "#FFF9DB", // light yellow (last prediction)
];

// Crest palette (dark blue to light blue) for hulls
export const CREST_PALETTE = [
  "#228BE6", // dark blue (first prediction)
  "#339AF0", // blue
  "#4DABF7", // medium blue
  "#74C0FC", // light blue
  "#A5D8FF", // sky blue
  "#D0EBFF", // pale blue
  "#E7F5FF", // light blue (last prediction)
];

export function getFlareColor(index: number, alpha: number = 1): string {
  const color = FLARE_PALETTE[index % FLARE_PALETTE.length];
  return hexToRgba(color, alpha);
}

export function getCrestColor(index: number, alpha: number = 0.5): string {
  const color = CREST_PALETTE[index % CREST_PALETTE.length];
  return hexToRgba(color, alpha);
}

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
