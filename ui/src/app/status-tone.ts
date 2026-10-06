/** Semantic tones shared by status glyphs, chips, progress segments and vocabulary labels. */
export type StatusTone =
  | "done"
  | "partial"
  | "progress"
  | "blocked"
  | "idle"
  | "review"
  | "warning"
  | "success"
  | "info"
  | "neutral";

export const STATUS_TONES: readonly StatusTone[] = [
  "done",
  "partial",
  "progress",
  "blocked",
  "idle",
  "review",
  "warning",
  "success",
  "info",
  "neutral",
];
