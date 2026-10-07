/**
 * The check's time limit at start (POST /v1/audits/{auditId}/start
 * `deadlineSeconds`): 24 hours by default, 7 days, no time limit (0) or a
 * custom number of hours between 0.01 and 8760 (365 days).
 */

export type TimeLimitChoice = "86400" | "604800" | "0" | "custom";

export const DEFAULT_TIME_LIMIT: TimeLimitChoice = "86400";

export const TIME_LIMIT_OPTIONS: readonly {
  value: TimeLimitChoice;
  label: string;
}[] = [
  { value: "86400", label: "24 hours" },
  { value: "604800", label: "7 days" },
  { value: "0", label: "No time limit" },
  { value: "custom", label: "Custom" },
];

export const MIN_CUSTOM_HOURS = 0.01;
export const MAX_CUSTOM_HOURS = 8760;

export interface TimeLimitState {
  choice: TimeLimitChoice;
  /** The custom duration as typed, in hours. */
  hours: string;
}

export function isTimeLimitChoice(value: string): value is TimeLimitChoice {
  return TIME_LIMIT_OPTIONS.some((option) => option.value === value);
}

/** Whole seconds for the start request, or undefined when invalid. */
export function timeLimitSeconds(state: TimeLimitState): number | undefined {
  if (state.choice !== "custom") return Number(state.choice);
  const text = state.hours.trim();
  if (text === "") return undefined;
  const hours = Number(text);
  if (
    !Number.isFinite(hours) ||
    hours < MIN_CUSTOM_HOURS ||
    hours > MAX_CUSTOM_HOURS
  )
    return undefined;
  const seconds = Math.round(hours * 3600);
  return seconds >= 1 && seconds <= MAX_CUSTOM_HOURS * 3600
    ? seconds
    : undefined;
}

function plural(count: number, unit: string): string {
  const text = Number.isInteger(count)
    ? count.toLocaleString("en-US")
    : count.toLocaleString("en-US", { maximumFractionDigits: 2 });
  return `${text} ${unit}${count === 1 ? "" : "s"}`;
}

/** "24 hours", "7 days", "1.5 hours", "36 seconds"; 0 is "no time limit". */
export function describeTimeLimit(seconds: number): string {
  if (seconds === 0) return "no time limit";
  if (seconds > 86400 && seconds % 86400 === 0)
    return plural(seconds / 86400, "day");
  if (seconds >= 3600) return plural(Math.round(seconds / 36) / 100, "hour");
  if (seconds % 60 === 0) return plural(seconds / 60, "minute");
  return plural(seconds, "second");
}
