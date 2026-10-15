/**
 * The check's time limit at start (POST /v1/audits/{auditId}/start
 * `deadlineSeconds`): 24 hours by default, 7 days, no time limit (0) or a
 * custom number of hours between 0.01 and 8760 (365 days).
 */

type TimeLimitChoice = "86400" | "604800" | "0" | "custom";

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
