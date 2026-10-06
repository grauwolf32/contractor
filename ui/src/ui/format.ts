import { formatTimestamp, middleTruncate } from "../app/format";

/**
 * Short form of an identifier for an IdChip: values up to 16 characters stay
 * as they are, longer ones keep the first 8 and the last 4 characters.
 */
export function shortenId(value: string): string {
  return middleTruncate(value, 8, 4);
}

const clockFormat = new Intl.DateTimeFormat("en-US", {
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
});

export interface ClockTime {
  /** HH:MM in local time, or the original text when it is not a date. */
  label: string;
  /** ISO timestamp for `<time dateTime>`; absent for free text. */
  dateTime?: string;
  /** Full absolute timestamp for the hover title; absent for free text. */
  title?: string;
}

/**
 * Local HH:MM for a timeline entry. A string that is not a date ("now",
 * "Today") is kept as written; an invalid Date or an empty value has no time.
 */
export function clockTime(
  value: string | Date | undefined,
): ClockTime | undefined {
  if (value === undefined || value === "") return undefined;
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.valueOf())) {
    return typeof value === "string" ? { label: value } : undefined;
  }
  const iso = date.toISOString();
  return {
    label: clockFormat.format(date),
    dateTime: iso,
    title: formatTimestamp(iso),
  };
}
