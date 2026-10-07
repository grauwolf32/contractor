import type { Audit } from "../../../api/audits";

const shortDateTime = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
});

const clock = new Intl.DateTimeFormat("en-US", {
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
});

function parse(value: string | undefined): Date | undefined {
  if (value === undefined) return undefined;
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? undefined : date;
}

/** "Oct 4, 23:51" in local time; the value as written when it is no date. */
export function formatShortDateTime(value: string): string {
  const date = parse(value);
  return date === undefined ? value : shortDateTime.format(date);
}

/**
 * "Oct 4, 23:53 to 23:56", or "Oct 4, 23:53 to Oct 5, 00:10" across days.
 */
export function formatSpan(start: string, end: string | undefined): string {
  const from = parse(start);
  const to = parse(end);
  if (from === undefined) return start;
  if (to === undefined) return shortDateTime.format(from);
  const sameDay = from.toDateString() === to.toDateString();
  return `${shortDateTime.format(from)} to ${sameDay ? clock.format(to) : shortDateTime.format(to)}`;
}

/** "Started Oct 4, 23:51", or when the check was created as a draft. */
export function startedText(audit: Audit): string {
  return audit.startedAt === undefined
    ? `Created ${formatShortDateTime(audit.createdAt)}`
    : `Started ${formatShortDateTime(audit.startedAt)}`;
}

const TERMINAL: readonly Audit["state"][] = [
  "completed",
  "cancelled",
  "failed",
  "deleting",
];

/**
 * The check's time limit in words, or undefined once it no longer matters
 * (ended checks).
 */
export function timeLimitText(audit: Audit): string | undefined {
  if (audit.state === "draft") return "The time limit is set when it starts.";
  if (audit.stopReason?.code === "deadline_exhausted")
    return "Time limit reached.";
  if (TERMINAL.includes(audit.state)) return undefined;
  if (audit.deadlineAt === undefined) return "No time limit.";
  if (audit.state === "paused") return "Paused: the remaining time is kept.";
  return `Time limit: stops starting new work by ${formatShortDateTime(audit.deadlineAt)}.`;
}
