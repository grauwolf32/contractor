import type { Audit } from "../../api/audits";
import { ENDED_AUDIT_STATES } from "../../api/states";
import { formatShortDateTime } from "../../app/format";

const TERMINAL: readonly Audit["state"][] = ENDED_AUDIT_STATES;

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
