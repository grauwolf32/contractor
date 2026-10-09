import type { Audit } from "../../../api/audits";
import type { StatusTone } from "../../../app/status-tone";

/** Short names of the stop reasons, for technical details. */
const STOP_REASON_LABELS: Readonly<Record<string, string>> = {
  deadline_exhausted: "Time limit reached",
  round_complete: "Round complete",
  cancel_requested: "Stopped on request",
  delete_requested: "Deletion requested",
  project_deleting: "Project is being deleted",
  report_rejected: "Report rejected",
  report_acceptance_expired: "Report acceptance expired",
  submission_budget_exhausted: "Run limit reached",
  item_budget_exhausted: "Item limit reached",
  item_attempt_budget_exhausted: "Item attempt limit reached",
  round_budget_exhausted: "Round limit reached",
  role_attempt_budget_exhausted: "Step attempt limit reached",
  proposal_scan_budget_exhausted: "Possible issue limit reached",
  proposal_inventory_limit_exceeded: "Possible issue inventory too large",
  next_round_contract_invalid: "Next round data invalid",
  controller_contract_invalid: "Check type configuration invalid",
  dispatch_contract_invalid: "Work data invalid",
  role_contract_invalid: "Step configuration invalid",
  role_dispatch_contract_invalid: "Step data invalid",
  role_dependency_unresolved: "Step dependency missing",
  role_execution_not_retryable: "Step cannot be retried",
};

/**
 * One plain sentence per stop reason the Server sends. Unknown
 * codes are never shown raw: the Server's own message stands in.
 */
const STOP_REASON_SENTENCES: Readonly<Record<string, string>> = {
  deadline_exhausted: "The time limit was reached, so no new work was started.",
  round_complete: "Every item of this round has a result.",
  cancel_requested: "The check was stopped on request.",
  delete_requested: "The check is being deleted on request.",
  project_deleting: "The check stopped because its project is being deleted.",
  report_rejected: "The check ended because its report was rejected.",
  report_acceptance_expired:
    "The check ended because its report was not accepted in time.",
  submission_budget_exhausted:
    "The check started every run it was allowed to start.",
  item_budget_exhausted: "The check reached its limit of items.",
  item_attempt_budget_exhausted: "An item used every attempt it was allowed.",
  round_budget_exhausted: "The check used every round it was allowed.",
  role_attempt_budget_exhausted: "A step used every attempt it was allowed.",
  proposal_scan_budget_exhausted:
    "The check reached its limit for following up possible issues.",
  proposal_inventory_limit_exceeded:
    "The next round could not start because its possible issue inventory exceeds the limit.",
  next_round_contract_invalid:
    "The next round could not start because its work data was invalid.",
  controller_contract_invalid:
    "The check stopped because its check type is configured incorrectly.",
  dispatch_contract_invalid:
    "The check stopped because the data to start its work was invalid.",
  role_contract_invalid:
    "The check stopped because one of its steps is configured incorrectly.",
  role_dispatch_contract_invalid:
    "The check stopped because the data to start one of its steps was invalid.",
  role_dependency_unresolved:
    "The check stopped because a step needs something that could not be found.",
  role_execution_not_retryable:
    "The check stopped because a step failed in a way that cannot be retried.",
};

/** Every stop reason code the UI explains in plain words. */
export const STOP_REASON_CODES: readonly string[] = Object.keys(
  STOP_REASON_SENTENCES,
);

const FAILURE_CODE =
  /_invalid$|_unresolved$|not_retryable$|_budget_exhausted$/u;

export interface StopReasonPresentation {
  /** Short name of the code, or undefined when the code is unknown. */
  label: string | undefined;
  /** One plain sentence; the Server's message when the code is unknown. */
  sentence: string;
  /** The Server's own message. */
  message: string;
  tone: "error" | "neutral";
  /** The time limit ended the work. */
  deadline: boolean;
}

/**
 * Describes a check's stop reason for display. Never surfaces the raw code:
 * unknown codes fall back to the Server's message alone.
 */
export function describeStopReason(
  audit: Pick<Audit, "state" | "stopReason">,
): StopReasonPresentation | null {
  const reason = audit.stopReason;
  if (reason === undefined) return null;
  const deadline = reason.code === "deadline_exhausted";
  const known = Object.hasOwn(STOP_REASON_SENTENCES, reason.code);
  const tone =
    audit.state === "failed" || (!deadline && FAILURE_CODE.test(reason.code))
      ? "error"
      : "neutral";
  let sentence = known ? STOP_REASON_SENTENCES[reason.code]! : reason.message;
  if (deadline)
    sentence =
      audit.state === "paused"
        ? "The time limit was reached. Continue with a longer limit or no time limit; collected results are kept."
        : "Stopped by the time limit: no new work was started after it.";
  return {
    label: Object.hasOwn(STOP_REASON_LABELS, reason.code)
      ? STOP_REASON_LABELS[reason.code]
      : undefined,
    sentence,
    message: reason.message,
    tone,
    deadline,
  };
}

/** Status tone of a stop reason banner. */
export function stopReasonTone(stop: StopReasonPresentation): StatusTone {
  if (stop.tone === "error") return "blocked";
  return stop.deadline ? "warning" : "info";
}
