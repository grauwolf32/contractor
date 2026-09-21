import type { Audit } from "../../../api/audits";

const STOP_REASON_LABELS: Record<string, string> = {
  deadline_exhausted: "Time limit reached",
  round_complete: "Round complete",
  owner_paused: "Paused by owner",
  cancel_requested: "Cancelled by owner",
  delete_requested: "Deletion requested",
  project_deleting: "Project is being deleted",
  report_rejected: "Report rejected",
  report_acceptance_expired: "Report approval expired",
  submission_budget_exhausted: "Run submission budget exhausted",
  item_budget_exhausted: "Check budget exhausted",
  item_attempt_budget_exhausted: "Check attempt budget exhausted",
  round_budget_exhausted: "Round budget exhausted",
  role_attempt_budget_exhausted: "Role attempt budget exhausted",
  proposal_scan_budget_exhausted: "Proposal scan budget exhausted",
  controller_contract_invalid: "Controller contract invalid",
  dispatch_contract_invalid: "Dispatch data invalid",
  role_contract_invalid: "Role configuration invalid",
  role_dispatch_contract_invalid: "Role dispatch data invalid",
  role_dependency_unresolved: "Role dependency unresolved",
  role_execution_not_retryable: "Role execution not retryable",
};

const FAILURE_CODE =
  /_invalid$|_unresolved$|not_retryable$|_budget_exhausted$/u;

export interface StopReasonPresentation {
  /** Human label for the code, or undefined when the code is unknown. */
  label: string | undefined;
  message: string;
  tone: "error" | "neutral";
  deadline: boolean;
}

/**
 * Describes an Audit stop reason for display. Never surfaces the raw enum:
 * unknown codes fall back to the server message alone.
 */
export function describeStopReason(
  audit: Pick<Audit, "state" | "stopReason">,
): StopReasonPresentation | null {
  const reason = audit.stopReason;
  if (reason === undefined) return null;
  const deadline = reason.code === "deadline_exhausted";
  const tone =
    audit.state === "failed" || (!deadline && FAILURE_CODE.test(reason.code))
      ? "error"
      : "neutral";
  return {
    label: STOP_REASON_LABELS[reason.code],
    message: reason.message,
    tone,
    deadline,
  };
}
