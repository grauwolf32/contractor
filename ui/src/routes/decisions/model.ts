/**
 * Pure rules of the decision components: which verdicts and actions exist,
 * the request bodies they send, client-side validation, pending-review
 * selection and the explanations shown when the Server refuses a decision.
 */
import {
  AUDIT_ID_PATTERN,
  type AuditAnalystVerdict,
  type AuditFinding,
  type AuditFindingSeverity,
  type AuditReviewAction,
  type AuditReviewRequest,
  type DecideAuditFindingRequest,
} from "../../api/audits";
import { PublicAPIError } from "../../api/error";
import type { components } from "../../api/generated/public";
import {
  reviewActionLabel,
  SEVERITY_LABELS,
  severityLabel,
  verdictLabel,
  type VocabularyLabel,
} from "../../app/vocabulary";

export type AuditReviewDecision = components["schemas"]["AuditReviewDecision"];
export type AuditFindingDecisionResult =
  components["schemas"]["AuditFindingDecisionResult"];
export type AuditActionDecisionResult =
  components["schemas"]["AuditActionDecisionResult"];

/** Rationale limit of every decision: 64 KiB of UTF-8 (S19 §14.1). */
export const MAX_RATIONALE_BYTES = 65_536;

/** Required-reason helper text (contract §3). */
export const RATIONALE_HELP = "Required. Saved with the decision.";

/** Button labels of the five finding decisions (contract §3). */
export const VERDICT_ACTIONS: Readonly<Record<AuditAnalystVerdict, string>> = {
  true_positive: "Confirm issue",
  false_positive: "Not an issue",
  needs_evidence: "Needs evidence",
  duplicate: "Duplicate…",
  reopen: "Reopen",
};

/** Button labels of the approval decisions, in display order. */
export const REVIEW_ACTIONS: readonly {
  action: AuditReviewAction;
  label: string;
}[] = [
  { action: "approve", label: "Approve" },
  { action: "reject", label: "Reject" },
  { action: "not_applicable", label: "Not applicable" },
];

export const SEVERITY_OPTIONS: readonly {
  value: AuditFindingSeverity;
  label: string;
}[] = (Object.keys(SEVERITY_LABELS) as AuditFindingSeverity[]).map((value) => ({
  value,
  label: SEVERITY_LABELS[value],
}));

export function isSeverity(value: string): value is AuditFindingSeverity {
  return Object.hasOwn(SEVERITY_LABELS, value);
}

/** What the user chose for a possible issue, before it is sent. */
export interface FindingDecisionDraft {
  verdict: AuditAnalystVerdict;
  severity?: AuditFindingSeverity | undefined;
  duplicateTargetId?: string | undefined;
  rationale: string;
}

export function utf8Length(text: string): number {
  return new TextEncoder().encode(text).length;
}

/**
 * What still blocks a finding decision that DecisionBar does not check
 * itself, or undefined when the draft can be sent. DecisionBar already
 * requires a verdict, a non-blank reason and a severity for Confirm issue.
 */
export function findingDraftProblem(
  draft: FindingDecisionDraft,
  findingId: string,
): string | undefined {
  const rationale = draft.rationale.trim();
  if (rationale === "") return "Write a short reason.";
  if (utf8Length(rationale) > MAX_RATIONALE_BYTES)
    return "The reason is too long. Keep it under 64 KiB.";
  if (draft.verdict === "true_positive" && draft.severity === undefined)
    return "Choose severity.";
  if (draft.verdict === "duplicate") {
    const target = draft.duplicateTargetId?.trim() ?? "";
    if (target === "") return "Choose the possible issue this one duplicates.";
    if (target === findingId)
      return "A possible issue cannot duplicate itself. Choose another one.";
    if (!AUDIT_ID_PATTERN.test(target))
      return "That is not a valid possible issue ID.";
  }
  return undefined;
}

/** Request body of a finding decision; the rationale is trimmed. */
export function findingDecisionBody(
  draft: FindingDecisionDraft,
): DecideAuditFindingRequest {
  const rationale = draft.rationale.trim();
  switch (draft.verdict) {
    case "true_positive":
      if (draft.severity === undefined)
        throw new TypeError("Confirm issue needs a severity");
      return { verdict: "true_positive", severity: draft.severity, rationale };
    case "duplicate": {
      const target = draft.duplicateTargetId?.trim() ?? "";
      if (target === "")
        throw new TypeError("Duplicate needs the original possible issue");
      return { verdict: "duplicate", duplicateTargetId: target, rationale };
    }
    default:
      return { verdict: draft.verdict, rationale };
  }
}

/** The review requests of one finding that are still open. */
function pendingTriage(
  reviews: readonly AuditReviewRequest[],
  findingId: string,
): AuditReviewRequest[] {
  return reviews.filter(
    (review) =>
      review.state === "pending" &&
      review.kind === "finding-triage" &&
      (review.findingId ?? review.subjectId) === findingId,
  );
}

export interface PendingReviewStatus {
  /** The open request bound to the finding's current revision. */
  review?: AuditReviewRequest | undefined;
  /**
   * An open request is bound to a newer revision than the finding the page
   * shows: the page is out of date and must not decide.
   */
  stale: boolean;
}

/**
 * The open finding-triage request a decision can reuse. Requests for an older
 * revision are ignored: they can no longer be decided, and opening a review
 * for the current revision replaces them.
 */
export function pickPendingReview(
  reviews: readonly AuditReviewRequest[],
  finding: Pick<AuditFinding, "findingId" | "revision">,
): PendingReviewStatus {
  const pending = pendingTriage(reviews, finding.findingId);
  const review = pending.find(
    (candidate) => candidate.subjectRevision === finding.revision,
  );
  return {
    review,
    stale:
      review === undefined &&
      pending.some((candidate) => candidate.subjectRevision > finding.revision),
  };
}

/** True when the request's expiry has passed by `now` (milliseconds). */
export function reviewExpired(
  review: Pick<AuditReviewRequest, "expiresAt">,
  now: number,
): boolean {
  if (review.expiresAt === undefined) return false;
  const expires = Date.parse(review.expiresAt);
  return Number.isFinite(expires) && expires <= now;
}

/** "Confirmed · High", "Not an issue", "Approved", … */
export function decisionOutcome(
  decision: Pick<AuditReviewDecision, "action" | "verdict" | "severity">,
): VocabularyLabel {
  if (decision.action !== undefined) return reviewActionLabel(decision.action);
  const verdict = verdictLabel(decision.verdict);
  return decision.severity === undefined
    ? verdict
    : {
        label: `${verdict.label} · ${severityLabel(decision.severity)}`,
        tone: verdict.tone,
      };
}

/** The most recent decision among review requests, if any was recorded. */
export function latestDecision(
  reviews: readonly AuditReviewRequest[],
  preferredVerdict?: AuditAnalystVerdict,
): AuditReviewDecision | undefined {
  const decisions = reviews.flatMap((review) =>
    review.decision === undefined ? [] : [review.decision],
  );
  const preferred =
    preferredVerdict === undefined
      ? []
      : decisions.filter((decision) => decision.verdict === preferredVerdict);
  const candidates = preferred.length > 0 ? preferred : decisions;
  let latest: AuditReviewDecision | undefined;
  for (const decision of candidates) {
    if (
      latest === undefined ||
      Date.parse(decision.createdAt) > Date.parse(latest.createdAt)
    )
      latest = decision;
  }
  return latest;
}

/** Which verdict produced a decided finding state, when one did. */
export function verdictForState(
  state: AuditFinding["state"],
): AuditAnalystVerdict | undefined {
  switch (state) {
    case "confirmed":
      return "true_positive";
    case "rejected":
      return "false_positive";
    case "duplicate":
      return "duplicate";
    case "needs-evidence":
      return "needs_evidence";
    default:
      return undefined;
  }
}

/** What a refused decision was about, for the explanation. */
export type DecisionSubject = "finding" | "request";

/**
 * Why a decision was not saved and what the page did about it. Every failed
 * decision refetches the current state; nothing is retried automatically.
 */
export function decisionErrorMessage(
  error: unknown,
  subject: DecisionSubject,
): string {
  if (!(error instanceof PublicAPIError)) {
    return error instanceof Error
      ? `Not saved: ${error.message}`
      : "Not saved: the request failed.";
  }
  if (error.status === 412)
    return subject === "finding"
      ? "Not saved: this possible issue or its check changed first. The latest version is now shown. Check it, then record your decision again."
      : "Not saved: this request changed or expired first. The latest version is now shown. Check it, then decide again.";
  if (error.status === 409)
    return "Not saved: the decision conflicts with the current state, so nothing was retried. The latest version is now shown.";
  if (error.status === 404)
    return subject === "finding"
      ? "Not saved: the review or the chosen possible issue no longer exists. The latest version is now shown."
      : "Not saved: this request no longer exists. The latest version is now shown.";
  if (
    error.status === 0 &&
    (error.code === "network_error" || error.code === "invalid_api_response")
  )
    return "The Server's answer did not arrive, so the decision may have been saved. Check the latest state before you record it again.";
  return `Not saved: ${error.message}`;
}
