/**
 * Pure rules of the decision components: which verdicts and actions exist,
 * the request bodies they send, client-side validation, pending-review
 * selection and the explanations shown when the Server refuses a decision.
 */
import {
  AUDIT_ID_PATTERN,
  type FindingVerdict,
  type FindingSeverity,
  type EditProjectFindingRequest,
} from "../../api/audits";
import { schemaLimit } from "../../api/contract";

import type { components } from "../../api/generated/public";

import { SEVERITY_LABELS } from "../../app/vocabulary";
export type ProjectFindingEditResult =
  components["schemas"]["EditProjectFindingResult"];

/** Rationale limit of every decision, in UTF-8 bytes (x-max-utf8-bytes). */
export const MAX_RATIONALE_BYTES = schemaLimit(
  "DecisionRationale",
  "x-max-utf8-bytes",
);

/** Button labels of the five finding decisions (contract §3). */
export const VERDICT_ACTIONS: Readonly<Record<FindingVerdict, string>> = {
  true_positive: "Confirm issue",
  false_positive: "Not an issue",
  needs_evidence: "Needs evidence",
  duplicate: "Duplicate…",
  reopen: "Reopen",
};

export const SEVERITY_OPTIONS: readonly {
  value: FindingSeverity;
  label: string;
}[] = (Object.keys(SEVERITY_LABELS) as FindingSeverity[]).map((value) => ({
  value,
  label: SEVERITY_LABELS[value],
}));

/** What the user chose for a possible issue, before it is sent. */
interface FindingDecisionDraft {
  verdict: FindingVerdict;
  severity?: FindingSeverity | undefined;
  duplicateTargetId?: string | undefined;
  rationale: string;
}

function utf8Length(text: string): number {
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
): EditProjectFindingRequest {
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
