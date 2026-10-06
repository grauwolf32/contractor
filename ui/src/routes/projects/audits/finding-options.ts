import type {
  AuditFindingSeverity,
  AuditFindingState,
  FindingPageRequest,
} from "../../../api/audits";
import {
  FINDING_STATE_LABELS,
  SEVERITY_LABELS,
  VERDICT_LABELS,
} from "../../../app/vocabulary";

/** Analyst severities, most severe first, for filters. */
export const FINDING_SEVERITIES: readonly AuditFindingSeverity[] = [
  "critical",
  "high",
  "medium",
  "low",
  "informational",
];

/** Possible issue states in display order: what needs review first. */
export const FINDING_STATES: readonly AuditFindingState[] = [
  "proposed",
  "confirmed",
  "rejected",
  "duplicate",
  "needs-evidence",
];

/** Decision filter values the findings endpoint accepts (`verdict`). */
export type VerdictFilter = NonNullable<FindingPageRequest["verdict"]>;

export const VERDICT_FILTERS: readonly VerdictFilter[] = [
  "unreviewed",
  "true_positive",
  "false_positive",
];

export function isFindingState(value: string): value is AuditFindingState {
  return (FINDING_STATES as readonly string[]).includes(value);
}

export function isFindingSeverity(
  value: string,
): value is AuditFindingSeverity {
  return (FINDING_SEVERITIES as readonly string[]).includes(value);
}

export function isVerdictFilter(value: string): value is VerdictFilter {
  return (VERDICT_FILTERS as readonly string[]).includes(value);
}

export interface FilterOption {
  value: string;
  label: string;
}

/** "All states" plus every state, labelled in the shared vocabulary. */
export const STATE_OPTIONS: readonly FilterOption[] = FINDING_STATES.map(
  (state) => ({ value: state, label: FINDING_STATE_LABELS[state].label }),
);

/** Analyst ratings only: an AI suggestion is never a severity filter. */
export const SEVERITY_OPTIONS: readonly FilterOption[] = [
  { value: "", label: "All severities" },
  ...FINDING_SEVERITIES.map((severity) => ({
    value: severity,
    label: SEVERITY_LABELS[severity],
  })),
];

/** The analyst's decision: none yet, confirmed or not an issue. */
export const DECISION_OPTIONS: readonly FilterOption[] = [
  { value: "", label: "All decisions" },
  ...VERDICT_FILTERS.map((verdict) => ({
    value: verdict,
    label: VERDICT_LABELS[verdict].label,
  })),
];
