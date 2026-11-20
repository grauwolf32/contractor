import type {
  FindingPageRequest,
  FindingSeverity,
  FindingState,
} from "../../../api/audits";
import { SEVERITY_LABELS, VERDICT_LABELS } from "../../../app/vocabulary";

/** Analyst severities, most severe first, for filters. */
const FINDING_SEVERITIES: readonly FindingSeverity[] = [
  "critical",
  "high",
  "medium",
  "low",
  "informational",
];

/** Possible issue states in display order: what needs review first. */
export const FINDING_STATES: readonly FindingState[] = [
  "proposed",
  "confirmed",
  "rejected",
  "duplicate",
  "needs-evidence",
];

/** Decision filter values the findings endpoint accepts (`verdict`). */
type VerdictFilter = NonNullable<FindingPageRequest["verdict"]>;

const VERDICT_FILTERS: readonly VerdictFilter[] = [
  "needs_review",
  "true_positive",
  "false_positive",
];

export function isFindingState(value: string): value is FindingState {
  return (FINDING_STATES as readonly string[]).includes(value);
}

export function isFindingSeverity(value: string): value is FindingSeverity {
  return (FINDING_SEVERITIES as readonly string[]).includes(value);
}

export function isVerdictFilter(value: string): value is VerdictFilter {
  return (VERDICT_FILTERS as readonly string[]).includes(value);
}

interface FilterOption {
  value: string;
  label: string;
}

/** Analyst severity when present, otherwise the canonical producer suggestion. */
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
    label:
      verdict === "needs_review"
        ? "Needs review"
        : VERDICT_LABELS[verdict].label,
  })),
];
