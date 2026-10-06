/**
 * User-facing vocabulary of the V3B UI (docs/design/ui/v3b-build-contract.md
 * §3): one term, one meaning. Pages take labels and tones from here instead of
 * writing their own strings.
 *
 * Every table is keyed by a type generated from the public OpenAPI document,
 * so a new or removed enum value is a compile error here rather than a blank
 * label on a page. A value the client does not know at run time (an older
 * client against a newer Server) still gets a readable label and the neutral
 * tone.
 */
import type {
  Audit,
  AuditAnalystVerdict,
  AuditFindingSeverity,
  AuditFindingState,
  AuditItem,
  AuditProfile,
  AuditReviewAction,
  AuditReviewRequest,
  AuditState,
} from "../api/audits";
import type { components } from "../api/generated/public";
import type { StatusTone } from "./status-tone";

export type AuditCoverageStatus = components["schemas"]["AuditCoverageStatus"];
export type AuditReportStatus = components["schemas"]["AuditReportStatus"];
export type AuditReviewState = components["schemas"]["AuditReviewState"];
export type AuditReviewKind = AuditReviewRequest["kind"];

/** A status word and the tone its glyph, chip or segment uses. */
export interface VocabularyLabel {
  readonly label: string;
  readonly tone: StatusTone;
}

/** User-facing nouns, lower case for use inside sentences. */
export const TERMS = {
  check: "check",
  checks: "checks",
  checkType: "check type",
  checkTypes: "check types",
  possibleIssue: "possible issue",
  possibleIssues: "possible issues",
  issue: "issue",
  issues: "issues",
  /** An artifact on project pages. */
  material: "material",
  materials: "materials",
  /** An artifact in the Library. */
  file: "file",
  files: "files",
  /** An eval assessment check. */
  criterion: "criterion",
  criteria: "criteria",
  report: "report",
  reports: "reports",
  /** Destination names are proper names and keep their capital. */
  inbox: "Inbox",
  library: "Library",
} as const;

/** Upper-cases the first letter, for headings built from TERMS. */
export function capitalize(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

// The label tables are exported for lists that iterate every value, such as
// filter chips; their key order is the display order.

export const CHECK_STATE_LABELS: Readonly<Record<AuditState, VocabularyLabel>> =
  {
    draft: { label: "Draft", tone: "idle" },
    active: { label: "Running", tone: "progress" },
    waiting_review: { label: "Waiting for you", tone: "review" },
    paused: { label: "Paused", tone: "warning" },
    finalizing: { label: "Finishing", tone: "progress" },
    cancelling: { label: "Stopping", tone: "warning" },
    completed: { label: "Finished", tone: "done" },
    cancelled: { label: "Stopped", tone: "neutral" },
    failed: { label: "Failed", tone: "blocked" },
    deleting: { label: "Deleting", tone: "neutral" },
  };

export const COVERAGE_STATUS_LABELS: Readonly<
  Record<AuditCoverageStatus, VocabularyLabel>
> = {
  "not-tested": { label: "Not checked yet", tone: "idle" },
  inconclusive: { label: "Inconclusive", tone: "warning" },
  satisfied: { label: "Met", tone: "done" },
  violated: { label: "Issue found", tone: "blocked" },
  "not-applicable": { label: "Not applicable", tone: "neutral" },
  blocked: { label: "Blocked", tone: "blocked" },
  excluded: { label: "Excluded", tone: "neutral" },
  "traced-complete": { label: "Fully traced", tone: "done" },
  "traced-partial": { label: "Partially traced", tone: "partial" },
  unmapped: { label: "Unmapped", tone: "neutral" },
};

export const FINDING_STATE_LABELS: Readonly<
  Record<AuditFindingState, VocabularyLabel>
> = {
  proposed: { label: "Needs review", tone: "review" },
  confirmed: { label: "Confirmed", tone: "success" },
  rejected: { label: "Not an issue", tone: "neutral" },
  duplicate: { label: "Duplicate", tone: "neutral" },
  "needs-evidence": { label: "Needs evidence", tone: "warning" },
};

/** An analyst verdict, or "unreviewed" for a finding without one. */
export type VerdictValue = AuditAnalystVerdict | "unreviewed";

export const VERDICT_LABELS: Readonly<Record<VerdictValue, VocabularyLabel>> = {
  true_positive: { label: "Confirmed", tone: "success" },
  false_positive: { label: "Not an issue", tone: "neutral" },
  needs_evidence: { label: "Needs evidence", tone: "warning" },
  duplicate: { label: "Duplicate", tone: "neutral" },
  reopen: { label: "Reopened", tone: "review" },
  unreviewed: { label: "Not reviewed", tone: "idle" },
};

export const REVIEW_ACTION_LABELS: Readonly<
  Record<AuditReviewAction, VocabularyLabel>
> = {
  approve: { label: "Approved", tone: "success" },
  reject: { label: "Rejected", tone: "neutral" },
  not_applicable: { label: "Not applicable", tone: "neutral" },
};

export const REVIEW_STATE_LABELS: Readonly<
  Record<AuditReviewState, VocabularyLabel>
> = {
  pending: { label: "Waiting for you", tone: "review" },
  decided: { label: "Decided", tone: "done" },
  expired: { label: "Expired", tone: "neutral" },
};

// A proposed report never reads as accepted (contract §3).
export const REPORT_STATUS_LABELS: Readonly<
  Record<AuditReportStatus, VocabularyLabel>
> = {
  pending: { label: "Not ready yet", tone: "idle" },
  proposed: { label: "Waiting for acceptance", tone: "review" },
  ready: { label: "Ready", tone: "done" },
  unavailable: { label: "Not available", tone: "neutral" },
};

export const REVIEW_KIND_LABELS: Readonly<Record<AuditReviewKind, string>> = {
  "finding-triage": "Possible issue",
  "active-check-approval": "Active test approval",
  "requirement-applicability": "Requirement applicability",
  "report-acceptance": "Report acceptance",
};

export const SEVERITY_LABELS: Readonly<Record<AuditFindingSeverity, string>> = {
  informational: "Informational",
  low: "Low",
  medium: "Medium",
  high: "High",
  critical: "Critical",
};

/** Readable fallback for a value this client does not know yet. */
function unknownLabel(value: string): string {
  const words = value.replaceAll(/[_-]+/g, " ").trim();
  return words === "" ? "Unknown" : capitalize(words);
}

function labelOf<K extends string>(
  table: Readonly<Record<K, VocabularyLabel>>,
  value: K,
): VocabularyLabel {
  return Object.hasOwn(table, value)
    ? table[value]
    : { label: unknownLabel(value), tone: "neutral" };
}

function wordOf<K extends string>(
  table: Readonly<Record<K, string>>,
  value: K,
): string {
  return Object.hasOwn(table, value) ? table[value] : unknownLabel(value);
}

/** Check (Audit) state: "Running", "Waiting for you", "Finished", … */
export function checkStateLabel(state: AuditState): VocabularyLabel {
  return labelOf(CHECK_STATE_LABELS, state);
}

/** Coverage status of one endpoint, requirement or scenario. */
export function coverageStatusLabel(
  status: AuditCoverageStatus,
): VocabularyLabel {
  return labelOf(COVERAGE_STATUS_LABELS, status);
}

/** Triage state of a possible issue: "Needs review", "Confirmed", … */
export function findingStateLabel(state: AuditFindingState): VocabularyLabel {
  return labelOf(FINDING_STATE_LABELS, state);
}

/** Analyst verdict of a decision; a missing verdict is "Not reviewed". */
export function verdictLabel(
  verdict: VerdictValue | null | undefined,
): VocabularyLabel {
  return labelOf(VERDICT_LABELS, verdict ?? "unreviewed");
}

/** Decision on an active test, requirement applicability or report. */
export function reviewActionLabel(action: AuditReviewAction): VocabularyLabel {
  return labelOf(REVIEW_ACTION_LABELS, action);
}

/** State of a review request: pending, decided or expired. */
export function reviewStateLabel(state: AuditReviewState): VocabularyLabel {
  return labelOf(REVIEW_STATE_LABELS, state);
}

/** Report status; "proposed" waits for acceptance and never reads as ready. */
export function reportStatusLabel(status: AuditReportStatus): VocabularyLabel {
  return labelOf(REPORT_STATUS_LABELS, status);
}

/** What a review request asks the user to decide on. */
export function reviewKindLabel(kind: AuditReviewKind): string {
  return wordOf(REVIEW_KIND_LABELS, kind);
}

/**
 * Analyst severity. An empty or missing severity (an unreviewed finding, or
 * a proposal without a suggestion) is "Not set"; a model suggestion is never
 * shown as the analyst's rating.
 */
export function severityLabel(
  severity: AuditFindingSeverity | "" | null | undefined,
): string {
  return severity === undefined || severity === null || severity === ""
    ? "Not set"
    : wordOf(SEVERITY_LABELS, severity);
}

/** What one unit of work in a check is called. */
export type ItemKind = "endpoint" | "requirement" | "scenario" | "item";

const ITEM_NOUNS: Readonly<
  Record<ItemKind, readonly [singular: string, plural: string]>
> = {
  endpoint: ["endpoint", "endpoints"],
  requirement: ["requirement", "requirements"],
  scenario: ["scenario", "scenarios"],
  item: ["item", "items"],
};

/** "endpoint" for one, "endpoints" for any other count (including 0). */
export function itemNoun(kind: ItemKind, count: number): string {
  const [singular, plural] = ITEM_NOUNS[kind];
  return count === 1 ? singular : plural;
}

/** "1 endpoint", "12 requirements", "1,204 items". */
export function itemCount(kind: ItemKind, count: number): string {
  return `${count.toLocaleString("en-US")} ${itemNoun(kind, count)}`;
}

type InventoryImplementation = AuditProfile["inventory"]["implementation"];

const IMPLEMENTATION_KINDS: Readonly<
  Record<InventoryImplementation, ItemKind>
> = {
  "openapi-operations@1": "endpoint",
  "openapi-scans@1": "endpoint",
  // Refined by the standards the check type references.
  "standard-mappings@1": "requirement",
  "checklist@1": "item",
  "finding-candidates@1": "item",
};

// Work item kinds the Server's inventories create (internal/auditdomain).
const WORK_ITEM_KINDS: Readonly<Record<string, ItemKind>> = {
  "operation-trace": "endpoint",
  "openapi-scan": "endpoint",
  "standard-mapping": "requirement",
  checklist: "item",
  "finding-verification": "item",
};

const WSTG = /\bwstg\b/i;

/** WSTG standards hold test scenarios; every other standard requirements. */
function standardsKind(schemes: readonly string[]): ItemKind | undefined {
  if (schemes.length === 0) return undefined;
  return schemes.every((scheme) => WSTG.test(scheme))
    ? "scenario"
    : "requirement";
}

/** Last resort for a check without a baseline: its check type's name. */
function checkTypeNameKind(name: string): ItemKind {
  if (name.startsWith("openapi-")) return "endpoint";
  if (WSTG.test(name)) return "scenario";
  if (/\b(asvs|top-?10)\b/i.test(name)) return "requirement";
  return "item";
}

/** What the UI knows about a check: a work item, the check type or the check. */
export type CheckItemKindSource =
  | Pick<AuditItem, "kind" | "origin">
  | Pick<AuditProfile, "inventory" | "standards">
  | Pick<Audit, "profile" | "baseline">;

/**
 * What the units of work of a check are called, from the most specific data
 * the caller has:
 *
 * 1. A work item (AuditItem): its kind. OpenAPI operation traces and scans
 *    are endpoints; a standard mapping is a scenario when its standard is
 *    WSTG and a requirement otherwise (OWASP Top 10, ASVS, any other
 *    standard); checklist and finding-verification items are items.
 * 2. A check type (AuditProfile): its inventory. `openapi-operations@1` and
 *    `openapi-scans@1` give endpoints; `standard-mappings@1` gives scenarios
 *    when every referenced standard is WSTG and requirements otherwise;
 *    checklists and finding candidates give items.
 * 3. A check (Audit): the standards pinned in its baseline decide as in 2
 *    (a standard selection without pinned standards means requirements).
 *    Before the baseline exists (a draft) or for a check type without
 *    standards, the check type's catalog name decides: `openapi-…` names are
 *    endpoints, names with `wstg` scenarios, names with `asvs` or `top10`
 *    requirements.
 *
 * Anything else is an item: the contract's word when the kind is unknown.
 */
export function checkItemKind(source: CheckItemKindSource): ItemKind {
  if ("inventory" in source) {
    const kind = Object.hasOwn(
      IMPLEMENTATION_KINDS,
      source.inventory.implementation,
    )
      ? IMPLEMENTATION_KINDS[source.inventory.implementation]
      : "item";
    return kind === "requirement"
      ? (standardsKind(source.standards.map((standard) => standard.scheme)) ??
          "requirement")
      : kind;
  }
  if ("origin" in source) {
    const kind = Object.hasOwn(WORK_ITEM_KINDS, source.kind)
      ? WORK_ITEM_KINDS[source.kind]
      : undefined;
    if (kind === "requirement") {
      const scheme = source.origin.standard?.scheme;
      return scheme === undefined
        ? "requirement"
        : (standardsKind([scheme]) ?? "requirement");
    }
    return kind ?? "item";
  }
  const baseline = source.baseline;
  const pinned = standardsKind(
    baseline?.standards.map((standard) => standard.reference.scheme) ?? [],
  );
  if (pinned !== undefined) return pinned;
  if (baseline?.inventory?.standardSelection !== undefined)
    return "requirement";
  return checkTypeNameKind(source.profile.name);
}
