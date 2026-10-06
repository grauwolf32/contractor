/**
 * Words, paths and progress lines for Inbox rows and details. Labels come
 * from the shared vocabulary (src/app/vocabulary.ts); nothing here reads the
 * Server.
 */
import type { AuditCollection } from "../../api/audit-collections";
import type { Audit, AuditItem, AuditWorkspace } from "../../api/audits";
import type {
  CrossProjectCheck,
  CrossProjectDecision,
} from "../../api/cross-project";
import type { RunStatus, RunSummary, WorkflowRunState } from "../../api/runs";
import {
  capitalize,
  checkItemKind,
  COVERAGE_STATUS_LABELS,
  itemCount,
  itemNoun,
  type ItemKind,
  type VocabularyLabel,
} from "../../app/vocabulary";
import type { ProgressSegment } from "../../ui";
import { auditPresetLabel } from "../projects/audits/labels";
import type { InboxRow, InboxSectionId } from "./model";
import {
  organizeRunOutputs,
  parseWorkflowIdentity,
  type OutputEntry,
  type WorkflowOutputSlot,
} from "../runs/output-model";

/** Section titles, asides and empty states (V3B home mockup). */
export const SECTIONS: Readonly<
  Record<InboxSectionId, { title: string; aside: string; empty: string }>
> = {
  decide: {
    title: "Decide",
    aside: "Only you can decide these",
    empty: "Nothing needs a decision. New possible issues land here.",
  },
  unblock: {
    title: "Unblock",
    aside: "Stopped and needs a nudge",
    empty: "Nothing is stuck. Failed Runs and stopped checks land here.",
  },
  ready: {
    title: "Ready",
    aside: "Finished results",
    empty:
      "Nothing finished in the last 7 days. Reports, finished checks and Runs land here.",
  },
  running: {
    title: "Running",
    aside: "Nothing to do yet",
    empty: "Nothing is running. Checks you start show their progress here.",
  },
};

/** The check type's name, e.g. "OpenAPI · Operation trace". */
export function checkTypeLabel(audit: Pick<Audit, "profile">): string {
  return auditPresetLabel(audit.profile.name);
}

/** "crapi-workshop: OpenAPI · Operation trace". */
export function checkName(check: CrossProjectCheck): string {
  return `${check.project.name}: ${checkTypeLabel(check.audit)}`;
}

const segment = encodeURIComponent;

export function projectPath(projectId: string): string {
  return `/projects/${segment(projectId)}`;
}

export function checkPath(projectId: string, auditId: string): string {
  return `${projectPath(projectId)}/audits/${segment(auditId)}`;
}

/**
 * The check's coverage filtered to the work that needs follow-up: blocked,
 * inconclusive, partially traced and unmapped items, the statuses the
 * workspace counts as gaps.
 */
export function followUpPath(projectId: string, auditId: string): string {
  return `${checkPath(projectId, auditId)}/coverage?result=uncertain`;
}

export function pendingReviewsPath(projectId: string, auditId: string) {
  return `${checkPath(projectId, auditId)}/reviews?state=pending`;
}

export function issuePath(auditId: string, findingId: string): string {
  return `/issues/${segment(auditId)}/${segment(findingId)}`;
}

export function reportPath(auditId: string): string {
  return `/reports/${segment(auditId)}`;
}

export function runPath(runId: string): string {
  return `/runs/${segment(runId)}`;
}

export const FAILED_RUNS_PATH = "/runs?view=completed&state=failed";
export const SUCCEEDED_RUNS_PATH = "/runs?view=completed&state=succeeded";

/** The item's own page, which Enter and "Open …" go to. */
export function rowPage(row: InboxRow): string {
  switch (row.type) {
    case "issue":
      return issuePath(row.issue.audit.auditId, row.issue.finding.findingId);
    case "review":
      return row.decision.review.kind === "report-acceptance"
        ? reportPath(row.decision.audit.auditId)
        : pendingReviewsPath(
            row.decision.project.projectId,
            row.decision.audit.auditId,
          );
    case "check":
      return checkPath(row.check.project.projectId, row.check.audit.auditId);
    case "report":
      return reportPath(row.report.audit.auditId);
    case "run":
      return runPath(row.run.runId);
  }
}

/** Progress of a check's current round from its workspace counters. */
export interface CheckProgress {
  kind: ItemKind;
  total: number;
  /** Concluded items, including those with an issue found. */
  done: number;
  issues: number;
  /** Blocked, inconclusive, partially traced or unmapped items. */
  gaps: number;
  unchecked: number;
  segments: ProgressSegment[];
  /** "3 of 5 endpoints done: 1 issue found, 1 needs follow-up, …". */
  label: string;
  /** "3 of 5 endpoints done". */
  summary: string;
}

function count(value: number): number {
  return Number.isSafeInteger(value) && value > 0 ? value : 0;
}

function repeat(entry: ProgressSegment, times: number): ProgressSegment[] {
  return Array.from({ length: times }, () => entry);
}

export const PROGRESS_LABELS = {
  done: "Done",
  issue: COVERAGE_STATUS_LABELS.violated.label,
  followUp: "Needs follow-up",
  unchecked: COVERAGE_STATUS_LABELS["not-tested"].label,
} as const;

/**
 * One segment per item of the current round, grouped by outcome: done, issue
 * found, needs follow-up, not checked yet. The workspace has counts only, so
 * the line keeps their proportions, not the items' list order.
 */
export function checkProgress(
  workspace: AuditWorkspace,
  kind: ItemKind,
): CheckProgress {
  const total = count(workspace.totalChecks);
  const done = Math.min(count(workspace.completedChecks), total);
  const issues = Math.min(count(workspace.issues), done);
  const gaps = Math.min(count(workspace.gaps), total - done);
  const unchecked = Math.max(0, total - done - gaps);
  const segments = [
    ...repeat({ tone: "done", label: PROGRESS_LABELS.done }, done - issues),
    ...repeat(
      {
        tone: COVERAGE_STATUS_LABELS.violated.tone,
        label: PROGRESS_LABELS.issue,
      },
      issues,
    ),
    ...repeat({ tone: "partial", label: PROGRESS_LABELS.followUp }, gaps),
    ...repeat(
      {
        tone: COVERAGE_STATUS_LABELS["not-tested"].tone,
        label: PROGRESS_LABELS.unchecked,
      },
      unchecked,
    ),
  ];
  const summary =
    total === 0
      ? `No ${itemNoun(kind, 0)} listed yet`
      : `${done.toLocaleString("en-US")} of ${itemCount(kind, total)} done`;
  const details = [
    issues > 0 ? `${issues.toLocaleString("en-US")} with an issue found` : "",
    gaps > 0
      ? `${gaps.toLocaleString("en-US")} ${gaps === 1 ? "needs" : "need"} follow-up`
      : "",
    unchecked > 0 ? `${unchecked.toLocaleString("en-US")} not checked yet` : "",
  ].filter((part) => part !== "");
  return {
    kind,
    total,
    done,
    issues,
    gaps,
    unchecked,
    segments,
    summary,
    label: details.length === 0 ? summary : `${summary}: ${details.join(", ")}`,
  };
}

/** "2 endpoints couldn't be fully checked". */
export function gapsTitle(gaps: number, kind: ItemKind): string {
  return `${itemCount(kind, gaps)} couldn't be fully checked`;
}

/** What a check's work items are called. */
export function checkKind(check: CrossProjectCheck): ItemKind {
  return checkItemKind(check.audit);
}

// OpenAPI operation keys are opaque digests; the ordinal names those items.
const OPAQUE_SUBJECT = /^op-[a-f0-9]{16,}$/i;

/** A work item for people: its key, or "Endpoint 3" for an opaque key. */
export function itemLabel(item: AuditItem): string {
  return OPAQUE_SUBJECT.test(item.subjectKey)
    ? `${capitalize(itemNoun(checkItemKind(item), 1))} ${item.ordinal + 1}`
    : item.subjectKey;
}

/**
 * The work item an approval or applicability request is about, when the
 * check's items have been read; undefined otherwise (and for a report).
 */
export function reviewSubjectItem(
  decision: CrossProjectDecision,
  items: ReadonlyMap<string, AuditCollection<AuditItem>>,
): AuditItem | undefined {
  if (decision.review.subjectKind !== "audit-item-action") return undefined;
  return items
    .get(decision.audit.auditId)
    ?.items.find((item) => item.itemId === decision.review.subjectId);
}

/** "Active test approval: WSTG-ATHN-01" or "Report acceptance: crapi". */
export function reviewTitle(
  decision: CrossProjectDecision,
  item: AuditItem | undefined,
  kindLabel: string,
): string {
  return `${kindLabel}: ${item === undefined ? decision.project.name : itemLabel(item)}`;
}

const RUN_STATE_LABELS: Readonly<Record<WorkflowRunState, VocabularyLabel>> = {
  initializing: { label: "Starting", tone: "progress" },
  pending: { label: "Queued", tone: "idle" },
  waiting: { label: "Waiting for the model", tone: "warning" },
  running: { label: "Running", tone: "progress" },
  cancelling: { label: "Cancelling", tone: "warning" },
  succeeded: { label: "Succeeded", tone: "done" },
  failed: { label: "Failed", tone: "blocked" },
  cancelled: { label: "Cancelled", tone: "neutral" },
};

/** Run state for people: "Failed", "Waiting for the model", … */
export function runStateLabel(state: WorkflowRunState): VocabularyLabel {
  return Object.hasOwn(RUN_STATE_LABELS, state)
    ? RUN_STATE_LABELS[state]
    : {
        label: capitalize(String(state).replaceAll("_", " ")),
        tone: "neutral",
      };
}

type RecoveryCode = NonNullable<RunStatus["recovery"]>["code"];

// The Run page's words for each recovery cause (routes/runs/recovery.tsx).
const RECOVERY_REASONS: Readonly<Record<RecoveryCode, string>> = {
  model_unavailable: "The model was unloaded or is unavailable.",
  gateway_unavailable: "The model gateway is unavailable.",
  gateway_timeout: "The model request timed out.",
  gateway_rate_limited: "The model gateway is rate limiting requests.",
};

/** Why a waiting Run waits, in the Run page's words. */
export function recoveryReason(code: RecoveryCode): string {
  return Object.hasOwn(RECOVERY_REASONS, code)
    ? RECOVERY_REASONS[code]
    : "The model connection is interrupted.";
}

export type OutputArtifact = NonNullable<OutputEntry["artifact"]>;

/**
 * A finished Run's primary result, from the Workflow's published `primary`
 * flag (UUS:102-103). Without a declared primary output nothing is picked in
 * its place, and a declared one the Run did not produce stays missing.
 */
export type PrimaryOutput =
  | { state: "loading" }
  | { state: "unavailable" }
  | { state: "none" }
  | { state: "missing"; slot: string }
  | { state: "present"; slot: string; artifact: OutputArtifact };

export function primaryOutput(
  run: Pick<RunSummary, "workflow">,
  status: RunStatus | undefined,
  statusFailed: boolean,
  /** Declared outputs by "name@version"; null when they cannot be read. */
  declarations: ReadonlyMap<string, Record<string, WorkflowOutputSlot> | null>,
): PrimaryOutput {
  const identity = parseWorkflowIdentity(run.workflow);
  if (identity === undefined || statusFailed) return { state: "unavailable" };
  const declared = declarations.get(`${identity.name}@${identity.version}`);
  if (declared === null) return { state: "unavailable" };
  if (status === undefined || declared === undefined)
    return { state: "loading" };
  const primaries = organizeRunOutputs(status.outputs, declared).filter(
    (entry) => entry.kind === "primary",
  );
  const present = primaries.find((entry) => entry.artifact !== undefined);
  if (present?.artifact !== undefined)
    return { state: "present", slot: present.slot, artifact: present.artifact };
  const missing = primaries[0];
  return missing === undefined
    ? { state: "none" }
    : { state: "missing", slot: missing.slot };
}

/** Text cut at a word boundary to at most `limit` characters. */
export function excerpt(text: string, limit: number): string {
  const flat = text.replace(/\s+/g, " ").trim();
  if (flat.length <= limit) return flat;
  const cut = flat.slice(0, limit - 1);
  const space = cut.lastIndexOf(" ");
  return `${(space > limit / 2 ? cut.slice(0, space) : cut).trimEnd()}…`;
}
