/**
 * What the Inbox lists and in which order (docs/design/ui/v3b-build-contract.md
 * §5, UUS:26-27): decisions first, then blocked work, finished results and
 * running checks. Everything here is pure, so the sections can be tested
 * without a Server.
 */
import type { AuditState, AuditWorkspace } from "../../api/audits";
import type {
  CrossProjectCheck,
  CrossProjectDecision,
  CrossProjectIssue,
  CrossProjectReport,
} from "../../api/cross-project";
import type { RunStatus, RunSummary } from "../../api/runs";

/** How far back failed and finished work counts as new: 7 days. */
export const RECENT_MS = 7 * 24 * 60 * 60 * 1000;
/** Rows per kind in Unblock and Ready before "See all …". */
export const SHOWN_PER_KIND = 5;

export type InboxSectionId = "decide" | "unblock" | "ready" | "running";

/** The selected item, as `?item=` holds it. */
export type InboxRef =
  | { kind: "issue"; auditId: string; findingId: string }
  | { kind: "review"; auditId: string; requestId: string }
  | { kind: "run"; runId: string }
  | { kind: "check"; auditId: string }
  | { kind: "report"; auditId: string };

/** Why a check is listed. */
export type CheckReason = "paused" | "failed" | "gaps" | "running" | "finished";

/** Why a Run is listed. */
export type RunReason = "model" | "failed" | "finished";

interface RowBase {
  /** `refKey(ref)`; a check can show in two sections under one key. */
  key: string;
  ref: InboxRef;
  section: InboxSectionId;
}

export type InboxRow =
  | (RowBase & { type: "issue"; issue: CrossProjectIssue })
  | (RowBase & { type: "review"; decision: CrossProjectDecision })
  | (RowBase & {
      type: "check";
      check: CrossProjectCheck;
      reason: CheckReason;
      workspace: AuditWorkspace | undefined;
    })
  | (RowBase & { type: "report"; report: CrossProjectReport })
  | (RowBase & {
      type: "run";
      run: RunSummary;
      reason: RunReason;
      status: RunStatus | undefined;
    });

export type InboxRowOf<T extends InboxRow["type"]> = Extract<
  InboxRow,
  { type: T }
>;

/** A list cut to SHOWN_PER_KIND rows; `hidden` more matched. */
export interface Shown<T> {
  shown: T[];
  hidden: number;
}

export interface InboxInput {
  /** Milliseconds since the epoch; the 7-day window ends here. */
  now: number;
  /** Possible issues that need a review, newest first. */
  issues: readonly CrossProjectIssue[];
  /** Pending approvals, applicability and report acceptance, newest first. */
  decisions: readonly CrossProjectDecision[];
  /** Listed checks, newest first. */
  checks: readonly CrossProjectCheck[];
  /** Ready reports, newest check first. */
  reports: readonly CrossProjectReport[];
  /** Workspace counters by check ID. */
  workspaces: ReadonlyMap<string, AuditWorkspace>;
  /** Failed Runs of the last 7 days (see `recentRuns`). */
  failedRuns: Shown<RunSummary>;
  /** Succeeded Runs of the last 7 days. */
  succeededRuns: Shown<RunSummary>;
  /** Runs waiting for the model whose status has been read. */
  waitingRuns: readonly RunSummary[];
  /** Run status by Run ID, where it has been read. */
  runStatuses: ReadonlyMap<string, RunStatus>;
}

export interface InboxModel {
  decide: InboxRow[];
  unblock: InboxRow[];
  ready: InboxRow[];
  running: InboxRow[];
  /** Every item once, in display order: what J and K move through. */
  order: InboxRow[];
  /** Matching items left out of Ready (reports, finished checks). */
  hiddenReports: number;
  hiddenFinishedChecks: number;
}

/** Checks the Inbox lists in one of its sections. */
export const LISTED_CHECK_STATES: readonly AuditState[] = [
  "active",
  "waiting_review",
  "paused",
  "finalizing",
  "completed",
  "failed",
];

/** Checks whose stopped items show in Unblock (the Inbox check states). */
const GAP_CHECK_STATES: ReadonlySet<AuditState> = new Set([
  "active",
  "waiting_review",
]);

/** Checks in Running: working or writing their report. */
export const RUNNING_CHECK_STATES: ReadonlySet<AuditState> = new Set([
  "active",
  "finalizing",
]);

/** Checks whose workspace counters the list shows. */
export const COUNTED_CHECK_STATES: ReadonlySet<AuditState> = new Set([
  "active",
  "waiting_review",
  "finalizing",
  "paused",
]);

// Server IDs (ResourceId); anything else in `?item=` is ignored.
const RESOURCE_ID = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$/;

function idsOf(ref: InboxRef): string[] {
  switch (ref.kind) {
    case "issue":
      return [ref.auditId, ref.findingId];
    case "review":
      return [ref.auditId, ref.requestId];
    case "run":
      return [ref.runId];
    case "check":
    case "report":
      return [ref.auditId];
  }
}

/**
 * "issue:<auditId>:<findingId>", "review:<auditId>:<requestId>",
 * "run:<runId>", "check:<auditId>" or "report:<auditId>". IDs may contain
 * ":", so each one is percent-encoded.
 */
export function refKey(ref: InboxRef): string {
  return [ref.kind, ...idsOf(ref).map(encodeURIComponent)].join(":");
}

/** The item a `?item=` value names; undefined for anything malformed. */
export function parseRefKey(
  value: string | null | undefined,
): InboxRef | undefined {
  if (value === null || value === undefined || value === "") return undefined;
  const [kind, ...parts] = value.split(":");
  let ids: string[];
  try {
    ids = parts.map((part) => decodeURIComponent(part));
  } catch {
    return undefined;
  }
  if (ids.some((id) => !RESOURCE_ID.test(id))) return undefined;
  const [first, second] = ids;
  if (first === undefined) return undefined;
  switch (kind) {
    case "issue":
      return ids.length === 2 && second !== undefined
        ? { kind, auditId: first, findingId: second }
        : undefined;
    case "review":
      return ids.length === 2 && second !== undefined
        ? { kind, auditId: first, requestId: second }
        : undefined;
    case "run":
      return ids.length === 1 ? { kind, runId: first } : undefined;
    case "check":
    case "report":
      return ids.length === 1 ? { kind, auditId: first } : undefined;
    default:
      return undefined;
  }
}

/**
 * The Inbox URL's search for an item: "?item=issue:a:f". The key holds only
 * ID characters and percent escapes, so only "%" needs escaping here.
 */
export function inboxSearch(ref: InboxRef): string {
  return `?item=${refKey(ref).replaceAll("%", "%25")}`;
}

function time(value: string | undefined): number {
  if (value === undefined) return Number.NEGATIVE_INFINITY;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : Number.NEGATIVE_INFINITY;
}

/** When a Run ended, or last changed while it runs. */
export function runTime(run: Pick<RunSummary, "finishedAt" | "updatedAt">) {
  return run.finishedAt ?? run.updatedAt;
}

/** When a check ended, or last changed while it runs. */
export function checkTime(check: CrossProjectCheck): string {
  return check.audit.finishedAt ?? check.audit.updatedAt;
}

/** Ended (or changed) in the last 7 days; a clock ahead of ours counts too. */
function isRecent(value: string | undefined, now: number): boolean {
  return time(value) >= now - RECENT_MS;
}

/**
 * Runs of one state that ended in the last 7 days, newest first, cut to
 * SHOWN_PER_KIND. The Server's first page is the newest Runs.
 */
export function recentRuns(
  runs: readonly RunSummary[],
  now: number,
): Shown<RunSummary> {
  const recent = runs
    .filter((run) => isRecent(runTime(run), now))
    .sort(
      (left, right) =>
        time(runTime(right)) - time(runTime(left)) ||
        (left.runId < right.runId ? -1 : left.runId > right.runId ? 1 : 0),
    );
  return {
    shown: recent.slice(0, SHOWN_PER_KIND),
    hidden: Math.max(0, recent.length - SHOWN_PER_KIND),
  };
}

/** Active Runs waiting for the model, newest first, at most SHOWN_PER_KIND. */
export function waitingRuns(runs: readonly RunSummary[]): RunSummary[] {
  return runs
    .filter((run) => run.state === "waiting")
    .sort((left, right) => time(right.updatedAt) - time(left.updatedAt))
    .slice(0, SHOWN_PER_KIND);
}

/** The model connection needs the user: automatic recovery has ended. */
export function needsModelRetry(status: RunStatus | undefined): boolean {
  return status?.recovery?.requiresRetry === true;
}

function cut<T>(items: readonly T[]): Shown<T> {
  return {
    shown: items.slice(0, SHOWN_PER_KIND),
    hidden: Math.max(0, items.length - SHOWN_PER_KIND),
  };
}

function checkRow(
  section: InboxSectionId,
  check: CrossProjectCheck,
  reason: CheckReason,
  workspaces: ReadonlyMap<string, AuditWorkspace>,
): InboxRow {
  const ref: InboxRef = { kind: "check", auditId: check.audit.auditId };
  return {
    key: refKey(ref),
    ref,
    section,
    type: "check",
    check,
    reason,
    workspace: workspaces.get(check.audit.auditId),
  };
}

function runRow(
  section: InboxSectionId,
  run: RunSummary,
  reason: RunReason,
  statuses: ReadonlyMap<string, RunStatus>,
): InboxRow {
  const ref: InboxRef = { kind: "run", runId: run.runId };
  return {
    key: refKey(ref),
    ref,
    section,
    type: "run",
    run,
    reason,
    status: statuses.get(run.runId),
  };
}

/** The four sections and the keyboard order over them. */
export function buildInbox(input: InboxInput): InboxModel {
  const { now, workspaces, runStatuses } = input;

  const decide: InboxRow[] = [
    ...input.issues.map((issue): InboxRow => {
      const ref: InboxRef = {
        kind: "issue",
        auditId: issue.audit.auditId,
        findingId: issue.finding.findingId,
      };
      return { key: refKey(ref), ref, section: "decide", type: "issue", issue };
    }),
    ...input.decisions.map((decision): InboxRow => {
      const ref: InboxRef = {
        kind: "review",
        auditId: decision.audit.auditId,
        requestId: decision.review.requestId,
      };
      return {
        key: refKey(ref),
        ref,
        section: "decide",
        type: "review",
        decision,
      };
    }),
  ];

  const checks = input.checks;
  const unblock: InboxRow[] = [
    ...input.waitingRuns
      .filter((run) => needsModelRetry(runStatuses.get(run.runId)))
      .map((run) => runRow("unblock", run, "model", runStatuses)),
    ...checks
      .filter((check) => check.audit.state === "paused")
      .map((check) => checkRow("unblock", check, "paused", workspaces)),
    ...checks
      .filter(
        (check) =>
          GAP_CHECK_STATES.has(check.audit.state) &&
          (workspaces.get(check.audit.auditId)?.gaps ?? 0) > 0,
      )
      .map((check) => checkRow("unblock", check, "gaps", workspaces)),
    ...checks
      .filter(
        (check) =>
          check.audit.state === "failed" && isRecent(checkTime(check), now),
      )
      .map((check) => checkRow("unblock", check, "failed", workspaces)),
    ...input.failedRuns.shown.map((run) =>
      runRow("unblock", run, "failed", runStatuses),
    ),
  ];

  const recentReports = input.reports.filter((entry) =>
    isRecent(checkTime(entry), now),
  );
  const reported = new Set(recentReports.map((entry) => entry.audit.auditId));
  const reports = cut(recentReports);
  const finished = cut(
    checks.filter(
      (check) =>
        check.audit.state === "completed" &&
        !reported.has(check.audit.auditId) &&
        isRecent(checkTime(check), now),
    ),
  );
  const ready: InboxRow[] = [
    ...reports.shown.map((report): InboxRow => {
      const ref: InboxRef = { kind: "report", auditId: report.audit.auditId };
      return {
        key: refKey(ref),
        ref,
        section: "ready",
        type: "report",
        report,
      };
    }),
    ...finished.shown.map((check) =>
      checkRow("ready", check, "finished", workspaces),
    ),
    ...input.succeededRuns.shown.map((run) =>
      runRow("ready", run, "finished", runStatuses),
    ),
  ];

  const running: InboxRow[] = checks
    .filter((check) => RUNNING_CHECK_STATES.has(check.audit.state))
    .map((check) => checkRow("running", check, "running", workspaces));

  const seen = new Set<string>();
  const order: InboxRow[] = [];
  for (const row of [...decide, ...unblock, ...ready, ...running]) {
    if (seen.has(row.key)) continue;
    seen.add(row.key);
    order.push(row);
  }
  return {
    decide,
    unblock,
    ready,
    running,
    order,
    hiddenReports: reports.hidden,
    hiddenFinishedChecks: finished.hidden,
  };
}

/** Distinct items of rows; a check listed twice counts once. */
export function distinctCount(rows: readonly InboxRow[]): number {
  return new Set(rows.map((row) => row.key)).size;
}

export interface SubtitleCounts {
  /** Items that need a decision; the rail badge's number. */
  decide: number;
  unblock: number;
  ready: number;
  /** Running checks. */
  running: number;
}

function phrase(count: number, one: string, many: string): string {
  return `${count.toLocaleString("en-US")} ${count === 1 ? one : many}`;
}

/** "2 need you, 1 is stuck, 2 are ready, 1 check is running." */
export function inboxSubtitle(counts: SubtitleCounts): string {
  const parts = [
    counts.decide > 0 ? phrase(counts.decide, "needs you", "need you") : "",
    counts.unblock > 0 ? phrase(counts.unblock, "is stuck", "are stuck") : "",
    counts.ready > 0 ? phrase(counts.ready, "is ready", "are ready") : "",
    counts.running > 0
      ? phrase(counts.running, "check is running", "checks are running")
      : "",
  ].filter((part) => part !== "");
  return parts.length === 0
    ? "Nothing needs you right now."
    : `${parts.join(", ")}.`;
}

/**
 * The next item to decide after `key` leaves the list: the one after it, or
 * the one before it when it was the last; undefined when none is left.
 */
export function nextToDecide(
  decide: readonly InboxRow[],
  key: string,
): InboxRow | undefined {
  const index = decide.findIndex((row) => row.key === key);
  if (index < 0) return decide[0];
  return decide[index + 1] ?? decide[index - 1];
}
