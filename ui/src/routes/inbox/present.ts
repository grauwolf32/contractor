/**
 * Words, paths and progress lines for Inbox rows and details. Labels come
 * from the shared vocabulary (src/app/vocabulary.ts); nothing here reads the
 * Server.
 */
import type { Audit, AuditWorkspace } from "../../api/audits";
import type { CrossProjectCheck } from "../../api/cross-project";
import type { RunStatus, RunSummary } from "../../api/runs";
import { formatNumber } from "../../app/format";
import type { ProgressSegment } from "../../ui";
import { auditPresetLabel } from "../../modules/checks/labels";
import type { InboxRow, InboxSectionId } from "./model";
import {
  organizeRunOutputs,
  parseWorkflowIdentity,
  type OutputEntry,
  type WorkflowOutputSlot,
} from "../runs/output-model";
import { checkPath } from "../../modules/checks/links";

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

/** The title of the section a row is listed in, or `fallback`. */
export function sectionTitle(
  row: InboxRow | undefined,
  fallback: string,
): string {
  return row === undefined ? fallback : SECTIONS[row.section].title;
}

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
      return issuePath(
        row.issue.finding.projectId,
        row.issue.finding.findingId,
      );
    case "check":
      return checkPath(row.check.project.projectId, row.check.audit.auditId);
    case "report":
      return reportPath(row.report.audit.auditId);
    case "run":
      return runPath(row.run.runId);
  }
}

/** Whole-check Task progress from its workspace counters. */
interface CheckProgress {
  total: number;
  /** Tasks settled without a technical gap. */
  done: number;
  /** Tasks settled with a technical gap. */
  gaps: number;
  unchecked: number;
  segments: ProgressSegment[];
  /** "3 of 5 tasks done: 1 with gaps, 1 pending". */
  label: string;
  /** "3 of 5 tasks done". */
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
  followUp: "Gaps",
  unchecked: "Pending",
} as const;

/** Task progress from one consistent Server workspace snapshot. */
export function checkProgress(workspace: AuditWorkspace): CheckProgress {
  const total = count(workspace.totalTasks);
  const done = Math.min(count(workspace.completedTasks), total);
  const gaps = Math.min(count(workspace.gaps), total - done);
  const unchecked = Math.max(0, total - done - gaps);
  const segments = [
    ...repeat({ tone: "done", label: PROGRESS_LABELS.done }, done),
    ...repeat({ tone: "partial", label: PROGRESS_LABELS.followUp }, gaps),
    ...repeat(
      {
        tone: "idle" as const,
        label: PROGRESS_LABELS.unchecked,
      },
      unchecked,
    ),
  ];
  const summary =
    total === 0
      ? "No tasks listed yet"
      : `${formatNumber(done)} of ${formatNumber(total)} ${total === 1 ? "task" : "tasks"} done`;
  const details = [
    gaps > 0 ? `${formatNumber(gaps)} with gaps` : "",
    unchecked > 0 ? `${formatNumber(unchecked)} pending` : "",
  ].filter((part) => part !== "");
  return {
    total,
    done,
    gaps,
    unchecked,
    segments,
    summary,
    label: details.length === 0 ? summary : `${summary}: ${details.join(", ")}`,
  };
}

/** Technical gaps in accepted Tasks, including preparation and generation. */
export function gapsTitle(gaps: number): string {
  return `${formatNumber(gaps)} ${gaps === 1 ? "task" : "tasks"} with gaps`;
}

type OutputArtifact = NonNullable<OutputEntry["artifact"]>;

/**
 * A finished Run's primary result, from the Workflow's published `primary`
 * flag (UUS:102-103). Without a declared primary output nothing is picked in
 * its place, and a declared one the Run did not produce stays missing.
 */
type PrimaryOutput =
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
