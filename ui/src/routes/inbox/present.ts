/**
 * Words, paths and progress lines for Inbox rows and details. Labels come
 * from the shared vocabulary (src/app/vocabulary.ts); nothing here reads the
 * Server.
 */
import type { Audit, AuditWorkspace } from "../../api/audits";
import type { CrossProjectCheck } from "../../api/cross-project";
import type { RunStatus, RunSummary } from "../../api/runs";
import { coordinationSummary } from "../projects/audits/check-model";
import { auditPresetLabel } from "../projects/audits/labels";
import type { InboxRow, InboxSectionId } from "./model";
import {
  organizeRunOutputs,
  parseWorkflowIdentity,
  type OutputEntry,
  type WorkflowOutputSlot,
} from "../runs/output-model";
import { checkPath } from "../projects/audits/check-links";

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

/** Coordination counts never infer check conclusions from Run outcomes. */
export function checkProgress(workspace: AuditWorkspace) {
  return {
    total: workspace.totalTasks,
    summary: coordinationSummary(workspace),
  };
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
