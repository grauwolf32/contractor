/**
 * The check page's view of its work items: coverage rows joined with the
 * item collection (kind, execution state, attempts) and the check's possible
 * issues. Pure functions, so the list, the item view and the progress header
 * read the same model.
 */
import type { AuditWorkspace } from "../../../api/audits";
import { formatNumber } from "../../../app/format";
import type { StatusTone } from "../../../app/status-tone";
import { type Group } from "./assessments";

/** One legend entry: a status, its tone, how many items have it. */
export interface LegendEntry {
  label: string;
  tone: StatusTone;
  /** The result filter its link opens; "all" when no one filter holds it. */
  group: Group;
  count: number;
}

/** "1 partially traced, 2 not checked yet" for an accessible summary. */
export function legendSentence(entries: readonly LegendEntry[]): string {
  return entries
    .map((entry) => `${entry.count} ${entry.label.toLocaleLowerCase()}`)
    .join(", ");
}

/** Work counts of a check, from its workspace snapshot. */
interface WorkCounts {
  total: number;
  done: number;
  issues: number;
  followUp: number;
  unchecked: number;
}

export function workCounts(workspace: AuditWorkspace): WorkCounts {
  return {
    total: workspace.totalChecks,
    done: Math.max(0, workspace.completedChecks - workspace.issues),
    issues: workspace.issues,
    followUp: workspace.gaps,
    unchecked: workspace.unchecked,
  };
}

/**
 * Legend of a check whose items are not listed: concluded work (met, traced,
 * not applicable), issues found, work that needs follow-up (including
 * excluded items) and unchecked work.
 */
export function countLegend(counts: WorkCounts): LegendEntry[] {
  return [
    // Done also holds not applicable work: no one filter.
    { label: "Done", tone: "done", group: "all", count: counts.done },
    {
      label: "Issues found",
      tone: "blocked",
      group: "issues",
      count: counts.issues,
    },
    {
      label: "Need follow-up",
      tone: "partial",
      group: "uncertain",
      count: counts.followUp,
    },
    {
      label: "Not checked yet",
      tone: "idle",
      group: "not-tested",
      count: counts.unchecked,
    },
  ];
}

/** One segment per counted item, in legend order. */
export function countSegments(
  entries: readonly LegendEntry[],
): { tone: StatusTone; label: string }[] {
  return entries.flatMap((entry) =>
    Array.from({ length: Math.max(0, entry.count) }, () => ({
      tone: entry.tone,
      label: entry.label,
    })),
  );
}

/** "3 of 10 done", from the Server's counts when it has them. */
export function doneSummary(done: number, total: number): string {
  return `${formatNumber(done)} of ${formatNumber(total)} done`;
}
