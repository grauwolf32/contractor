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
  followUp: number;
  unchecked: number;
}

export function workCounts(workspace: AuditWorkspace): WorkCounts {
  return {
    total: workspace.totalTasks,
    done: workspace.completedTasks,
    followUp: workspace.gaps,
    unchecked: workspace.pendingTasks,
  };
}

/** Technical Task progress. Findings and hypothesis verdicts are separate. */
export function countLegend(counts: WorkCounts): LegendEntry[] {
  return [
    // Done also holds not applicable work: no one filter.
    { label: "Done", tone: "done", group: "all", count: counts.done },
    {
      label: "Gaps",
      tone: "partial",
      group: "uncertain",
      count: counts.followUp,
    },
    {
      label: "Pending",
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
