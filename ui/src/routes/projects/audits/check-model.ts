import type { AuditWorkspace } from "../../../api/audits";
import { formatNumber } from "../../../app/format";

/** Coordination counters do not imply semantic check outcomes. */
export function coordinationSummary(workspace: AuditWorkspace): string {
  return `${formatNumber(workspace.totalTasks)} tasks · ${formatNumber(workspace.outstandingRuns)} runs in progress`;
}
