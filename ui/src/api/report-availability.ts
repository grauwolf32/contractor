import { getAuditReport, type AuditReport } from "./audits";
import type { PublicAPI } from "./client";
import { PublicAPIError } from "./error";
import { queryKeys } from "./query-keys";

/** Availability is shared across lists and pinned to the check revision. */
export function reportAvailabilityKey(auditId: string, revision: number) {
  return [...queryKeys.crossProject.reportOf(auditId), revision] as const;
}

/** Keep report bodies in the selected reader's cache only. */
export async function loadReportAvailability(
  api: PublicAPI,
  auditId: string,
  signal: AbortSignal,
): Promise<Pick<AuditReport, "status"> | null> {
  try {
    const report = await getAuditReport(api, auditId, signal);
    return { status: report.status };
  } catch (error) {
    // Deletion between reading the inventory and its report is ordinary absence.
    if (error instanceof PublicAPIError && error.status === 404) return null;
    throw error;
  }
}
