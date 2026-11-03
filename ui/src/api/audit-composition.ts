import { invalidAPIResponse } from "./error";
import type { components } from "./generated/public";
type Audit = components["schemas"]["Audit"];
/** Validate relationships between the common Audit counters. */
export function requireAuditComposition(audit: Audit, status: number): Audit {
  if (
    audit.submittedRunCount > audit.reservedRunCount ||
    audit.outstandingRunCount > audit.reservedRunCount ||
    audit.taskCount > audit.limits.maxItemsTotal
  ) {
    throw invalidAPIResponse(
      status,
      "Server returned inconsistent Audit counters",
    );
  }
  return audit;
}
