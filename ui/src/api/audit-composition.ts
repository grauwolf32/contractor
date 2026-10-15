import { invalidAPIResponse } from "./error";
import type { components } from "./generated/public";

type Audit = components["schemas"]["Audit"];

/**
 * Returns `audit` after checking the preparation invariants its schema cannot
 * express: a role never used more attempts than it allows, an accepted role's
 * outputs come from its own execution and Run, and an Audit past preparation
 * has accepted every role.
 */
export function requireAuditComposition(audit: Audit, status: number): Audit {
  const roles = Object.values(audit.preparation?.roles ?? {});
  const prepared = audit.phase === "inventory" || audit.phase === "rounds";
  if (
    roles.some(
      (role) =>
        role.attempts > role.maxAttempts ||
        (prepared && role.status !== "accepted") ||
        Object.values(role.outputs).some(
          (output) =>
            output.executionId !== role.executionId ||
            output.runId !== role.runId,
        ),
    )
  ) {
    throw invalidAPIResponse(
      status,
      "Server returned an Audit whose preparation is inconsistent",
    );
  }
  return audit;
}
