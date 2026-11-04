// Fixtures for the decision component tests, built on the shared,
// schema-checked builders in src/test.
// Fixtures for the decision component tests, built on the shared,
// schema-checked builders in src/test.
import type { ProjectFinding } from "../../api/audits";
import * as build from "../../test/builders";

export const AUDIT_ID = "audit_1";
export const FINDING_ID = "finding_1";
export const { DIGEST, NOW } = build;

/** A possible issue of AUDIT_ID at revision 3 unless overridden. */
export function makeFinding(
  overrides: Partial<ProjectFinding> = {},
  document: Partial<NonNullable<ProjectFinding["claim"]>> = {},
): ProjectFinding {
  return build.finding(
    overrides.projectId ?? "project_1",
    overrides.findingId ?? FINDING_ID,
    { revision: 3, ...overrides },
    document,
  );
}
