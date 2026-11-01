// Fixtures for the decision component tests, built on the shared,
// schema-checked builders in src/test.
import type {
  ProjectFinding,
  AuditReport,
  AuditReviewRequest,
} from "../../api/audits";
import * as build from "../../test/builders";
import type { AuditReviewDecision } from "./model";

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

/** A pending active test approval for item_1. */
export function makeActionReview(
  overrides: Partial<AuditReviewRequest> = {},
): AuditReviewRequest {
  return build.review(AUDIT_ID, "review_action", "active-check-approval", {
    subjectId: "item_1",
    revision: 2,
    ...overrides,
  });
}

/** A true-positive, high-severity decision on the triage request. */
export function makeDecision(
  overrides: Partial<AuditReviewDecision> = {},
): AuditReviewDecision {
  return build.decision(makeActionReview(), {
    action: "approve",
    ...overrides,
  });
}

export function proposedReport(review: AuditReviewRequest): AuditReport {
  return build.report(AUDIT_ID, "proposed", { review, summary: "# Report" });
}
