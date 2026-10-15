// Fixtures for the decision component tests, built on the shared,
// schema-checked builders in src/test.
import type {
  AuditFinding,
  AuditReport,
  AuditReviewRequest,
} from "../../api/audits";
import * as build from "../../test/builders";
import type { AuditReviewDecision } from "./model";

export const AUDIT_ID = "audit_1";
export const FINDING_ID = "finding_1";
export const { ALL_VERDICTS, DIGEST, NOW } = build;

/** A possible issue of AUDIT_ID at revision 3 unless overridden. */
export function makeFinding(
  overrides: Partial<AuditFinding> = {},
  document: Partial<AuditFinding["firstProposal"]["document"]> = {},
): AuditFinding {
  return build.finding(
    overrides.auditId ?? AUDIT_ID,
    overrides.findingId ?? FINDING_ID,
    { revision: 3, ...overrides },
    document,
  );
}

/** A pending triage request for FINDING_ID. */
export function makeTriageReview(
  overrides: Partial<AuditReviewRequest> = {},
): AuditReviewRequest {
  return build.review(AUDIT_ID, "review_open", "finding-triage", {
    findingId: FINDING_ID,
    subjectId: FINDING_ID,
    subjectRevision: 3,
    ...overrides,
  });
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
  return build.conforming("AuditReviewDecision", {
    decisionId: "decision_1",
    requestId: "review_open",
    auditId: AUDIT_ID,
    findingId: FINDING_ID,
    actorId: "user_analyst",
    verdict: "true_positive",
    severity: "high",
    rationale: "Confirmed against the **retained** source.",
    subjectRevision: 3,
    subjectDigest: DIGEST,
    createdAt: NOW,
    ...overrides,
  });
}

export function proposedReport(review: AuditReviewRequest): AuditReport {
  return build.conforming("AuditReport", {
    status: "proposed",
    review,
    summary: "# Report",
    summaryArtifact: {
      ref: { namespace: "audit-reports", name: "summary", revision: "r1" },
      digest: DIGEST,
      mediaType: "text/markdown",
    },
  });
}
