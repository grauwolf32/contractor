import type { FindingPreview } from "../api/finding-correlations";
import type { components } from "../api/generated/public";
import { audit, finding, conforming, DIGEST } from "./builders";

type CandidatesPage = components["schemas"]["FindingCandidatesPage"];
type MembersPage = components["schemas"]["FindingMembersPage"];

export const CORRELATION_GROUP = "a".repeat(64);
export const CORRELATION_PROJECT = "project-correlation";
export const correlationAudit = audit(
  CORRELATION_PROJECT,
  "audit-left",
  "completed",
);
export const correlationFinding = finding(
  "audit-left",
  "finding-left",
  {
    correlation: {
      schema: "contractor.finding.correlation.v1",
      policyVersion: "qualified-snapshot/v1",
      scope: { kind: "project", projectId: CORRELATION_PROJECT },
      projectRevision: 3,
      availability: "available",
      groupKey: CORRELATION_GROUP,
      basis: {
        algorithm: "contractor/producer-issue/v1",
        value: CORRELATION_GROUP,
      },
      matchedDomain: "snapshot",
      representative: { auditId: "audit-left", findingId: "finding-left" },
      automaticDuplicate: false,
      hasOtherObservations: true,
    },
  },
  { title: "Repeated source observation" },
);

export function observation(
  findingId: string,
  overrides: Partial<FindingPreview> = {},
): FindingPreview {
  return conforming("FindingComparisonPreview", {
    findingId,
    auditId: "audit-right",
    revision: 1,
    preview: true,
    title: `Observation ${findingId}`,
    titleTruncated: false,
    state: "rejected",
    severitySuggestion: "high",
    analystVerdict: "false_positive",
    origin: {
      receiptId: `receipt-${findingId}`,
      runId: "run-source",
      runDeleted: true,
      logicalAgentName: "source-reviewer",
      workflowName: "source-review",
      workflowVersion: "1",
      workflowClosureDigest: DIGEST,
    },
    ...overrides,
  });
}
export function members(items: FindingPreview[], more = false): MembersPage {
  return conforming("FindingMembersPage", {
    policyVersion: "qualified-snapshot/v1",
    scope: { kind: "project", projectId: CORRELATION_PROJECT, auditIds: [] },
    fence: { projectId: CORRELATION_PROJECT, projectRevision: 3, audits: [] },
    groupKey: CORRELATION_GROUP,
    representative: { auditId: "audit-left", findingId: "finding-left" },
    items,
    page: more
      ? { hasMore: true, nextCursor: "members-next" }
      : { hasMore: false },
  });
}
export function candidates(
  items: CandidatesPage["items"] = [],
  more = false,
): CandidatesPage {
  return conforming("FindingCandidatesPage", {
    policyVersion: "qualified-snapshot/v1",
    scope: { kind: "project", projectId: CORRELATION_PROJECT, auditIds: [] },
    fence: { projectId: CORRELATION_PROJECT, projectRevision: 3, audits: [] },
    source: {
      auditId: "audit-left",
      findingId: "finding-left",
      receiptId: correlationFinding.firstProposal.receiptId,
      firstReceiptId: correlationFinding.firstProposal.receiptId,
      hintsAvailability: "available",
      familiesAvailable: true,
    },
    targetBasis: "immutable-first-receipt",
    items,
    rowsExamined: more ? 897 : items.length,
    workLimited: more,
    page: more
      ? { hasMore: true, nextCursor: "search-next" }
      : { hasMore: false },
  });
}
export function provenance(
  more = false,
): components["schemas"]["AuditFindingProvenancePage"] {
  return conforming("AuditFindingProvenancePage", {
    auditRevision: 1,
    findingRevision: 1,
    items: more
      ? []
      : [
          {
            recordId: "record-verification",
            kind: "check-attempt",
            receiptId: correlationFinding.firstProposal.receiptId,
            relation: "verification",
            proposal: correlationFinding.firstProposal.proposal,
            origin: correlationFinding.firstProposal.origin,
            supportsCurrentAssessment: false,
            createdAt: correlationFinding.createdAt,
          },
          {
            recordId: "record-supporting",
            kind: "source-proposal",
            receiptId: "receipt-supporting",
            relation: "contributing",
            proposal: correlationFinding.firstProposal.proposal,
            origin: correlationFinding.firstProposal.origin,
            supportsCurrentAssessment: false,
            createdAt: correlationFinding.createdAt,
          },
        ],
    page: more
      ? { hasMore: true, nextCursor: "source-next" }
      : { hasMore: false },
  });
}
