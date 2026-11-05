// Response fixtures built on the generated types and checked against the
// generated schemas when they are built, so a test cannot rely on a
// response the Server would never send.
// Response fixtures built on the generated types and checked against the
// generated schemas when they are built, so a test cannot rely on a
// response the Server would never send.
// Response fixtures built on the generated types and checked against the
// generated schemas when they are built, so a test cannot rely on a
// response the Server would never send.
// Response fixtures built on the generated types and checked against the
// generated schemas when they are built, so a test cannot rely on a
// response the Server would never send.
// Response fixtures built on the generated types and checked against the
// generated schemas when they are built, so a test cannot rely on a
// response the Server would never send.
// Response fixtures built on the generated types and checked against the
// generated schemas when they are built, so a test cannot rely on a
// response the Server would never send.
// Response fixtures built on the generated types and checked against the
// generated schemas when they are built, so a test cannot rely on a
// response the Server would never send.
// Response fixtures built on the generated types and checked against the
// generated schemas when they are built, so a test cannot rely on a
// response the Server would never send.
import { schemaViolation } from "../api/contract";
import type { components } from "../api/generated/public";
import type { PublicSchemaName } from "../api/generated/response-schemas";

type Schemas = components["schemas"];

export const DIGEST = `sha256:${"a".repeat(64)}`;
export const NOW = "2026-10-05T10:00:00Z";
export const ARTIFACT_STORAGE_TIMESTAMPS = {
  scopeCreatedAt: NOW,
  blobCreatedAt: NOW,
  versionCreatedAt: NOW,
  bindingUpdatedAt: NOW,
};

/** `value`, after checking that it conforms to the public schema `name`. */
export function conforming<N extends PublicSchemaName & keyof Schemas>(
  name: N,
  value: Schemas[N],
): Schemas[N] {
  const violation = schemaViolation(name, value);
  if (violation !== undefined) {
    throw new Error(`${name} fixture breaks the contract at ${violation}`);
  }
  return value;
}

/** Overrides in which `undefined` removes the member. */
type Overrides<T> = { [K in keyof T]?: T[K] | undefined };

function merge<T extends object>(base: T, overrides: Overrides<T>): T {
  const result: Record<string, unknown> = { ...base, ...overrides };
  for (const [key, value] of Object.entries(result)) {
    if (value === undefined) delete result[key];
  }
  return result as T;
}

export function project(
  projectId: string,
  overrides: Overrides<Schemas["Project"]> = {},
): Schemas["Project"] {
  return conforming(
    "Project",
    merge(
      {
        projectId,
        kind: "project",
        name: projectId,
        description: "",
        lifecycle: "active",
        activeChecks: "automatic",
        revision: "1",
        createdAt: NOW,
        updatedAt: NOW,
      },
      overrides,
    ),
  );
}

/** A source archive an Audit pinned as an input. */
export function auditInput(
  name = "service-source",
  overrides: Overrides<Schemas["AuditExactArtifact"]> = {},
): Schemas["AuditExactArtifact"] {
  return conforming(
    "AuditExactArtifact",
    merge(
      {
        ref: { namespace: "sources", name, revision: "r1" },
        digest: DIGEST,
        mediaType: "application/zip",
        sizeBytes: 2048,
      },
      overrides,
    ),
  );
}

/**
 * An Audit in `state`: a draft has not started; any other state is in its
 * first round.
 */
export function audit(
  projectId: string,
  auditId: string,
  state: Schemas["AuditState"],
  overrides: Overrides<Schemas["Audit"]> = {},
): Schemas["Audit"] {
  const draft = state === "draft";
  return conforming(
    "Audit",
    merge<Schemas["Audit"]>(
      {
        auditId,
        projectId,
        profile: {
          name: "openapi-operation-trace",
          version: "1",
          digest: DIGEST,
        },
        inputs: { source: auditInput() },
        scope: {},
        runtimeLabels: [],
        state,
        taskCount: 0,
        iteration: 0,
        revision: 1,
        dispatchState: state === "active" ? "open" : "closed",
        holdState: draft ? "pending" : "held",
        limits: {
          maxRounds: 1,
          batchSize: 1,
          maxItemsPerRound: 8,
          maxItemsTotal: 8,
          maxSubmittedRuns: 8,
          maxItemRunAttempts: 2,
          maxEvidenceBytes: 1_048_576,
        },
        reservedRunCount: Math.max(
          overrides.submittedRunCount ?? 0,
          overrides.outstandingRunCount ?? 0,
        ),
        submittedRunCount: 0,
        outstandingRunCount: 0,
        collectionPending: false,
        retainedEvidenceBytes: 0,
        eventSequence: 1,
        createdAt: NOW,
        updatedAt: NOW,
      },
      overrides,
    ),
  );
}

/** One canonical, initially unassessed Project Finding. */
export function finding(
  projectId: string,
  findingId: string,
  overrides: Overrides<Schemas["ProjectFinding"]> = {},
  claim: Partial<Schemas["FindingClaim"]> = {},
): Schemas["ProjectFinding"] {
  const title = claim.title ?? "Missing object authorization";
  return conforming(
    "ProjectFinding",
    merge<Schemas["ProjectFinding"]>(
      {
        projectId,
        findingId,
        firstReceiptId: `receipt_${findingId}`,
        firstProposal: {
          ref: {
            namespace: "finding-materials",
            name: findingId,
            revision: "r1",
          },
          digest: DIGEST,
          mediaType: "application/json",
          sizeBytes: 512,
        },
        title,
        claim: {
          title,
          description:
            "The **order endpoint** may read another owner's record.\n\nIt never compares `ownerId` with the caller.",
          subject: null,
          preconditions: [],
          standard_refs: [],
          severity_suggestion: "high",
          limitations: [],
          ...claim,
        },
        state: null,
        analystVerdict: null,
        analystSeverity: null,
        duplicateTargetId: null,
        rationale: null,
        actorId: null,
        assessedAt: null,
        revision: 1,
        createdAt: NOW,
        updatedAt: NOW,
      },
      overrides,
    ),
  );
}

/** The compact list projection returned without hydrating any source. */
export function findingSummary(
  value: Schemas["ProjectFinding"],
): Schemas["ProjectFindingSummary"] {
  return conforming("ProjectFindingSummary", {
    findingId: value.findingId,
    projectId: value.projectId,
    title: value.title.slice(0, 256),
    titleTruncated: value.title.length > 256,
    subjectKind: value.claim?.subject?.kind ?? "",
    subjectKey: value.claim?.subject?.key ?? "",
    subjectTruncated: false,
    severitySuggestion: value.claim?.severity_suggestion ?? "",
    state: value.state,
    analystVerdict: value.analystVerdict,
    analystSeverity: value.analystSeverity,
    duplicateTargetId: value.duplicateTargetId,
    revision: value.revision,
    createdAt: value.createdAt,
    updatedAt: value.updatedAt,
  });
}

/** Immutable producer receipt retained by its Project. */
export function findingSource(
  value: Schemas["ProjectFinding"],
  overrides: Overrides<Schemas["FindingProposalReceipt"]> = {},
  document: Partial<Schemas["FindingProposalDocument"]> = {},
): Schemas["FindingProposalReceipt"] {
  return conforming(
    "FindingProposalReceipt",
    merge<Schemas["FindingProposalReceipt"]>(
      {
        receiptId: value.firstReceiptId,
        proposalId: `proposal_${value.findingId}`,
        requestDigest: DIGEST,
        clientKey: value.findingId,
        projectId: value.projectId,
        findingId: value.findingId,
        publicationRequested: true,
        proposal: value.firstProposal,
        projectProposal: value.firstProposal,
        document: {
          schema: "contractor.audit.finding-proposal.v1",
          client_key: value.findingId,
          title: value.title,
          description: "Original observation",
          subject: null,
          preconditions: [],
          standard_refs: [],
          evidence_ids: [],
          proposed_checks: [],
          severity_suggestion: "",
          limitations: [],
          ...value.claim,
          ...document,
        },
        evidence: [],
        projectEvidence: [],
        retention: "project-held",
        auditHolds: [],
        origin: {
          runId: `run_${value.findingId}`,
          stageExecutionId: "stage",
          allocationId: "allocation",
          invocationId: "invocation",
          logicalAgentName: "reviewer",
          workflow: {
            name: "source-review",
            version: "1",
            schemaVersion: "contractor/v1alpha1",
            configurationRef: { name: "source-review", version: "1" },
            closureDigest: DIGEST,
          },
          runDeleted: true,
        },
        createdAt: value.createdAt,
      },
      overrides,
    ),
  );
}
export function findingSourcePreview(
  source: Schemas["FindingProposalReceipt"],
): Schemas["ProjectFindingSourcePreview"] {
  return conforming("ProjectFindingSourcePreview", {
    receiptId: source.receiptId,
    proposalId: source.proposalId,
    projectId: source.projectId!,
    ...(source.findingId === undefined ? {} : { findingId: source.findingId }),
    proposal: source.projectProposal!,
    origin: source.origin,
    evidenceCount: source.projectEvidence?.length ?? 0,
    publicationRequested: source.publicationRequested,
    createdAt: source.createdAt,
  });
}

export const ALL_VERDICTS: Schemas["FindingVerdict"][] = [
  "true_positive",
  "false_positive",
  "duplicate",
  "reopen",
  "needs_evidence",
];

/** A proposed or ready report with both files; others carry only a status. */
export function report(
  auditId: string,
  status: Schemas["AuditReport"]["status"],
  options: {
    summary?: string | undefined;
  } = {},
): Schemas["AuditReport"] {
  if (status !== "ready") {
    return conforming("AuditReport", { status });
  }
  return conforming("AuditReport", {
    status,
    machineArtifact: {
      ref: {
        namespace: `audit-${auditId}`,
        name: "report.json",
        revision: "r1",
      },
      digest: `sha256:${"7".repeat(64)}`,
      mediaType: "application/json",
      sizeBytes: 64,
    },
    summaryArtifact: {
      ref: { namespace: `audit-${auditId}`, name: "report.md", revision: "r1" },
      digest: `sha256:${"8".repeat(64)}`,
      mediaType: "text/markdown",
      sizeBytes: 32,
    },
    machine: { conclusion: "completed-with-gaps", certification: false },
    summary: options.summary ?? `# Summary of ${auditId}\n\nTwo issues.`,
  });
}

/** The workspace summary of `value`, with zero counts unless given. */
export function workspace(
  value: Schemas["Audit"],
  counts: Partial<
    Pick<
      Schemas["AuditWorkspace"],
      | "totalTasks"
      | "completedTasks"
      | "gaps"
      | "pendingTasks"
      | "findings"
      | "unreviewedFindings"
    >
  > = {},
): Schemas["AuditWorkspace"] {
  return conforming("AuditWorkspace", {
    auditId: value.auditId,
    auditRevision: value.revision,
    asOf: "2026-10-05T10:40:00Z",

    iteration: 1,
    executionState: value.state,
    outstandingRuns: value.outstandingRunCount,
    totalTasks: 0,
    completedTasks: 0,
    gaps: 0,
    pendingTasks: 0,
    findings: 0,
    unreviewedFindings: 0,

    ...counts,
  });
}

/** The current revision `r1` of a 64-byte ZIP at `namespace/name`. */
export function artifactMetadata(
  namespace: string,
  name: string,
  overrides: Overrides<Schemas["ArtifactMetadata"]> = {},
): Schemas["ArtifactMetadata"] {
  return conforming(
    "ArtifactMetadata",
    merge<Schemas["ArtifactMetadata"]>(
      {
        storageTimestamps: ARTIFACT_STORAGE_TIMESTAMPS,
        canUploadNewVersion:
          overrides.current !== false && overrides.frozen !== true,
        artifact: { namespace, name, revision: "r1" },
        mediaType: "application/zip",
        size: 64,
        current: true,
        frozen: false,
        createdAt: NOW,
      },
      overrides,
    ),
  );
}

/** The write answer that stored `metadata`. */
export function artifactWrite(
  metadata: Schemas["ArtifactMetadata"],
): Schemas["ArtifactWriteResponse"] {
  return conforming("ArtifactWriteResponse", {
    artifact: metadata.artifact,
    mediaType: metadata.mediaType,
    size: metadata.size,
  });
}
