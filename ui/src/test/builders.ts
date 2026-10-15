// Response fixtures built on the generated types and checked against the
// generated schemas when they are built, so a test cannot rely on a
// response the Server would never send.
import { schemaViolation } from "../api/contract";
import type { PublicSchemaName } from "../api/generated/response-schemas";
import type { components } from "../api/generated/public";

type Schemas = components["schemas"];

export const DIGEST = `sha256:${"a".repeat(64)}`;
export const NOW = "2026-10-05T10:00:00Z";

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
        ...(draft
          ? { phase: "not-started" as const }
          : { phase: "rounds" as const, currentRoundId: "round_1" }),
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
        reservedRunCount: 0,
        submittedRunCount: 0,
        outstandingRunCount: 0,
        retainedEvidenceBytes: 0,
        eventSequence: 1,
        createdAt: NOW,
        updatedAt: NOW,
      },
      overrides,
    ),
  );
}

/** A possible issue proposed by one Run. */
export function finding(
  auditId: string,
  findingId: string,
  overrides: Overrides<Schemas["AuditFinding"]> = {},
  document: Partial<Schemas["AuditFinding"]["firstProposal"]["document"]> = {},
): Schemas["AuditFinding"] {
  return conforming(
    "AuditFinding",
    merge<Schemas["AuditFinding"]>(
      {
        findingId,
        auditId,
        state: "proposed",
        firstProposal: {
          receiptId: `receipt_${findingId}`,
          proposalId: `proposal_${findingId}`,
          requestDigest: DIGEST,
          clientKey: findingId,
          proposal: {
            ref: {
              namespace: "audit-findings",
              name: findingId,
              revision: "r1",
            },
            digest: DIGEST,
            mediaType: "application/json",
            sizeBytes: 512,
          },
          document: {
            schema: "contractor.audit.finding-proposal.v1",
            client_key: findingId,
            title: "Missing object authorization",
            description:
              "The **order endpoint** may read another owner's record.\n\nIt never compares `ownerId` with the caller.",
            subject: null,
            preconditions: [],
            standard_refs: [],
            evidence_ids: [],
            proposed_checks: [],
            severity_suggestion: "high",
            limitations: [],
            ...document,
          },
          evidence: [],
          origin: {
            runId: "run_1",
            stageExecutionId: "stage_1",
            allocationId: "allocation_1",
            invocationId: "invocation_1",
            logicalAgentName: "reviewer",
            workflow: {
              name: "source-review",
              version: "1",
              schemaVersion: "contractor/v1alpha1",
              configurationRef: { name: "source-review", version: "1" },
              closureDigest: DIGEST,
            },
            runDeleted: false,
          },
          retention: "audit-held",
          auditHolds: [],
          createdAt: NOW,
        },
        revision: 1,
        createdAt: NOW,
        updatedAt: NOW,
      },
      overrides,
    ),
  );
}

export const ALL_VERDICTS: Schemas["AuditReviewRequest"]["requestedActions"] = [
  "true_positive",
  "false_positive",
  "duplicate",
  "reopen",
  "needs_evidence",
];

/** A pending review request of `kind`. */
export function review(
  auditId: string,
  requestId: string,
  kind: Schemas["AuditReviewRequest"]["kind"],
  overrides: Overrides<Schemas["AuditReviewRequest"]> = {},
): Schemas["AuditReviewRequest"] {
  const subject = {
    "finding-triage": {
      subjectKind: "finding",
      requestedActions: ALL_VERDICTS,
    },
    "active-check-approval": {
      subjectKind: "audit-item-action",
      requestedActions: ["approve", "reject"],
    },
    "requirement-applicability": {
      subjectKind: "audit-item-action",
      requestedActions: ["approve", "reject", "not_applicable"],
    },
    "report-acceptance": {
      subjectKind: "audit-report",
      requestedActions: ["approve", "reject"],
    },
  } as const;
  return conforming(
    "AuditReviewRequest",
    merge<Schemas["AuditReviewRequest"]>(
      {
        requestId,
        auditId,
        subjectKind: subject[kind].subjectKind,
        subjectId: kind === "report-acceptance" ? auditId : "subject_1",
        kind,
        subjectRevision: 1,
        subjectDigest: DIGEST,
        requestedActions: [...subject[kind].requestedActions],
        state: "pending",
        revision: 1,
        createdAt: NOW,
        updatedAt: NOW,
      },
      overrides,
    ),
  );
}

/** The decision recorded for `request`. */
export function decision(
  request: Schemas["AuditReviewRequest"],
  overrides: Overrides<Schemas["AuditReviewDecision"]> = {},
): Schemas["AuditReviewDecision"] {
  return conforming(
    "AuditReviewDecision",
    merge<Schemas["AuditReviewDecision"]>(
      {
        decisionId: `decision_${request.requestId}`,
        requestId: request.requestId,
        auditId: request.auditId,
        ...(request.findingId === undefined
          ? {}
          : { findingId: request.findingId }),
        actorId: "user_analyst",
        rationale: "Confirmed against the **retained** source.",
        subjectRevision: request.subjectRevision,
        subjectDigest: request.subjectDigest,
        createdAt: NOW,
      },
      overrides,
    ),
  );
}

/** A proposed or ready report with both files; others carry only a status. */
export function report(
  auditId: string,
  status: Schemas["AuditReport"]["status"],
  options: {
    summary?: string | undefined;
    review?: Schemas["AuditReviewRequest"] | undefined;
  } = {},
): Schemas["AuditReport"] {
  if (status !== "proposed" && status !== "ready") {
    return conforming("AuditReport", { status });
  }
  return conforming("AuditReport", {
    status,
    ...(status === "proposed"
      ? {
          review:
            options.review ??
            review(auditId, `review_${auditId}`, "report-acceptance"),
        }
      : {}),
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

type Attempt = Schemas["AuditItemAttempt"];

/**
 * One attempt of `itemId`, settled with an accepted result unless
 * overridden; an override of `undefined` removes the member (a running
 * attempt has no outcome, disposition or collection time).
 */
export function itemAttempt(
  itemId: string,
  n: number,
  overrides: Overrides<Attempt> = {},
): Attempt {
  return conforming(
    "AuditItemAttempt",
    merge<Attempt>(
      {
        executionItemId: `exec_${itemId}_${n}`,
        executionId: "round_1",
        itemId,
        itemAttempt: n,
        role: "check",
        state: "settled",
        terminalOutcome: "succeeded",
        collectionDisposition: "accepted-result",
        runId: `run_${itemId}_${n}`,
        runDeleted: false,
        createdAt: `2026-10-05T10:0${n + 1}:00Z`,
        collectedAt: `2026-10-05T10:0${n + 2}:00Z`,
      },
      overrides,
    ),
  );
}

/** A work item of the first Round, settled without attempts. */
export function auditItem(
  itemId: string,
  ordinal: number,
  kind: string,
  overrides: Overrides<Schemas["AuditItem"]> = {},
): Schemas["AuditItem"] {
  const task = auditInput();
  return conforming(
    "AuditItem",
    merge<Schemas["AuditItem"]>(
      {
        itemId,
        roundId: "round_1",
        itemKey: `key_${itemId}`,
        ordinal,
        kind,
        subjectKey: `subject_${itemId}`,
        task,
        origin: {
          schema: "contractor.audit.item-origin.v1",
          entryKey: `entry_${itemId}`,
          sourceRef: task.ref,
          sourceContentDigest: DIGEST,
          sourceMediaType: "application/zip",
          canonicalInventoryDigest: DIGEST,
        },
        workflowRole: "check",
        state: "settled",
        approvalKind: "none",
        attempts: [],
        createdAt: NOW,
        updatedAt: NOW,
      },
      overrides,
    ),
  );
}

/** The workspace summary of `value`, with zero counts unless given. */
export function workspace(
  value: Schemas["Audit"],
  counts: Partial<
    Pick<
      Schemas["AuditWorkspace"],
      | "totalChecks"
      | "completedChecks"
      | "issues"
      | "gaps"
      | "unchecked"
      | "findings"
      | "unreviewedFindings"
      | "pendingReviews"
    >
  > = {},
): Schemas["AuditWorkspace"] {
  return conforming("AuditWorkspace", {
    auditId: value.auditId,
    auditRevision: value.revision,
    asOf: "2026-10-05T10:40:00Z",
    ...(value.currentRoundId === undefined
      ? {}
      : { roundId: value.currentRoundId }),
    executionState: value.state,
    outstandingRuns: value.outstandingRunCount,
    totalChecks: 0,
    completedChecks: 0,
    issues: 0,
    gaps: 0,
    unchecked: 0,
    findings: 0,
    unreviewedFindings: 0,
    pendingReviews: 0,
    ...counts,
  });
}

/** The first Round of an Audit, executing one item. */
export function auditRound(
  overrides: Overrides<Schemas["AuditRound"]> = {},
): Schemas["AuditRound"] {
  return conforming(
    "AuditRound",
    merge<Schemas["AuditRound"]>(
      {
        roundId: "round_1",
        ordinal: 1,
        manifest: auditInput("round-manifest", {
          mediaType: "application/json",
        }),
        state: "executing",
        expectedItemCount: 1,
        revision: 1,
        createdAt: NOW,
        updatedAt: NOW,
      },
      overrides,
    ),
  );
}
