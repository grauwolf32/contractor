// Fixtures and a fake Server for the Checks area tests (the Checks list, the
// check page and the project's Checks tab). Not imported by the application.
import { render } from "@testing-library/react";
import { createMemoryRouter } from "react-router";
import { vi } from "vitest";

import type {
  Audit,
  AuditCoverageRow,
  AuditFinding,
  AuditItem,
  AuditReviewRequest,
  AuditState,
} from "../../../api/audits";
import { PublicAPI } from "../../../api/client";
import type { Project } from "../../../api/projects";
import { Application } from "../../../app/application";
import { applicationRoutes } from "../../../app/router";
import type { RuntimeConfig } from "../../../config/runtime-config";

export const DIGEST = `sha256:${"a".repeat(64)}`;
export const BASE_TIME = "2026-10-05T10:00:00Z";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

export const session = {
  principal: {
    userId: "user_local",
    username: "owner",
    capabilities: ["user"],
  },
  csrfToken: "a".repeat(43),
  idleExpiresAt: "2026-10-05T20:00:00Z",
  absoluteExpiresAt: "2026-10-06T12:00:00Z",
};

export function makeProject(
  projectId: string,
  name: string,
  overrides: Partial<Project> = {},
): Project {
  return {
    projectId,
    kind: "project",
    name,
    description: "",
    lifecycle: "active",
    revision: "1",
    createdAt: BASE_TIME,
    updatedAt: BASE_TIME,
    ...overrides,
  };
}

const source = {
  ref: { namespace: "sources", name: "service-source", revision: "r1" },
  digest: DIGEST,
  mediaType: "application/zip",
  sizeBytes: 2048,
};

export function makeAudit(
  auditId: string,
  projectId: string,
  state: AuditState,
  overrides: Partial<Audit> & { profileName?: string } = {},
): Audit {
  const { profileName, ...rest } = overrides;
  const draft = state === "draft";
  const base = {
    auditId,
    projectId,
    profile: {
      name: profileName ?? "openapi-operation-trace",
      version: "1",
      digest: DIGEST,
    },
    inputs: { source },
    scope: { objective: "Find broken object authorization" },
    runtimeLabels: [],
    state,
    revision: 3,
    dispatchState: state === "active" ? ("open" as const) : ("closed" as const),
    holdState: draft ? ("pending" as const) : ("held" as const),
    limits: {
      maxRounds: 1,
      batchSize: 1,
      maxItemsPerRound: 10,
      maxItemsTotal: 10,
      maxSubmittedRuns: 20,
      maxItemRunAttempts: 3,
      maxEvidenceBytes: 1_048_576,
    },
    reservedRunCount: 0,
    submittedRunCount: draft ? 0 : 2,
    outstandingRunCount: 0,
    retainedEvidenceBytes: 0,
    eventSequence: 3,
    createdAt: BASE_TIME,
    updatedAt: BASE_TIME,
    ...(draft ? {} : { startedAt: "2026-10-05T10:01:00Z" }),
  };
  return draft
    ? ({ ...base, phase: "not-started", ...rest } as Audit)
    : ({
        ...base,
        phase: "rounds",
        currentRoundId: "round_1",
        ...rest,
      } as Audit);
}

export function endpointRow(
  itemId: string,
  ordinal: number,
  method: string,
  path: string,
  status: AuditCoverageRow["coverage"]["status"],
  overrides: Partial<AuditCoverageRow["coverage"]> & {
    resultSummary?: string;
    evidence?: { id: string; kind: string; summary: string }[];
  } = {},
): AuditCoverageRow {
  const { resultSummary, evidence, ...coverage } = overrides;
  return {
    roundId: "round_1",
    itemId,
    ordinal,
    itemKey: `op-${"a".repeat(16)}${ordinal}`,
    subjectKey: `op-${"a".repeat(16)}${ordinal}`,
    coverage: {
      status,
      requested: ["operation-resolution"],
      completed: status === "not-tested" ? [] : ["operation-resolution"],
      gaps: [],
      ...coverage,
    },
    details: {
      objective: `${method} ${path}\n\nReads ${path}.`,
      methods: ["source-analysis"],
      taskDocument: {
        schema: "contractor.audit.item-task.v1",
        operation: { method: method.toLowerCase(), path },
      },
      ...(resultSummary === undefined ? {} : { resultSummary }),
      evidence: evidence ?? [],
    },
    updatedAt: "2026-10-05T10:20:00Z",
  };
}

export function requirementRow(
  itemId: string,
  ordinal: number,
  key: string,
  statement: string,
  status: AuditCoverageRow["coverage"]["status"],
  overrides: Partial<AuditCoverageRow["coverage"]> & {
    resultSummary?: string;
  } = {},
): AuditCoverageRow {
  const { resultSummary, ...coverage } = overrides;
  return {
    roundId: "round_1",
    itemId,
    ordinal,
    itemKey: key,
    subjectKey: key,
    coverage: {
      status,
      requested: ["source-analysis"],
      completed: status === "not-tested" ? [] : ["source-analysis"],
      gaps: [],
      ...coverage,
    },
    details: {
      objective: statement,
      methods: ["source-analysis"],
      taskDocument: {
        schema: "contractor.audit.item-task.v1",
        checklist: { statement },
      },
      ...(resultSummary === undefined ? {} : { resultSummary }),
      evidence: [],
    },
    updatedAt: "2026-10-05T10:20:00Z",
  };
}

type Attempt = AuditItem["attempts"][number];

/**
 * One attempt, settled with an accepted result unless overridden. An
 * override of `undefined` removes the field (an attempt still running has
 * no outcome, disposition or collection time).
 */
export function makeAttempt(
  itemId: string,
  n: number,
  overrides: { [K in keyof Attempt]?: Attempt[K] | undefined } = {},
): Attempt {
  const attempt: Record<string, unknown> = {
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
    ...overrides,
  };
  for (const [key, value] of Object.entries(attempt))
    if (value === undefined) delete attempt[key];
  return attempt as Attempt;
}

export function makeItem(
  itemId: string,
  ordinal: number,
  kind: string,
  overrides: Partial<AuditItem> = {},
): AuditItem {
  return {
    itemId,
    roundId: "round_1",
    itemKey: `key_${itemId}`,
    ordinal,
    kind,
    subjectKey: `subject_${itemId}`,
    task: source,
    origin: {
      schema: "contractor.audit.item-origin.v1",
      entryKey: `entry_${itemId}`,
      sourceRef: source.ref,
      sourceContentDigest: DIGEST,
      sourceMediaType: "application/zip",
      canonicalInventoryDigest: DIGEST,
    },
    workflowRole: "check",
    state: "settled",
    approvalKind: "none",
    attempts: [],
    createdAt: BASE_TIME,
    updatedAt: BASE_TIME,
    ...overrides,
  };
}

export function makeFinding(
  auditId: string,
  findingId: string,
  title: string,
  subjectKey: string | null,
  overrides: Partial<AuditFinding> = {},
): AuditFinding {
  return {
    findingId,
    auditId,
    state: "proposed",
    firstProposal: {
      receiptId: `receipt_${findingId}`,
      proposalId: `proposal_${findingId}`,
      requestDigest: DIGEST,
      clientKey: findingId,
      proposal: {
        ref: { namespace: "audit-findings", name: findingId, revision: "r1" },
        digest: DIGEST,
        mediaType: "application/json",
        sizeBytes: 10,
      },
      document: {
        schema: "contractor.audit.finding-proposal.v1",
        client_key: findingId,
        title,
        description: "Reads another owner's record.",
        subject:
          subjectKey === null ? null : { kind: "operation", key: subjectKey },
        preconditions: [],
        standard_refs: [
          { scheme: "CWE", version: "4.14", requirement_id: "CWE-639" },
        ],
        evidence_ids: [],
        proposed_checks: [],
        severity_suggestion: "high",
        limitations: [],
      },
      evidence: [],
      origin: {
        runId: "run_source",
        stageExecutionId: "stage_source",
        allocationId: "allocation_source",
        invocationId: "invocation_source",
        logicalAgentName: "tracer",
        workflow: {
          name: "trace",
          version: "1",
          schemaVersion: "contractor/v1alpha1",
          configurationRef: { name: "trace", version: "1" },
          closureDigest: DIGEST,
        },
        runDeleted: false,
      },
      retention: "audit-held",
      auditHolds: [],
      createdAt: "2026-10-05T10:15:00Z",
    },
    revision: 1,
    createdAt: "2026-10-05T10:15:00Z",
    updatedAt: "2026-10-05T10:15:00Z",
    ...overrides,
  };
}

export function makeReview(
  auditId: string,
  requestId: string,
  overrides: Partial<AuditReviewRequest> = {},
): AuditReviewRequest {
  return {
    requestId,
    auditId,
    subjectKind: "audit-item-action",
    subjectId: "item_1",
    kind: "active-check-approval",
    subjectRevision: 1,
    subjectDigest: DIGEST,
    requestedActions: ["approve", "reject"],
    state: "pending",
    revision: 1,
    createdAt: "2026-10-05T10:30:00Z",
    updatedAt: "2026-10-05T10:30:00Z",
    ...overrides,
  };
}

export function workspaceOf(
  audit: Audit,
  counts: Partial<{
    totalChecks: number;
    completedChecks: number;
    issues: number;
    gaps: number;
    unchecked: number;
    findings: number;
    unreviewedFindings: number;
    pendingReviews: number;
  }> = {},
) {
  return {
    auditId: audit.auditId,
    auditRevision: audit.revision,
    asOf: "2026-10-05T10:40:00Z",
    roundId: "round_1",
    executionState: audit.state,
    outstandingRuns: audit.outstandingRunCount,
    totalChecks: 0,
    completedChecks: 0,
    issues: 0,
    gaps: 0,
    unchecked: 0,
    findings: 0,
    unreviewedFindings: 0,
    pendingReviews: 0,
    ...counts,
  };
}

export function jsonResponse(
  value: unknown,
  options: ResponseInit = {},
): Response {
  const headers = new Headers(options.headers);
  headers.set("content-type", "application/json");
  headers.set("X-Contractor-API-Version", "contractor.public.v1");
  return new Response(JSON.stringify(value), { ...options, headers });
}

export function emptyPage(extra: Record<string, unknown> = {}): Response {
  return jsonResponse({ items: [], page: { hasMore: false }, ...extra });
}

export type Handler = (
  request: Request,
  url: URL,
) => Response | undefined | Promise<Response | undefined>;

/**
 * A PublicAPI whose Server answers with `handle`; anything it leaves
 * unanswered gets an empty page, so the shell's own reads stay quiet.
 */
export function fakeAPI(handle: Handler): {
  api: PublicAPI;
  requests: Request[];
} {
  const requests: Request[] = [];
  const api = new PublicAPI(
    runtimeConfig,
    vi.fn(async (input: RequestInfo | URL) => {
      const request = input instanceof Request ? input : new Request(input);
      requests.push(request.clone());
      const url = new URL(request.url);
      if (url.pathname === "/v1/auth/session") return jsonResponse(session);
      const answer = await handle(request, url);
      return answer ?? emptyPage();
    }),
  );
  return { api, requests };
}

export function renderApplication(api: PublicAPI, path: string) {
  const router = createMemoryRouter(applicationRoutes(), {
    initialEntries: [path],
  });
  return {
    ...render(<Application api={api} publicAPI={api} router={router} />),
    router,
  };
}
