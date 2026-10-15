// Fixtures and a fake Server for the Checks area tests (the Checks list, the
// check page and the project's Checks tab). Not imported by the application.

import type {
  Audit,
  AuditCoverageRow,
  AuditFinding,
  AuditItem,
  AuditReviewRequest,
  AuditState,
} from "../../../api/audits";
import type { PublicAPI } from "../../../api/client";
import type { Project } from "../../../api/projects";
import * as build from "../../../test/builders";
import {
  emptyPageOrNotFound,
  fakeServer,
  json,
  page,
  type Fallback,
} from "../../../test/fake-server";
import { testAPI, testSession } from "../../../test/render";

export const { DIGEST } = build;
export const BASE_TIME = "2026-10-05T10:00:00Z";

export function makeProject(
  projectId: string,
  name: string,
  overrides: Partial<Project> = {},
): Project {
  return build.project(projectId, {
    name,
    createdAt: BASE_TIME,
    updatedAt: BASE_TIME,
    ...overrides,
  });
}

const source = build.auditInput("service-source");

export function makeAudit(
  auditId: string,
  projectId: string,
  state: AuditState,
  overrides: Partial<Audit> & { profileName?: string } = {},
): Audit {
  const { profileName, ...rest } = overrides;
  const draft = state === "draft";
  return build.audit(projectId, auditId, state, {
    profile: {
      name: profileName ?? "openapi-operation-trace",
      version: "1",
      digest: DIGEST,
    },
    inputs: { source },
    scope: { objective: "Find broken object authorization" },
    revision: 3,
    limits: {
      maxRounds: 1,
      batchSize: 1,
      maxItemsPerRound: 10,
      maxItemsTotal: 10,
      maxSubmittedRuns: 20,
      maxItemRunAttempts: 3,
      maxEvidenceBytes: 1_048_576,
    },
    submittedRunCount: draft ? 0 : 2,
    eventSequence: 3,
    createdAt: BASE_TIME,
    updatedAt: BASE_TIME,
    ...(draft ? {} : { startedAt: "2026-10-05T10:01:00Z" }),
    ...rest,
  });
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
  return build.conforming("AuditCoverageRow", {
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
  });
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
  return build.conforming("AuditCoverageRow", {
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
  });
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
  return build.itemAttempt(itemId, n, overrides);
}

export function makeItem(
  itemId: string,
  ordinal: number,
  kind: string,
  overrides: Partial<AuditItem> = {},
): AuditItem {
  return build.auditItem(itemId, ordinal, kind, {
    task: source,
    createdAt: BASE_TIME,
    updatedAt: BASE_TIME,
    ...overrides,
  });
}

export function makeFinding(
  auditId: string,
  findingId: string,
  title: string,
  subjectKey: string | null,
  overrides: Partial<AuditFinding> = {},
): AuditFinding {
  const finding = build.finding(
    auditId,
    findingId,
    {
      createdAt: "2026-10-05T10:15:00Z",
      updatedAt: "2026-10-05T10:15:00Z",
      ...overrides,
    },
    {
      title,
      description: "Reads another owner's record.",
      subject:
        subjectKey === null ? null : { kind: "operation", key: subjectKey },
      standard_refs: [
        { scheme: "CWE", version: "4.14", requirement_id: "CWE-639" },
      ],
    },
  );
  return overrides.firstProposal === undefined
    ? {
        ...finding,
        firstProposal: {
          ...finding.firstProposal,
          createdAt: "2026-10-05T10:15:00Z",
          origin: {
            ...finding.firstProposal.origin,
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
          },
        },
      }
    : finding;
}

export function makeReview(
  auditId: string,
  requestId: string,
  overrides: Partial<AuditReviewRequest> = {},
): AuditReviewRequest {
  return build.review(auditId, requestId, "active-check-approval", {
    subjectId: "item_1",
    createdAt: "2026-10-05T10:30:00Z",
    updatedAt: "2026-10-05T10:30:00Z",
    ...overrides,
  });
}

/** The check workspace summary of `audit`. */
export const workspaceOf = build.workspace;

/** An empty keyset page. */
export function emptyPage(extra: Record<string, unknown> = {}): Response {
  return json({ ...page([]), ...extra });
}

export type Handler = Fallback;

/**
 * A PublicAPI whose Server answers with `handle`; anything it leaves
 * unanswered gets an empty page where the operation pages, so the shell's
 * own reads stay quiet, and 404 otherwise.
 */
export function fakeAPI(
  handle: Handler,
  options: { capabilities?: Parameters<typeof testSession>[0] } = {},
): {
  api: PublicAPI;
  requests: Request[];
} {
  const { api, server } = testAPI(
    fakeServer(
      { "GET /v1/auth/session": () => json(testSession(options.capabilities)) },
      async (request, url) =>
        (await handle(request, url)) ?? emptyPageOrNotFound(request, url),
    ),
  );
  return { api, requests: server.requests };
}
