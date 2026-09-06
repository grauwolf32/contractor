import { describe, expect, it, vi } from "vitest";

import type { RuntimeConfig } from "../config/runtime-config";
import { PublicAPI } from "./client";
import {
  createAudit,
  createAuditFindingReview,
  decideAuditFinding,
  getAudit,
  getAuditFinding,
  getAuditProfile,
  listAuditCoverage,
  listAuditFindingProvenance,
  listAuditFindings,
  listAuditItems,
  listAuditProfiles,
  listAuditReviews,
  listProjectAudits,
  mutateAudit,
  type Audit,
  type AuditFinding,
  type AuditProfile,
  type AuditReviewRequest,
} from "./audits";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

const exactInput = {
  ref: { namespace: "sources", name: "service", revision: "revision-1" },
  digest: `sha256:${"1".repeat(64)}`,
  mediaType: "application/zip",
  sizeBytes: 123,
};

const profile: AuditProfile = {
  ref: {
    name: "source-checklist",
    version: "1",
    digest: `sha256:${"2".repeat(64)}`,
  },
  mode: "custom-checklist",
  standards: [],
  inputs: { sources: { required: true, mediaTypes: ["application/zip"] } },
  inventory: {
    implementation: "checklist@1",
    sourceInput: "sources",
    itemWorkflowRole: "check",
  },
  execution: {
    roundMode: "fixed-barrier",
    maxRounds: 1,
    batchSize: 1,
    maxItemsPerRound: 8,
    maxItemsTotal: 8,
    maxSubmittedRuns: 8,
    maxItemRunAttempts: 2,
    deadlineSeconds: 600,
    maxEvidenceBytes: 1_048_576,
    incompleteRound: "assess-with-gaps",
  },
  interaction: {
    activeChecks: "prohibited",
    findingConfirmation: "disabled",
    notApplicable: "profile-rule",
    reportAcceptance: "automatic",
  },
  serverCompatible: true,
  requiresInputValidation: true,
  compatibilityReasons: [],
};

const audit: Audit = {
  auditId: "audit_example",
  projectId: "project_example",
  profile: { ...profile.ref },
  inputs: { sources: exactInput },
  scope: {},
  runtimeLabels: [],
  state: "draft",
  revision: 1,
  dispatchState: "closed",
  holdState: "pending",
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
  createdAt: "2026-09-06T10:00:00Z",
  updatedAt: "2026-09-06T10:00:00Z",
};

const finding: AuditFinding = {
  findingId: "finding_example",
  auditId: audit.auditId,
  state: "proposed",
  firstProposal: {
    receiptId: "receipt_example",
    proposalId: "proposal_example",
    requestDigest: `sha256:${"3".repeat(64)}`,
    clientKey: "candidate-example",
    proposal: {
      ref: {
        namespace: "audit-findings",
        name: "candidate-example",
        revision: "proposal-r1",
      },
      digest: `sha256:${"4".repeat(64)}`,
      mediaType: "application/json",
      sizeBytes: 128,
    },
    document: {
      schema: "contractor.audit.finding-proposal.v1",
      client_key: "candidate-example",
      title: "Missing authorization",
      description: "The object read path may omit an owner check.",
      subject: { kind: "component", key: "orders" },
      preconditions: [],
      standard_refs: [],
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
      logicalAgentName: "reviewer",
      workflow: {
        name: "source-review",
        version: "1",
        schemaVersion: "contractor/v1alpha1",
        configurationRef: { name: "source-review", version: "1" },
        closureDigest: `sha256:${"5".repeat(64)}`,
      },
      runDeleted: false,
    },
    retention: "audit-held",
    auditHolds: [],
    createdAt: audit.createdAt,
  },
  revision: 1,
  createdAt: audit.createdAt,
  updatedAt: audit.updatedAt,
};

const review: AuditReviewRequest = {
  requestId: "review_example",
  auditId: audit.auditId,
  findingId: finding.findingId,
  kind: "finding-triage",
  subjectRevision: finding.revision,
  subjectDigest: `sha256:${"6".repeat(64)}`,
  requestedActions: [
    "true_positive",
    "false_positive",
    "duplicate",
    "reopen",
    "needs_evidence",
  ],
  state: "pending",
  revision: 1,
  createdAt: audit.createdAt,
  updatedAt: audit.updatedAt,
};

function response(value: unknown, options: ResponseInit = {}): Response {
  const headers = new Headers(options.headers);
  headers.set("content-type", "application/json");
  headers.set("X-Contractor-API-Version", "contractor.public.v1");
  return new Response(JSON.stringify(value), { ...options, headers });
}

describe("Audit API", () => {
  it("reads exact profile versions and validates digest ETags", async () => {
    const requests: Request[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request);
        const url = new URL(request.url);
        if (url.pathname === "/v1/audit-profiles") {
          return response({ items: [profile], page: { hasMore: false } });
        }
        return response(profile, {
          headers: { ETag: `"${profile.ref.digest}"` },
        });
      }),
    );

    await expect(listAuditProfiles(api, { cursor: "page-2" })).resolves.toEqual(
      {
        items: [profile],
        page: { hasMore: false },
      },
    );
    await expect(
      getAuditProfile(api, profile.ref.name, profile.ref.version),
    ).resolves.toEqual(profile);
    const listURL = new URL(requests[0]!.url);
    expect(listURL.searchParams.get("limit")).toBe("50");
    expect(listURL.searchParams.get("cursor")).toBe("page-2");
  });

  it("creates a draft and mutates with replay and revision headers", async () => {
    const requests: Request[] = [];
    const active = { ...audit, state: "active" as const, revision: 2 };
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request.clone());
        if (new URL(request.url).pathname.endsWith("/start")) {
          return response(
            {
              audit: active,
              round: {
                roundId: "round_example",
                ordinal: 1,
                manifest: exactInput,
                state: "accepted",
                expectedItemCount: 0,
                revision: 1,
                createdAt: active.createdAt,
                updatedAt: active.updatedAt,
              },
              items: [],
            },
            { headers: { ETag: '"2"' } },
          );
        }
        return response(audit, {
          status: 201,
          headers: { ETag: '"1"' },
        });
      }),
    );
    api.csrf.replace("a".repeat(43));
    const request = {
      profile: { name: profile.ref.name, version: profile.ref.version },
      inputs: { sources: exactInput.ref },
    };

    await expect(
      createAudit(api, {
        projectId: audit.projectId,
        request,
        idempotencyKey: "create-audit-example",
      }),
    ).resolves.toEqual(audit);
    await expect(
      mutateAudit(api, "start", {
        auditId: audit.auditId,
        expectedRevision: 1,
        idempotencyKey: "start-audit-example",
      }),
    ).resolves.toMatchObject({ audit: active });

    expect(requests[0]?.headers.get("Idempotency-Key")).toBe(
      "create-audit-example",
    );
    expect(requests[1]?.headers.get("Idempotency-Key")).toBe(
      "start-audit-example",
    );
    expect(requests[1]?.headers.get("If-Match")).toBe('"1"');
    expect(requests[1]?.headers.get("X-CSRF-Token")).toBe("a".repeat(43));
    await expect(requests[0]?.json()).resolves.toEqual(request);
    await expect(requests[1]?.text()).resolves.toBe("");
  });

  it("reads authoritative Audit, item and coverage projections", async () => {
    const requests: Request[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request);
        const path = new URL(request.url).pathname;
        if (path.endsWith("/items") || path.endsWith("/coverage")) {
          return response({ items: [], page: { hasMore: false } });
        }
        if (path.endsWith("/audits")) {
          return response({ items: [audit], page: { hasMore: false } });
        }
        return response(audit, { headers: { ETag: '"1"' } });
      }),
    );

    await expect(getAudit(api, audit.auditId)).resolves.toEqual(audit);
    await expect(
      listProjectAudits(api, { projectId: audit.projectId }),
    ).resolves.toMatchObject({ items: [audit] });
    await expect(listAuditItems(api, audit.auditId)).resolves.toMatchObject({
      items: [],
    });
    await expect(listAuditCoverage(api, audit.auditId)).resolves.toMatchObject({
      items: [],
    });
    expect(requests.map((request) => new URL(request.url).pathname)).toEqual([
      "/v1/audits/audit_example",
      "/v1/projects/project_example/audits",
      "/v1/audits/audit_example/items",
      "/v1/audits/audit_example/coverage",
    ]);
  });

  it("reads finding provenance and sends exact review mutation headers", async () => {
    const requests: Request[] = [];
    const decided = {
      ...review,
      state: "decided" as const,
      revision: 2,
      decision: {
        decisionId: "decision_example",
        requestId: review.requestId,
        auditId: audit.auditId,
        findingId: finding.findingId,
        actorId: "user_local",
        verdict: "true_positive" as const,
        severity: "high" as const,
        rationale: "Confirmed from exact evidence.",
        subjectRevision: finding.revision,
        subjectDigest: review.subjectDigest,
        createdAt: audit.createdAt,
      },
    };
    const confirmed = {
      ...finding,
      state: "confirmed" as const,
      revision: 2,
      analystVerdict: "true_positive" as const,
      analystSeverity: "high" as const,
      analystDecision: decided.decision,
    };
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request.clone());
        const path = new URL(request.url).pathname;
        if (path.endsWith("/provenance")) {
          return response({
            findingRevision: finding.revision,
            auditRevision: audit.revision,
            items: [],
            page: { hasMore: false },
          });
        }
        if (path.endsWith("/decisions")) {
          return response({
            finding: confirmed,
            request: decided,
            decision: decided.decision,
            replayed: false,
          });
        }
        if (path.endsWith("/reviews") && request.method === "POST") {
          return response(review, { status: 201, headers: { ETag: '"1"' } });
        }
        if (path.endsWith("/reviews")) {
          return response({ items: [review], page: { hasMore: false } });
        }
        if (path.endsWith(`/findings/${finding.findingId}`)) {
          return response(finding, { headers: { ETag: '"1"' } });
        }
        return response({ items: [finding], page: { hasMore: false } });
      }),
    );
    api.csrf.replace("a".repeat(43));

    await expect(listAuditFindings(api, audit.auditId)).resolves.toMatchObject({
      items: [finding],
    });
    await expect(
      getAuditFinding(api, audit.auditId, finding.findingId),
    ).resolves.toEqual(finding);
    await expect(listAuditReviews(api, audit.auditId)).resolves.toMatchObject({
      items: [review],
    });
    await expect(
      createAuditFindingReview(api, {
        auditId: audit.auditId,
        findingId: finding.findingId,
        expectedRevision: finding.revision,
        idempotencyKey: "create-review-example",
      }),
    ).resolves.toEqual(review);
    await expect(
      decideAuditFinding(api, {
        auditId: audit.auditId,
        requestId: review.requestId,
        expectedRevision: review.revision,
        idempotencyKey: "decide-review-example",
        decision: {
          verdict: "true_positive",
          severity: "high",
          rationale: "Confirmed from exact evidence.",
        },
      }),
    ).resolves.toMatchObject({ finding: confirmed });
    await expect(
      listAuditFindingProvenance(api, audit.auditId, finding.findingId, {
        auditRevision: audit.revision,
        findingRevision: finding.revision,
      }),
    ).resolves.toMatchObject({ items: [] });

    const create = requests.find(
      (request) =>
        request.method === "POST" &&
        new URL(request.url).pathname.endsWith(
          "/findings/finding_example/reviews",
        ),
    );
    const decision = requests.find((request) =>
      new URL(request.url).pathname.endsWith(
        "/reviews/review_example/decisions",
      ),
    );
    expect(create?.headers.get("If-Match")).toBe('"1"');
    expect(create?.headers.get("Idempotency-Key")).toBe(
      "create-review-example",
    );
    expect(decision?.headers.get("If-Match")).toBe('"1"');
    const provenanceURL = new URL(requests.at(-1)!.url);
    expect(provenanceURL.searchParams.get("auditRevision")).toBe("1");
    expect(provenanceURL.searchParams.get("findingRevision")).toBe("1");
  });

  it("rejects a profile response whose digest ETag does not match", async () => {
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async () => response(profile, { headers: { ETag: '"1"' } })),
    );
    await expect(
      getAuditProfile(api, profile.ref.name, profile.ref.version),
    ).rejects.toMatchObject({ code: "invalid_api_response" });
  });
});
