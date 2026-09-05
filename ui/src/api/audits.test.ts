import { describe, expect, it, vi } from "vitest";

import type { RuntimeConfig } from "../config/runtime-config";
import { PublicAPI } from "./client";
import {
  createAudit,
  getAudit,
  getAuditProfile,
  listAuditCoverage,
  listAuditItems,
  listAuditProfiles,
  listProjectAudits,
  mutateAudit,
  type Audit,
  type AuditProfile,
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
