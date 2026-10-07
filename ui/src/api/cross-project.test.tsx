import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";

import type {
  Audit,
  AuditFinding,
  AuditFindingSeverity,
  AuditFindingState,
  AuditReport,
  AuditReviewRequest,
  AuditState,
} from "./audits";
import { PublicAPI } from "./client";
import { PublicAPIProvider } from "./context";
import {
  INBOX_CHECK_STATES,
  invalidateCrossProject,
  useAllChecks,
  useAllPossibleIssues,
  useAllReports,
  useInboxSummary,
  usePendingDecisions,
  useProjectsIndex,
} from "./cross-project";
import type { Project } from "./projects";
import { collectOwnerPages } from "./owner-pages";

const digest = `sha256:${"a".repeat(64)}`;

function project(projectId: string, overrides: Partial<Project> = {}): Project {
  return {
    projectId,
    kind: "project",
    name: projectId,
    description: "",
    lifecycle: "active",
    revision: "1",
    createdAt: "2026-10-01T10:00:00Z",
    updatedAt: "2026-10-01T10:00:00Z",
    ...overrides,
  };
}

function audit(
  projectId: string,
  auditId: string,
  state: AuditState,
  options: { revision?: number; updatedAt?: string } = {},
): Audit {
  const base = {
    auditId,
    projectId,
    profile: { name: "openapi-operation-trace", version: "1", digest },
    inputs: {},
    scope: {},
    runtimeLabels: [],
    state,
    revision: options.revision ?? 1,
    dispatchState: "closed" as const,
    holdState: "pending" as const,
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
    createdAt: "2026-10-01T10:00:00Z",
    updatedAt: options.updatedAt ?? "2026-10-01T10:00:00Z",
  };
  return state === "draft"
    ? { ...base, phase: "not-started" }
    : { ...base, phase: "rounds", currentRoundId: "round_1" };
}

function finding(
  auditId: string,
  findingId: string,
  options: {
    state?: AuditFindingState;
    createdAt?: string;
    verdict?: "true_positive" | "false_positive";
    severity?: AuditFindingSeverity;
  } = {},
): AuditFinding {
  const createdAt = options.createdAt ?? "2026-10-02T10:00:00Z";
  return {
    findingId,
    auditId,
    state: options.state ?? "proposed",
    firstProposal: {
      receiptId: `receipt_${findingId}`,
      proposalId: `proposal_${findingId}`,
      requestDigest: digest,
      clientKey: findingId,
      proposal: {
        ref: { namespace: "audit-findings", name: findingId, revision: "r1" },
        digest,
        mediaType: "application/json",
        sizeBytes: 1,
      },
      document: {
        schema: "contractor.audit.finding-proposal.v1",
        client_key: findingId,
        title: `Finding ${findingId}`,
        description: "",
        subject: null,
        preconditions: [],
        standard_refs: [],
        evidence_ids: [],
        proposed_checks: [],
        severity_suggestion: "",
        limitations: [],
      },
      evidence: [],
      origin: {
        runId: "run_1",
        stageExecutionId: "stage_1",
        allocationId: "allocation_1",
        invocationId: "invocation_1",
        logicalAgentName: "reviewer",
        workflow: {
          name: "trace",
          version: "1",
          schemaVersion: "contractor/v1alpha1",
          configurationRef: { name: "trace", version: "1" },
          closureDigest: digest,
        },
        runDeleted: false,
      },
      retention: "audit-held",
      auditHolds: [],
      createdAt,
    },
    ...(options.verdict === undefined
      ? {}
      : { analystVerdict: options.verdict }),
    ...(options.severity === undefined
      ? {}
      : { analystSeverity: options.severity }),
    revision: 1,
    createdAt,
    updatedAt: createdAt,
  };
}

function review(
  auditId: string,
  requestId: string,
  kind: AuditReviewRequest["kind"],
  createdAt = "2026-10-03T10:00:00Z",
): AuditReviewRequest {
  return {
    requestId,
    auditId,
    ...(kind === "finding-triage" ? { findingId: "finding_1" } : {}),
    subjectKind:
      kind === "finding-triage"
        ? "finding"
        : kind === "report-acceptance"
          ? "audit-report"
          : "audit-item-action",
    subjectId: "subject_1",
    kind,
    subjectRevision: 1,
    subjectDigest: digest,
    requestedActions:
      kind === "finding-triage"
        ? ["true_positive", "false_positive"]
        : ["approve", "reject"],
    state: "pending",
    revision: 1,
    createdAt,
    updatedAt: createdAt,
  };
}

function report(status: AuditReport["status"]): AuditReport {
  return status === "ready" || status === "proposed"
    ? {
        status,
        summary: "# Report",
        summaryArtifact: {
          ref: { namespace: "reports", name: "summary", revision: "r1" },
          digest,
          mediaType: "text/markdown",
        },
      }
    : { status };
}

interface ServerState {
  projects: Project[];
  audits: Record<string, Audit[]>;
  findings?: Record<string, AuditFinding[]>;
  reviews?: Record<string, AuditReviewRequest[]>;
  reports?: Record<string, AuditReport>;
  failures?: Record<string, boolean>;
  continuationFails?: string;
}

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "content-type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
    },
  });
}
function setup(state: ServerState) {
  const requests: URL[] = [];
  const fetch = vi.fn(async (input: RequestInfo | URL) => {
    const url = new URL(input instanceof Request ? input.url : String(input));
    requests.push(url);
    if (
      state.failures?.[url.pathname] ||
      (state.continuationFails === url.pathname &&
        url.searchParams.has("cursor"))
    ) {
      return json(
        {
          code: "internal_error",
          message: "Server failed",
          retryable: false,
          requestId: "req_test",
        },
        500,
      );
    }
    const audits = Object.values(state.audits).flat();
    const wanted = (name: string, value: string) =>
      !url.searchParams.has(name) ||
      url.searchParams.get(name)!.split(",").includes(value);
    const inScope = (auditId: string) =>
      audits.some(
        (a) => a.auditId === auditId && wanted("auditState", a.state),
      );
    let items: unknown[];
    if (url.pathname === "/v1/projects") items = state.projects;
    else if (url.pathname === "/v1/audits")
      items = audits.filter((a) => wanted("state", a.state));
    else if (url.pathname === "/v1/findings")
      items = Object.values(state.findings ?? {})
        .flat()
        .filter(
          (f) =>
            inScope(f.auditId) &&
            wanted("state", f.state) &&
            wanted("verdict", f.analystVerdict ?? "unreviewed") &&
            (!url.searchParams.has("severity") ||
              wanted("severity", f.analystSeverity ?? "")),
        );
    else if (url.pathname === "/v1/reviews")
      items = Object.values(state.reviews ?? {})
        .flat()
        .filter((r) => inScope(r.auditId) && wanted("state", r.state));
    else {
      const match = /^\/v1\/audits\/([^/]+)\/report$/.exec(url.pathname);
      const found = match === null ? undefined : state.reports?.[match[1]!];
      return found === undefined
        ? json(
            {
              code: "not_found",
              message: "Not found",
              retryable: false,
              requestId: "req_test",
            },
            404,
          )
        : json(found);
    }
    const limit = Number(url.searchParams.get("limit") ?? 50);
    const start = Number(url.searchParams.get("cursor") ?? 0);
    const hasMore = start + limit < items.length;
    return json({
      items: items.slice(start, start + limit),
      page: {
        hasMore,
        ...(hasMore ? { nextCursor: String(start + limit) } : {}),
      },
    });
  });
  const api = new PublicAPI(
    {
      uiVersion: "0.1.0",
      supportedApiVersions: ["contractor.public.v1"],
      apiBaseUrl: "http://127.0.0.1:8080",
    },
    fetch,
  );
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <PublicAPIProvider api={api}>{children}</PublicAPIProvider>
    </QueryClientProvider>
  );
  const count = (path: string) =>
    requests.filter((url) => url.pathname === path).length;
  return { requests, queryClient, wrapper, count };
}
async function settled(result: { current: { isPending: boolean } }) {
  await waitFor(() => expect(result.current.isPending).toBe(false));
}
const base = (): ServerState => ({
  projects: [project("project_a"), project("project_b")],
  audits: {
    project_a: [audit("project_a", "audit_a", "active")],
    project_b: [audit("project_b", "audit_b", "completed")],
  },
});

describe("owner-wide lists", () => {
  it("reads projects beyond page 1 and checks beyond 200 without per-project requests", async () => {
    const state = base();
    state.projects = Array.from({ length: 51 }, (_, i) =>
      project(`project_${i}`),
    );
    state.audits = {
      project_50: Array.from({ length: 201 }, (_, i) =>
        audit("project_50", `audit_${i}`, "active"),
      ),
    };
    const server = setup(state);
    const { result } = renderHook(() => useAllChecks(), {
      wrapper: server.wrapper,
    });
    await settled(result);
    expect(result.current.checks).toHaveLength(201);
    expect(result.current.checks[0]?.project.projectId).toBe("project_50");
    expect(result.current).toMatchObject({
      partial: false,
      truncated: false,
      error: null,
    });
    expect(server.count("/v1/projects")).toBe(2);
    expect(server.count("/v1/audits")).toBe(2);
    expect(
      server.requests.every(
        (r) => !r.pathname.match(/projects\/[^/]+\/audits/),
      ),
    ).toBe(true);
  });
  it("filters multiple check states on the Server and shares canonical keys", async () => {
    const server = setup(base());
    const { result } = renderHook(
      () => ({
        a: useAllChecks({ states: ["active", "completed"] }),
        b: useAllChecks({ states: ["completed", "active", "active"] }),
      }),
      { wrapper: server.wrapper },
    );
    await waitFor(() =>
      expect(result.current.a.isPending || result.current.b.isPending).toBe(
        false,
      ),
    );
    expect(result.current.a.checks).toHaveLength(2);
    expect(server.count("/v1/audits")).toBe(1);
    expect(
      server.requests
        .find((r) => r.pathname === "/v1/audits")
        ?.searchParams.get("state"),
    ).toBe("active,completed");
  });
  it("excludes deleting and evaluation Projects", async () => {
    const state = base();
    state.projects[1] = project("project_b", { kind: "evaluation" });
    state.projects.push(
      project("deleted", {
        lifecycle: "deleting",
        deletion: { phase: "draining", requestedAt: "2026-10-01T00:00:00Z" },
      }),
    );
    const { result } = renderHook(() => useProjectsIndex(), {
      wrapper: setup(state).wrapper,
    });
    await settled(result);
    expect(result.current.projects.map((p) => p.projectId)).toEqual([
      "project_a",
    ]);
  });
  it("reports an initial list failure without claiming an empty result is complete", async () => {
    const state = base();
    state.failures = { "/v1/audits": true };
    const { result } = renderHook(() => useAllChecks(), {
      wrapper: setup(state).wrapper,
    });
    await settled(result);
    expect(result.current.error).toBeInstanceOf(Error);
    expect(result.current.partial).toBe(true);
  });
  it("keeps the last complete collection on a failed head refresh", async () => {
    const state = base();
    const server = setup(state);
    const { result } = renderHook(() => useAllChecks(), {
      wrapper: server.wrapper,
    });
    await settled(result);
    state.failures = { "/v1/audits": true };
    await act(() => result.current.refetch());
    expect(result.current.checks).toHaveLength(2);
    await waitFor(() => expect(result.current.partial).toBe(true));
    expect(result.current.error).toBeNull();
  });
  it("keeps settled pages and marks a failed continuation partial", async () => {
    const state = base();
    state.audits.project_a = Array.from({ length: 201 }, (_, i) =>
      audit("project_a", `audit_${i}`, "active"),
    );
    state.audits.project_b = [];
    state.continuationFails = "/v1/audits";
    const server = setup(state);
    const { result } = renderHook(() => useAllChecks(), {
      wrapper: server.wrapper,
    });
    await settled(result);
    expect(result.current.checks).toHaveLength(200);
    expect(result.current.partial).toBe(true);
    delete state.continuationFails;
    await act(() => result.current.refetch());
    await waitFor(() => expect(result.current.checks).toHaveLength(201));
    expect(result.current.partial).toBe(false);
  });
  it("loads every finding page and computes a complete Inbox count", async () => {
    const state = base();
    state.findings = {
      audit_a: Array.from({ length: 201 }, (_, i) =>
        finding("audit_a", `finding_${i}`),
      ),
    };
    const server = setup(state);
    const { result } = renderHook(
      () => ({
        issues: useAllPossibleIssues({
          checkStates: INBOX_CHECK_STATES,
          states: ["proposed"],
        }),
        summary: useInboxSummary(),
      }),
      { wrapper: server.wrapper },
    );
    await waitFor(() => expect(result.current.summary.needsDecision).toBe(201));
    expect(result.current.issues.issues).toHaveLength(201);
    expect(result.current.issues.total).toBe(201);
    expect(
      result.current.summary.partial || result.current.summary.truncated,
    ).toBe(false);
    expect(server.count("/v1/findings")).toBe(2);
  });
  it("applies finding verdict and analyst severity filters", async () => {
    const state = base();
    state.findings = {
      audit_a: [
        finding("audit_a", "unreviewed"),
        finding("audit_a", "high", {
          state: "confirmed",
          verdict: "true_positive",
          severity: "high",
        }),
        finding("audit_a", "low", {
          state: "confirmed",
          verdict: "true_positive",
          severity: "low",
        }),
      ],
    };
    const server = setup(state);
    const { result } = renderHook(
      () =>
        useAllPossibleIssues({
          verdicts: ["true_positive"],
          severities: ["high"],
        }),
      { wrapper: server.wrapper },
    );
    await settled(result);
    expect(result.current.issues.map((i) => i.finding.findingId)).toEqual([
      "high",
    ]);
    expect(
      server.requests
        .find((r) => r.pathname === "/v1/findings")
        ?.searchParams.get("severity"),
    ).toBe("high");
  });
  it("does not request an impossible unreviewed severity pair", async () => {
    const server = setup(base());
    const { result } = renderHook(
      () =>
        useAllPossibleIssues({
          verdicts: ["unreviewed"],
          severities: ["high"],
        }),
      { wrapper: server.wrapper },
    );
    await settled(result);
    expect(result.current.issues).toEqual([]);
    expect(server.count("/v1/findings")).toBe(0);
  });
  it("reads all pending request pages and shares requests between kind filters", async () => {
    const state = base();
    state.reviews = {
      audit_a: Array.from({ length: 201 }, (_, i) =>
        review(
          "audit_a",
          `review_${i}`,
          i % 2 ? "active-check-approval" : "finding-triage",
        ),
      ),
    };
    const server = setup(state);
    const { result } = renderHook(
      () => ({
        all: usePendingDecisions(),
        approvals: usePendingDecisions({ kinds: ["active-check-approval"] }),
      }),
      { wrapper: server.wrapper },
    );
    await waitFor(() =>
      expect(
        result.current.all.isPending || result.current.approvals.isPending,
      ).toBe(false),
    );
    expect(result.current.all.decisions).toHaveLength(201);
    expect(result.current.approvals.decisions).toHaveLength(100);
    expect(server.count("/v1/reviews")).toBe(2);
  });
  it("does not count triage requests twice in the Inbox", async () => {
    const state = base();
    state.findings = { audit_a: [finding("audit_a", "one")] };
    state.reviews = {
      audit_a: [
        review("audit_a", "triage", "finding-triage"),
        review("audit_a", "approval", "active-check-approval"),
      ],
    };
    const { result } = renderHook(() => useInboxSummary(), {
      wrapper: setup(state).wrapper,
    });
    await waitFor(() => expect(result.current.needsDecision).toBe(2));
    expect(result.current.otherDecisions).toBe(1);
  });
  it("refreshes owner lists after decisions without reloading unchanged report payloads", async () => {
    const state = base();
    state.reports = { audit_b: report("ready") };
    state.findings = { audit_a: [finding("audit_a", "one")] };
    const server = setup(state);
    const { result } = renderHook(
      () => ({ reports: useAllReports(), issues: useAllPossibleIssues() }),
      { wrapper: server.wrapper },
    );
    await waitFor(() =>
      expect(
        result.current.reports.isPending || result.current.issues.isPending,
      ).toBe(false),
    );
    state.findings.audit_a = [];
    await act(() => invalidateCrossProject(server.queryClient));
    await waitFor(() => expect(result.current.issues.issues).toHaveLength(0));
    expect(server.count("/v1/findings")).toBe(2);
    expect(server.count("/v1/audits/audit_b/report")).toBe(1);
    state.audits.project_b![0] = audit("project_b", "audit_b", "completed", {
      revision: 2,
    });
    await act(() => invalidateCrossProject(server.queryClient));
    await waitFor(() =>
      expect(server.count("/v1/audits/audit_b/report")).toBe(2),
    );
  });
  it("keeps other reports when a payload fails and treats missing reports as normal", async () => {
    const state = base();
    state.audits.project_a![0] = audit("project_a", "audit_a", "completed");
    state.failures = { "/v1/audits/audit_a/report": true };
    state.reports = { audit_b: report("ready") };
    const { result } = renderHook(() => useAllReports(), {
      wrapper: setup(state).wrapper,
    });
    await settled(result);
    expect(result.current.reports).toHaveLength(1);
    expect(result.current.partial).toBe(true);
  });
  it("keeps disabled lists idle", async () => {
    const server = setup(base());
    const { result } = renderHook(
      () => ({
        checks: useAllChecks({ enabled: false }),
        issues: useAllPossibleIssues({ enabled: false }),
        decisions: usePendingDecisions({ enabled: false }),
        reports: useAllReports({ enabled: false }),
      }),
      { wrapper: server.wrapper },
    );
    expect(result.current.checks.isPending).toBe(true);
    expect(server.requests).toEqual([]);
  });
});

describe("collection continuations", () => {
  it("stops on a repeated cursor and exposes the failure", async () => {
    const load = vi.fn(async () => ({
      items: [1],
      page: { hasMore: true, nextCursor: "same" },
    }));
    const result = await collectOwnerPages(load);
    expect(load).toHaveBeenCalledTimes(2);
    expect(result.error?.message).toMatch(/invalid page/);
  });
  it("rejects a hasMore page without a cursor", async () => {
    const result = await collectOwnerPages(async () => ({
      items: [],
      page: { hasMore: true },
    }));
    expect(result.error).toBeInstanceOf(Error);
  });
});
