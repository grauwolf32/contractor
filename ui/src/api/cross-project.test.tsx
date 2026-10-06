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
  CROSS_PROJECT_LIMITS,
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
import { queryKeys } from "./query-keys";

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
  projectsHasMore?: boolean;
  projectsFail?: boolean;
  audits: Record<string, Audit[]>;
  auditsHasMore?: string[];
  failingProjects?: string[];
  findings?: Record<string, AuditFinding[]>;
  failingFindings?: string[];
  reviews?: Record<string, AuditReviewRequest[]>;
  reports?: Record<string, AuditReport>;
  failingReports?: string[];
  /** Holds responses for a path until the returned promise resolves. */
  hold?: (url: URL) => Promise<void> | undefined;
}

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "Content-Type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
    },
  });
}

function failure(status: number): Response {
  const [code, message] =
    status === 404
      ? ["not_found", "Not found"]
      : status === 400
        ? ["invalid_argument", "Invalid request"]
        : ["internal_error", "Server failed"];
  return json(
    {
      code,
      message,
      retryable: false,
      requestId: "request_test",
    },
    status,
  );
}

function page<T>(items: T[], url: URL, hasMore = false) {
  const limit = Number(url.searchParams.get("limit") ?? "50");
  return {
    items: items.slice(0, limit),
    page: { hasMore: hasMore || items.length > limit },
  };
}

function setup(state: ServerState) {
  const requests: URL[] = [];
  const findAudit = (auditId: string) =>
    Object.values(state.audits)
      .flat()
      .find((candidate) => candidate.auditId === auditId);
  const fetch = vi.fn(async (input: RequestInfo | URL) => {
    const request = input instanceof Request ? input : new Request(input);
    const url = new URL(request.url);
    requests.push(url);
    await state.hold?.(url);
    const path = url.pathname;
    if (path === "/v1/projects") {
      if (state.projectsFail === true) return failure(500);
      return json(page(state.projects, url, state.projectsHasMore));
    }
    const projectAudits = /^\/v1\/projects\/([^/]+)\/audits$/.exec(path);
    if (projectAudits !== null) {
      const projectId = decodeURIComponent(projectAudits[1]!);
      if (state.failingProjects?.includes(projectId)) return failure(500);
      const wanted = url.searchParams.get("state");
      const items = (state.audits[projectId] ?? []).filter(
        (candidate) => wanted === null || candidate.state === wanted,
      );
      return json(
        page(items, url, state.auditsHasMore?.includes(projectId) ?? false),
      );
    }
    const perAudit = /^\/v1\/audits\/([^/]+)\/(findings|reviews|report)$/.exec(
      path,
    );
    if (perAudit === null) return failure(404);
    const auditId = decodeURIComponent(perAudit[1]!);
    const current = findAudit(auditId);
    if (current === undefined) return failure(404);
    const summary = {
      auditRevision: current.revision,
      asOf: current.updatedAt,
    };
    if (perAudit[2] === "findings") {
      if (state.failingFindings?.includes(auditId)) return failure(500);
      const wantedState = url.searchParams.get("state");
      const wantedVerdict = url.searchParams.get("verdict");
      const wantedSeverity = url.searchParams.get("severity");
      // validateFindingList: an unreviewed finding has no analyst severity.
      if (wantedVerdict === "unreviewed" && wantedSeverity !== null)
        return failure(400);
      const items = (state.findings?.[auditId] ?? []).filter(
        (candidate) =>
          (wantedState === null || candidate.state === wantedState) &&
          (wantedVerdict === null ||
            (wantedVerdict === "unreviewed"
              ? candidate.analystVerdict === undefined
              : candidate.analystVerdict === wantedVerdict)) &&
          (wantedSeverity === null ||
            candidate.analystSeverity === wantedSeverity),
      );
      return json({ ...summary, total: items.length, ...page(items, url) });
    }
    if (perAudit[2] === "reviews") {
      const wanted = url.searchParams.get("state");
      const items = (state.reviews?.[auditId] ?? []).filter(
        (candidate) => wanted === null || candidate.state === wanted,
      );
      return json({ ...summary, total: items.length, ...page(items, url) });
    }
    if (state.failingReports?.includes(auditId)) return failure(500);
    const found = state.reports?.[auditId];
    return found === undefined ? failure(404) : json(found);
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
  const requested = (pattern: RegExp) =>
    requests.filter((url) => pattern.test(url.pathname));
  return { requests, requested, queryClient, wrapper };
}

async function settle(result: { current: { isPending: boolean } }) {
  await waitFor(() => expect(result.current.isPending).toBe(false));
}

describe("useProjectsIndex", () => {
  it("lists the first page of active projects and polls it", async () => {
    const server = setup({
      projects: [
        project("project_a"),
        project("project_gone", {
          lifecycle: "deleting",
          deletion: {
            phase: "draining",
            requestedAt: "2026-10-04T10:00:00Z",
          },
        }),
      ],
      projectsHasMore: true,
      audits: {},
    });
    const { result } = renderHook(() => useProjectsIndex(), {
      wrapper: server.wrapper,
    });
    await settle(result);

    expect(result.current.projects.map((item) => item.projectId)).toEqual([
      "project_a",
    ]);
    expect(result.current.truncated).toBe(true);
    expect(result.current.error).toBeNull();
    const [request] = server.requested(/^\/v1\/projects$/);
    expect(request?.searchParams.get("kind")).toBe("project");
    expect(request?.searchParams.get("limit")).toBe(
      String(CROSS_PROJECT_LIMITS.projects),
    );

    const options = server.queryClient
      .getQueryCache()
      .find({ queryKey: queryKeys.crossProject.projects })
      ?.observers[0]?.options;
    expect(options?.refetchInterval).toBe(CROSS_PROJECT_LIMITS.pollMs);
    expect(options?.refetchIntervalInBackground).toBe(false);
    expect(options?.refetchOnWindowFocus).toBe(false);
    expect(options?.retry).toBe(false);
  });
});

describe("useAllChecks", () => {
  const state = (): ServerState => ({
    projects: [project("project_a"), project("project_b")],
    audits: {
      project_a: [
        audit("project_a", "audit_a1", "active", {
          updatedAt: "2026-10-05T10:00:00Z",
        }),
        audit("project_a", "audit_a2", "completed", {
          updatedAt: "2026-10-02T10:00:00Z",
        }),
      ],
      project_b: [
        audit("project_b", "audit_b1", "waiting_review", {
          updatedAt: "2026-10-04T10:00:00Z",
        }),
        audit("project_b", "audit_b2", "draft", {
          updatedAt: "2026-10-01T10:00:00Z",
        }),
      ],
    },
  });

  it("fans out over projects and merges checks newest first", async () => {
    const server = setup(state());
    const { result } = renderHook(() => useAllChecks(), {
      wrapper: server.wrapper,
    });
    await settle(result);

    expect(
      result.current.checks.map(
        ({ project, audit }) => `${project.projectId}/${audit.auditId}`,
      ),
    ).toEqual([
      "project_a/audit_a1",
      "project_b/audit_b1",
      "project_a/audit_a2",
      "project_b/audit_b2",
    ]);
    expect(result.current).toMatchObject({
      truncated: false,
      partial: false,
      errors: [],
      error: null,
    });
    const pages = server.requested(/^\/v1\/projects\/[^/]+\/audits$/);
    expect(pages.map((url) => url.pathname).sort()).toEqual([
      "/v1/projects/project_a/audits",
      "/v1/projects/project_b/audits",
    ]);
    for (const url of pages) {
      expect(url.searchParams.get("limit")).toBe(
        String(CROSS_PROJECT_LIMITS.auditsPerProject),
      );
      expect(url.searchParams.has("state")).toBe(false);
    }
  });

  it("keeps the other projects when one project fails", async () => {
    const server = setup({ ...state(), failingProjects: ["project_b"] });
    const { result } = renderHook(() => useAllChecks(), {
      wrapper: server.wrapper,
    });
    await settle(result);

    expect(result.current.checks.map(({ audit }) => audit.auditId)).toEqual([
      "audit_a1",
      "audit_a2",
    ]);
    expect(result.current.partial).toBe(true);
    expect(result.current.error).toBeNull();
    expect(result.current.errors).toHaveLength(1);
    expect(result.current.errors[0]).toMatchObject({
      scope: "project",
      projectId: "project_b",
      error: { status: 500 },
    });
    expect(result.current.errors[0]).not.toHaveProperty("auditId");
  });

  it("reports a failed project index as the error", async () => {
    const server = setup({ ...state(), projectsFail: true });
    const { result } = renderHook(() => useAllChecks(), {
      wrapper: server.wrapper,
    });
    await settle(result);

    expect(result.current.error).toMatchObject({ status: 500 });
    expect(result.current.checks).toEqual([]);
    expect(result.current.partial).toBe(false);
    expect(result.current.errors).toEqual([]);
  });

  it("keeps the last project index when a refresh fails and lists the failure", async () => {
    const current = state();
    const server = setup(current);
    const { result } = renderHook(
      () => ({ index: useProjectsIndex(), checks: useAllChecks() }),
      { wrapper: server.wrapper },
    );
    await waitFor(() => expect(result.current.checks.isPending).toBe(false));
    expect(result.current.checks.errors).toEqual([]);

    current.projectsFail = true;
    await act(() =>
      server.queryClient.refetchQueries({
        queryKey: queryKeys.crossProject.projects,
      }),
    );
    await waitFor(() => expect(result.current.checks.partial).toBe(true));

    const failed = {
      scope: "index",
      error: expect.objectContaining({ status: 500 }),
    };
    expect(result.current.index).toMatchObject({
      partial: true,
      error: null,
      errors: [failed],
    });
    expect(result.current.index.projects).toHaveLength(2);
    expect(result.current.checks.error).toBeNull();
    expect(result.current.checks.errors).toEqual([failed]);
    expect(result.current.checks.checks).toHaveLength(4);
  });

  it("flags truncated project and check pages", async () => {
    const server = setup({ ...state(), auditsHasMore: ["project_a"] });
    const { result } = renderHook(() => useAllChecks(), {
      wrapper: server.wrapper,
    });
    await settle(result);
    expect(result.current.truncated).toBe(true);

    const more = setup({ ...state(), projectsHasMore: true });
    const index = renderHook(() => useAllChecks(), { wrapper: more.wrapper });
    await settle(index.result);
    expect(index.result.current.truncated).toBe(true);
  });

  it("filters one state on the Server and several on the client", async () => {
    const server = setup(state());
    const single = renderHook(() => useAllChecks({ states: ["active"] }), {
      wrapper: server.wrapper,
    });
    await settle(single.result);
    expect(
      single.result.current.checks.map(({ audit }) => audit.auditId),
    ).toEqual(["audit_a1"]);
    expect(
      server
        .requested(/\/audits$/)
        .every((url) => url.searchParams.get("state") === "active"),
    ).toBe(true);

    const other = setup(state());
    const several = renderHook(
      () => useAllChecks({ states: ["completed", "waiting_review"] }),
      { wrapper: other.wrapper },
    );
    await settle(several.result);
    expect(
      several.result.current.checks.map(({ audit }) => audit.auditId),
    ).toEqual(["audit_b1", "audit_a2"]);
    expect(
      other.requested(/\/audits$/).some((url) => url.searchParams.has("state")),
    ).toBe(false);
  });

  it("keeps result references while the data is unchanged", async () => {
    const server = setup(state());
    const { result, rerender } = renderHook(
      () => useAllChecks({ states: ["active", "completed"] }),
      { wrapper: server.wrapper },
    );
    await settle(result);
    const first = result.current;
    rerender();
    expect(result.current).toBe(first);

    await act(() =>
      server.queryClient.refetchQueries({
        queryKey: queryKeys.crossProject.all,
      }),
    );
    expect(server.requested(/^\/v1\/projects$/)).toHaveLength(2);
    expect(result.current.checks).toBe(first.checks);
  });
});

describe("useAllPossibleIssues", () => {
  const state = (): ServerState => ({
    projects: [project("project_a"), project("project_b")],
    audits: {
      project_a: [
        audit("project_a", "audit_a1", "active", { revision: 3 }),
        audit("project_a", "audit_draft", "draft"),
        audit("project_a", "audit_deleting", "deleting"),
      ],
      project_b: [
        audit("project_b", "audit_b1", "completed"),
        audit("project_b", "audit_b2", "cancelling"),
      ],
    },
    findings: {
      audit_a1: [
        finding("audit_a1", "finding_a1_old", {
          createdAt: "2026-10-02T09:00:00Z",
        }),
        finding("audit_a1", "finding_a1_confirmed", {
          state: "confirmed",
          verdict: "true_positive",
          createdAt: "2026-10-02T11:00:00Z",
        }),
      ],
      audit_b1: [
        finding("audit_b1", "finding_b1", {
          createdAt: "2026-10-02T10:00:00Z",
        }),
      ],
      audit_b2: [
        finding("audit_b2", "finding_b2_rejected", {
          state: "rejected",
          verdict: "false_positive",
          createdAt: "2026-10-02T08:00:00Z",
        }),
      ],
    },
  });

  it("reads every check that ran and lists issues newest first", async () => {
    const server = setup(state());
    const { result } = renderHook(() => useAllPossibleIssues(), {
      wrapper: server.wrapper,
    });
    await settle(result);

    expect(
      result.current.issues.map(({ finding }) => finding.findingId),
    ).toEqual([
      "finding_a1_confirmed",
      "finding_b1",
      "finding_a1_old",
      "finding_b2_rejected",
    ]);
    expect(result.current.issues[1]).toMatchObject({
      project: { projectId: "project_b" },
      audit: { auditId: "audit_b1" },
    });
    expect(result.current.total).toBe(4);
    expect(
      server
        .requested(/\/findings$/)
        .map((url) => url.pathname)
        .sort(),
    ).toEqual([
      "/v1/audits/audit_a1/findings",
      "/v1/audits/audit_b1/findings",
      "/v1/audits/audit_b2/findings",
    ]);
  });

  it("filters states and verdicts on the Server", async () => {
    const server = setup(state());
    const proposed = renderHook(
      () => useAllPossibleIssues({ states: ["proposed"] }),
      { wrapper: server.wrapper },
    );
    await settle(proposed.result);
    expect(
      proposed.result.current.issues.map(({ finding }) => finding.findingId),
    ).toEqual(["finding_b1", "finding_a1_old"]);
    expect(
      server
        .requested(/\/findings$/)
        .every(
          (url) =>
            url.searchParams.get("state") === "proposed" &&
            !url.searchParams.has("verdict"),
        ),
    ).toBe(true);

    const other = setup(state());
    const decided = renderHook(
      () =>
        useAllPossibleIssues({
          states: ["confirmed", "rejected"],
          verdicts: ["true_positive"],
        }),
      { wrapper: other.wrapper },
    );
    await settle(decided.result);
    expect(
      decided.result.current.issues.map(({ finding }) => finding.findingId),
    ).toEqual(["finding_a1_confirmed"]);
    const filters = other
      .requested(/\/audits\/audit_a1\/findings$/)
      .map(
        (url) =>
          `${url.searchParams.get("state")}+${url.searchParams.get("verdict")}`,
      )
      .sort();
    expect(filters).toEqual([
      "confirmed+true_positive",
      "rejected+true_positive",
    ]);
  });

  it("keeps the other checks when one check fails", async () => {
    const server = setup({ ...state(), failingFindings: ["audit_b1"] });
    const { result } = renderHook(() => useAllPossibleIssues(), {
      wrapper: server.wrapper,
    });
    await settle(result);

    expect(
      result.current.issues.map(({ finding }) => finding.findingId),
    ).toEqual([
      "finding_a1_confirmed",
      "finding_a1_old",
      "finding_b2_rejected",
    ]);
    expect(result.current.partial).toBe(true);
    expect(result.current.errors).toEqual([
      {
        scope: "check",
        projectId: "project_b",
        auditId: "audit_b1",
        error: expect.objectContaining({ status: 500 }),
      },
    ]);
    // A failed check read retries on the poll interval and stays fetchable;
    // a settled one is pinned to its revision and never polls.
    const failed = server.queryClient.getQueryCache().findAll({
      queryKey: queryKeys.crossProject.findingsOf("audit_b1", null, null, null),
    })[0]!;
    const settled = server.queryClient.getQueryCache().findAll({
      queryKey: queryKeys.crossProject.findingsOf("audit_a1", null, null, null),
    })[0]!;
    const interval = failed.observers[0]!.options.refetchInterval as (
      query: unknown,
    ) => number | false;
    expect(interval(failed)).toBe(CROSS_PROJECT_LIMITS.pollMs);
    expect(interval(settled)).toBe(false);
    expect(failed.isStatic()).toBe(false);
    expect(settled.isStatic()).toBe(true);
  });

  it("refresh refetches the heads and failed check reads, not settled ones", async () => {
    const current: ServerState = { ...state(), failingFindings: ["audit_b1"] };
    const server = setup(current);
    const { result } = renderHook(() => useAllPossibleIssues(), {
      wrapper: server.wrapper,
    });
    await settle(result);
    expect(result.current.partial).toBe(true);

    current.failingFindings = [];
    await act(() => result.current.refetch());
    await waitFor(() => expect(result.current.partial).toBe(false));

    expect(
      result.current.issues.map(({ finding }) => finding.findingId),
    ).toContain("finding_b1");
    expect(server.requested(/^\/v1\/projects$/)).toHaveLength(2);
    expect(server.requested(/project_a\/audits$/)).toHaveLength(2);
    expect(server.requested(/project_b\/audits$/)).toHaveLength(2);
    expect(server.requested(/audit_b1\/findings$/)).toHaveLength(2);
    expect(server.requested(/audit_a1\/findings$/)).toHaveLength(1);
    expect(server.requested(/audit_b2\/findings$/)).toHaveLength(1);
  });

  it("keeps issue references across renders and unchanged refetches", async () => {
    const server = setup(state());
    const { result, rerender } = renderHook(() => useAllPossibleIssues(), {
      wrapper: server.wrapper,
    });
    await settle(result);
    const first = result.current;
    rerender();
    expect(result.current).toBe(first);

    await act(() =>
      server.queryClient.refetchQueries({
        queryKey: queryKeys.crossProject.all,
      }),
    );
    // The heads refetched with unchanged data; check reads stay pinned.
    expect(server.requested(/^\/v1\/projects$/)).toHaveLength(2);
    expect(server.requested(/audit_a1\/findings$/)).toHaveLength(1);
    expect(result.current.issues).toBe(first.issues);
  });

  it("looks up previous revisions once per change of the checks, not per render", async () => {
    const current = state();
    let release = () => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    current.hold = (url) =>
      url.pathname.endsWith("/audit_a1/findings") ? held : undefined;
    const server = setup(current);
    const { result, rerender } = renderHook(() => useAllPossibleIssues(), {
      wrapper: server.wrapper,
    });
    await waitFor(() =>
      expect(
        result.current.issues.map(({ finding }) => finding.findingId),
      ).toEqual(["finding_b1", "finding_b2_rejected"]),
    );
    expect(result.current.isPending).toBe(true);

    // Rendering while a check read loads never scans the query cache.
    const scan = vi.spyOn(server.queryClient.getQueryCache(), "findAll");
    rerender();
    rerender();
    rerender();
    expect(scan).not.toHaveBeenCalled();

    release();
    await settle(result);
    expect(result.current.issues).toHaveLength(4);
  });

  it("caps each check's oldest page and counts the rest on the Server", async () => {
    // The Server lists possible issues oldest first.
    const many = Array.from(
      { length: CROSS_PROJECT_LIMITS.findingsPerAudit + 2 },
      (_, index) =>
        finding("audit_a1", `finding_many_${String(index).padStart(3, "0")}`, {
          createdAt: new Date(Date.UTC(2026, 9, 2, 0, index)).toISOString(),
        }),
    );
    const server = setup({ ...state(), findings: { audit_a1: many } });
    const { result } = renderHook(() => useAllPossibleIssues(), {
      wrapper: server.wrapper,
    });
    await settle(result);

    const listed = result.current.issues.map(
      ({ finding }) => finding.findingId,
    );
    expect(listed).toHaveLength(CROSS_PROJECT_LIMITS.findingsPerAudit);
    // The newest two are the ones left out; the check says where to look.
    expect(listed).not.toContain("finding_many_051");
    expect(listed).not.toContain("finding_many_050");
    expect(listed[0]).toBe("finding_many_049");
    expect(result.current.truncated).toBe(true);
    expect(result.current.truncatedAuditIds).toEqual(["audit_a1"]);
    expect(result.current.total).toBe(
      CROSS_PROJECT_LIMITS.findingsPerAudit + 2,
    );
  });

  it("reads only the wanted check states that can hold possible issues", async () => {
    const server = setup(state());
    const { result } = renderHook(
      () =>
        useAllPossibleIssues({ checkStates: ["active", "completed", "draft"] }),
      { wrapper: server.wrapper },
    );
    await settle(result);
    expect(
      result.current.issues.map(({ finding }) => finding.findingId),
    ).toEqual(["finding_a1_confirmed", "finding_b1", "finding_a1_old"]);
    expect(
      server
        .requested(/\/findings$/)
        .map((url) => url.pathname)
        .sort(),
    ).toEqual(["/v1/audits/audit_a1/findings", "/v1/audits/audit_b1/findings"]);

    // No wanted state can hold possible issues: nothing is listed.
    const other = setup(state());
    const none = renderHook(
      () => useAllPossibleIssues({ checkStates: ["draft", "deleting"] }),
      { wrapper: other.wrapper },
    );
    await settle(none.result);
    expect(none.result.current.issues).toEqual([]);
    expect(none.result.current.total).toBe(0);
    expect(none.result.current.partial).toBe(false);
    expect(other.requested(/\/findings$/)).toEqual([]);
  });

  it("filters analyst severities on the Server and never with unreviewed", async () => {
    const server = setup({
      ...state(),
      findings: {
        audit_a1: [
          finding("audit_a1", "finding_high", {
            state: "confirmed",
            verdict: "true_positive",
            severity: "high",
          }),
          finding("audit_a1", "finding_low", {
            state: "confirmed",
            verdict: "true_positive",
            severity: "low",
          }),
          finding("audit_a1", "finding_open"),
        ],
      },
    });
    const { result } = renderHook(
      () =>
        useAllPossibleIssues({
          verdicts: ["true_positive", "unreviewed"],
          severities: ["high", "critical"],
        }),
      { wrapper: server.wrapper },
    );
    await settle(result);

    expect(
      result.current.issues.map(({ finding }) => finding.findingId),
    ).toEqual(["finding_high"]);
    expect(result.current.partial).toBe(false);
    expect(result.current.total).toBe(1);
    const filters = server
      .requested(/\/audits\/audit_a1\/findings$/)
      .map(
        (url) =>
          `${url.searchParams.get("verdict")}+${url.searchParams.get("severity")}`,
      )
      .sort();
    expect(filters).toEqual(["true_positive+critical", "true_positive+high"]);
    expect(
      server.queryClient.getQueryCache().findAll({
        queryKey: queryKeys.crossProject.findingsOf(
          "audit_a1",
          null,
          "true_positive",
          "high",
        ),
      }),
    ).toHaveLength(1);

    // An unreviewed possible issue has no analyst severity: nothing to read.
    const other = setup(state());
    const none = renderHook(
      () =>
        useAllPossibleIssues({
          verdicts: ["unreviewed"],
          severities: ["high"],
        }),
      { wrapper: other.wrapper },
    );
    await settle(none.result);
    expect(none.result.current.issues).toEqual([]);
    expect(none.result.current.total).toBe(0);
    expect(other.requested(/\/findings$/)).toEqual([]);
  });

  it("refetches only checks whose revision changed and keeps showing them meanwhile", async () => {
    const current = state();
    const server = setup(current);
    const { result } = renderHook(() => useAllPossibleIssues(), {
      wrapper: server.wrapper,
    });
    await settle(result);
    const listed = () =>
      result.current.issues.map(({ finding }) => finding.findingId);
    const before = listed();
    expect(server.requested(/audit_a1\/findings$/)).toHaveLength(1);
    expect(server.requested(/audit_b1\/findings$/)).toHaveLength(1);

    let release = () => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    current.hold = (url) =>
      url.pathname.endsWith("/audit_a1/findings") ? held : undefined;
    current.audits.project_a![0] = audit("project_a", "audit_a1", "active", {
      revision: 4,
    });
    current.findings!.audit_a1 = [
      ...current.findings!.audit_a1!,
      finding("audit_a1", "finding_a1_new", {
        createdAt: "2026-10-06T10:00:00Z",
      }),
    ];

    // A poll of the check pages reveals the new revision.
    await act(() =>
      server.queryClient.refetchQueries({
        queryKey: queryKeys.crossProject.allAudits,
      }),
    );
    await waitFor(() =>
      expect(server.requested(/audit_a1\/findings$/)).toHaveLength(2),
    );
    // The previous revision's issues stay listed with the updated check.
    expect(result.current.isPending).toBe(false);
    expect(listed()).toEqual(before);
    expect(
      result.current.issues.find(({ audit }) => audit.auditId === "audit_a1")
        ?.audit.revision,
    ).toBe(4);

    release();
    await waitFor(() =>
      expect(result.current.issues[0]?.finding.findingId).toBe(
        "finding_a1_new",
      ),
    );
    expect(server.requested(/audit_b1\/findings$/)).toHaveLength(1);
    expect(server.requested(/audit_b2\/findings$/)).toHaveLength(1);
  });
});

describe("usePendingDecisions", () => {
  const state = (): ServerState => ({
    projects: [project("project_a")],
    audits: {
      project_a: [
        audit("project_a", "audit_active", "active"),
        audit("project_a", "audit_waiting", "waiting_review"),
        audit("project_a", "audit_paused", "paused"),
        audit("project_a", "audit_done", "completed"),
        audit("project_a", "audit_draft", "draft"),
      ],
    },
    reviews: {
      audit_active: [
        review(
          "audit_active",
          "review_triage",
          "finding-triage",
          "2026-10-03T09:00:00Z",
        ),
        review(
          "audit_active",
          "review_active_test",
          "active-check-approval",
          "2026-10-03T11:00:00Z",
        ),
      ],
      audit_waiting: [
        review(
          "audit_waiting",
          "review_report",
          "report-acceptance",
          "2026-10-03T10:00:00Z",
        ),
      ],
      audit_done: [review("audit_done", "review_late", "finding-triage")],
    },
  });

  it("reads pending requests of running and waiting checks", async () => {
    const server = setup(state());
    const { result } = renderHook(() => usePendingDecisions(), {
      wrapper: server.wrapper,
    });
    await settle(result);

    expect(
      result.current.decisions.map(({ review }) => review.requestId),
    ).toEqual(["review_active_test", "review_report", "review_triage"]);
    const reads = server.requested(/\/reviews$/);
    expect(reads.map((url) => url.pathname).sort()).toEqual([
      "/v1/audits/audit_active/reviews",
      "/v1/audits/audit_paused/reviews",
      "/v1/audits/audit_waiting/reviews",
    ]);
    expect(
      reads.every((url) => url.searchParams.get("state") === "pending"),
    ).toBe(true);
  });

  it("filters kinds on the client from the same reads", async () => {
    const server = setup(state());
    const { result } = renderHook(
      () => ({
        all: usePendingDecisions(),
        other: usePendingDecisions({
          kinds: ["active-check-approval", "report-acceptance"],
        }),
      }),
      { wrapper: server.wrapper },
    );
    await waitFor(() => expect(result.current.other.isPending).toBe(false));

    expect(
      result.current.other.decisions.map(({ review }) => review.kind),
    ).toEqual(["active-check-approval", "report-acceptance"]);
    expect(server.requested(/audit_active\/reviews$/)).toHaveLength(1);
  });

  it("reads only the wanted check states that can ask for decisions", async () => {
    const server = setup(state());
    const { result } = renderHook(
      () => usePendingDecisions({ checkStates: INBOX_CHECK_STATES }),
      { wrapper: server.wrapper },
    );
    await settle(result);
    expect(
      result.current.decisions.map(({ review }) => review.requestId),
    ).toEqual(["review_active_test", "review_report", "review_triage"]);
    expect(
      server
        .requested(/\/reviews$/)
        .map((url) => url.pathname)
        .sort(),
    ).toEqual([
      "/v1/audits/audit_active/reviews",
      "/v1/audits/audit_waiting/reviews",
    ]);

    // Finished checks cannot ask for decisions: nothing is listed.
    const other = setup(state());
    const none = renderHook(
      () => usePendingDecisions({ checkStates: ["completed"] }),
      { wrapper: other.wrapper },
    );
    await settle(none.result);
    expect(none.result.current.decisions).toEqual([]);
    expect(other.requested(/\/reviews$/)).toEqual([]);
  });

  it("names the checks whose oldest page of requests was capped", async () => {
    const many = Array.from(
      { length: CROSS_PROJECT_LIMITS.findingsPerAudit + 1 },
      (_, index) =>
        review(
          "audit_waiting",
          `review_many_${String(index).padStart(3, "0")}`,
          "requirement-applicability",
          new Date(Date.UTC(2026, 9, 3, 0, index)).toISOString(),
        ),
    );
    const base = state();
    const server = setup({
      ...base,
      reviews: { ...base.reviews, audit_waiting: many },
    });
    const { result } = renderHook(() => usePendingDecisions(), {
      wrapper: server.wrapper,
    });
    await settle(result);

    const listed = result.current.decisions
      .filter(({ audit }) => audit.auditId === "audit_waiting")
      .map(({ review }) => review.requestId);
    expect(listed).toHaveLength(CROSS_PROJECT_LIMITS.findingsPerAudit);
    expect(listed).not.toContain("review_many_050");
    expect(result.current.truncated).toBe(true);
    expect(result.current.truncatedAuditIds).toEqual(["audit_waiting"]);
  });
});

describe("useAllReports", () => {
  const state = (): ServerState => ({
    projects: [project("project_a")],
    audits: {
      project_a: [
        audit("project_a", "audit_ready", "completed", {
          updatedAt: "2026-10-05T10:00:00Z",
        }),
        audit("project_a", "audit_proposed", "waiting_review", {
          updatedAt: "2026-10-04T10:00:00Z",
        }),
        audit("project_a", "audit_finishing", "finalizing", {
          updatedAt: "2026-10-03T10:00:00Z",
        }),
        audit("project_a", "audit_failed", "failed", {
          updatedAt: "2026-10-02T10:00:00Z",
        }),
        audit("project_a", "audit_stopped", "cancelled", {
          updatedAt: "2026-10-01T12:00:00Z",
        }),
        audit("project_a", "audit_running", "active"),
      ],
    },
    reports: {
      audit_ready: report("ready"),
      audit_proposed: report("proposed"),
      audit_finishing: report("pending"),
      audit_failed: report("unavailable"),
      // audit_stopped answers 404.
    },
  });

  it("lists existing reports and treats a missing one as no report", async () => {
    const server = setup(state());
    const { result } = renderHook(() => useAllReports(), {
      wrapper: server.wrapper,
    });
    await settle(result);

    expect(
      result.current.reports.map(
        ({ audit, report }) => `${audit.auditId}:${report.status}`,
      ),
    ).toEqual(["audit_ready:ready", "audit_proposed:proposed"]);
    expect(result.current.partial).toBe(false);
    expect(result.current.errors).toEqual([]);
    expect(
      server
        .requested(/\/report$/)
        .map((url) => url.pathname)
        .sort(),
    ).toEqual([
      "/v1/audits/audit_failed/report",
      "/v1/audits/audit_finishing/report",
      "/v1/audits/audit_proposed/report",
      "/v1/audits/audit_ready/report",
      "/v1/audits/audit_stopped/report",
    ]);
  });

  it("lists other statuses on request and reports failures", async () => {
    const server = setup({ ...state(), failingReports: ["audit_ready"] });
    const { result } = renderHook(
      () => useAllReports({ statuses: ["pending", "unavailable"] }),
      { wrapper: server.wrapper },
    );
    await settle(result);

    expect(
      result.current.reports.map(
        ({ audit, report }) => `${audit.auditId}:${report.status}`,
      ),
    ).toEqual(["audit_finishing:pending", "audit_failed:unavailable"]);
    expect(result.current.partial).toBe(true);
    expect(result.current.errors).toEqual([
      expect.objectContaining({
        projectId: "project_a",
        auditId: "audit_ready",
      }),
    ]);
  });

  it("lists every status for an empty status filter", async () => {
    const server = setup(state());
    const { result } = renderHook(() => useAllReports({ statuses: [] }), {
      wrapper: server.wrapper,
    });
    await settle(result);

    expect(result.current.reports.map(({ report }) => report.status)).toEqual([
      "ready",
      "proposed",
      "pending",
      "unavailable",
    ]);
  });
});

describe("enabled: false", () => {
  it("keeps every read idle", async () => {
    const server = setup({
      projects: [project("project_a")],
      audits: { project_a: [audit("project_a", "audit_a1", "active")] },
    });
    const { result } = renderHook(
      () => ({
        checks: useAllChecks({ enabled: false }),
        issues: useAllPossibleIssues({ enabled: false }),
        decisions: usePendingDecisions({ enabled: false }),
        reports: useAllReports({ enabled: false }),
      }),
      { wrapper: server.wrapper },
    );
    await act(async () => {
      await Promise.resolve();
    });

    expect(server.requests).toEqual([]);
    expect(result.current.checks.isPending).toBe(true);
    expect(result.current.issues.issues).toEqual([]);
  });
});

describe("useInboxSummary", () => {
  it("counts possible issues to review and other decisions of running and waiting checks", async () => {
    const many = Array.from(
      { length: CROSS_PROJECT_LIMITS.findingsPerAudit + 1 },
      (_, index) => finding("audit_waiting", `finding_waiting_${index}`),
    );
    const server = setup({
      projects: [project("project_a"), project("project_b")],
      audits: {
        project_a: [
          audit("project_a", "audit_active", "active"),
          audit("project_a", "audit_waiting", "waiting_review"),
          audit("project_a", "audit_paused", "paused"),
        ],
        project_b: [
          audit("project_b", "audit_done", "completed"),
          audit("project_b", "audit_finishing", "finalizing"),
        ],
      },
      findings: {
        audit_active: [
          finding("audit_active", "finding_open"),
          finding("audit_active", "finding_confirmed", {
            state: "confirmed",
            verdict: "true_positive",
          }),
          finding("audit_active", "finding_evidence", {
            state: "needs-evidence",
          }),
        ],
        audit_waiting: many,
        audit_paused: [finding("audit_paused", "finding_paused")],
        audit_done: [finding("audit_done", "finding_done")],
        audit_finishing: [finding("audit_finishing", "finding_finishing")],
      },
      reviews: {
        audit_active: [
          review("audit_active", "review_triage", "finding-triage"),
          review("audit_active", "review_test", "active-check-approval"),
        ],
        audit_waiting: [
          review("audit_waiting", "review_report", "report-acceptance"),
          review("audit_waiting", "review_scope", "requirement-applicability"),
        ],
        audit_paused: [
          review("audit_paused", "review_paused", "active-check-approval"),
        ],
        audit_finishing: [
          review("audit_finishing", "review_finishing", "report-acceptance"),
        ],
      },
    });
    const { result } = renderHook(() => useInboxSummary(), {
      wrapper: server.wrapper,
    });
    expect(result.current.needsDecision).toBeUndefined();

    await waitFor(() => expect(result.current.needsDecision).toBeDefined());
    expect(result.current).toEqual({
      needsDecision: 1 + many.length + 3,
      possibleIssues: 1 + many.length,
      otherDecisions: 3,
      partial: false,
      truncated: false,
    });
    const inboxChecks = ["/v1/audits/audit_active", "/v1/audits/audit_waiting"];
    expect(
      server
        .requested(/\/findings$/)
        .map((url) => url.pathname)
        .sort(),
    ).toEqual(inboxChecks.map((path) => `${path}/findings`));
    expect(
      server
        .requested(/\/findings$/)
        .every((url) => url.searchParams.get("state") === "proposed"),
    ).toBe(true);
    expect(
      server
        .requested(/\/reviews$/)
        .map((url) => url.pathname)
        .sort(),
    ).toEqual(inboxChecks.map((path) => `${path}/reviews`));
  });

  it("does not read possible issues of finished checks", async () => {
    const server = setup({
      projects: [project("project_a")],
      audits: {
        project_a: [
          audit("project_a", "audit_done", "completed"),
          audit("project_a", "audit_failed", "failed"),
          audit("project_a", "audit_stopped", "cancelled"),
        ],
      },
      findings: {
        audit_done: [finding("audit_done", "finding_untriaged")],
        audit_failed: [finding("audit_failed", "finding_failed")],
      },
    });
    const { result } = renderHook(() => useInboxSummary(), {
      wrapper: server.wrapper,
    });
    await waitFor(() => expect(result.current.needsDecision).toBeDefined());

    expect(result.current).toMatchObject({
      needsDecision: 0,
      possibleIssues: 0,
      otherDecisions: 0,
    });
    expect(server.requested(/\/findings$/)).toEqual([]);
    expect(server.requested(/\/reviews$/)).toEqual([]);
  });

  it("stays unknown when the project index fails", async () => {
    const server = setup({ projects: [], projectsFail: true, audits: {} });
    const { result } = renderHook(() => useInboxSummary(), {
      wrapper: server.wrapper,
    });
    await waitFor(() =>
      expect(server.requested(/^\/v1\/projects$/).length).toBeGreaterThan(0),
    );
    await waitFor(() =>
      expect(
        server.queryClient
          .getQueryCache()
          .find({ queryKey: queryKeys.crossProject.projects })?.state.status,
      ).toBe("error"),
    );
    expect(result.current.needsDecision).toBeUndefined();
  });
});

describe("invalidateCrossProject", () => {
  it("refetches the polled heads and only the checks that changed", async () => {
    const current: ServerState = {
      projects: [project("project_a")],
      audits: {
        project_a: [
          audit("project_a", "audit_one", "active", { revision: 5 }),
          audit("project_a", "audit_two", "waiting_review", { revision: 7 }),
        ],
      },
      findings: {
        audit_one: [finding("audit_one", "finding_one")],
        audit_two: [finding("audit_two", "finding_two")],
      },
    };
    const server = setup(current);
    const useLists = () => ({
      issues: useAllPossibleIssues({ states: ["proposed"] }),
      summary: useInboxSummary(),
    });
    const { result, unmount } = renderHook(useLists, {
      wrapper: server.wrapper,
    });
    await waitFor(() => expect(result.current.summary.needsDecision).toBe(2));

    // The user confirms finding_one: the Server advances audit_one.
    current.findings!.audit_one = [
      finding("audit_one", "finding_one", {
        state: "confirmed",
        verdict: "true_positive",
      }),
    ];
    current.audits.project_a![0] = audit("project_a", "audit_one", "active", {
      revision: 6,
    });
    await act(() => invalidateCrossProject(server.queryClient));

    await waitFor(() => expect(result.current.summary.needsDecision).toBe(1));
    expect(
      result.current.issues.issues.map(({ finding }) => finding.findingId),
    ).toEqual(["finding_two"]);
    expect(server.requested(/^\/v1\/projects$/)).toHaveLength(2);
    expect(server.requested(/project_a\/audits$/)).toHaveLength(2);
    expect(server.requested(/audit_one\/findings$/)).toHaveLength(2);
    expect(server.requested(/audit_two\/findings$/)).toHaveLength(1);
    // The unchanged check's read keeps its data and is not invalidated...
    const unchanged = server.queryClient.getQueryCache().find({
      queryKey: [
        ...queryKeys.crossProject.findingsOf(
          "audit_two",
          "proposed",
          null,
          null,
        ),
        7,
      ],
    });
    expect(unchanged?.state.isInvalidated).toBe(false);

    // ...so lists mounted again (fresh heads, pinned check reads) read
    // nothing.
    const reads = server.requests.length;
    unmount();
    const again = renderHook(useLists, { wrapper: server.wrapper });
    await waitFor(() =>
      expect(again.result.current.summary.needsDecision).toBe(1),
    );
    expect(again.result.current.issues.isPending).toBe(false);
    await act(async () => {
      await Promise.resolve();
    });
    expect(server.requests).toHaveLength(reads);
  });
});
