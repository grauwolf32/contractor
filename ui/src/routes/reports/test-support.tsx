// Fixtures and a fake Server for the Reports tests.
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import type { ReactElement } from "react";
import { createMemoryRouter, type RouteObject } from "react-router";
import { RouterProvider } from "react-router/dom";
import { vi } from "vitest";

import type {
  Audit,
  AuditReport,
  AuditReviewAction,
  AuditReviewRequest,
  AuditState,
} from "../../api/audits";
import { PublicAPI } from "../../api/client";
import { PublicAPIProvider } from "../../api/context";
import type { Project } from "../../api/projects";
import type { RuntimeConfig } from "../../config/runtime-config";
import { ReportsRoute } from "./index";

export const DIGEST = `sha256:${"a".repeat(64)}`;
export const NOW = "2026-10-05T10:00:00Z";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

export function project(projectId: string, name: string): Project {
  return {
    projectId,
    kind: "project",
    name,
    description: "",
    lifecycle: "active",
    revision: "1",
    createdAt: NOW,
    updatedAt: NOW,
  };
}

export function audit(
  projectId: string,
  auditId: string,
  state: AuditState,
  options: { profile?: string; updatedAt?: string; revision?: number } = {},
): Audit {
  return {
    auditId,
    projectId,
    profile: {
      name: options.profile ?? "owasp-top10-2025-source-risk",
      version: "1",
      digest: DIGEST,
    },
    inputs: {},
    scope: {},
    runtimeLabels: [],
    state,
    phase: "rounds",
    currentRoundId: "round_1",
    revision: options.revision ?? 4,
    dispatchState: "closed",
    holdState: "held",
    limits: {
      maxRounds: 1,
      batchSize: 1,
      maxItemsPerRound: 8,
      maxItemsTotal: 8,
      maxSubmittedRuns: 8,
      maxItemRunAttempts: 2,
      maxEvidenceBytes: 1_048_576,
    },
    reservedRunCount: 1,
    submittedRunCount: 1,
    outstandingRunCount: 0,
    retainedEvidenceBytes: 0,
    eventSequence: 4,
    createdAt: NOW,
    updatedAt: options.updatedAt ?? NOW,
  };
}

export function acceptanceReview(
  auditId: string,
  overrides: Partial<AuditReviewRequest> = {},
): AuditReviewRequest {
  return {
    requestId: `review_${auditId}`,
    auditId,
    subjectKind: "audit-report",
    subjectId: auditId,
    kind: "report-acceptance",
    subjectRevision: 4,
    subjectDigest: DIGEST,
    requestedActions: ["approve", "reject"],
    state: "pending",
    revision: 1,
    createdAt: NOW,
    updatedAt: NOW,
    ...overrides,
  };
}

/** A proposed or ready report with both files; others carry only a status. */
export function report(
  auditId: string,
  status: AuditReport["status"],
  options: {
    summary?: string | undefined;
    review?: AuditReviewRequest | undefined;
  } = {},
): AuditReport {
  if (status !== "proposed" && status !== "ready") return { status };
  return {
    status,
    ...(status === "proposed"
      ? { review: options.review ?? acceptanceReview(auditId) }
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
  };
}

function json(
  value: unknown,
  status = 200,
  headers: Record<string, string> = {},
): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "Content-Type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
      ...headers,
    },
  });
}

export function failure(status: number, message = "Server failed"): Response {
  return json(
    {
      code: status === 404 ? "not_found" : "internal_error",
      message,
      retryable: false,
      requestId: "request_test",
    },
    status,
  );
}

export interface ServerState {
  projects: Project[];
  audits: Audit[];
  /** By check; a function answers per request (e.g. to fail a refetch). */
  reports: Record<string, AuditReport | (() => Response)>;
  failingProjects?: string[];
  indexFails?: boolean;
  /**
   * Answers reads of review requests instead of the stored request (e.g. to
   * fail them); answering undefined falls back to the stored request.
   */
  reviewRead?:
    (() => Response | undefined | Promise<Response | undefined>) | undefined;
  /** Requests whose path matches wait for `until` before they are answered. */
  hold?: { path: RegExp; until: Promise<void> } | undefined;
}

/** A promise for ServerState.hold and the function that settles it. */
export function gate(): { until: Promise<void>; release: () => void } {
  let release: () => void = () => undefined;
  const until = new Promise<void>((resolve) => {
    release = resolve;
  });
  return { until, release };
}

/**
 * A Server with projects, checks and reports. Deciding a report acceptance
 * request finishes the check: approve makes the report ready, reject makes
 * it unavailable, and the report no longer carries the request.
 */
export function fakeServer(state: ServerState) {
  const requests: Request[] = [];
  const reviews = new Map<string, AuditReviewRequest>();
  for (const value of Object.values(state.reports))
    if (typeof value !== "function" && value.review !== undefined)
      reviews.set(value.review.requestId, value.review);
  const auditOf = (auditId: string) =>
    state.audits.find((candidate) => candidate.auditId === auditId);

  async function handle(request: Request): Promise<Response> {
    const url = new URL(request.url);
    const path = url.pathname;
    if (state.hold?.path.test(path) === true) await state.hold.until;
    if (path === "/v1/projects") {
      if (state.indexFails === true) return failure(500, "Index failed");
      return json({ items: state.projects, page: { hasMore: false } });
    }
    let match = /^\/v1\/projects\/([^/]+)$/.exec(path);
    if (match !== null) {
      const found = state.projects.find(
        (candidate) => candidate.projectId === match?.[1],
      );
      return found === undefined
        ? failure(404, "not found")
        : json(found, 200, { ETag: `"${found.revision}"` });
    }
    match = /^\/v1\/projects\/([^/]+)\/audits$/.exec(path);
    if (match !== null) {
      const projectId = match[1] ?? "";
      if (state.failingProjects?.includes(projectId)) return failure(500);
      const wanted = url.searchParams.get("state");
      return json({
        items: state.audits.filter(
          (candidate) =>
            candidate.projectId === projectId &&
            (wanted === null || candidate.state === wanted),
        ),
        page: { hasMore: false },
      });
    }
    match = /^\/v1\/audits\/([^/]+)\/reviews\/([^/]+)\/decisions$/.exec(path);
    if (match !== null && request.method === "POST") {
      const review = reviews.get(match[2] ?? "");
      const target = auditOf(match[1] ?? "");
      if (review === undefined || target === undefined)
        return failure(404, "not found");
      const body = (await request.clone().json()) as {
        action: AuditReviewAction;
        rationale: string;
      };
      const decision = {
        decisionId: "decision_1",
        requestId: review.requestId,
        auditId: review.auditId,
        action: body.action,
        actorId: "user_owner",
        rationale: body.rationale,
        subjectRevision: review.subjectRevision,
        subjectDigest: review.subjectDigest,
        createdAt: NOW,
      };
      const decided: AuditReviewRequest = {
        ...review,
        state: "decided",
        revision: review.revision + 1,
        decision,
      };
      reviews.set(review.requestId, decided);
      const finished: Audit = {
        ...target,
        state: body.action === "approve" ? "completed" : "failed",
        revision: target.revision + 1,
      };
      state.audits = state.audits.map((candidate) =>
        candidate.auditId === finished.auditId ? finished : candidate,
      );
      state.reports[finished.auditId] =
        body.action === "approve"
          ? report(finished.auditId, "ready", {
              summary: (state.reports[finished.auditId] as AuditReport).summary,
            })
          : { status: "unavailable" };
      return json({ request: decided, decision, replayed: false });
    }
    match = /^\/v1\/audits\/([^/]+)\/reviews\/([^/]+)$/.exec(path);
    if (match !== null) {
      const answer = await state.reviewRead?.();
      if (answer !== undefined) return answer;
      const review = reviews.get(match[2] ?? "");
      return review === undefined
        ? failure(404, "not found")
        : json(review, 200, { ETag: `"${review.revision}"` });
    }
    match = /^\/v1\/audits\/([^/]+)(\/report)?$/.exec(path);
    if (match !== null) {
      const found = auditOf(match[1] ?? "");
      if (found === undefined) return failure(404, "not found");
      if (match[2] === undefined)
        return json(found, 200, { ETag: `"${found.revision}"` });
      const value = state.reports[found.auditId];
      if (value === undefined) return failure(404, "not found");
      return typeof value === "function" ? value() : json(value);
    }
    throw new Error(`unexpected ${request.method} ${path}`);
  }

  return {
    state,
    requests,
    handle,
    reviews,
    /** Requests with this method whose path ends with `suffix`. */
    sent: (method: string, suffix: string) =>
      requests.filter(
        (request) =>
          request.method === method &&
          new URL(request.url).pathname.endsWith(suffix),
      ),
  };
}

export type FakeServer = ReturnType<typeof fakeServer>;

/** Renders `routes` (default: /reports and /reports/:auditId) at `path`. */
export function renderRoutes(
  path: string,
  server: FakeServer,
  routes: RouteObject[] = [
    { path: "/reports", element: <ReportsRoute /> },
    { path: "/reports/:auditId", element: <ReportsRoute /> },
  ],
) {
  const api = new PublicAPI(
    runtimeConfig,
    vi.fn(async (input: RequestInfo | URL) => {
      const request = input instanceof Request ? input : new Request(input);
      server.requests.push(request.clone());
      return server.handle(request);
    }),
  );
  api.csrf.replace("a".repeat(43));
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, refetchOnWindowFocus: false },
      mutations: { retry: false },
    },
  });
  const router = createMemoryRouter(
    // Links out of the page land here; `location()` tells where.
    [...routes, { path: "*", element: <p>Elsewhere</p> }],
    { initialEntries: [path] },
  );
  const wrap = (node: ReactElement) => (
    <QueryClientProvider client={queryClient}>
      <PublicAPIProvider api={api}>{node}</PublicAPIProvider>
    </QueryClientProvider>
  );
  const view = render(wrap(<RouterProvider router={router} />));
  return {
    ...view,
    api,
    router,
    queryClient,
    /** The current path and query string. */
    location: () =>
      router.state.location.pathname + router.state.location.search,
  };
}
