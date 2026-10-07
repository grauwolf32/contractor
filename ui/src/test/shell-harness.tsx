import { ownerListResponse } from "./owner-lists";
import { makeFinding } from "../routes/decisions/test-support";
/**
 * Renders the application shell around a probe page against a small fake
 * Server, for the shell, account menu and command palette tests.
 */
import { QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import { createMemoryRouter, useLocation } from "react-router";
import { RouterProvider } from "react-router/dom";
import { vi } from "vitest";

import type {
  Audit,
  AuditProfile,
  AuditReviewRequest,
  AuditState,
} from "../api/audits";
import { PublicAPI, type AuthSession } from "../api/client";
import { PublicAPIProvider } from "../api/context";
import type { Project } from "../api/projects";
import { queryKeys } from "../api/query-keys";
import type { WorkflowSummary } from "../api/workflows";
import { createApplicationQueryClient } from "../app/query-client";
import { ApplicationShell } from "../app/shell";
import { SessionProvider, type SessionAPI } from "../auth/session";

type Capability = AuthSession["principal"]["capabilities"][number];

const digest = `sha256:${"a".repeat(64)}`;

export function projectFixture(
  projectId: string,
  overrides: Partial<Project> = {},
): Project {
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

export function auditFixture(
  projectId: string,
  auditId: string,
  state: AuditState,
  objective?: string,
): Audit {
  const base = {
    auditId,
    projectId,
    profile: { name: "openapi-operation-trace", version: "1", digest },
    inputs: {},
    scope: objective === undefined ? {} : { objective },
    runtimeLabels: [],
    state,
    revision: 1,
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
    updatedAt: "2026-10-01T10:00:00Z",
  };
  return state === "draft"
    ? { ...base, phase: "not-started" }
    : { ...base, phase: "rounds", currentRoundId: "round_1" };
}

export function reviewFixture(
  auditId: string,
  requestId: string,
  kind: AuditReviewRequest["kind"],
): AuditReviewRequest {
  return {
    requestId,
    auditId,
    subjectKind: "audit-item-action",
    subjectId: "subject_1",
    kind,
    subjectRevision: 1,
    subjectDigest: digest,
    requestedActions: ["approve", "reject"],
    state: "pending",
    revision: 1,
    createdAt: "2026-10-03T10:00:00Z",
    updatedAt: "2026-10-03T10:00:00Z",
  };
}

export function workflowFixture(
  name: string,
  version: string,
  displayName?: string,
): WorkflowSummary {
  return {
    ref: { name, version },
    ...(displayName === undefined
      ? {}
      : { presentation: { displayName, description: "" } }),
    entryStage: "execute",
    parameters: {},
    inputs: {},
    outputs: {},
  };
}

/** What the fake Server holds. Everything else answers an empty page. */
export interface FakeServer {
  projects?: Project[];
  audits?: Record<string, Audit[]>;
  /** Possible issues in state `proposed` per check, as the Server counts them. */
  proposedFindings?: Record<string, number>;
  reviews?: Record<string, AuditReviewRequest[]>;
  profiles?: AuditProfile[];
  workflows?: WorkflowSummary[];
}

export interface ShellOptions {
  capabilities?: Capability[];
  server?: FakeServer;
  logout?: SessionAPI["logout"];
}

function json(value: unknown): Response {
  return new Response(JSON.stringify(value), {
    headers: {
      "Content-Type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
    },
  });
}

function answer(server: FakeServer, url: URL): Response {
  const page = (items: unknown[]) => ({ items, page: { hasMore: false } });
  const path = url.pathname;
  const ownerResponse = ownerListResponse(url, {
    projects: server.projects ?? [],
    audits: Object.values(server.audits ?? {}).flat(),
    findings: Object.entries(server.proposedFindings ?? {}).flatMap(
      ([auditId, count]) =>
        Array.from({ length: count }, (_, i) =>
          makeFinding({ auditId, findingId: `finding_${auditId}_${i}` }),
        ),
    ),
    reviews: Object.values(server.reviews ?? {}).flat(),
  });
  if (ownerResponse !== undefined) return ownerResponse;
  if (path === "/v1/projects") return json(page(server.projects ?? []));
  if (path === "/v1/audit-profiles") return json(page(server.profiles ?? []));
  if (path === "/v1/workflows") return json(page(server.workflows ?? []));
  const projectAudits = /^\/v1\/projects\/([^/]+)\/audits$/.exec(path);
  if (projectAudits !== null) {
    const projectId = decodeURIComponent(projectAudits[1] ?? "");
    return json(page(server.audits?.[projectId] ?? []));
  }
  const perAudit = /^\/v1\/audits\/([^/]+)\/(findings|reviews)$/.exec(path);
  if (perAudit !== null) {
    const auditId = decodeURIComponent(perAudit[1] ?? "");
    const summary = { auditRevision: 1, asOf: "2026-10-03T10:00:00Z" };
    if (perAudit[2] === "findings") {
      const total =
        url.searchParams.get("state") === "proposed"
          ? (server.proposedFindings?.[auditId] ?? 0)
          : 0;
      return json({ ...summary, total, ...page([]) });
    }
    const reviews = server.reviews?.[auditId] ?? [];
    return json({ ...summary, total: reviews.length, ...page(reviews) });
  }
  return json(page([]));
}

export const TEST_USERNAME = "owner";

export function sessionFixture(capabilities: Capability[]): AuthSession {
  return {
    principal: { userId: "user_local", username: TEST_USERNAME, capabilities },
    csrfToken: "a".repeat(43),
    idleExpiresAt: "2026-10-06T20:00:00Z",
    absoluteExpiresAt: "2026-10-07T12:00:00Z",
  };
}

/** The shell at `path` with a probe page that prints its location. */
export function renderShell(path: string, options: ShellOptions = {}) {
  const session = sessionFixture(
    options.capabilities ?? ["user", "operations"],
  );
  const server = options.server ?? {};
  const requests: URL[] = [];
  const publicAPI = new PublicAPI(
    {
      uiVersion: "0.1.0",
      supportedApiVersions: ["contractor.public.v1"],
      apiBaseUrl: "http://127.0.0.1:8080",
    },
    vi.fn(async (input: RequestInfo | URL) => {
      const request = input instanceof Request ? input : new Request(input);
      const url = new URL(request.url);
      requests.push(url);
      return answer(server, url);
    }),
  );
  const sessionAPI: SessionAPI = {
    getSession: vi.fn(async () => session),
    login: vi.fn(async () => session),
    logout: vi.fn(options.logout ?? (async () => undefined)),
  };
  const queryClient = createApplicationQueryClient();
  queryClient.setQueryData(queryKeys.session, session);

  function PageProbe() {
    const location = useLocation();
    return (
      <p>
        Page at {location.pathname}
        {location.search}
      </p>
    );
  }

  const router = createMemoryRouter(
    [
      {
        element: <ApplicationShell />,
        children: [{ path: "*", element: <PageProbe /> }],
      },
    ],
    { initialEntries: [path] },
  );
  const view = render(
    <QueryClientProvider client={queryClient}>
      <PublicAPIProvider api={publicAPI}>
        <SessionProvider api={sessionAPI}>
          <RouterProvider router={router} />
        </SessionProvider>
      </PublicAPIProvider>
    </QueryClientProvider>,
  );
  return { ...view, router, requests, sessionAPI, queryClient };
}
