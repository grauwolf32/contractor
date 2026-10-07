// A fake Server and router around the Start page, for its tests.
import { QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { RouterProvider } from "react-router/dom";
import { vi } from "vitest";

import type { ArtifactMetadata } from "../../../api/artifacts";
import type { Audit, AuditProfile } from "../../../api/audits";
import { PublicAPI } from "../../../api/client";
import { PublicAPIProvider } from "../../../api/context";
import type { Project } from "../../../api/projects";
import { createApplicationQueryClient } from "../../../app/query-client";
import type { RuntimeConfig } from "../../../config/runtime-config";
import { StartCheckRoute } from "../start";
import { draftAudit, projectFixture, startResponse } from "./test-fixtures";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

export const CSRF_TOKEN = "a".repeat(43);

export function json(
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

export function failure(status: number, code: string, message: string) {
  return json({ code, message, retryable: false, requestId: "req_1" }, status);
}

/** A gateway's error page: no API version header, no error envelope. */
export function gatewayFailure(status: number): Response {
  return new Response("Gateway error", {
    status,
    headers: { "Content-Type": "text/plain" },
  });
}

export function page(items: unknown[], more?: { nextCursor: string }) {
  return {
    items,
    page:
      more === undefined
        ? { hasMore: false }
        : { hasMore: true, nextCursor: more.nextCursor },
  };
}

type Answer = (request: Request, url: URL) => Response | Promise<Response>;

export interface FakeStartServer {
  /** Projects listed by the picker. */
  projects?: Project[];
  /** The project read by ID; null answers 404. */
  project?: Project | null;
  profiles?: AuditProfile[];
  /** GET /v1/audit-profiles; one page of `profiles` otherwise. */
  profileList?: Answer;
  /** Detail responses by "name@version"; the list item otherwise. */
  details?: Record<string, AuditProfile>;
  materials?: ArtifactMetadata[] | Answer;
  /** POST /v1/projects/{id}/audits; a draft of the request's profile otherwise. */
  create?: Answer;
  /** POST /v1/audits/{id}/start; the started draft otherwise. */
  start?: Answer;
  /**
   * GET /v1/audits/{id}; otherwise the check as the default create and start
   * answers left it (a draft at revision 1 when they were replaced).
   */
  audit?: Answer;
}

function auditResponse(audit: Audit): Response {
  return json(audit, 200, { ETag: `"${audit.revision}"` });
}

export function renderStart(path: string, server: FakeStartServer = {}) {
  const requests: Request[] = [];
  const project =
    server.project === undefined ? projectFixture() : server.project;
  const profiles = server.profiles ?? [];
  // Checks as the default create and start answers left them.
  const audits = new Map<string, Audit>();

  async function answer(request: Request, url: URL): Promise<Response> {
    const path = url.pathname;
    if (path === "/v1/projects") return json(page(server.projects ?? []));
    const projectRead = /^\/v1\/projects\/([^/]+)$/.exec(path);
    if (projectRead !== null) {
      const id = decodeURIComponent(projectRead[1] ?? "");
      return project === null || project.projectId !== id
        ? failure(404, "not_found", "Project not found")
        : json(project, 200, { ETag: `"${project.revision}"` });
    }
    if (path === "/v1/audit-profiles")
      return server.profileList === undefined
        ? json(page(profiles))
        : server.profileList(request, url);
    const detail = /^\/v1\/audit-profiles\/([^/]+)\/versions\/([^/]+)$/.exec(
      path,
    );
    if (detail !== null) {
      const key = `${decodeURIComponent(detail[1] ?? "")}@${decodeURIComponent(detail[2] ?? "")}`;
      const found =
        server.details?.[key] ??
        profiles.find(
          (profile) => `${profile.ref.name}@${profile.ref.version}` === key,
        );
      return found === undefined
        ? failure(404, "not_found", "Check type not found")
        : json(found, 200, { ETag: `"${found.ref.digest}"` });
    }
    if (/^\/v1\/projects\/[^/]+\/artifacts$/.test(path)) {
      const materials = server.materials ?? [];
      return typeof materials === "function"
        ? materials(request, url)
        : json(page(materials));
    }
    if (/^\/v1\/projects\/[^/]+\/audits$/.test(path)) {
      if (request.method !== "POST") return json(page([]));
      if (server.create !== undefined) return server.create(request, url);
      const body = (await request.clone().json()) as {
        profile: { name: string; version: string };
      };
      const profile = profiles.find(
        (candidate) =>
          candidate.ref.name === body.profile.name &&
          candidate.ref.version === body.profile.version,
      );
      if (profile === undefined)
        return failure(404, "not_found", "Check type not found");
      const draft = draftAudit(profile);
      audits.set(draft.auditId, draft);
      return json(draft, 201, { ETag: '"1"' });
    }
    const start = /^\/v1\/audits\/([^/]+)\/start$/.exec(path);
    if (start !== null && request.method === "POST") {
      if (server.start !== undefined) return server.start(request, url);
      const auditId = decodeURIComponent(start[1] ?? "");
      const draft =
        audits.get(auditId) ?? draftAudit(profiles[0]!, { auditId });
      const started = startResponse(draft);
      audits.set(auditId, started.audit);
      return json(started, 200, { ETag: `"${started.audit.revision}"` });
    }
    const auditRead = /^\/v1\/audits\/([^/]+)$/.exec(path);
    if (auditRead !== null && request.method === "GET") {
      if (server.audit !== undefined) return server.audit(request, url);
      const auditId = decodeURIComponent(auditRead[1] ?? "");
      return auditResponse(
        audits.get(auditId) ?? draftAudit(profiles[0]!, { auditId }),
      );
    }
    return failure(404, "not_found", `No fake route for ${path}`);
  }

  const api = new PublicAPI(
    runtimeConfig,
    vi.fn(async (input: RequestInfo | URL) => {
      const request = input instanceof Request ? input : new Request(input);
      requests.push(request.clone());
      return answer(request, new URL(request.url));
    }),
  );
  api.csrf.replace(CSRF_TOKEN);
  const queryClient = createApplicationQueryClient();
  const router = createMemoryRouter(
    [
      { path: "/checks/new", element: <StartCheckRoute /> },
      // Elsewhere: tests read router.state.location.
      { path: "*", element: <p>Another page</p> },
    ],
    { initialEntries: [path] },
  );
  const user = userEvent.setup();
  const view = render(
    <QueryClientProvider client={queryClient}>
      <PublicAPIProvider api={api}>
        <RouterProvider router={router} />
      </PublicAPIProvider>
    </QueryClientProvider>,
  );
  return {
    ...view,
    user,
    router,
    queryClient,
    requests,
    /** Requests with this method whose path ends with `suffix`. */
    sent: (method: string, suffix: string) =>
      requests.filter(
        (request) =>
          request.method === method &&
          new URL(request.url).pathname.endsWith(suffix),
      ),
  };
}
