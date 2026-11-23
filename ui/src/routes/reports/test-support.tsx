// Fixtures and a fake Server for the Reports tests.
import type { RouteObject } from "react-router";

import type { Audit, AuditReport, AuditState } from "../../api/audits";
import type { Project } from "../../api/projects";
import * as build from "../../test/builders";
import { fakeServer, type Route } from "../../test/fake-server";
import { failure, json, page } from "../../test/responses";
import { ownerListRoutes } from "../../test/owner-lists";
import { renderWithAPI } from "../../test/render";
import { ReportsRoute } from "./index";

export function project(projectId: string, name: string): Project {
  return build.project(projectId, { name });
}

export function audit(
  projectId: string,
  auditId: string,
  state: AuditState,
  options: { profile?: string; updatedAt?: string; revision?: number } = {},
): Audit {
  return build.audit(projectId, auditId, state, {
    profile: {
      name: options.profile ?? "owasp-top10-2025-source-risk",
      version: "1",
      digest: build.DIGEST,
    },
    revision: options.revision ?? 4,
    holdState: "held",
    dispatchState: "closed",
    reservedRunCount: 1,
    submittedRunCount: 1,
    eventSequence: 4,
    updatedAt: options.updatedAt ?? build.NOW,
  });
}

/**
 * A ready report has both exact files; other responses carry only a status.
 */
export function report(
  auditId: string,
  status: AuditReport["status"],
  options: { summary?: string | undefined } = {},
): AuditReport {
  return build.report(auditId, status, options);
}

export interface ServerState {
  projects: Project[];
  audits: Audit[];
  /** By check; a function answers per request (e.g. to fail a refetch). */
  reports: Record<string, AuditReport | (() => Response)>;
  failingProjects?: string[];
  indexFails?: boolean; /** Requests whose path matches wait for `until` before they are answered. */
  hold?: { path: RegExp; until: Promise<void> } | undefined;
}

/**
 * A Server with projects, checks and exact retained reports.
 */
export function reportsServer(state: ServerState) {
  const auditOf = (auditId: string | undefined) =>
    state.audits.find((candidate) => candidate.auditId === auditId);
  /** Requests matching `state.hold` wait for it before they are answered. */
  const held =
    (route: Route): Route =>
    async (call) => {
      if (state.hold?.path.test(call.url.pathname) === true)
        await state.hold.until;
      return route(call);
    };
  const server = fakeServer({
    ...ownerListRoutes(() => ({
      projects: state.projects,
      audits: state.audits,
      findings: [],
    })),
    "GET /v1/projects": held(() =>
      state.indexFails === true
        ? failure(500, "internal_error", "Index failed")
        : json(page(state.projects)),
    ),
    "GET /v1/projects/{projectId}": held(({ params }) => {
      const found = state.projects.find(
        (candidate) => candidate.projectId === params.projectId,
      );
      return found === undefined
        ? failure(404, "not_found", "not found")
        : json(found, 200, { ETag: `"${found.revision}"` });
    }),
    "GET /v1/projects/{projectId}/audits": held(({ params, url }) => {
      const projectId = params.projectId ?? "";
      if (state.failingProjects?.includes(projectId)) return failure(500);
      const wanted = url.searchParams.get("state");
      return json(
        page(
          state.audits.filter(
            (candidate) =>
              candidate.projectId === projectId &&
              (wanted === null || candidate.state === wanted),
          ),
        ),
      );
    }),
    "GET /v1/audits/{auditId}": held(({ params }) => {
      const found = auditOf(params.auditId);
      return found === undefined
        ? failure(404, "not_found", "not found")
        : json(found, 200, { ETag: `"${found.revision}"` });
    }),
    "GET /v1/audits/{auditId}/report": held(({ params }) => {
      const found = auditOf(params.auditId);
      if (found === undefined) return failure(404, "not_found", "not found");
      if (state.failingProjects?.includes(found.projectId)) return failure(500);
      const value = state.reports[found.auditId];
      if (value === undefined) return failure(404, "not_found", "not found");
      return typeof value === "function" ? value() : json(value);
    }),
  });
  return { ...server, state };
}

export type ReportsServer = ReturnType<typeof reportsServer>;

/** Renders `routes` (default: /reports and /reports/:auditId) at `path`. */
export function renderRoutes(
  path: string,
  server: ReportsServer,
  routes: RouteObject[] = [
    { path: "/reports", element: <ReportsRoute /> },
    { path: "/reports/:auditId", element: <ReportsRoute /> },
  ],
) {
  return renderWithAPI(null, { path, server, routes });
}
