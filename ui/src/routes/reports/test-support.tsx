// Fixtures and a fake Server for the Reports tests.
import type { RouteObject } from "react-router";

import type {
  Audit,
  AuditReport,
  AuditReviewAction,
  AuditReviewRequest,
  AuditState,
} from "../../api/audits";
import type { Project } from "../../api/projects";
import * as build from "../../test/builders";
import {
  failure,
  fakeServer,
  json,
  page,
  type FakeCall,
  type Route,
} from "../../test/fake-server";
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

export function acceptanceReview(
  auditId: string,
  overrides: Partial<AuditReviewRequest> = {},
): AuditReviewRequest {
  return build.review(auditId, `review_${auditId}`, "report-acceptance", {
    subjectRevision: 4,
    ...overrides,
  });
}

/**
 * A proposed or ready report with both files, a proposed one carrying
 * acceptanceReview; others carry only a status.
 */
export function report(
  auditId: string,
  status: AuditReport["status"],
  options: { summary?: string | undefined; review?: AuditReviewRequest } = {},
): AuditReport {
  return build.report(auditId, status, {
    summary: options.summary,
    review: options.review ?? acceptanceReview(auditId),
  });
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

/**
 * A Server with projects, checks and reports. Deciding a report acceptance
 * request finishes the check: approve makes the report ready, reject makes
 * it unavailable, and the report no longer carries the request.
 */
export function reportsServer(state: ServerState) {
  const reviews = new Map<string, AuditReviewRequest>();
  for (const value of Object.values(state.reports))
    if (typeof value !== "function" && value.review !== undefined)
      reviews.set(value.review.requestId, value.review);
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
      reviews: [...reviews.values()],
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
    "POST /v1/audits/{auditId}/reviews/{requestId}/decisions": held(
      async ({ params, request }: FakeCall) => {
        const review = reviews.get(params.requestId ?? "");
        const target = auditOf(params.auditId);
        if (review === undefined || target === undefined)
          return failure(404, "not_found", "not found");
        const body = (await request.clone().json()) as {
          action: AuditReviewAction;
          rationale: string;
        };
        const decision = build.decision(review, {
          decisionId: "decision_1",
          action: body.action,
          actorId: "user_owner",
          rationale: body.rationale,
        });
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
                summary: (state.reports[finished.auditId] as AuditReport)
                  .summary,
              })
            : { status: "unavailable" };
        return json({ request: decided, decision, replayed: false });
      },
    ),
    "GET /v1/audits/{auditId}/reviews/{requestId}": held(async ({ params }) => {
      const answer = await state.reviewRead?.();
      if (answer !== undefined) return answer;
      const review = reviews.get(params.requestId ?? "");
      return review === undefined
        ? failure(404, "not_found", "not found")
        : json(review, 200, { ETag: `"${review.revision}"` });
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
  return { ...server, state, reviews };
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
