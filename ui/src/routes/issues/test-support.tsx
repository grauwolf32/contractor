// A fake Public API for the Issues tests: projects, checks, possible
// issues and review requests, with the filters, ETags and decisions the
// Server applies.
import type {
  Audit,
  AuditFinding,
  AuditReviewRequest,
  AuditState,
} from "../../api/audits";
import type { ResponseOperation } from "../../api/contract";
import type { Project } from "../../api/projects";
import * as build from "../../test/builders";
import {
  fakeServer,
  type FakeCall,
  type Route,
  type Routes,
} from "../../test/fake-server";
import { failure, json, page } from "../../test/responses";
import { ownerListRoutes } from "../../test/owner-lists";
import { renderWithAPI } from "../../test/render";
import { IssuesRoute } from ".";

const { DIGEST } = build;

/** A held check of `projectId` over the shop's source archive. */
export function check(
  projectId: string,
  auditId: string,
  state: AuditState = "completed",
  profile = "openapi-operation-trace",
): Audit {
  return build.audit(projectId, auditId, state, {
    profile: { name: profile, version: "1", digest: DIGEST },
    inputs: {
      source: build.auditInput("shop-source", {
        ref: { namespace: "sources", name: "shop-source", revision: "src-r1" },
      }),
    },
    holdState: "held",
    dispatchState: "closed",
  });
}

/** A possible issue of `auditId`, proposed at `createdAt`. */
export function issue(
  auditId: string,
  findingId: string,
  title: string,
  createdAt: string,
  overrides: Partial<AuditFinding> = {},
  document: Partial<AuditFinding["firstProposal"]["document"]> = {},
): AuditFinding {
  const base = build.finding(
    auditId,
    findingId,
    { createdAt, updatedAt: createdAt, revision: 1, ...overrides },
    { title, ...document },
  );
  return build.conforming("AuditFinding", {
    ...base,
    firstProposal: { ...base.firstProposal, createdAt },
  });
}

/** What the fake Server of the Issues pages holds. */
export interface IssuesServer {
  projects: Project[];
  audits: Audit[];
  findings: AuditFinding[];
  reviews: AuditReviewRequest[];
  /** Checks whose possible-issue reads fail. */
  failing: Set<string>;
  /** Checks whose possible-issue pages report more. */
  more: Set<string>;
  /** Possible-issue list reads wait for this (exact reads do not). */
  gate?: Promise<void> | undefined;
  /** Recording a decision waits for this. */
  decisionGate?: Promise<void> | undefined;
  /**
   * Every request waits for the promise this returns, if any: tests that
   * answer reads in a chosen order.
   */
  hold?: ((call: FakeCall) => Promise<void> | undefined) | undefined;
}

function bump(server: IssuesServer, auditId: string) {
  server.audits = server.audits.map((candidate) =>
    candidate.auditId === auditId
      ? {
          ...candidate,
          revision: candidate.revision + 1,
          updatedAt: new Date(
            Date.parse(candidate.updatedAt) + 60_000,
          ).toISOString(),
        }
      : candidate,
  );
}

const VERDICT_STATES = {
  true_positive: "confirmed",
  false_positive: "rejected",
  needs_evidence: "needs-evidence",
  duplicate: "duplicate",
  reopen: "proposed",
} as const;

/** The routes of the Issues pages over `server`. */
function routes(server: IssuesServer) {
  const auditOf = (auditId: string | undefined) =>
    server.audits.find((candidate) => candidate.auditId === auditId);
  const summary = (audit: Audit) => ({
    auditRevision: audit.revision,
    asOf: audit.updatedAt,
  });
  const owner = ownerListRoutes(() => server);
  return {
    ...owner,
    "GET /v1/findings": async (call: FakeCall) => {
      await server.gate;
      if (server.failing.size > 0)
        return failure(503, "unavailable", "Findings unavailable");
      return owner["GET /v1/findings"]!(call);
    },
    "GET /v1/projects": () => json(page(server.projects)),
    "GET /v1/projects/{projectId}": ({ params }: FakeCall) => {
      const found = server.projects.find(
        (candidate) => candidate.projectId === params.projectId,
      );
      return found === undefined
        ? failure(404, "not_found", "Project not found")
        : json(found, 200, { ETag: '"1"' });
    },
    "GET /v1/projects/{projectId}/audits": ({ params, url }: FakeCall) => {
      const state = url.searchParams.get("state");
      return json(
        page(
          server.audits.filter(
            (candidate) =>
              candidate.projectId === params.projectId &&
              (state === null || candidate.state === state),
          ),
        ),
      );
    },
    "GET /v1/audits/{auditId}": ({ params }: FakeCall) => {
      const audit = auditOf(params.auditId);
      return audit === undefined
        ? failure(404, "not_found", "No check")
        : json(audit, 200, { ETag: `"${audit.revision}"` });
    },
    "GET /v1/audits/{auditId}/findings": async ({ params, url }: FakeCall) => {
      const audit = auditOf(params.auditId);
      if (audit === undefined) return failure(404, "not_found", "No check");
      await server.gate;
      if (server.failing.has(audit.auditId))
        return failure(503, "unavailable", "Findings unavailable");
      const state = url.searchParams.get("state");
      const verdict = url.searchParams.get("verdict");
      const severity = url.searchParams.get("severity");
      const items = server.findings.filter(
        (candidate) =>
          candidate.auditId === audit.auditId &&
          (state === null || candidate.state === state) &&
          (verdict === null ||
            (verdict === "unreviewed"
              ? candidate.analystVerdict === undefined
              : candidate.analystVerdict === verdict)) &&
          (severity === null || candidate.analystSeverity === severity),
      );
      // A check in `more` has a second page; it holds no further match.
      const more = server.more.has(audit.auditId);
      const next = url.searchParams.get("cursor") === "more";
      return json({
        ...summary(audit),
        total: items.length + (more ? 50 : 0),
        ...page(next ? [] : items, more && !next ? "more" : undefined),
      });
    },
    "GET /v1/audits/{auditId}/findings/{findingId}": ({ params }: FakeCall) => {
      const finding = server.findings.find(
        (candidate) =>
          candidate.auditId === params.auditId &&
          candidate.findingId === params.findingId,
      );
      return finding === undefined
        ? failure(404, "not_found", "Possible issue not found")
        : json(finding, 200, { ETag: `"${finding.revision}"` });
    },
    "GET /v1/audits/{auditId}/findings/{findingId}/provenance": ({
      params,
    }: FakeCall) => {
      const audit = auditOf(params.auditId);
      const finding = server.findings.find(
        (candidate) =>
          candidate.auditId === params.auditId &&
          candidate.findingId === params.findingId,
      );
      if (audit === undefined || finding === undefined)
        return failure(404, "not_found", "Possible issue not found");
      return json({
        auditRevision: audit.revision,
        findingRevision: finding.revision,
        items: [
          {
            recordId: "source:receipt_1",
            kind: "source-proposal",
            relation: "first",
            receiptId: "receipt_1",
            proposal: finding.firstProposal.proposal,
            origin: finding.firstProposal.origin,
            supportsCurrentAssessment: false,
            createdAt: finding.createdAt,
          },
        ],
        page: { hasMore: false },
      });
    },
    "POST /v1/audits/{auditId}/findings/{findingId}/reviews": ({
      params,
    }: FakeCall) => {
      const finding = server.findings.find(
        (candidate) =>
          candidate.auditId === params.auditId &&
          candidate.findingId === params.findingId,
      );
      if (finding === undefined)
        return failure(404, "not_found", "Possible issue not found");
      const created = build.review(
        finding.auditId,
        `review_${finding.findingId}`,
        "finding-triage",
        {
          findingId: finding.findingId,
          subjectId: finding.findingId,
          subjectRevision: finding.revision,
        },
      );
      server.reviews.push(created);
      return json(created, 201, { ETag: '"1"' });
    },
    "GET /v1/audits/{auditId}/reviews": ({ params, url }: FakeCall) => {
      const audit = auditOf(params.auditId);
      if (audit === undefined) return failure(404, "not_found", "No check");
      const finding = url.searchParams.get("finding");
      const state = url.searchParams.get("state");
      const items = server.reviews.filter(
        (candidate) =>
          candidate.auditId === audit.auditId &&
          (finding === null || candidate.findingId === finding) &&
          (state === null || candidate.state === state),
      );
      return json({ ...summary(audit), total: items.length, ...page(items) });
    },
    "GET /v1/audits/{auditId}/reviews/{requestId}": ({ params }: FakeCall) => {
      const review = server.reviews.find(
        (candidate) => candidate.requestId === params.requestId,
      );
      return review === undefined
        ? failure(404, "not_found", "Review not found")
        : json(review, 200, { ETag: `"${review.revision}"` });
    },
    "POST /v1/audits/{auditId}/reviews/{requestId}/decisions": async ({
      params,
      request,
    }: FakeCall) => {
      await server.decisionGate;
      const auditId = params.auditId ?? "";
      const review = server.reviews.find(
        (candidate) => candidate.requestId === params.requestId,
      );
      if (review === undefined)
        return failure(404, "not_found", "Review not found");
      const body = (await request.json()) as {
        verdict: AuditFinding["analystVerdict"];
        severity?: AuditFinding["analystSeverity"];
        rationale: string;
      };
      const decision = build.decision(review, {
        auditId,
        ...(body.verdict === undefined ? {} : { verdict: body.verdict }),
        ...(body.severity === undefined ? {} : { severity: body.severity }),
        rationale: body.rationale,
        subjectDigest: DIGEST,
      });
      let decided: AuditFinding | undefined;
      server.findings = server.findings.map((candidate) => {
        if (candidate.findingId !== review.findingId) return candidate;
        decided = {
          ...candidate,
          state: VERDICT_STATES[body.verdict ?? "reopen"],
          revision: candidate.revision + 1,
          ...(body.verdict === undefined
            ? {}
            : { analystVerdict: body.verdict }),
          ...(body.severity === undefined
            ? {}
            : { analystSeverity: body.severity }),
          analystDecision: decision,
        };
        return decided;
      });
      const updated = {
        ...review,
        state: "decided" as const,
        revision: 2,
        decision,
      };
      server.reviews = server.reviews.map((candidate) =>
        candidate.requestId === review.requestId ? updated : candidate,
      );
      bump(server, auditId);
      return json({
        finding: decided,
        request: updated,
        decision,
        replayed: false,
      });
    },
  };
}

/** `table` with every route answering once `server.hold` lets it. */
function held(server: IssuesServer, table: Routes): Routes {
  const answers: Routes = {};
  for (const [operation, route] of Object.entries(table) as [
    ResponseOperation,
    Route,
  ][])
    answers[operation] = async (call) => {
      await server.hold?.(call);
      return route(call);
    };
  return answers;
}

/** Renders the Issues route at `path` against `server`. */
export function renderIssues(server: IssuesServer, path: string) {
  return renderWithAPI(null, {
    path,
    server: fakeServer(held(server, routes(server))),
    routes: [
      { path: "/issues", element: <IssuesRoute /> },
      { path: "/issues/:auditId/:findingId", element: <IssuesRoute /> },
      {
        path: "/projects/:projectId/audits/:auditId/:section",
        element: <p>Check page</p>,
      },
    ],
  });
}

export function issuesServer(
  initial: Partial<IssuesServer> & Pick<IssuesServer, "projects" | "audits">,
): IssuesServer {
  return {
    findings: [],
    reviews: [],
    failing: new Set(),
    more: new Set(),
    ...initial,
  };
}
