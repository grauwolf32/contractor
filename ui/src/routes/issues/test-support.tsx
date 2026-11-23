// Canonical Project Findings and independent immutable producer receipts.
import type {
  Audit,
  ProjectFinding,
  FindingProposalReceipt,
  AuditState,
  EditProjectFindingRequest,
} from "../../api/audits";
import type { Project } from "../../api/projects";
import type { ResponseOperation } from "../../api/contract";
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
export function check(
  projectId: string,
  auditId: string,
  state: AuditState = "completed",
  profile = "openapi-operation-trace",
): Audit {
  return build.audit(projectId, auditId, state, {
    profile: { name: profile, version: "1", digest: build.DIGEST },
    holdState: "held",
    dispatchState: "closed",
  });
}
export function issue(
  projectId: string,
  findingId: string,
  title: string,
  createdAt: string,
  overrides: Partial<ProjectFinding> = {},
  claim: Partial<NonNullable<ProjectFinding["claim"]>> = {},
): ProjectFinding {
  return build.finding(
    projectId,
    findingId,
    { createdAt, updatedAt: createdAt, ...overrides },
    { title, ...claim },
  );
}
export interface IssuesServer {
  projects: Project[];
  audits: Audit[];
  findings: ProjectFinding[];
  sources: FindingProposalReceipt[];
  failing: Set<string>;
  more: Set<string>;
  gate?: Promise<void> | undefined;
  decisionGate?: Promise<void> | undefined;
  hold?: ((call: FakeCall) => Promise<void> | undefined) | undefined;
}
const VERDICT_STATES = {
  true_positive: "confirmed",
  false_positive: "rejected",
  needs_evidence: "needs-evidence",
  duplicate: "duplicate",
  reopen: "proposed",
} as const;
function routes(server: IssuesServer): Routes {
  const owner = ownerListRoutes(() => server);
  const finding = (p: Record<string, string | undefined>) =>
    server.findings.find(
      (f) => f.projectId === p.projectId && f.findingId === p.findingId,
    );
  const sources = () => server.sources;
  return {
    ...owner,
    "GET /v1/findings": async (call) => {
      await server.gate;
      if (server.failing.size > 0)
        return failure(
          503,
          "storage_transaction_conflict",
          "Findings unavailable",
        );
      return owner["GET /v1/findings"]!(call);
    },
    "GET /v1/projects": () => json(page(server.projects)),
    "GET /v1/projects/{projectId}": ({ params }) => {
      const p = server.projects.find((p) => p.projectId === params.projectId);
      return p === undefined ? failure(404) : json(p, 200, { ETag: '"1"' });
    },
    "GET /v1/projects/{projectId}/audits": ({ params }) =>
      json(page(server.audits.filter((a) => a.projectId === params.projectId))),
    "GET /v1/audits/{auditId}": ({ params }) => {
      const a = server.audits.find((a) => a.auditId === params.auditId);
      return a === undefined
        ? failure(404)
        : json(a, 200, { ETag: `"${a.revision}"` });
    },
    "GET /v1/projects/{projectId}/findings": ({ params }) =>
      json({
        projectRevision: 1,
        ...page(
          server.findings
            .filter((f) => f.projectId === params.projectId)
            .map(build.findingSummary),
        ),
      }),
    "GET /v1/projects/{projectId}/findings/{findingId}": ({ params }) => {
      const f = finding(params);
      return f === undefined
        ? failure(404)
        : json(f, 200, { ETag: `"${f.revision}"` });
    },
    "GET /v1/projects/{projectId}/finding-sources": ({ params, url }) =>
      json({
        projectRevision: 1,
        ...page(
          sources()
            .filter(
              (s) =>
                s.projectId === params.projectId &&
                (!url.searchParams.has("findingId") ||
                  s.findingId === url.searchParams.get("findingId")),
            )
            .map(build.findingSourcePreview),
        ),
      }),
    "GET /v1/projects/{projectId}/finding-sources/{receiptId}": ({
      params,
    }) => {
      const source = sources().find(
        (s) =>
          s.projectId === params.projectId && s.receiptId === params.receiptId,
      );
      return source === undefined ? failure(404) : json(source);
    },
    "GET /v1/projects/{projectId}/finding-sources/{receiptId}/history": () =>
      json({ projectRevision: 1, ...page([]) }),
    "PATCH /v1/projects/{projectId}/findings/{findingId}": async ({
      params,
      request,
    }) => {
      await server.decisionGate;
      const f = finding(params);
      if (f === undefined) return failure(404);
      if (request.headers.get("If-Match") !== `"${f.revision}"`)
        return failure(412, "precondition_failed", "The Finding changed first");
      const body = (await request.json()) as EditProjectFindingRequest;
      const next: ProjectFinding = {
        ...f,
        state: VERDICT_STATES[body.verdict],
        analystVerdict: body.verdict,
        analystSeverity:
          body.verdict === "true_positive" ? body.severity : null,
        duplicateTargetId:
          body.verdict === "duplicate" ? body.duplicateTargetId : null,
        rationale: body.rationale,
        actorId: "user_analyst",
        assessedAt: build.NOW,
        revision: f.revision + 1,
      };
      server.findings = server.findings.map((value) =>
        value === f ? next : value,
      );
      return json({ finding: next, replayed: false }, 200, {
        ETag: `"${next.revision}"`,
      });
    },
  };
}
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
export function renderIssues(server: IssuesServer, path: string) {
  return renderWithAPI(null, {
    path,
    server: fakeServer(held(server, routes(server))),
    routes: [
      { path: "/issues", element: <IssuesRoute /> },
      { path: "/issues/:projectId/:findingId", element: <IssuesRoute /> },
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
  const findings = initial.findings ?? [];
  return {
    findings,
    sources: findings.map((f) => build.findingSource(f)),
    failing: new Set(),
    more: new Set(),
    ...initial,
  };
}
