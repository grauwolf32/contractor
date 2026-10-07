import { ownerListResponse } from "../../test/owner-lists";
// A fake Public API for the Issues tests: projects, checks, possible
// issues and review requests, with the filters, ETags and decisions the
// Server applies.
import { createMemoryRouter, RouterProvider } from "react-router";

import type {
  Audit,
  AuditFinding,
  AuditReviewRequest,
  AuditState,
} from "../../api/audits";
import type { Project } from "../../api/projects";
import type { AuditReviewDecision } from "../decisions";
import {
  ALL_VERDICTS,
  DIGEST,
  json,
  failure,
  makeFinding,
  renderWithServer,
} from "../decisions/test-support";
import { IssuesRoute } from ".";

export const NOW = "2026-10-05T10:00:00Z";

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

export function check(
  projectId: string,
  auditId: string,
  state: AuditState = "completed",
  profile = "openapi-operation-trace",
): Audit {
  return {
    auditId,
    projectId,
    profile: { name: profile, version: "1", digest: DIGEST },
    inputs: {
      source: {
        ref: { namespace: "sources", name: "shop-source", revision: "src-r1" },
        digest: DIGEST,
        mediaType: "application/zip",
        sizeBytes: 2048,
      },
    },
    scope: {},
    runtimeLabels: [],
    state,
    phase: "rounds",
    currentRoundId: "round_1",
    revision: 1,
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
    reservedRunCount: 0,
    submittedRunCount: 0,
    outstandingRunCount: 0,
    retainedEvidenceBytes: 0,
    eventSequence: 1,
    createdAt: NOW,
    updatedAt: NOW,
  };
}

/** A possible issue of `auditId`, created at `createdAt`. */
export function issue(
  auditId: string,
  findingId: string,
  title: string,
  createdAt: string,
  overrides: Partial<AuditFinding> = {},
  document: Partial<AuditFinding["firstProposal"]["document"]> = {},
): AuditFinding {
  const base = makeFinding(
    { findingId, auditId, createdAt, updatedAt: createdAt, revision: 1 },
    { title, ...document },
  );
  base.firstProposal.createdAt = createdAt;
  return { ...base, ...overrides };
}

export interface FakeServer {
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
}

function page<T>(items: T[], hasMore = false) {
  return { items, page: { hasMore } };
}

function bump(server: FakeServer, auditId: string) {
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

/** Handles the requests of the Issues pages against `server`. */
export function handler(server: FakeServer) {
  return async (request: Request, url: URL): Promise<Response> => {
    const path = url.pathname;
    if (path === "/v1/findings") {
      await server.gate;
      if (server.failing.size > 0)
        return failure(503, "unavailable", "Findings unavailable");
    }

    const ownerResponse = ownerListResponse(url, {
      projects: server.projects,
      audits: server.audits,
      findings: server.findings,
      reviews: server.reviews,
    });
    if (ownerResponse !== undefined) return ownerResponse;

    if (path === "/v1/projects") return json(page(server.projects));
    const projectPath = /^\/v1\/projects\/([^/]+)$/.exec(path);
    if (projectPath !== null) {
      const found = server.projects.find(
        (candidate) => candidate.projectId === projectPath[1],
      );
      return found === undefined
        ? failure(404, "not_found", "Project not found")
        : json(found, 200, { ETag: '"1"' });
    }
    const projectAudits = /^\/v1\/projects\/([^/]+)\/audits$/.exec(path);
    if (projectAudits !== null) {
      const state = url.searchParams.get("state");
      return json(
        page(
          server.audits.filter(
            (candidate) =>
              candidate.projectId === projectAudits[1] &&
              (state === null || candidate.state === state),
          ),
        ),
      );
    }
    const auditPath = /^\/v1\/audits\/([^/]+)(\/.*)?$/.exec(path);
    if (auditPath === null) return failure(404, "not_found", "Not found");
    const auditId = decodeURIComponent(auditPath[1]!);
    const rest = auditPath[2] ?? "";
    const audit = server.audits.find(
      (candidate) => candidate.auditId === auditId,
    );
    if (audit === undefined) return failure(404, "not_found", "No check");
    const summary = { auditRevision: audit.revision, asOf: audit.updatedAt };
    if (rest === "") return json(audit, 200, { ETag: `"${audit.revision}"` });
    if (rest === "/findings") {
      await server.gate;
      if (server.failing.has(auditId))
        return failure(503, "unavailable", "Findings unavailable");
      const state = url.searchParams.get("state");
      const verdict = url.searchParams.get("verdict");
      const severity = url.searchParams.get("severity");
      const items = server.findings.filter(
        (candidate) =>
          candidate.auditId === auditId &&
          (state === null || candidate.state === state) &&
          (verdict === null ||
            (verdict === "unreviewed"
              ? candidate.analystVerdict === undefined
              : candidate.analystVerdict === verdict)) &&
          (severity === null || candidate.analystSeverity === severity),
      );
      const more = server.more.has(auditId);
      return json({
        ...summary,
        total: items.length + (more ? 50 : 0),
        ...page(items, more),
      });
    }
    const findingPath = /^\/findings\/([^/]+)(\/.*)?$/.exec(rest);
    if (findingPath !== null) {
      const findingId = decodeURIComponent(findingPath[1]!);
      const finding = server.findings.find(
        (candidate) =>
          candidate.auditId === auditId && candidate.findingId === findingId,
      );
      if (finding === undefined)
        return failure(404, "not_found", "Possible issue not found");
      if (findingPath[2] === undefined)
        return json(finding, 200, { ETag: `"${finding.revision}"` });
      if (findingPath[2] === "/provenance")
        return json({
          auditRevision: audit.revision,
          findingRevision: finding.revision,
          items: [
            {
              recordId: "source:receipt_1",
              kind: "source-proposal",
              receiptId: "receipt_1",
              proposal: finding.firstProposal.proposal,
              origin: finding.firstProposal.origin,
              supportsCurrentAssessment: false,
              createdAt: finding.createdAt,
            },
          ],
          page: { hasMore: false },
        });
      if (findingPath[2] === "/reviews" && request.method === "POST") {
        const created: AuditReviewRequest = {
          requestId: `review_${findingId}`,
          auditId,
          findingId,
          subjectKind: "finding",
          subjectId: findingId,
          kind: "finding-triage",
          subjectRevision: finding.revision,
          subjectDigest: DIGEST,
          requestedActions: ALL_VERDICTS,
          state: "pending",
          revision: 1,
          createdAt: NOW,
          updatedAt: NOW,
        };
        server.reviews.push(created);
        return json(created, 201, { ETag: '"1"' });
      }
    }
    if (rest === "/reviews") {
      const finding = url.searchParams.get("finding");
      const state = url.searchParams.get("state");
      const items = server.reviews.filter(
        (candidate) =>
          candidate.auditId === auditId &&
          (finding === null || candidate.findingId === finding) &&
          (state === null || candidate.state === state),
      );
      return json({ ...summary, total: items.length, ...page(items) });
    }
    const reviewPath = /^\/reviews\/([^/]+)(\/decisions)?$/.exec(rest);
    if (reviewPath !== null) {
      const review = server.reviews.find(
        (candidate) => candidate.requestId === reviewPath[1],
      );
      if (review === undefined)
        return failure(404, "not_found", "Review not found");
      if (reviewPath[2] === undefined)
        return json(review, 200, { ETag: `"${review.revision}"` });
      const body = (await request.json()) as {
        verdict: AuditFinding["analystVerdict"];
        severity?: AuditFinding["analystSeverity"];
        rationale: string;
      };
      const decision: AuditReviewDecision = {
        decisionId: `decision_${review.requestId}`,
        requestId: review.requestId,
        auditId,
        ...(review.findingId === undefined
          ? {}
          : { findingId: review.findingId }),
        actorId: "user_analyst",
        ...(body.verdict === undefined ? {} : { verdict: body.verdict }),
        ...(body.severity === undefined ? {} : { severity: body.severity }),
        rationale: body.rationale,
        subjectRevision: review.subjectRevision,
        subjectDigest: DIGEST,
        createdAt: NOW,
      };
      const states = {
        true_positive: "confirmed",
        false_positive: "rejected",
        needs_evidence: "needs-evidence",
        duplicate: "duplicate",
        reopen: "proposed",
      } as const;
      let decided: AuditFinding | undefined;
      server.findings = server.findings.map((candidate) => {
        if (candidate.findingId !== review.findingId) return candidate;
        decided = {
          ...candidate,
          state: states[body.verdict ?? "reopen"],
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
    }
    return failure(404, "not_found", `Unexpected ${request.method} ${path}`);
  };
}

/** Renders the Issues route at `path` against `server`. */
export function renderIssues(server: FakeServer, path: string) {
  const router = createMemoryRouter(
    [
      { path: "/issues", element: <IssuesRoute /> },
      { path: "/issues/:auditId/:findingId", element: <IssuesRoute /> },
      {
        path: "/projects/:projectId/audits/:auditId/:section",
        element: <p>Check page</p>,
      },
      { path: "*", element: <p>Elsewhere</p> },
    ],
    { initialEntries: [path] },
  );
  const view = renderWithServer(
    <RouterProvider router={router} />,
    handler(server),
  );
  return { ...view, router };
}

export function server(
  initial: Partial<FakeServer> & Pick<FakeServer, "projects" | "audits">,
): FakeServer {
  return {
    findings: [],
    reviews: [],
    failing: new Set(),
    more: new Set(),
    ...initial,
  };
}
