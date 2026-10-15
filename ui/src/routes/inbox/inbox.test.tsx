import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter, type RouteObject } from "react-router";
import { describe, expect, it } from "vitest";

import type {
  Audit,
  AuditAnalystVerdict,
  AuditFinding,
  AuditItem,
  AuditReport,
  AuditReviewAction,
  AuditReviewRequest,
  AuditWorkspace,
} from "../../api/audits";
import type { Project } from "../../api/projects";
import type { RunStatus, RunSummary } from "../../api/runs";
import type { WorkflowResource } from "../../api/workflows";
import { Application } from "../../app/application";
import { applicationRoutes } from "../../app/router";
import * as build from "../../test/builders";
import {
  runStatus,
  stageAttempt,
  workflowResource,
} from "../../test/builders-execution";
import {
  emptyPageOrNotFound,
  failure,
  fakeServer,
  json,
  page,
  type Route,
  type Routes,
} from "../../test/fake-server";
import { ownerListRoutes } from "../../test/owner-lists";
import { testAPI, testSession } from "../../test/render";

const NOW = Date.now();

/** An ISO time `minutes` ago. */
function ago(minutes: number): string {
  return new Date(NOW - minutes * 60_000).toISOString();
}

/** A check last changed 5 minutes ago. */
function audit(
  projectId: string,
  auditId: string,
  state: Audit["state"],
  overrides: Partial<Audit> = {},
): Audit {
  return build.audit(projectId, auditId, state, {
    updatedAt: ago(5),
    ...overrides,
  });
}

/** A possible issue proposed `minutes` ago. */
function finding(
  auditId: string,
  findingId: string,
  title: string,
  minutes: number,
  document: Partial<AuditFinding["firstProposal"]["document"]> = {},
): AuditFinding {
  return build.finding(
    auditId,
    findingId,
    { revision: 3, createdAt: ago(minutes), updatedAt: ago(minutes) },
    { title, ...document },
  );
}

/** A pending request made `minutes` ago, about work item item_1 by default. */
function review(
  auditId: string,
  requestId: string,
  kind: AuditReviewRequest["kind"],
  minutes: number,
  overrides: Partial<AuditReviewRequest> = {},
): AuditReviewRequest {
  return build.review(auditId, requestId, kind, {
    ...(kind === "report-acceptance" ? {} : { subjectId: "item_1" }),
    createdAt: ago(minutes),
    updatedAt: ago(minutes),
    ...overrides,
  });
}

/** Workspace counters of `check`: five items, one done, unless given. */
function workspace(
  check: Audit,
  counts: Parameters<typeof build.workspace>[1] = {},
): AuditWorkspace {
  return build.workspace(check, {
    totalChecks: 5,
    completedChecks: 1,
    unchecked: 4,
    ...counts,
  });
}

/** A Run of the owner's list that changed `minutes` ago. */
function summaryOf(
  runId: string,
  workflow: string,
  state: RunSummary["state"],
  minutes: number,
  projectId?: string,
): RunSummary {
  const terminal = ["succeeded", "failed", "cancelled"].includes(state);
  return build.conforming("RunSummary", {
    runId,
    workflow,
    state,
    deletable: false,
    labels: {},
    ...(projectId === undefined ? {} : { projectId }),
    createdAt: ago(minutes + 5),
    updatedAt: ago(minutes),
    ...(terminal ? { finishedAt: ago(minutes) } : {}),
  });
}

/** The status of a listed Run, as the Server reads it. */
function statusOf(run: RunSummary, extra: Partial<RunStatus> = {}): RunStatus {
  return runStatus(run.runId, {
    workflow: run.workflow,
    state: run.state,
    deletable: false,
    createdAt: run.createdAt,
    updatedAt: run.updatedAt,
    ...(run.finishedAt === undefined ? {} : { finishedAt: run.finishedAt }),
    ...extra,
  });
}

const FAILED_ATTEMPT = stageAttempt("stage_failed_1", {
  stage: "analysis",
  state: "failed",
  termination: {
    outcome: "interrupted",
    code: "worker_gateway_unavailable",
    message: "The model gateway refused the connection.",
    retryable: true,
    phase: "running",
    occurredAt: ago(90),
  },
  createdAt: ago(95),
  updatedAt: ago(90),
});

/** What the fake Server holds; it changes as the page acts on it. */
interface InboxServer {
  projects: Project[];
  audits: Record<string, Audit[]>;
  /** Possible issues in state `proposed`, by check. */
  findings: Record<string, AuditFinding[]>;
  /** Pending review requests, by check. */
  reviews: Record<string, AuditReviewRequest[]>;
  reports: Record<string, AuditReport>;
  workspaces: Record<string, AuditWorkspace>;
  items: Record<string, AuditItem[]>;
  runs: RunSummary[];
  statuses: Record<string, RunStatus>;
  /** Exact Workflow versions by "name@version". */
  workflows: Record<string, WorkflowResource>;
  paused: boolean;
  /** Answers a request before the fake does, e.g. with a failure. */
  override?: (url: URL, request: Request) => Response | undefined;
  /** Holds a request until the returned promise settles. */
  delay?: (url: URL, request: Request) => Promise<void> | undefined;
}

/**
 * A promise the test settles: a Server answer that takes a while.
 * V385: needs shared deferred helper in src/test.
 */
function gate(): { wait: Promise<void>; release: () => void } {
  let release!: () => void;
  const wait = new Promise<void>((resolve) => {
    release = resolve;
  });
  return { wait, release };
}

/** The check with this ID. */
function auditOf(server: InboxServer, auditId: string): Audit {
  const found = Object.values(server.audits)
    .flat()
    .find((entry) => entry.auditId === auditId);
  if (found === undefined) throw new Error(`No check ${auditId}`);
  return found;
}

/** A recorded decision advances its check's revision, as the Server does. */
function advance(server: InboxServer, auditId: string) {
  const entry = auditOf(server, auditId);
  entry.revision += 1;
  entry.updatedAt = ago(0);
}

function emptyServer(): InboxServer {
  return {
    projects: [],
    audits: {},
    findings: {},
    reviews: {},
    reports: {},
    workspaces: {},
    items: {},
    runs: [],
    statuses: {},
    workflows: {},
    paused: false,
  };
}

/** Two projects with something in every section. */
function busyServer(): InboxServer {
  const runs = [
    summaryOf("run_failed", "review@2", "failed", 90),
    summaryOf("run_old_failure", "review@2", "failed", 60 * 24 * 10),
    summaryOf("run_waiting", "triage@3", "waiting", 3),
    summaryOf("run_active", "triage@3", "running", 2),
    summaryOf("run_ok", "generate@1", "succeeded", 30, "project_identity"),
  ];
  const byId = Object.fromEntries(runs.map((run) => [run.runId, run]));
  const trace = audit("project_shop", "audit_trace", "active", {
    revision: 3,
  });
  const wait = audit("project_shop", "audit_wait", "waiting_review");
  const paused = audit("project_shop", "audit_paused", "paused", {
    pausedAt: ago(40),
    stopReason: {
      code: "deadline_exhausted",
      message: "The check used its time limit.",
    },
  });
  const done = audit("project_identity", "audit_done", "completed", {
    finishedAt: ago(60),
  });
  const acceptance = review(
    "audit_wait",
    "review_report",
    "report-acceptance",
    15,
  );
  return {
    ...emptyServer(),
    projects: [
      build.project("project_shop", { name: "crapi-workshop" }),
      build.project("project_identity", { name: "crapi-identity" }),
    ],
    audits: {
      project_shop: [trace, wait, paused],
      project_identity: [
        done,
        audit("project_identity", "audit_failed", "failed", {
          finishedAt: ago(70),
          stopReason: {
            code: "role_execution_not_retryable",
            message: "The trace role failed for good.",
          },
        }),
        audit("project_identity", "audit_finished", "completed", {
          finishedAt: ago(80),
        }),
      ],
    },
    findings: {
      audit_trace: [
        finding(
          "audit_trace",
          "finding_idor",
          "Any user can read any report",
          10,
          {
            standard_refs: [
              { scheme: "CWE", version: "4.14", requirement_id: "CWE-639" },
            ],
          },
        ),
        finding(
          "audit_trace",
          "finding_sqli",
          "Order search builds raw SQL",
          20,
        ),
      ],
    },
    reviews: {
      audit_wait: [acceptance],
      audit_trace: [
        review("audit_trace", "review_approval", "active-check-approval", 25),
      ],
    },
    reports: {
      audit_wait: build.report("audit_wait", "proposed", {
        review: acceptance,
        summary: "# Report\n\nThree endpoints were traced.",
      }),
      audit_done: build.report("audit_done", "ready", {
        summary: "# Report\n\nAll five endpoints were traced.",
      }),
    },
    workspaces: {
      audit_trace: workspace(trace, { gaps: 2, unchecked: 2 }),
      audit_wait: workspace(wait, { completedChecks: 5, unchecked: 0 }),
      audit_paused: workspace(paused, { completedChecks: 2, unchecked: 3 }),
      audit_done: workspace(done, { completedChecks: 5, unchecked: 0 }),
    },
    items: {
      audit_trace: [
        build.auditItem("item_1", 0, "standard-mapping", {
          itemKey: "WSTG-ATHN-01",
          subjectKey: "WSTG-ATHN-01",
          state: "awaiting_review",
          approvalKind: "active-check-approval",
          createdAt: ago(30),
          updatedAt: ago(25),
        }),
      ],
    },
    runs,
    statuses: {
      run_failed: statusOf(byId.run_failed!, { attempts: [FAILED_ATTEMPT] }),
      run_waiting: statusOf(byId.run_waiting!, {
        recovery: {
          code: "model_unavailable",
          since: ago(20),
          automaticUntil: ago(10),
          requiresRetry: true,
        },
      }),
      run_active: statusOf(byId.run_active!),
      run_ok: statusOf(byId.run_ok!, {
        projectId: "project_identity",
        outputs: {
          report: { namespace: "outputs", name: "report", revision: "r1" },
        },
      }),
    },
    workflows: {
      "generate@1": workflowResource("generate", "1", {
        outputs: {
          report: {
            required: true,
            mediaTypes: ["text/markdown"],
            primary: true,
          },
          log: { required: false, mediaTypes: ["text/plain"] },
        },
      }),
    },
  };
}

const TERMINAL = new Set(["succeeded", "failed", "cancelled"]);

/** The header of a per-check page: the revision it was read at. */
const CHECK_PAGE = { auditRevision: 1, asOf: ago(1) };

/** A decision the Server records, by action or verdict. */
interface DecisionBody {
  action?: AuditReviewAction;
  verdict?: AuditAnalystVerdict;
  rationale: string;
}

/** Records the decision on `requestId` and answers as the Server does. */
async function decide(
  server: InboxServer,
  auditId: string,
  requestId: string,
  body: DecisionBody,
): Promise<Response> {
  // The decision leaves the list and advances the check's revision.
  advance(server, auditId);
  const decided = (request: AuditReviewRequest) => {
    const decision = build.decision(request, {
      decisionId: "decision_1",
      actorId: "user_local",
      action: body.action,
      verdict: body.verdict,
      rationale: body.rationale,
      createdAt: ago(0),
    });
    return {
      decision,
      request: {
        ...request,
        state: "decided" as const,
        revision: request.revision + 1,
        decision,
        updatedAt: ago(0),
      },
    };
  };
  const pending = server.reviews[auditId]?.find(
    (candidate) => candidate.requestId === requestId,
  );
  if (pending !== undefined) {
    // An approval, applicability or report acceptance request.
    server.reviews[auditId] = server.reviews[auditId]!.filter(
      (candidate) => candidate !== pending,
    );
    return json({ ...decided(pending), replayed: false });
  }
  const findingId = requestId.replace(/^review_/, "");
  // A finding a refresh already removed is still decided by the Server.
  const subject =
    server.findings[auditId]?.find(
      (candidate) => candidate.findingId === findingId,
    ) ?? finding(auditId, findingId, "Decided possible issue", 0);
  server.findings[auditId] = (server.findings[auditId] ?? []).filter(
    (candidate) => candidate.findingId !== findingId,
  );
  const { decision, request } = decided(triageRequest(subject));
  return json({
    finding: {
      ...subject,
      state: "rejected",
      analystVerdict: body.verdict,
      analystDecision: decision,
      revision: subject.revision + 1,
    },
    request,
    decision,
    replayed: false,
  });
}

/** The triage request the page opens to decide on a possible issue. */
function triageRequest(subject: AuditFinding): AuditReviewRequest {
  return build.review(
    subject.auditId,
    `review_${subject.findingId}`,
    "finding-triage",
    {
      findingId: subject.findingId,
      subjectId: subject.findingId,
      subjectRevision: subject.revision,
      createdAt: ago(0),
      updatedAt: ago(0),
    },
  );
}

/** `found`, or the Server's 404 for a resource that is not there. */
function found<T>(value: T | undefined, headers?: Record<string, string>) {
  return value === undefined ? failure(404) : json(value, 200, headers);
}

/** The Server's answers over `server`, which they read and change. */
function inboxRoutes(server: InboxServer): Routes {
  return {
    ...ownerListRoutes(() => ({
      projects: server.projects,
      audits: Object.values(server.audits).flat(),
      findings: Object.values(server.findings).flat(),
      reviews: Object.values(server.reviews).flat(),
    })),
    "GET /v1/auth/session": () => json(testSession(["user", "operations"])),
    "GET /v1/queue/control": () =>
      server.paused
        ? json({ paused: true, revision: "4", updatedAt: ago(10) }, 200, {
            ETag: '"4"',
          })
        : json({ paused: false, revision: "0" }, 200, { ETag: '"0"' }),
    "GET /v1/projects": () => json(page(server.projects)),
    "GET /v1/projects/{projectId}/audits": ({ params, url }) => {
      const state = url.searchParams.get("state");
      const audits = server.audits[params.projectId!] ?? [];
      return json(
        page(state === null ? audits : audits.filter((a) => a.state === state)),
      );
    },
    "GET /v1/audits/{auditId}": ({ params }) => {
      const check = Object.values(server.audits)
        .flat()
        .find((entry) => entry.auditId === params.auditId);
      return found(check, { ETag: `"${check?.revision}"` });
    },
    "GET /v1/audits/{auditId}/findings": ({ params, url }) => {
      const items =
        url.searchParams.get("state") === "proposed"
          ? (server.findings[params.auditId!] ?? [])
          : [];
      return json({ ...CHECK_PAGE, total: items.length, ...page(items) });
    },
    "GET /v1/audits/{auditId}/findings/{findingId}": ({ params }) => {
      const subject = server.findings[params.auditId!]?.find(
        (candidate) => candidate.findingId === params.findingId,
      );
      return found(subject, { ETag: `"${subject?.revision}"` });
    },
    "POST /v1/audits/{auditId}/findings/{findingId}/reviews": ({ params }) => {
      const subject =
        server.findings[params.auditId!]?.find(
          (candidate) => candidate.findingId === params.findingId,
        ) ?? finding(params.auditId!, params.findingId!, "Removed", 0);
      return json(triageRequest(subject), 201);
    },
    "GET /v1/audits/{auditId}/reviews": ({ params, url }) => {
      const subject = url.searchParams.get("finding");
      const items = (server.reviews[params.auditId!] ?? []).filter(
        (candidate) => subject === null || candidate.findingId === subject,
      );
      return json({ ...CHECK_PAGE, total: items.length, ...page(items) });
    },
    "POST /v1/audits/{auditId}/reviews/{requestId}/decisions": async ({
      params,
      request,
    }) =>
      decide(
        server,
        params.auditId!,
        params.requestId!,
        (await request.json()) as DecisionBody,
      ),
    "GET /v1/audits/{auditId}/report": ({ params }) =>
      found(server.reports[params.auditId!]),
    "GET /v1/audits/{auditId}/workspace": ({ params }) =>
      found(server.workspaces[params.auditId!]),
    "GET /v1/audits/{auditId}/items": ({ params, url }) => {
      const state = url.searchParams.get("state");
      return json(
        page(
          (server.items[params.auditId!] ?? []).filter(
            (item) => state === null || item.state === state,
          ),
        ),
      );
    },
    "GET /v1/runs": ({ url }) => {
      const state = url.searchParams.get("state");
      const lifecycle = url.searchParams.get("lifecycle");
      return json(
        page(
          server.runs.filter(
            (run) =>
              (state === null || run.state === state) &&
              (lifecycle === null ||
                (lifecycle === "active") !== TERMINAL.has(run.state)),
          ),
        ),
      );
    },
    "GET /v1/runs/{runId}": ({ params }) =>
      found(server.statuses[params.runId!]),
    "POST /v1/runs/{runId}/retry-gateway": ({ params }) => {
      const runId = params.runId!;
      const status = server.statuses[runId];
      if (status !== undefined) {
        delete status.recovery;
        status.state = "running";
      }
      server.runs = server.runs.map((run) =>
        run.runId === runId ? { ...run, state: "running" } : run,
      );
      return json({ runId }, 202);
    },
    "GET /v1/workflows/{name}/versions/{version}": ({ params }) =>
      found(server.workflows[`${params.name}@${params.version}`]),
  };
}

/**
 * The fake Server of `server`: every answer, the fallback's too, first
 * passes its override and delay.
 */
function inboxServer(server: InboxServer) {
  const held = async (request: Request, url: URL) => {
    const overridden = server.override?.(url, request);
    if (overridden !== undefined) return overridden;
    await server.delay?.(url, request);
    return undefined;
  };
  const routes = Object.fromEntries(
    Object.entries(inboxRoutes(server)).map(([operation, route]) => [
      operation,
      (async (call) =>
        (await held(call.request, call.url)) ?? route(call)) satisfies Route,
    ]),
  );
  return fakeServer(
    routes,
    async (request, url) =>
      (await held(request, url)) ?? emptyPageOrNotFound(request, url),
  );
}

/** Holds a route's lazy code until `until` settles: a slow next page. */
function delayRoute(
  routes: RouteObject[],
  path: string,
  until: Promise<void>,
): boolean {
  for (const route of routes) {
    const { lazy } = route;
    if (route.path === path && typeof lazy === "function") {
      route.lazy = async () => {
        await until;
        return lazy();
      };
      return true;
    }
    if (route.children !== undefined && delayRoute(route.children, path, until))
      return true;
  }
  return false;
}

/**
 * The application at `path` over `server`.
 * V385: needs shared application harness in src/test.
 */
function renderInbox(
  server: InboxServer,
  path = "/",
  slowRoute?: { path: string; until: Promise<void> },
) {
  const { api, server: fake } = testAPI(inboxServer(server));
  const routes = applicationRoutes();
  if (
    slowRoute !== undefined &&
    !delayRoute(routes, slowRoute.path, slowRoute.until)
  )
    throw new Error(`No lazy route ${slowRoute.path}`);
  const router = createMemoryRouter(routes, {
    initialEntries: [path],
  });
  const view = render(
    <Application api={api} publicAPI={api} router={router} />,
  );
  return { ...view, router, requests: fake.requests, sent: fake.sent };
}

/** The detail pane, once the application has rendered it. */
function findDetail(): Promise<HTMLElement> {
  return screen.findByRole("region", { name: "Selected item" });
}

/** The page-level status region that outlives a decision bar. */
function pageStatus(): HTMLElement {
  const region = document.querySelector<HTMLElement>(
    'main > p.ui-visually-hidden[role="status"]',
  );
  if (region === null) throw new Error("The page status region is missing");
  return region;
}

/** The list pane's section with this heading. */
async function findSection(name: string): Promise<HTMLElement> {
  const heading = await screen.findByRole("heading", { name, level: 2 });
  const container = heading.closest("section");
  if (container === null) throw new Error(`${name} has no section`);
  return container;
}

/** The value of a fact (`<dt>` term, `<dd>` value) in the detail pane. */
function fact(container: HTMLElement, term: string): HTMLElement {
  const value = within(container).getByText(term, {
    selector: "dt",
  }).nextElementSibling;
  if (!(value instanceof HTMLElement)) throw new Error(`No fact ${term}`);
  return value;
}

/** The detail bar's "where": section and kind of the item shown. */
function detailWhere(): HTMLElement {
  const where = document.querySelector<HTMLElement>(".inbox-detail-where");
  if (where === null) throw new Error("The detail bar is missing");
  return where;
}

type User = ReturnType<typeof userEvent.setup>;

/** Records "Not an issue" on the possible issue shown in the detail pane. */
async function decideNotAnIssue(user: User) {
  const pane = await findDetail();
  const reject = await within(pane).findByRole("button", {
    name: "Not an issue",
  });
  await waitFor(() => expect(reject).toBeEnabled());
  await user.click(reject);
  await user.type(
    within(pane).getByRole("textbox", { name: "Why" }),
    "Only administrators reach this view.",
  );
  await user.click(
    within(pane).getByRole("button", { name: "Record decision" }),
  );
}

/** The Server's answer to a decision on "Any user can read any report". */
const FIRST_DECIDED =
  "Decision recorded on “Any user can read any report”: Not an issue.";

describe("Inbox", () => {
  it("lists decisions, blocked work, finished results and running checks", async () => {
    const { sent } = renderInbox(busyServer());

    expect(
      await screen.findByRole("heading", { name: "Inbox", level: 1 }),
    ).toBeVisible();
    const decide = await findSection("Decide");
    expect(
      await within(decide).findByRole("link", {
        name: "Any user can read any report",
      }),
    ).toHaveAttribute("href", "/?item=issue:audit_trace:finding_idor");
    expect(
      within(decide).getByRole("link", { name: "Order search builds raw SQL" }),
    ).toBeInTheDocument();
    expect(
      await within(decide).findByRole("link", {
        name: "Report acceptance: crapi-workshop",
      }),
    ).toHaveAttribute("href", "/?item=review:audit_wait:review_report");
    expect(
      await within(decide).findByRole("link", {
        name: "Active test approval: WSTG-ATHN-01",
      }),
    ).toBeInTheDocument();
    expect(within(decide).getByText("CWE-639")).toBeInTheDocument();

    const unblock = await findSection("Unblock");
    expect(
      await within(unblock).findByText(
        "The model gateway refused the connection.",
      ),
    ).toBeInTheDocument();
    expect(
      within(unblock).getByRole("link", { name: "review@2" }),
    ).toHaveAttribute("href", "/?item=run:run_failed");
    expect(
      within(unblock).queryByRole("link", { name: /run_old_failure/ }),
    ).not.toBeInTheDocument();
    expect(
      within(unblock).getByRole("link", { name: "See all failed Runs" }),
    ).toHaveAttribute("href", "/runs?view=completed&state=failed");
    expect(
      await within(unblock).findByRole("link", { name: "triage@3" }),
    ).toHaveAttribute("href", "/?item=run:run_waiting");
    expect(
      within(unblock).getByText("Needs a model retry"),
    ).toBeInTheDocument();
    expect(within(unblock).getByText("Time limit reached")).toBeInTheDocument();
    expect(
      await within(unblock).findByRole("link", {
        name: "2 endpoints couldn't be fully checked",
      }),
    ).toHaveAttribute("href", "/?item=check:audit_trace");
    expect(
      within(unblock).getByRole("link", {
        name: "crapi-identity: OpenAPI · Operation trace",
      }),
    ).toHaveAttribute("href", "/?item=check:audit_failed");

    const ready = await findSection("Ready");
    expect(
      await within(ready).findByRole("link", {
        name: "crapi-identity report is ready",
      }),
    ).toHaveAttribute("href", "/?item=report:audit_done");
    expect(
      within(ready).getByRole("link", {
        name: "crapi-identity check finished",
      }),
    ).toHaveAttribute("href", "/?item=check:audit_finished");
    expect(
      await within(ready).findByText("Primary result: report"),
    ).toBeInTheDocument();
    expect(
      within(ready).getByRole("link", { name: "See all finished Runs" }),
    ).toHaveAttribute("href", "/runs?view=completed&state=succeeded");

    const running = await findSection("Running");
    expect(
      within(running).getByRole("link", {
        name: "crapi-workshop: OpenAPI · Operation trace",
      }),
    ).toHaveAttribute("href", "/?item=check:audit_trace");
    expect(
      await within(running).findByRole("img", {
        name: "1 of 5 endpoints done: 2 need follow-up, 2 not checked yet",
      }),
    ).toBeInTheDocument();
    expect(
      within(running).getByRole("link", { name: "2 Runs in progress" }),
    ).toHaveAttribute("href", "/runs");
    // One running check and the active Runs row.
    expect(running.querySelector(".ui-list-section-count")).toHaveTextContent(
      /^2$/,
    );

    // Reports are read for checks that finished in the last 7 days only.
    expect(sent("GET", "/v1/audits/audit_done/report")).toHaveLength(1);
    for (const auditId of ["audit_failed", "audit_wait", "audit_paused"])
      expect(sent("GET", `/v1/audits/${auditId}/report`)).toHaveLength(0);
    // A request's subject comes from the items that wait for a decision.
    const items = sent("GET", "/v1/audits/audit_trace/items");
    expect(items.length).toBeGreaterThan(0);
    for (const request of items)
      expect(new URL(request.url).searchParams.get("state")).toBe(
        "awaiting_review",
      );

    // The subtitle counts what needs a decision as the rail badge does.
    expect(
      await screen.findByText(
        "4 need you, 5 are stuck, 3 are ready, 1 check is running.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: "Inbox (4 need your decision)" }),
    ).toHaveAttribute("aria-current", "page");
  });

  it("lists and counts only decisions that can still be made", async () => {
    const server = busyServer();
    // Stored as pending, but past its expiry: the Server's pending list
    // leaves it out before anything has expired it.
    server.reviews.audit_trace!.push(
      review("audit_trace", "review_expired", "requirement-applicability", 5, {
        subjectId: "item_2",
        expiresAt: ago(1),
      }),
      review("audit_trace", "review_open", "requirement-applicability", 4, {
        subjectId: "item_3",
        expiresAt: new Date(Date.now() + 60 * 60 * 1000).toISOString(),
      }),
    );
    const { sent } = renderInbox(server);

    const decide = await findSection("Decide");
    await waitFor(() =>
      expect(
        decide.querySelector('a[href="/?item=review:audit_trace:review_open"]'),
      ).not.toBeNull(),
    );
    expect(
      decide.querySelector(
        'a[href="/?item=review:audit_trace:review_expired"]',
      ),
    ).toBeNull();
    // The badge counts the same decisions: two possible issues, the report,
    // the approval and the open applicability request.
    expect(
      await screen.findByRole("link", { name: "Inbox (5 need your decision)" }),
    ).toBeInTheDocument();
    const reads = sent("GET", "/v1/reviews");
    expect(reads.length).toBeGreaterThan(0);
    for (const request of reads)
      expect(new URL(request.url).searchParams.get("state")).toBe("pending");
  });

  it("says plainly when a section is empty", async () => {
    renderInbox(emptyServer());

    expect(
      await screen.findByText("Nothing needs you right now."),
    ).toBeInTheDocument();
    expect(
      await within(await findSection("Decide")).findByText(
        "Nothing needs a decision. New possible issues land here.",
      ),
    ).toBeInTheDocument();
    expect(
      await within(await findSection("Unblock")).findByText(
        "Nothing is stuck. Failed Runs and stopped checks land here.",
      ),
    ).toBeInTheDocument();
    expect(
      await within(await findSection("Ready")).findByText(
        "Nothing finished in the last 7 days. Reports, finished checks and Runs land here.",
      ),
    ).toBeInTheDocument();
    expect(
      await within(await findSection("Running")).findByText(
        "Nothing is running. Checks you start show their progress here.",
      ),
    ).toBeInTheDocument();
    // Nothing selected: the detail pane gives the overview.
    expect(
      within(await findDetail()).getByRole("heading", { name: "Overview" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByText(/Queue admission is paused/),
    ).not.toBeInTheDocument();
    expect(screen.queryByText(/could not be read/)).not.toBeInTheDocument();
  });

  it("opens the item named in the URL", async () => {
    renderInbox(busyServer(), "/?item=issue:audit_trace:finding_sqli");

    const pane = await findDetail();
    expect(
      await within(pane).findByText("Possible issue 2 of 2"),
    ).toBeInTheDocument();
    expect(
      within(pane).getByRole("heading", {
        name: "Order search builds raw SQL",
        level: 2,
      }),
    ).toBeInTheDocument();
    expect(
      within(pane).getByRole("link", { name: "Open full review" }),
    ).toHaveAttribute("href", "/issues/audit_trace/finding_sqli");
    expect(
      within(pane).getByRole("button", { name: "Previous item" }),
    ).toBeEnabled();
    expect(
      screen.getByRole("link", { name: "Order search builds raw SQL" }),
    ).toHaveAttribute("aria-current", "true");
    // One-pane screens go back to the list with this link.
    expect(screen.getByRole("link", { name: "Inbox" })).toHaveAttribute(
      "href",
      "/",
    );
  });

  it("moves through the items with J and K and opens one with Enter", async () => {
    const { router } = renderInbox(busyServer());
    const user = userEvent.setup();
    const first = await screen.findByRole("link", {
      name: "Any user can read any report",
    });

    await user.keyboard("j");
    await waitFor(() =>
      expect(router.state.location.search).toBe(
        "?item=issue:audit_trace:finding_idor",
      ),
    );
    await user.keyboard("j");
    await waitFor(() =>
      expect(router.state.location.search).toBe(
        "?item=issue:audit_trace:finding_sqli",
      ),
    );
    await user.keyboard("k");
    await waitFor(() =>
      expect(router.state.location.search).toBe(
        "?item=issue:audit_trace:finding_idor",
      ),
    );
    await waitFor(() => expect(first).toHaveAttribute("aria-current", "true"));

    first.focus();
    await user.keyboard("{Enter}");
    await waitFor(() =>
      expect(router.state.location.pathname).toBe(
        "/issues/audit_trace/finding_idor",
      ),
    );
  });

  it("records a decision, announces it and moves to the next decision", async () => {
    const { router, sent } = renderInbox(
      busyServer(),
      "/?item=issue:audit_trace:finding_idor",
    );
    const user = userEvent.setup();
    const pane = await findDetail();
    expect(
      await within(pane).findByText("Possible issue 1 of 2"),
    ).toBeInTheDocument();
    const reject = await within(pane).findByRole("button", {
      name: "Not an issue",
    });
    await waitFor(() => expect(reject).toBeEnabled());

    // R chooses "Not an issue" and moves to the reason; Ctrl+Enter records.
    await user.keyboard("r");
    const reason = within(pane).getByRole("textbox", { name: "Why" });
    await waitFor(() => expect(reason).toHaveFocus());
    expect(reject).toHaveAttribute("aria-pressed", "true");
    await user.keyboard("Only administrators reach this view.");
    await user.keyboard("{Control>}{Enter}{/Control}");

    await waitFor(() =>
      expect(router.state.location.search).toBe(
        "?item=issue:audit_trace:finding_sqli",
      ),
    );
    await waitFor(() =>
      expect(pageStatus()).toHaveTextContent(
        "Decision recorded: Not an issue.",
      ),
    );
    // Focus follows to the next item, so the keyboard flow continues there.
    await waitFor(() =>
      expect(document.activeElement).toHaveClass("inbox-detail-where"),
    );
    expect(sent("POST", "/findings/finding_idor/reviews")).toHaveLength(1);
    const decisions = sent("POST", "/reviews/review_finding_idor/decisions");
    expect(decisions).toHaveLength(1);
    expect(await decisions[0]!.json()).toEqual({
      verdict: "false_positive",
      rationale: "Only administrators reach this view.",
    });
    // The decided possible issue leaves the list once the Server says so.
    expect(
      await within(await findDetail()).findByText("Possible issue 1 of 1"),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("link", { name: "Any user can read any report" }),
    ).not.toBeInTheDocument();
  });

  it("moves focus once after a decision and drops the message when the user moves on", async () => {
    const { router } = renderInbox(
      busyServer(),
      "/?item=issue:audit_trace:finding_idor",
    );
    const user = userEvent.setup();
    await decideNotAnIssue(user);

    await waitFor(() =>
      expect(router.state.location.search).toBe(
        "?item=issue:audit_trace:finding_sqli",
      ),
    );
    await waitFor(() => expect(detailWhere()).toHaveFocus());
    expect(
      within(await findDetail()).getByText("Decision recorded: Not an issue."),
    ).toBeInTheDocument();

    // The URL changes without the list (a link elsewhere, then Back): the
    // message ends, and coming back neither shows it nor moves focus again.
    await act(() => router.navigate("/?item=review:audit_wait:review_report"));
    await waitFor(() =>
      expect(detailWhere()).toHaveTextContent("Decide Report acceptance"),
    );
    expect(pageStatus().textContent).toBe("");
    act(() => {
      (document.activeElement as HTMLElement | null)?.blur();
    });
    await act(() => router.navigate(-1));
    await waitFor(() =>
      expect(detailWhere()).toHaveTextContent(/^Decide Possible issue/),
    );
    expect(detailWhere()).not.toHaveFocus();
    expect(pageStatus().textContent).toBe("");
    expect(
      screen.queryByText("Decision recorded: Not an issue."),
    ).not.toBeInTheDocument();

    // A row the user opens leaves focus in the list.
    const report = screen.getByRole("link", {
      name: "Report acceptance: crapi-workshop",
    });
    await user.click(report);
    await waitFor(() =>
      expect(router.state.location.search).toBe(
        "?item=review:audit_wait:review_report",
      ),
    );
    expect(report).toHaveFocus();

    // Opening the item the decision moved to again leaves focus in the list.
    const sqli = screen.getByRole("link", {
      name: "Order search builds raw SQL",
    });
    await user.click(sqli);
    await waitFor(() =>
      expect(router.state.location.search).toBe(
        "?item=issue:audit_trace:finding_sqli",
      ),
    );
    expect(
      await within(await findDetail()).findByText("Possible issue 1 of 1"),
    ).toBeInTheDocument();
    expect(sqli).toHaveFocus();
  });

  it("decides an approval in place and moves to the next decision", async () => {
    const { router, sent } = renderInbox(
      busyServer(),
      "/?item=review:audit_trace:review_approval",
    );
    const user = userEvent.setup();
    const pane = await findDetail();

    expect(
      await within(pane).findByRole("heading", {
        name: "Active test approval: WSTG-ATHN-01",
        level: 2,
      }),
    ).toBeInTheDocument();
    expect(detailWhere()).toHaveTextContent("Decide Active test approval");
    expect(
      within(fact(pane, "Check")).getByRole("link", {
        name: "OpenAPI · Operation trace",
      }),
    ).toHaveAttribute("href", "/projects/project_shop/audits/audit_trace");
    expect(
      within(fact(pane, "Project")).getByRole("link", {
        name: "crapi-workshop",
      }),
    ).toHaveAttribute("href", "/projects/project_shop");
    expect(fact(pane, "Subject")).toHaveTextContent("WSTG-ATHN-01");
    expect(
      within(pane).getByRole("link", { name: "Open in check" }),
    ).toHaveAttribute(
      "href",
      "/projects/project_shop/audits/audit_trace/reviews?state=pending",
    );

    await user.click(within(pane).getByRole("button", { name: "Approve" }));
    await user.type(
      within(pane).getByRole("textbox", { name: "Why" }),
      "The test account is disposable.",
    );
    await user.click(
      within(pane).getByRole("button", { name: "Record decision" }),
    );

    // The approval was the last decision: the one before it is next.
    await waitFor(() =>
      expect(router.state.location.search).toBe(
        "?item=review:audit_wait:review_report",
      ),
    );
    await waitFor(() =>
      expect(pageStatus()).toHaveTextContent("Decision recorded: Approved."),
    );
    await waitFor(() => expect(detailWhere()).toHaveFocus());
    const decisions = sent("POST", "/reviews/review_approval/decisions");
    expect(decisions).toHaveLength(1);
    expect(await decisions[0]!.json()).toEqual({
      action: "approve",
      rationale: "The test account is disposable.",
    });
    await waitFor(() =>
      expect(
        screen.queryByRole("link", {
          name: "Active test approval: WSTG-ATHN-01",
        }),
      ).not.toBeInTheDocument(),
    );
  });

  it("shows a proposed report next to its acceptance request", async () => {
    renderInbox(busyServer(), "/?item=review:audit_wait:review_report");
    const pane = await findDetail();

    expect(
      await within(pane).findByRole("heading", {
        name: "Report acceptance: crapi-workshop",
        level: 2,
      }),
    ).toBeInTheDocument();
    expect(fact(pane, "Subject")).toHaveTextContent(
      "The check's proposed report",
    );
    expect(
      within(fact(pane, "Check")).getByRole("link", {
        name: "OpenAPI · Operation trace",
      }),
    ).toHaveAttribute("href", "/projects/project_shop/audits/audit_wait");
    expect(
      within(fact(pane, "Project")).getByRole("link", {
        name: "crapi-workshop",
      }),
    ).toHaveAttribute("href", "/projects/project_shop");
    expect(
      await within(pane).findByText("Three endpoints were traced."),
    ).toBeInTheDocument();
    // A proposed report never reads as accepted.
    expect(
      within(pane).getByText(/This report is proposed and not accepted yet\./),
    ).toHaveTextContent(
      "A report is not a security or compliance certification.",
    );
    const links = within(pane).getAllByRole("link", { name: "Open report" });
    expect(links.length).toBeGreaterThan(0);
    for (const link of links)
      expect(link).toHaveAttribute("href", "/reports/audit_wait");
    expect(
      within(pane).getByText(/Approving accepts this report/),
    ).toBeInTheDocument();
    expect(
      await within(pane).findByRole("button", { name: "Approve" }),
    ).toBeEnabled();
  });

  it("says once when a proposed report cannot be read", async () => {
    const server = busyServer();
    server.override = (url) =>
      url.pathname === "/v1/audits/audit_wait/report"
        ? failure(500)
        : undefined;
    renderInbox(server, "/?item=review:audit_wait:review_report");
    const pane = await findDetail();

    expect(
      await within(pane).findByText(
        /The report could not be loaded, so this request cannot be decided here yet\./,
      ),
    ).toBeInTheDocument();
    expect(within(pane).getAllByRole("alert")).toHaveLength(1);
    expect(
      within(pane).queryByRole("heading", { name: "Proposed report" }),
    ).not.toBeInTheDocument();
    expect(
      within(pane).queryByRole("button", { name: "Approve" }),
    ).not.toBeInTheDocument();
  });

  it("keeps the row the user clicked when a decision is answered later", async () => {
    const server = busyServer();
    const answer = gate();
    server.delay = (url, request) =>
      request.method === "POST" && url.pathname.endsWith("/decisions")
        ? answer.wait
        : undefined;
    const { router, sent } = renderInbox(
      server,
      "/?item=issue:audit_trace:finding_idor",
    );
    const user = userEvent.setup();
    await decideNotAnIssue(user);
    await waitFor(() =>
      expect(
        sent("POST", "/reviews/review_finding_idor/decisions"),
      ).toHaveLength(1),
    );

    const approval = screen.getByRole("link", {
      name: "Active test approval: WSTG-ATHN-01",
    });
    await user.click(approval);
    await waitFor(() =>
      expect(router.state.location.search).toBe(
        "?item=review:audit_trace:review_approval",
      ),
    );
    answer.release();

    // The decision is announced by name, and the user stays where they are.
    await waitFor(() => expect(pageStatus()).toHaveTextContent(FIRST_DECIDED));
    expect(router.state.location.search).toBe(
      "?item=review:audit_trace:review_approval",
    );
    expect(approval).toHaveFocus();
    await waitFor(() =>
      expect(
        screen.queryByRole("link", { name: "Any user can read any report" }),
      ).not.toBeInTheDocument(),
    );
    expect(router.state.location.search).toBe(
      "?item=review:audit_trace:review_approval",
    );
  });

  it("keeps the item J moved to when a decision is answered later", async () => {
    const server = busyServer();
    const answer = gate();
    server.delay = (url, request) =>
      request.method === "POST" && url.pathname.endsWith("/decisions")
        ? answer.wait
        : undefined;
    const { router, sent } = renderInbox(
      server,
      "/?item=issue:audit_trace:finding_idor",
    );
    const user = userEvent.setup();
    await decideNotAnIssue(user);
    await waitFor(() =>
      expect(
        sent("POST", "/reviews/review_finding_idor/decisions"),
      ).toHaveLength(1),
    );

    // The reason field keeps J to itself; the user leaves it first.
    act(() => {
      (document.activeElement as HTMLElement | null)?.blur();
    });
    await user.keyboard("j");
    await waitFor(() =>
      expect(router.state.location.search).toBe(
        "?item=issue:audit_trace:finding_sqli",
      ),
    );
    await user.keyboard("j");
    await waitFor(() =>
      expect(router.state.location.search).toBe(
        "?item=review:audit_wait:review_report",
      ),
    );
    answer.release();

    await waitFor(() => expect(pageStatus()).toHaveTextContent(FIRST_DECIDED));
    expect(router.state.location.search).toBe(
      "?item=review:audit_wait:review_report",
    );
    expect(detailWhere()).not.toHaveFocus();
  });

  it("leaves another page alone when a decision is answered after the user left", async () => {
    const server = busyServer();
    const answer = gate();
    server.delay = (url, request) =>
      request.method === "POST" && url.pathname.endsWith("/decisions")
        ? answer.wait
        : undefined;
    const { router, sent } = renderInbox(
      server,
      "/?item=issue:audit_trace:finding_idor",
    );
    const user = userEvent.setup();
    await decideNotAnIssue(user);
    await waitFor(() =>
      expect(
        sent("POST", "/reviews/review_finding_idor/decisions"),
      ).toHaveLength(1),
    );

    const navigation = screen.getByRole("navigation", {
      name: "Primary navigation",
    });
    await user.click(within(navigation).getByRole("link", { name: "Runs" }));
    await waitFor(() => expect(router.state.location.pathname).toBe("/runs"));
    const refreshes = sent("GET", "/v1/projects").length;
    answer.release();

    // The decision's refresh runs (the rail badge reads the same lists),
    // then nothing takes the user back to the Inbox.
    await waitFor(() =>
      expect(sent("GET", "/v1/projects").length).toBeGreaterThan(refreshes),
    );
    await act(async () => {
      await new Promise((resolve) => window.setTimeout(resolve, 200));
    });
    expect(router.state.location.pathname).toBe("/runs");
    expect(router.state.location.search).not.toContain("item=");
  });

  it("lets the user leave while the next page loads when a decision is answered meanwhile", async () => {
    const server = busyServer();
    const answer = gate();
    server.delay = (url, request) =>
      request.method === "POST" && url.pathname.endsWith("/decisions")
        ? answer.wait
        : undefined;
    const runsCode = gate();
    const { router, sent } = renderInbox(
      server,
      "/?item=issue:audit_trace:finding_idor",
      { path: "/runs", until: runsCode.wait },
    );
    const user = userEvent.setup();
    await decideNotAnIssue(user);
    await waitFor(() =>
      expect(
        sent("POST", "/reviews/review_finding_idor/decisions"),
      ).toHaveLength(1),
    );

    const navigation = screen.getByRole("navigation", {
      name: "Primary navigation",
    });
    await user.click(within(navigation).getByRole("link", { name: "Runs" }));
    // Runs' code is still loading: the Inbox stays on screen meanwhile.
    await waitFor(() =>
      expect(router.state.navigation.location?.pathname).toBe("/runs"),
    );
    expect(await findDetail()).toBeInTheDocument();
    const refreshes = sent("GET", "/v1/projects").length;
    answer.release();

    // The decision's refresh runs, and the user is still on the way to Runs.
    await waitFor(() =>
      expect(sent("GET", "/v1/projects").length).toBeGreaterThan(refreshes),
    );
    expect(router.state.navigation.location?.pathname).toBe("/runs");
    runsCode.release();
    await waitFor(() => expect(router.state.location.pathname).toBe("/runs"));
    await act(async () => {
      await new Promise((resolve) => window.setTimeout(resolve, 200));
    });
    expect(router.state.location.pathname).toBe("/runs");
    expect(router.state.location.search).not.toContain("item=");
  });

  it("moves to the decision after one a refresh already removed", async () => {
    const server = busyServer();
    const answer = gate();
    // The Server records the decision at once but answers late; a refresh
    // in between already leaves the decided possible issue out.
    server.delay = (url, request) =>
      request.method === "POST" && url.pathname.endsWith("/decisions")
        ? answer.wait
        : undefined;
    const { router, sent } = renderInbox(
      server,
      "/?item=issue:audit_trace:finding_sqli",
    );
    const user = userEvent.setup();
    await decideNotAnIssue(user);
    await waitFor(() =>
      expect(
        sent("POST", "/reviews/review_finding_sqli/decisions"),
      ).toHaveLength(1),
    );
    server.findings.audit_trace = server.findings.audit_trace!.filter(
      (entry) => entry.findingId !== "finding_sqli",
    );
    advance(server, "audit_trace");
    await user.click(screen.getByRole("button", { name: "Refresh" }));
    await waitFor(() =>
      expect(
        screen.queryByRole("link", { name: "Order search builds raw SQL" }),
      ).not.toBeInTheDocument(),
    );
    answer.release();

    // Not the top of the list: the item that followed the decided one.
    await waitFor(() =>
      expect(router.state.location.search).toBe(
        "?item=review:audit_wait:review_report",
      ),
    );
    await waitFor(() =>
      expect(pageStatus()).toHaveTextContent(
        "Decision recorded: Not an issue.",
      ),
    );
  });

  it("selects the row the user chose for a check listed twice", async () => {
    const { router } = renderInbox(busyServer(), "/?item=check:audit_trace");
    const user = userEvent.setup();
    const stuck = await within(await findSection("Unblock")).findByRole(
      "link",
      { name: "2 endpoints couldn't be fully checked" },
    );
    const running = within(await findSection("Running")).getByRole("link", {
      name: "crapi-workshop: OpenAPI · Operation trace",
    });
    // A URL names the check: its first row is selected.
    await waitFor(() => expect(stuck).toHaveAttribute("aria-current", "true"));
    expect(running).not.toHaveAttribute("aria-current");
    expect(detailWhere()).toHaveTextContent("Unblock Check");

    await user.click(running);
    await waitFor(() =>
      expect(running).toHaveAttribute("aria-current", "true"),
    );
    expect(stuck).not.toHaveAttribute("aria-current");
    expect(router.state.location.search).toBe("?item=check:audit_trace");
    expect(detailWhere()).toHaveTextContent("Running Check");

    // The arrow keys move from the row as shown: up into Ready.
    await user.keyboard("{ArrowUp}");
    await waitFor(() =>
      expect(router.state.location.search).toBe("?item=run:run_ok"),
    );
    await waitFor(() =>
      expect(
        within(screen.getByRole("region", { name: "Inbox" })).getByRole(
          "link",
          { name: "generate@1" },
        ),
      ).toHaveFocus(),
    );
  });

  it("keeps a check's progress shown while its new counts load", async () => {
    const server = busyServer();
    const { sent } = renderInbox(server, "/?item=check:audit_paused");
    const user = userEvent.setup();
    const pane = await findDetail();
    expect(
      await within(pane).findByText("2 of 5 endpoints done"),
    ).toBeInTheDocument();

    // The check moves on; its new counts take a while.
    const counts = gate();
    server.delay = (url) =>
      url.pathname === "/v1/audits/audit_paused/workspace"
        ? counts.wait
        : undefined;
    advance(server, "audit_paused");
    server.workspaces.audit_paused = workspace(
      auditOf(server, "audit_paused"),
      { completedChecks: 3, unchecked: 2 },
    );
    const reads = sent("GET", "/v1/audits/audit_paused/workspace").length;
    await user.click(screen.getByRole("button", { name: "Refresh" }));
    await waitFor(() =>
      expect(
        sent("GET", "/v1/audits/audit_paused/workspace").length,
      ).toBeGreaterThan(reads),
    );
    expect(
      within(pane).queryByText("Loading progress…"),
    ).not.toBeInTheDocument();
    expect(within(pane).getByText("2 of 5 endpoints done")).toBeInTheDocument();

    counts.release();
    expect(
      await within(pane).findByText("3 of 5 endpoints done"),
    ).toBeInTheDocument();
  });

  it("shows a stopped check's progress, stop reason and follow-up", async () => {
    const server = busyServer();
    server.workspaces.audit_paused = workspace(
      auditOf(server, "audit_paused"),
      {
        completedChecks: 2,
        issues: 1,
        gaps: 1,
        unchecked: 2,
        unreviewedFindings: 1,
      },
    );
    renderInbox(server, "/?item=check:audit_paused");

    const pane = await findDetail();
    expect(
      await within(pane).findByText("2 of 5 endpoints done"),
    ).toBeInTheDocument();
    expect(within(pane).getByText("Time limit reached.")).toBeInTheDocument();
    expect(
      within(pane).getByText("The check used its time limit."),
    ).toBeInTheDocument();
    expect(
      within(pane).getByRole("img", {
        name: "2 of 5 endpoints done: 1 with an issue found, 1 needs follow-up, 2 not checked yet",
      }),
    ).toBeInTheDocument();
    for (const link of within(pane).getAllByRole("link", {
      name: "Open check",
    }))
      expect(link).toHaveAttribute(
        "href",
        "/projects/project_shop/audits/audit_paused",
      );
    expect(
      within(pane).getByRole("link", {
        name: "Show endpoints that need follow-up",
      }),
    ).toHaveAttribute(
      "href",
      "/projects/project_shop/audits/audit_paused/coverage?result=uncertain",
    );
    expect(
      within(pane).getByText(
        "Continue, Stop and Delete are on the check page.",
      ),
    ).toBeInTheDocument();
  });

  it("says when the item in the URL has left the Inbox", async () => {
    renderInbox(busyServer(), "/?item=issue:audit_trace:finding_gone");

    const pane = await findDetail();
    expect(
      await within(pane).findByText("This item is no longer in your Inbox"),
    ).toBeInTheDocument();
    expect(
      within(pane).getByRole("link", { name: "Open full review" }),
    ).toHaveAttribute("href", "/issues/audit_trace/finding_gone");
  });

  it("shows a failed Run's cause and leaves its recovery to the Run page", async () => {
    renderInbox(busyServer(), "/?item=run:run_failed");

    const pane = await findDetail();
    expect(
      await within(pane).findByRole("heading", { name: "review@2", level: 2 }),
    ).toBeInTheDocument();
    expect(
      within(pane).getByRole("heading", { name: "Primary cause" }),
    ).toBeInTheDocument();
    expect(
      within(pane).getByText("The model gateway refused the connection."),
    ).toBeInTheDocument();
    expect(
      within(pane).getByRole("link", { name: "Open Run" }),
    ).toHaveAttribute("href", "/runs/run_failed");
    expect(
      within(pane).getByText(
        /Continue from failed stage and Configure another Run are separate actions/,
      ),
    ).toBeInTheDocument();
    expect(
      within(pane).queryByRole("button", { name: "Retry model connection" }),
    ).not.toBeInTheDocument();
    expect(
      within(pane).queryByRole("button", {
        name: /Continue from failed stage/,
      }),
    ).not.toBeInTheDocument();
  });

  it("retries the model connection only when the Server asks for it", async () => {
    const { sent } = renderInbox(busyServer(), "/?item=run:run_waiting");
    const user = userEvent.setup();
    const pane = await findDetail();

    expect(
      await within(pane).findByText(
        "The model was unloaded or is unavailable.",
      ),
    ).toBeInTheDocument();
    await user.click(
      await within(pane).findByRole("button", {
        name: "Retry model connection",
      }),
    );
    // The same confirmation as on the Run page; nothing is sent before it.
    const dialog = screen.getByRole("alertdialog", {
      name: "Retry the model connection?",
    });
    expect(sent("POST", "/v1/runs/run_waiting/retry-gateway")).toHaveLength(0);
    await user.click(
      within(dialog).getByRole("button", { name: "Confirm retry" }),
    );

    expect(
      await within(pane).findByText(
        "Retry requested. The Run continues once the model answers.",
      ),
    ).toBeInTheDocument();
    expect(sent("POST", "/v1/runs/run_waiting/retry-gateway")).toHaveLength(1);
    await waitFor(() =>
      expect(
        within(pane).queryByRole("button", { name: "Retry model connection" }),
      ).not.toBeInTheDocument(),
    );
  });

  it("shows the queue admission notice only while admission is paused", async () => {
    const server = busyServer();
    server.paused = true;
    renderInbox(server);

    expect(
      await screen.findByText(
        /Queue admission is paused: your Runs, including running ones, start no new stages until you resume it in Runs\./,
      ),
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open Runs" })).toHaveAttribute(
      "href",
      "/runs",
    );
  });

  it("says when part of the Inbox could not be read and retries it", async () => {
    const server = busyServer();
    let failing = true;
    server.override = (url) =>
      failing && url.pathname === "/v1/audits" ? failure(500) : undefined;
    renderInbox(server);
    const user = userEvent.setup();

    expect(
      await screen.findByText(
        /Some projects, checks or Runs could not be read, so the Inbox may be incomplete\./,
      ),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("link", { name: "crapi-identity report is ready" }),
    ).not.toBeInTheDocument();

    failing = false;
    await user.click(screen.getByRole("button", { name: "Retry" }));

    expect(
      await screen.findByRole("link", {
        name: "crapi-identity report is ready",
      }),
    ).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.queryByText(/could not be read/)).not.toBeInTheDocument(),
    );
  });
});
