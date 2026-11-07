import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type {
  Audit,
  ProjectFinding,
  AuditReport,
  AuditWorkspace,
} from "../../api/audits";
import type { Project } from "../../api/projects";
import type { RunStatus, RunSummary } from "../../api/runs";
import type { WorkflowResource } from "../../api/workflows";
import { renderApplication } from "../../test/application";
import * as build from "../../test/builders";
import {
  runStatus,
  stageAttempt,
  workflowResource,
} from "../../test/builders-execution";
import { deferred } from "../../test/deferred";
import { fakeServer, type Route, type Routes } from "../../test/fake-server";
import { emptyPageOrNotFound, failure, json, page } from "../../test/responses";
import { ownerListRoutes } from "../../test/owner-lists";
import { testSession } from "../../test/render";

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
  document: Partial<NonNullable<ProjectFinding["claim"]>> = {},
): ProjectFinding {
  return build.finding(
    auditId,
    findingId,
    { revision: 3, createdAt: ago(minutes), updatedAt: ago(minutes) },
    { title, ...document },
  );
}

/** Workspace counters of `check`: five items, one done, unless given. */
function workspace(
  check: Audit,
  counts: Parameters<typeof build.workspace>[1] = {},
): AuditWorkspace {
  return build.workspace(check, {
    totalTasks: 5,
    completedTasks: 1,
    pendingTasks: 4,
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
  findings: Record<string, ProjectFinding[]>;
  reports: Record<string, AuditReport>;
  workspaces: Record<string, AuditWorkspace>;
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
    reports: {},
    workspaces: {},
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
  const wait = audit("project_shop", "audit_wait", "completed");
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
          "project_shop",
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
          "project_shop",
          "finding_sqli",
          "Order search builds raw SQL",
          20,
        ),
      ],
    },
    reports: {
      audit_wait: build.report("audit_wait", "ready", {
        summary: "# Report\n\nThree endpoints were traced.",
      }),
      audit_done: build.report("audit_done", "ready", {
        summary: "# Report\n\nAll five endpoints were traced.",
      }),
    },
    workspaces: {
      audit_trace: workspace(trace, { gaps: 2, pendingTasks: 2 }),
      audit_wait: workspace(wait, { completedTasks: 5, pendingTasks: 0 }),
      audit_paused: workspace(paused, { completedTasks: 2, pendingTasks: 3 }),
      audit_done: workspace(done, { completedTasks: 5, pendingTasks: 0 }),
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
      return json({
        ...CHECK_PAGE,
        total: items.length,
        ...page(items.map(build.findingSummary)),
      });
    },
    "GET /v1/audits/{auditId}/findings/{findingId}": ({ params }) => {
      const subject = server.findings[params.auditId!]?.find(
        (candidate) => candidate.findingId === params.findingId,
      );
      return found(subject, { ETag: `"${subject?.revision}"` });
    },

    "GET /v1/projects/{projectId}/findings/{findingId}": ({ params }) => {
      const value = Object.values(server.findings)
        .flat()
        .find(
          (f) =>
            f.projectId === params.projectId &&
            f.findingId === params.findingId,
        );
      return found(value, { ETag: `"${value?.revision}"` });
    },
    "PATCH /v1/projects/{projectId}/findings/{findingId}": async ({
      params,
      request,
    }) => {
      const value = Object.values(server.findings)
        .flat()
        .find(
          (f) =>
            f.projectId === params.projectId &&
            f.findingId === params.findingId,
        );
      if (value === undefined) return failure(404);
      if (request.headers.get("If-Match") !== `"${value.revision}"`)
        return failure(412, "precondition_failed", "Changed first");
      const body =
        (await request.json()) as import("../../api/audits").EditProjectFindingRequest;
      const states = {
        true_positive: "confirmed",
        false_positive: "rejected",
        duplicate: "duplicate",
        reopen: "proposed",
        needs_evidence: "needs-evidence",
      } as const;
      const next: ProjectFinding = {
        ...value,
        state: states[body.verdict],
        analystVerdict: body.verdict,
        analystSeverity:
          body.verdict === "true_positive" ? body.severity : null,
        duplicateTargetId:
          body.verdict === "duplicate" ? body.duplicateTargetId : null,
        rationale: body.rationale,
        actorId: "user_local",
        assessedAt: ago(0),
        revision: value.revision + 1,
      };
      for (const [key, values] of Object.entries(server.findings))
        server.findings[key] = values.map((f) =>
          f.findingId === next.findingId ? next : f,
        );
      return json({ finding: next, replayed: false }, 200, {
        ETag: `"${next.revision}"`,
      });
    },
    "GET /v1/audits/{auditId}/report": ({ params }) =>
      found(server.reports[params.auditId!]),
    "GET /v1/audits/{auditId}/workspace": ({ params }) =>
      found(server.workspaces[params.auditId!]),
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
      (async (call) => {
        if (call.request.method === "PATCH") {
          const response = await route(call);
          return (await held(call.request, call.url)) ?? response;
        }
        return (await held(call.request, call.url)) ?? route(call);
      }) satisfies Route,
    ]),
  );
  return fakeServer(
    routes,
    async (request, url) =>
      (await held(request, url)) ?? emptyPageOrNotFound(request, url),
  );
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
  it("labels a partial decision count as a lower bound and keeps the list selected", async () => {
    const server = emptyServer();
    server.projects = [build.project("project_shop")];
    const one = finding(
      "project_shop",
      "finding_partial",
      "A retained possible issue",
      1,
    );
    server.override = (url) => {
      if (
        url.pathname !== "/v1/findings" ||
        url.searchParams.get("verdict") !== "unreviewed"
      )
        return undefined;
      return url.searchParams.has("cursor")
        ? failure(503, "storage_transaction_conflict")
        : json(page([build.findingSummary(one)], "later"));
    };
    const view = renderApplication("/", { server: inboxServer(server) });
    expect(await screen.findByText("At least 1 needs you.")).toBeVisible();
    expect(screen.queryByText("Nothing needs you right now.")).toBeNull();
    expect(view.router.state.location.search).toBe("");
  });

  it("says plainly when a section is empty", async () => {
    renderApplication("/", { server: inboxServer(emptyServer()) });

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
      within(await findDetail()).getByRole("heading", {
        name: "Your Inbox is clear",
      }),
    ).toBeInTheDocument();
    expect(
      screen.queryByText(/Queue admission is paused/),
    ).not.toBeInTheDocument();
    expect(screen.queryByText(/could not be read/)).not.toBeInTheDocument();
  });

  it("reads the complete report in place and retains unavailable task-result navigation", async () => {
    const server = busyServer();
    server.reports.audit_done = build.report("audit_done", "ready", {
      summary:
        "# Retained report\n\n" +
        "Bounded observation. ".repeat(60) +
        "\n\n## Follow-up\n\nRead the exact retained results.",
    });
    renderApplication("/?item=report:audit_done", {
      server: inboxServer(server),
    });
    const pane = await findDetail();
    expect(
      await within(pane).findByText("Read the exact retained results."),
    ).toBeInTheDocument();
    expect(
      within(pane).getByRole("button", { name: "Download summary" }),
    ).toBeEnabled();
    expect(screen.queryByText("Runs and capacity")).not.toBeInTheDocument();
    expect(
      screen.queryByText(/acceptance|Active test approval/),
    ).not.toBeInTheDocument();
    server.reports.audit_done = build.report("audit_done", "unavailable");
    advance(server, "audit_done");
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Refresh" }));
    expect(
      await within(pane).findByText(
        "This check completed without a summary report.",
      ),
    ).toBeInTheDocument();
    expect(
      within(pane).getByRole("link", { name: "View task results →" }),
    ).toHaveAttribute("href", "/projects/project_identity/audits/audit_done");
    expect(
      within(pane).queryByRole("button", { name: "Download summary" }),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Refresh" })).toHaveFocus();
  });

  it("opens the item named in the URL", async () => {
    renderApplication("/?item=issue:project_shop:finding_sqli", {
      server: inboxServer(busyServer()),
    });

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
      within(pane).getByRole("link", { name: "Open in Issues" }),
    ).toHaveAttribute("href", "/issues/project_shop/finding_sqli");
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
    const { router } = renderApplication("/", {
      server: inboxServer(busyServer()),
    });
    const user = userEvent.setup();
    const first = await screen.findByRole("link", {
      name: "Any user can read any report",
    });

    await waitFor(() =>
      expect(router.state.location.search).toBe(
        "?item=issue:project_shop:finding_idor",
      ),
    );
    await waitFor(() => expect(first).toHaveAttribute("aria-current", "true"));
    await user.keyboard("j");
    await waitFor(() =>
      expect(router.state.location.search).toBe(
        "?item=issue:project_shop:finding_sqli",
      ),
    );
    await user.keyboard("k");
    await waitFor(() =>
      expect(router.state.location.search).toBe(
        "?item=issue:project_shop:finding_idor",
      ),
    );
    await waitFor(() => expect(first).toHaveAttribute("aria-current", "true"));

    first.focus();
    await user.keyboard("{Enter}");
    await waitFor(() =>
      expect(router.state.location.pathname).toBe(
        "/issues/project_shop/finding_idor",
      ),
    );
  });

  it("moves focus once after a decision and drops the message when the user moves on", async () => {
    const { router } = renderApplication(
      "/?item=issue:project_shop:finding_idor",
      { server: inboxServer(busyServer()) },
    );
    const user = userEvent.setup();
    await decideNotAnIssue(user);

    await waitFor(() =>
      expect(router.state.location.search).toBe(
        "?item=issue:project_shop:finding_sqli",
      ),
    );
    await waitFor(() => expect(detailWhere()).toHaveFocus());
    expect(
      within(await findDetail()).getByText("Decision recorded: Not an issue."),
    ).toBeInTheDocument();

    // The URL changes without the list (a link elsewhere, then Back): the
    // message ends, and coming back neither shows it nor moves focus again.
    await act(() => router.navigate("/?item=run:run_ok"));
    await waitFor(() =>
      expect(detailWhere()).toHaveTextContent("Ready Finished Run"),
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
      name: "generate@1",
    });
    await user.click(report);
    await waitFor(() =>
      expect(router.state.location.search).toBe("?item=run:run_ok"),
    );
    expect(report).toHaveFocus();

    // Opening the item the decision moved to again leaves focus in the list.
    const sqli = screen.getByRole("link", {
      name: "Order search builds raw SQL",
    });
    await user.click(sqli);
    await waitFor(() =>
      expect(router.state.location.search).toBe(
        "?item=issue:project_shop:finding_sqli",
      ),
    );
    expect(
      await within(await findDetail()).findByText("Possible issue 1 of 1"),
    ).toBeInTheDocument();
    expect(sqli).toHaveFocus();
  });

  it("keeps the row the user clicked when a decision is answered later", async () => {
    const server = busyServer();
    const answer = deferred();
    server.delay = (url, request) =>
      request.method === "PATCH" && url.pathname.includes("/findings/")
        ? answer.promise
        : undefined;
    const { router, sent } = renderApplication(
      "/?item=issue:project_shop:finding_idor",
      { server: inboxServer(server) },
    );
    const user = userEvent.setup();
    await decideNotAnIssue(user);
    await waitFor(() =>
      expect(sent("PATCH", "/findings/finding_idor")).toHaveLength(1),
    );

    const approval = screen.getByRole("link", {
      name: "Order search builds raw SQL",
    });
    await user.click(approval);
    await waitFor(() =>
      expect(router.state.location.search).toBe(
        "?item=issue:project_shop:finding_sqli",
      ),
    );
    answer.resolve();

    // The decision is announced by name, and the user stays where they are.
    await waitFor(() => expect(pageStatus()).toHaveTextContent(FIRST_DECIDED));
    expect(router.state.location.search).toBe(
      "?item=issue:project_shop:finding_sqli",
    );
    expect(approval).toHaveFocus();
    await waitFor(() =>
      expect(
        screen.queryByRole("link", { name: "Any user can read any report" }),
      ).not.toBeInTheDocument(),
    );
    expect(router.state.location.search).toBe(
      "?item=issue:project_shop:finding_sqli",
    );
  });

  it("keeps the item J moved to when a decision is answered later", async () => {
    const server = busyServer();
    const answer = deferred();
    server.delay = (url, request) =>
      request.method === "PATCH" && url.pathname.includes("/findings/")
        ? answer.promise
        : undefined;
    const { router, sent } = renderApplication(
      "/?item=issue:project_shop:finding_idor",
      { server: inboxServer(server) },
    );
    const user = userEvent.setup();
    await decideNotAnIssue(user);
    await waitFor(() =>
      expect(sent("PATCH", "/findings/finding_idor")).toHaveLength(1),
    );

    // The reason field keeps J to itself; the user leaves it first.
    act(() => {
      (document.activeElement as HTMLElement | null)?.blur();
    });
    await user.keyboard("j");
    await waitFor(() =>
      expect(router.state.location.search).toBe(
        "?item=issue:project_shop:finding_sqli",
      ),
    );
    answer.resolve();

    await waitFor(() => expect(pageStatus()).toHaveTextContent(FIRST_DECIDED));
    expect(router.state.location.search).toBe(
      "?item=issue:project_shop:finding_sqli",
    );
    expect(detailWhere()).not.toHaveFocus();
  });

  it("leaves another page alone when a decision is answered after the user left", async () => {
    const server = busyServer();
    const answer = deferred();
    server.delay = (url, request) =>
      request.method === "PATCH" && url.pathname.includes("/findings/")
        ? answer.promise
        : undefined;
    const { router, sent } = renderApplication(
      "/?item=issue:project_shop:finding_idor",
      { server: inboxServer(server) },
    );
    const user = userEvent.setup();
    await decideNotAnIssue(user);
    await waitFor(() =>
      expect(sent("PATCH", "/findings/finding_idor")).toHaveLength(1),
    );

    const navigation = screen.getByRole("navigation", {
      name: "Primary navigation",
    });
    await user.click(within(navigation).getByRole("link", { name: "Runs" }));
    await waitFor(() => expect(router.state.location.pathname).toBe("/runs"));
    const refreshes = sent("GET", "/v1/findings").length;
    answer.resolve();

    // The decision's refresh runs (the rail badge reads the same lists),
    // then nothing takes the user back to the Inbox.
    await waitFor(() =>
      expect(sent("GET", "/v1/findings").length).toBeGreaterThan(refreshes),
    );
    await act(async () => {
      await new Promise((resolve) => window.setTimeout(resolve, 200));
    });
    expect(router.state.location.pathname).toBe("/runs");
    expect(router.state.location.search).not.toContain("item=");
  });

  it("lets the user leave while the next page loads when a decision is answered meanwhile", async () => {
    const server = busyServer();
    const answer = deferred();
    server.delay = (url, request) =>
      request.method === "PATCH" && url.pathname.includes("/findings/")
        ? answer.promise
        : undefined;
    const runsCode = deferred();
    const { router, sent } = renderApplication(
      "/?item=issue:project_shop:finding_idor",
      {
        server: inboxServer(server),
        slowRoutes: { "/runs": runsCode.promise },
      },
    );
    const user = userEvent.setup();
    await decideNotAnIssue(user);
    await waitFor(() =>
      expect(sent("PATCH", "/findings/finding_idor")).toHaveLength(1),
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
    const refreshes = sent("GET", "/v1/findings").length;
    answer.resolve();

    // The decision's refresh runs, and the user is still on the way to Runs.
    await waitFor(() =>
      expect(sent("GET", "/v1/findings").length).toBeGreaterThan(refreshes),
    );
    expect(router.state.navigation.location?.pathname).toBe("/runs");
    runsCode.resolve();
    await waitFor(() => expect(router.state.location.pathname).toBe("/runs"));
    await act(async () => {
      await new Promise((resolve) => window.setTimeout(resolve, 200));
    });
    expect(router.state.location.pathname).toBe("/runs");
    expect(router.state.location.search).not.toContain("item=");
  });

  it("moves to the decision after one a refresh already removed", async () => {
    const server = busyServer();
    const answer = deferred();
    // The Server records the decision at once but answers late; a refresh
    // in between already leaves the decided possible issue out.
    server.delay = (url, request) =>
      request.method === "PATCH" && url.pathname.includes("/findings/")
        ? answer.promise
        : undefined;
    const { router, sent } = renderApplication(
      "/?item=issue:project_shop:finding_sqli",
      { server: inboxServer(server) },
    );
    const user = userEvent.setup();
    await decideNotAnIssue(user);
    await waitFor(() =>
      expect(sent("PATCH", "/findings/finding_sqli")).toHaveLength(1),
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
    answer.resolve();

    // Not the top of the list: the item that followed the decided one.
    await waitFor(() =>
      expect(router.state.location.search).toBe(
        "?item=issue:project_shop:finding_idor",
      ),
    );
    await waitFor(() =>
      expect(pageStatus()).toHaveTextContent(
        "Decision recorded: Not an issue.",
      ),
    );
  });

  it.each(["openapi-observe", "asvs", "wstg"])(
    "labels preparation and pending checks as tasks for %s",
    async (profile) => {
      const server = busyServer();
      const audit = auditOf(server, "audit_trace");
      audit.profile.name = profile;
      server.workspaces.audit_trace = workspace(audit, {
        totalTasks: 3,
        completedTasks: 1,
        gaps: 0,
        pendingTasks: 2,
        findings: 0,
        unreviewedFindings: 0,
      });
      renderApplication("/?item=check:audit_trace", {
        server: inboxServer(server),
      });
      const pane = await findDetail();
      expect(
        await within(pane).findByText("1 of 3 tasks done"),
      ).toBeInTheDocument();
      expect(
        within(await findSection("Running")).getByText("1 of 3 tasks done"),
      ).toBeInTheDocument();
      expect(
        within(pane).getByRole("img", { name: "1 of 3 tasks done: 2 pending" }),
      ).toBeInTheDocument();
    },
  );

  it("selects the row the user chose for a check listed twice", async () => {
    const { router } = renderApplication("/?item=check:audit_trace", {
      server: inboxServer(busyServer()),
    });
    const user = userEvent.setup();
    const stuck = await within(await findSection("Unblock")).findByRole(
      "link",
      { name: "2 tasks with gaps" },
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
    const { sent } = renderApplication("/?item=check:audit_paused", {
      server: inboxServer(server),
    });
    const user = userEvent.setup();
    const pane = await findDetail();
    expect(
      await within(pane).findByText("2 of 5 tasks done"),
    ).toBeInTheDocument();

    // The check moves on; its new counts take a while.
    const counts = deferred();
    server.delay = (url) =>
      url.pathname === "/v1/audits/audit_paused/workspace"
        ? counts.promise
        : undefined;
    advance(server, "audit_paused");
    server.workspaces.audit_paused = workspace(
      auditOf(server, "audit_paused"),
      { completedTasks: 3, pendingTasks: 2 },
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
    expect(within(pane).getByText("2 of 5 tasks done")).toBeInTheDocument();

    counts.resolve();
    expect(
      await within(pane).findByText("3 of 5 tasks done"),
    ).toBeInTheDocument();
  });

  it("shows a stopped check's progress, stop reason and follow-up", async () => {
    const server = busyServer();
    server.workspaces.audit_paused = workspace(
      auditOf(server, "audit_paused"),
      {
        completedTasks: 2,
        gaps: 1,
        pendingTasks: 2,
        unreviewedFindings: 1,
      },
    );
    renderApplication("/?item=check:audit_paused", {
      server: inboxServer(server),
    });

    const pane = await findDetail();
    expect(
      await within(pane).findByText("2 of 5 tasks done"),
    ).toBeInTheDocument();
    expect(within(pane).getByText("Time limit reached.")).toBeInTheDocument();
    expect(
      within(pane).getByText("The check used its time limit."),
    ).toBeInTheDocument();
    expect(
      within(pane).getByRole("img", {
        name: "2 of 5 tasks done: 1 with gaps, 2 pending",
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
      within(pane).queryByRole("link", { name: /need follow-up/ }),
    ).not.toBeInTheDocument();
    expect(
      within(pane).getByText(
        "Continue, Stop and Delete are on the check page.",
      ),
    ).toBeInTheDocument();
  });

  it("says when the item in the URL has left the Inbox", async () => {
    renderApplication("/?item=issue:project_shop:finding_gone", {
      server: inboxServer(busyServer()),
    });

    const pane = await findDetail();
    expect(
      await within(pane).findByText("This item is no longer in your Inbox"),
    ).toBeInTheDocument();
    expect(
      within(pane).getByRole("link", { name: "Open possible issue" }),
    ).toHaveAttribute("href", "/issues/project_shop/finding_gone");
  });

  it("shows a failed Run's cause and leaves its recovery to the Run page", async () => {
    renderApplication("/?item=run:run_failed", {
      server: inboxServer(busyServer()),
    });

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
    const { sent } = renderApplication("/?item=run:run_waiting", {
      server: inboxServer(busyServer()),
    });
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
    renderApplication("/", { server: inboxServer(server) });

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
    renderApplication("/", { server: inboxServer(server) });
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
