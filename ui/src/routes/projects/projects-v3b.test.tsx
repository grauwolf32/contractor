import { ownerListResponse } from "../../test/owner-lists";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";

import type { ArtifactMetadata } from "../../api/artifacts";
import type { Audit, AuditFinding, AuditProfile } from "../../api/audits";
import { PublicAPI } from "../../api/client";
import type { Project } from "../../api/projects";
import type { RunSummary } from "../../api/runs";
import type { WorkflowSummary } from "../../api/workflows";
import { Application } from "../../app/application";
import * as queryClientFactory from "../../app/query-client";
import { applicationRoutes } from "../../app/router";
import type { RuntimeConfig } from "../../config/runtime-config";
import { auditFixture, projectFixture } from "../../test/shell-harness";
import { makeFinding } from "../decisions/test-support";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

function session(capabilities: string[] = ["user", "operations"]) {
  return {
    principal: { userId: "user_local", username: "owner", capabilities },
    csrfToken: "a".repeat(43),
    idleExpiresAt: "2026-10-06T20:00:00Z",
    absoluteExpiresAt: "2026-10-07T12:00:00Z",
  };
}

const DIGEST = `sha256:${"a".repeat(64)}`;
const RECENT = new Date(Date.now() - 5 * 60_000).toISOString();

function json(value: unknown, init: ResponseInit = {}): Response {
  const headers = new Headers(init.headers);
  headers.set("content-type", "application/json");
  headers.set("X-Contractor-API-Version", "contractor.public.v1");
  return new Response(JSON.stringify(value), { ...init, headers });
}

function page(items: unknown[], hasMore = false) {
  return {
    items,
    page: hasMore ? { hasMore, nextCursor: "next" } : { hasMore },
  };
}

function check(
  projectId: string,
  auditId: string,
  state: Audit["state"],
  profile = "openapi-operation-trace",
  extra: Partial<Audit> = {},
): Audit {
  const base = auditFixture(projectId, auditId, state);
  return {
    ...base,
    profile: { ...base.profile, name: profile },
    updatedAt: RECENT,
    ...extra,
  };
}

function finding(auditId: string, findingId: string, title: string) {
  return makeFinding(
    { auditId, findingId, createdAt: RECENT },
    { title, subject: { kind: "operation", key: "GET /orders/{id}" } },
  );
}

function material(
  namespace: string,
  name: string,
  mediaType: string,
): ArtifactMetadata {
  return {
    artifact: { namespace, name, revision: `${name}-r1` },
    mediaType,
    size: 2048,
    current: true,
    frozen: false,
    createdAt: RECENT,
  };
}

function run(
  runId: string,
  state: RunSummary["state"],
  labels: Record<string, string> = {},
): RunSummary {
  return {
    runId,
    projectId: "project_a",
    workflow: "openapi-from-source@1",
    state,
    deletable: true,
    labels,
    createdAt: RECENT,
    updatedAt: RECENT,
    ...(state === "succeeded" ? { finishedAt: RECENT } : {}),
  };
}

const profileBase: Omit<AuditProfile, "ref" | "inputs"> = {
  mode: "operation-tracing",
  standards: [],
  inventory: {
    implementation: "openapi-operations@1",
    source: { source: "audit-input", name: "openapi" },
    itemWorkflowRole: "check",
  },
  execution: {
    roundMode: "fixed-barrier",
    maxRounds: 1,
    batchSize: 1,
    maxItemsPerRound: 10,
    maxItemsTotal: 10,
    maxSubmittedRuns: 30,
    maxItemRunAttempts: 3,
    deadlineSeconds: 600,
    maxEvidenceBytes: 1048576,
    incompleteRound: "assess-with-gaps",
  },
  interaction: {
    activeChecks: "prohibited",
    findingConfirmation: "human-required",
    notApplicable: "human-required",
    reportAcceptance: "automatic",
  },
  serverCompatible: true,
  requiresInputValidation: true,
  compatibilityReasons: [],
};

const workflow: WorkflowSummary = {
  ref: { name: "openapi-from-source", version: "1" },
  entryStage: "analyze",
  parameters: {},
  inputs: { source: { required: true, mediaTypes: ["application/zip"] } },
  outputs: {
    openapi: {
      required: true,
      mediaTypes: ["application/yaml"],
      primary: true,
    },
  },
};

interface FakeServer {
  projects: Project[];
  audits?: Record<string, Audit[]>;
  /** Checks of a project have a further page. */
  moreAudits?: boolean;
  findings?: Record<string, AuditFinding[]>;
  runs?: RunSummary[];
  outputs?: Record<string, Record<string, unknown>>;
  artifacts?: ArtifactMetadata[];
  profiles?: AuditProfile[];
  capabilities?: string[];
  /** Answers POST /v1/projects. */
  createProject?: (request: Request) => Response | Promise<Response>;
  /** Answers PATCH /v1/projects/:projectId. */
  updateProject?: (request: Request) => Response | Promise<Response>;
  /** Answers POST /v1/audits/:auditId/{start,pause,resume,cancel}. */
  mutateAudit?: (auditId: string, action: string) => Response;
  /** Holds a check's possible-issue read until the promise settles. */
  holdFindings?: (auditId: string) => Promise<void> | undefined;
  /** Leaves every read of a project's checks unanswered while true. */
  holdAudits?: () => boolean;
}

function serve(server: FakeServer) {
  const requests: Request[] = [];
  const api = new PublicAPI(
    runtimeConfig,
    vi.fn(async (input) => {
      const request = input instanceof Request ? input : new Request(input);
      requests.push(request.clone());
      const url = new URL(request.url);
      const path = url.pathname;
      if (path === "/v1/audits" && server.holdAudits?.())
        await new Promise(() => {});
      if (path === "/v1/findings")
        await Promise.all(
          Object.keys(server.findings ?? {}).map((id) =>
            server.holdFindings?.(id),
          ),
        );
      const ownerResponse = ownerListResponse(url, {
        projects: server.projects,
        audits: Object.values(server.audits ?? {}).flat(),
        findings: Object.values(server.findings ?? {}).flat(),
      });
      if (ownerResponse !== undefined) return ownerResponse;
      const query = url.searchParams;
      if (path === "/v1/auth/session")
        return json(session(server.capabilities));
      if (path === "/v1/projects" && request.method === "POST") {
        if (server.createProject === undefined)
          throw new Error("unexpected POST /v1/projects");
        return server.createProject(request);
      }
      if (path === "/v1/projects") return json(page(server.projects));
      let match = /^\/v1\/projects\/([^/]+)$/.exec(path);
      if (match !== null && request.method === "PATCH") {
        if (server.updateProject === undefined)
          throw new Error(`unexpected PATCH ${path}`);
        return server.updateProject(request);
      }
      if (match !== null) {
        const found = server.projects.find(
          (candidate) => candidate.projectId === match![1],
        );
        return found === undefined
          ? json(
              { code: "not_found", message: "not found", retryable: false },
              { status: 404 },
            )
          : json(found, { headers: { ETag: `"${found.revision}"` } });
      }
      match = /^\/v1\/projects\/([^/]+)\/audits$/.exec(path);
      if (match !== null) {
        if (server.holdAudits?.() === true)
          return new Promise<Response>(() => undefined);
        const state = query.get("state");
        const items = (server.audits?.[match[1]!] ?? []).filter(
          (audit) => state === null || audit.state === state,
        );
        return json(page(items, server.moreAudits === true && state === null));
      }
      match = /^\/v1\/audits\/([^/]+)\/(start|pause|resume|cancel)$/.exec(path);
      if (match !== null && request.method === "POST") {
        if (server.mutateAudit === undefined)
          throw new Error(`unexpected POST ${path}`);
        return server.mutateAudit(match[1]!, match[2]!);
      }
      match = /^\/v1\/audits\/([^/]+)$/.exec(path);
      if (match !== null && request.method === "GET") {
        const auditId = match[1];
        const found = Object.values(server.audits ?? {})
          .flat()
          .find((audit) => audit.auditId === auditId);
        return found === undefined
          ? json(
              { code: "not_found", message: "not found", retryable: false },
              { status: 404 },
            )
          : json(found, { headers: { ETag: `"${found.revision}"` } });
      }
      match = /^\/v1\/audits\/([^/]+)\/(findings|reviews)$/.exec(path);
      if (match !== null) {
        if (match[2] === "findings") await server.holdFindings?.(match[1]!);
        const items =
          match[2] === "findings" && query.get("state") === "proposed"
            ? (server.findings?.[match[1]!] ?? [])
            : [];
        return json({
          ...page(items),
          total: items.length,
          auditRevision: 1,
          asOf: RECENT,
        });
      }
      match = /^\/v1\/projects\/([^/]+)\/runs$/.exec(path);
      if (match !== null) {
        const state = query.get("state");
        const limit = Number(query.get("limit") ?? "50");
        return json(
          page(
            (server.runs ?? [])
              .filter((item) => state === null || item.state === state)
              .slice(0, limit),
          ),
        );
      }
      match = /^\/v1\/runs\/([^/]+)$/.exec(path);
      if (match !== null) {
        const summary = (server.runs ?? []).find(
          (item) => item.runId === match![1],
        );
        return json({
          ...summary,
          runtimeLabels: [],
          runtimeConfiguration: {
            default: {
              label: "default",
              bindingRevision: "1",
              config: { name: "empty", version: "1", digest: DIGEST },
            },
            labels: [],
          },
          attempts: [],
          transitions: [],
          outputs: server.outputs?.[match[1]!] ?? {},
          outputPublications: [],
        });
      }
      if (path === "/v1/workflows") return json(page([workflow]));
      if (path === "/v1/workflows/openapi-from-source/versions/1")
        return json({ ...workflow, stages: {} });
      match = /^\/v1\/projects\/([^/]+)\/artifacts$/.exec(path);
      if (match !== null) {
        const namespace = query.get("namespace");
        const limit = Number(query.get("limit") ?? "50");
        const items = (server.artifacts ?? []).filter(
          (item) => namespace === null || item.artifact.namespace === namespace,
        );
        return json(page(items.slice(0, limit), items.length > limit));
      }
      if (path === "/v1/audit-profiles")
        return json(page(server.profiles ?? []));
      return json(page([]));
    }),
  );
  return { api, requests };
}

/** Renders the application at `path`, after `earlier` history entries. */
function renderAt(server: FakeServer, path: string, earlier: string[] = []) {
  const { api, requests } = serve(server);
  const router = createMemoryRouter(applicationRoutes(), {
    initialEntries: [...earlier, path],
    initialIndex: earlier.length,
  });
  const queryClient = queryClientFactory.createApplicationQueryClient();
  const factory = vi
    .spyOn(queryClientFactory, "createApplicationQueryClient")
    .mockReturnValueOnce(queryClient);
  render(<Application api={api} publicAPI={api} router={router} />);
  factory.mockRestore();
  return { router, requests, queryClient, user: userEvent.setup() };
}

/** GET requests to `pathname`. */
function reads(requests: Request[], pathname: string) {
  return requests.filter(
    (request) =>
      request.method === "GET" && new URL(request.url).pathname === pathname,
  ).length;
}

const projectA = projectFixture("project_a", {
  name: "Project A",
  description: "Shop and mechanic APIs",
});
const projectB = projectFixture("project_b", { name: "Project B" });
const projectC = projectFixture("project_c", { name: "Project C" });

describe("Projects list pane", () => {
  it("says in one line what each project needs or did last", async () => {
    const deleting = projectFixture("project_d", {
      name: "Project D",
      lifecycle: "deleting",
      revision: "2",
      deletion: { phase: "draining", requestedAt: RECENT },
    });
    const quiet = projectFixture("project_e", { name: "Project E" });
    const ending = projectFixture("project_f", { name: "Project F" });
    const finished = projectFixture("project_g", { name: "Project G" });
    renderAt(
      {
        projects: [
          projectA,
          projectB,
          projectC,
          deleting,
          quiet,
          ending,
          finished,
        ],
        audits: {
          project_a: [
            check("project_a", "audit_running", "active"),
            check("project_a", "audit_done", "completed"),
          ],
          project_b: [
            check("project_b", "audit_finished", "completed", undefined, {
              finishedAt: RECENT,
            }),
          ],
          project_c: [check("project_c", "audit_waiting", "waiting_review")],
          project_f: [
            check("project_f", "audit_stopping", "cancelling"),
            check("project_f", "audit_finishing", "finalizing"),
          ],
          project_g: [
            check("project_g", "audit_old", "completed", undefined, {
              finishedAt: RECENT,
            }),
          ],
        },
        findings: {
          audit_running: [
            finding("audit_running", "finding_1", "First"),
            finding("audit_running", "finding_2", "Second"),
          ],
          // Finished checks' possible issues count too, as on the overview.
          audit_done: [finding("audit_done", "finding_3", "Third")],
          audit_waiting: [finding("audit_waiting", "finding_4", "Fourth")],
          audit_old: [finding("audit_old", "finding_5", "Fifth")],
        },
      },
      "/projects",
    );
    const row = async (name: string) =>
      (await screen.findByRole("link", { name })).closest("li")!;
    expect(
      await screen.findByRole("heading", { name: "Projects" }),
    ).toBeVisible();
    await waitFor(async () =>
      expect(await row("Project A")).toHaveTextContent(
        "Check running·3 possible issues to review",
      ),
    );
    await waitFor(async () =>
      expect(await row("Project B")).toHaveTextContent(/Check finished/),
    );
    await waitFor(async () =>
      expect(await row("Project C")).toHaveTextContent(
        "Check waiting for you·1 possible issue to review",
      ),
    );
    // A stopped check is not running: each state says so in its own words.
    await waitFor(async () =>
      expect(await row("Project F")).toHaveTextContent(
        "Check finishing·Check stopping",
      ),
    );
    expect(await row("Project F")).not.toHaveTextContent(/running/);
    // Unreviewed possible issues of a finished check still need the user.
    await waitFor(async () =>
      expect(await row("Project G")).toHaveTextContent(
        "1 possible issue to review",
      ),
    );
    expect(await row("Project D")).toHaveTextContent(/Deleting·requested/);
    expect(await row("Project E")).toHaveTextContent(/Updated/);
    expect(screen.getByText("7 projects, newest first")).toBeVisible();
  });

  it("counts a project's possible issues as its overview does", async () => {
    const { router } = renderAt(
      {
        projects: [projectA],
        audits: {
          project_a: [
            check("project_a", "audit_running", "active"),
            check("project_a", "audit_done", "completed"),
          ],
        },
        findings: {
          audit_running: [finding("audit_running", "finding_1", "First")],
          audit_done: [
            finding("audit_done", "finding_2", "Second"),
            finding("audit_done", "finding_3", "Third"),
          ],
        },
      },
      "/projects/project_a",
    );
    const glance = await screen.findByRole("region", { name: "At a glance" });
    await within(glance).findByRole("link", { name: "3 to review" });
    const row = screen.getByRole("link", { name: "Project A" }).closest("li")!;
    await waitFor(() =>
      expect(row).toHaveTextContent("3 possible issues to review"),
    );
    expect(router.state.location.pathname).toBe("/projects/project_a");
  });

  it("keeps the selection in the path and moves it with J and K where the section has no list", async () => {
    const { router, user } = renderAt(
      { projects: [projectA, projectB, projectC] },
      "/projects/project_a",
    );
    await screen.findByRole("heading", { name: "Project A", level: 2 });
    expect(screen.getByRole("link", { name: "Project A" })).toHaveAttribute(
      "aria-current",
      "true",
    );
    await user.keyboard("j");
    await waitFor(() =>
      expect(router.state.location.pathname).toBe("/projects/project_b"),
    );
    expect(
      await screen.findByRole("heading", { name: "Project B", level: 2 }),
    ).toBeVisible();
    expect(screen.getByRole("link", { name: "Project B" })).toHaveAttribute(
      "aria-current",
      "true",
    );
    await user.keyboard("k");
    await waitFor(() =>
      expect(router.state.location.pathname).toBe("/projects/project_a"),
    );
    // Runs has its own list: J and K stay with the section.
    await act(() => router.navigate("/projects/project_a/runs"));
    await screen.findByRole("group", { name: "Run views" });
    await user.keyboard("j");
    expect(router.state.location.pathname).toBe("/projects/project_a/runs");
  });

  it("keeps J and K from leaving an unsaved draft", async () => {
    const { router, user } = renderAt(
      { projects: [projectA, projectB] },
      "/projects/project_a/settings",
    );
    await screen.findByRole("region", { name: "Project details" });
    await user.click(screen.getByRole("button", { name: "Edit metadata" }));
    await user.type(screen.getByLabelText("Name"), " renamed");
    // Focus leaves the field: the draft is still unsaved.
    await user.click(screen.getByRole("heading", { name: "Live target" }));
    await user.keyboard("j");
    expect(router.state.location.pathname).toBe("/projects/project_a/settings");
    expect(screen.getByLabelText("Name")).toHaveValue("Project A renamed");
    expect(screen.queryByText(/move between projects/)).toBeNull();
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    await user.keyboard("j");
    await waitFor(() =>
      expect(router.state.location.pathname).toBe("/projects/project_b"),
    );

    // An objective typed into the composer is a draft as well.
    await act(() => router.navigate("/projects/project_a"));
    const objective = await screen.findByRole("textbox", {
      name: "What do you want to check?",
    });
    await user.type(objective, "Can one customer read another's orders?");
    await user.click(screen.getByRole("heading", { name: "Timeline" }));
    await user.keyboard("j");
    expect(router.state.location.pathname).toBe("/projects/project_a");
    await user.clear(objective);
    await user.click(screen.getByRole("heading", { name: "Timeline" }));
    await user.keyboard("j");
    await waitFor(() =>
      expect(router.state.location.pathname).toBe("/projects/project_b"),
    );
  });

  it("keeps the list pane mounted while a project is chosen", async () => {
    const { router, user } = renderAt(
      { projects: [projectA, projectB, projectC] },
      "/projects",
    );
    const filter = await screen.findByRole("searchbox", {
      name: "Filter projects",
    });
    await user.type(filter, "project");
    await user.click(screen.getByRole("link", { name: "Project B" }));
    await waitFor(() =>
      expect(router.state.location.pathname).toBe("/projects/project_b"),
    );
    expect(
      await screen.findByRole("heading", { name: "Project B", level: 2 }),
    ).toBeVisible();
    // Same component for /projects and /projects/:projectId: the list keeps
    // its state.
    expect(
      screen.getByRole("searchbox", { name: "Filter projects" }),
    ).toHaveValue("project");
    expect(screen.getByRole("link", { name: "Project B" })).toHaveAttribute(
      "aria-current",
      "true",
    );
  });

  it("refreshes the list on request", async () => {
    const { requests, user } = renderAt({ projects: [projectA] }, "/projects");
    await screen.findByRole("link", { name: "Project A" });
    const listReads = () =>
      requests.filter(
        (request) =>
          new URL(request.url).pathname === "/v1/projects" &&
          request.method === "GET",
      ).length;
    const before = listReads();
    await user.click(screen.getByRole("button", { name: "Refresh projects" }));
    await waitFor(() => expect(listReads()).toBeGreaterThan(before));
  });

  it("filters the loaded projects by name", async () => {
    const { user } = renderAt(
      { projects: [projectA, projectB, projectC] },
      "/projects",
    );
    await screen.findByRole("link", { name: "Project B" });
    await user.type(
      screen.getByRole("searchbox", { name: "Filter projects" }),
      "project b",
    );
    expect(screen.getByRole("link", { name: "Project B" })).toBeVisible();
    expect(screen.queryByRole("link", { name: "Project A" })).toBeNull();
    expect(screen.getByText("1 of 3 projects")).toBeVisible();
  });
});

describe("New project", () => {
  const created = projectFixture("project_new", { name: "Payment service" });

  function createdResponse(server: FakeServer): Response {
    server.projects = [created, ...server.projects];
    return json(created, { status: 201, headers: { ETag: '"1"' } });
  }

  it("replaces ?new=1 once the project exists, so Back does not reopen the dialog", async () => {
    const server: FakeServer = {
      projects: [projectA],
      createProject: () => createdResponse(server),
    };
    const { router, user } = renderAt(server, "/projects?new=1", ["/projects"]);
    const dialog = await screen.findByRole("dialog", { name: "New project" });
    await user.type(within(dialog).getByLabelText("Name"), "Payment service");
    await user.click(
      within(dialog).getByRole("button", { name: "Create project" }),
    );
    expect(
      await screen.findByRole("heading", {
        name: "Payment service",
        level: 2,
      }),
    ).toBeVisible();
    expect(router.state.location.pathname).toBe("/projects/project_new");
    expect(router.state.historyAction).toBe("REPLACE");
    expect(screen.queryByRole("dialog")).toBeNull();
    await act(() => router.navigate(-1));
    expect(await screen.findByText("Choose a project")).toBeVisible();
    expect(router.state.location.pathname).toBe("/projects");
    expect(router.state.location.search).toBe("");
    expect(screen.queryByRole("dialog", { name: "New project" })).toBeNull();
  });

  it("reuses the Idempotency-Key after the dialog was closed and opened again", async () => {
    const creates: Request[] = [];
    let unavailable = true;
    const server: FakeServer = {
      projects: [projectA],
      createProject: (request) => {
        creates.push(request.clone());
        return unavailable
          ? json(
              {
                code: "unavailable",
                message: "Project store unavailable",
                retryable: true,
              },
              { status: 503 },
            )
          : createdResponse(server);
      },
    };
    const { router, user } = renderAt(server, "/projects");
    async function submitNewProject() {
      await user.click(
        (await screen.findAllByRole("button", { name: "New project" }))[0]!,
      );
      const dialog = screen.getByRole("dialog", { name: "New project" });
      await user.type(within(dialog).getByLabelText("Name"), "Payment service");
      await user.type(within(dialog).getByLabelText("Description"), "Shop");
      await user.click(
        within(dialog).getByRole("button", { name: "Create project" }),
      );
      return dialog;
    }
    const dialog = await submitNewProject();
    expect(
      await within(dialog).findByText("Project store unavailable"),
    ).toBeVisible();
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    // The response was lost: the same request again carries the same key.
    unavailable = false;
    await submitNewProject();
    await waitFor(() =>
      expect(router.state.location.pathname).toBe("/projects/project_new"),
    );
    expect(creates).toHaveLength(2);
    expect(creates[1]?.headers.get("Idempotency-Key")).toBe(
      creates[0]?.headers.get("Idempotency-Key"),
    );
    // Once a project exists, the same values make a new request.
    await submitNewProject();
    await waitFor(() => expect(creates).toHaveLength(3));
    expect(creates[2]?.headers.get("Idempotency-Key")).not.toBe(
      creates[1]?.headers.get("Idempotency-Key"),
    );
  });

  it("opens the new project without waiting for every cross-project list", async () => {
    let created = false;
    const server: FakeServer = {
      projects: [projectA],
      // After the write, no project's checks answer (a slow Server).
      holdAudits: () => created,
      createProject: () => {
        created = true;
        return createdResponse(server);
      },
    };
    const { router, user } = renderAt(server, "/projects?new=1");
    const dialog = await screen.findByRole("dialog", { name: "New project" });
    await user.type(within(dialog).getByLabelText("Name"), "Payment service");
    await user.click(
      within(dialog).getByRole("button", { name: "Create project" }),
    );
    await waitFor(() =>
      expect(router.state.location.pathname).toBe("/projects/project_new"),
    );
  });
});

describe("Project deletion progress", () => {
  it.each([
    ["draining", "Waiting for Runtime release", 1],
    ["purging_runs", "Removing Run history", 2],
  ] as const)(
    "lists the phases done and the current one while %s",
    async (phase, heading, current) => {
      const deleting = projectFixture("project_d", {
        name: "Project D",
        lifecycle: "deleting",
        revision: "2",
        deletion: { phase, requestedAt: RECENT },
      });
      renderAt({ projects: [deleting] }, "/projects/project_d");
      expect(
        await screen.findByRole("heading", { name: heading }),
      ).toBeVisible();
      const phases = within(
        screen.getByRole("list", { name: "Deletion phases" }),
      ).getAllByRole("listitem");
      expect(phases.map((item) => item.textContent)).toEqual(
        [
          "Cancelling active Runs",
          "Waiting for Runtime release",
          "Removing Run history",
          "Removing Project Artifacts",
        ].map((label, index) => (index < current ? `${label}, done` : label)),
      );
      phases.forEach((item, index) => {
        if (index === current)
          expect(item).toHaveAttribute("aria-current", "step");
        else expect(item).not.toHaveAttribute("aria-current");
      });
    },
  );
});

describe("Project sections", () => {
  it.each([
    ["", "Overview"],
    ["artifacts", "Materials"],
    ["audits", "Checks"],
    ["findings", "Issues"],
    ["settings", "Settings"],
    ["runs", "Runs"],
    ["workflows", "Workflows"],
  ])("opens /projects/project_a/%s directly as %s", async (segment, label) => {
    renderAt(
      { projects: [projectA] },
      `/projects/project_a${segment === "" ? "" : `/${segment}`}`,
    );
    const nav = await screen.findByRole("navigation", {
      name: "Project sections",
    });
    await waitFor(() =>
      expect(within(nav).getByRole("link", { name: label })).toHaveAttribute(
        "aria-current",
        "page",
      ),
    );
    expect(
      screen.getByRole("combobox", { name: "Project section" }),
    ).toHaveValue(segment);
  });

  it("names the sections in V3B words and keeps Runs and Workflows under Advanced", async () => {
    const { router, user } = renderAt(
      { projects: [projectA] },
      "/projects/project_a",
    );
    const nav = await screen.findByRole("navigation", {
      name: "Project sections",
    });
    expect(
      within(nav)
        .getAllByRole("link")
        .map((link) => link.textContent),
    ).toEqual([
      "Overview",
      "Materials",
      "Checks",
      "Issues",
      "Settings",
      "Runs",
      "Workflows",
    ]);
    const advanced = within(nav).getByRole("group", { name: "Advanced" });
    expect(
      within(advanced)
        .getAllByRole("link")
        .map((link) => link.getAttribute("href")),
    ).toEqual(["/projects/project_a/runs", "/projects/project_a/workflows"]);
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Project section" }),
      "findings",
    );
    await waitFor(() =>
      expect(router.state.location.pathname).toBe(
        "/projects/project_a/findings",
      ),
    );
    expect(screen.getByText("Project ID")).toBeVisible();
    expect(
      screen.getByRole("button", { name: "Copy project ID" }),
    ).toBeVisible();
  });
});

describe("Project overview", () => {
  const overviewServer: FakeServer = {
    projects: [projectA],
    audits: {
      project_a: [
        check("project_a", "audit_running", "active"),
        check("project_a", "audit_wait", "waiting_review", "source-checklist"),
        check(
          "project_a",
          "audit_done",
          "completed",
          "owasp-top10-2025-source-risk",
          {
            finishedAt: RECENT,
          },
        ),
      ],
    },
    findings: {
      audit_running: [finding("audit_running", "finding_1", "Order IDOR")],
      audit_done: [
        finding("audit_done", "finding_2", "SQL in search"),
        finding("audit_done", "finding_3", "Hard-coded key"),
      ],
    },
    runs: [
      run("run_ok", "succeeded"),
      run("run_missing", "succeeded"),
      run("run_item", "running", { "audit.id": "audit_running" }),
    ],
    outputs: {
      run_ok: {
        openapi: { namespace: "outputs", name: "openapi", revision: "r1" },
      },
    },
    artifacts: [
      material("sources", "shop", "application/zip"),
      material("openapi", "shop-api", "application/yaml"),
    ],
    profiles: [
      {
        ...profileBase,
        ref: { name: "openapi-operation-trace", version: "1", digest: DIGEST },
        inputs: {
          openapi: { required: true, mediaTypes: ["application/yaml"] },
          source: { required: true, mediaTypes: ["application/zip"] },
        },
      },
      {
        ...profileBase,
        ref: {
          name: "owasp-wstg-4-2-active-http",
          version: "1",
          digest: DIGEST,
        },
        inputs: { source: { required: true, mediaTypes: ["application/zip"] } },
        interaction: { ...profileBase.interaction, activeChecks: "automatic" },
      },
      {
        ...profileBase,
        ref: { name: "docs-review", version: "1", digest: DIGEST },
        inputs: { docs: { required: true, mediaTypes: ["text/markdown"] } },
      },
    ],
  };

  it("counts possible issues of every listed check and links each cell", async () => {
    renderAt(overviewServer, "/projects/project_a");
    const glance = await screen.findByRole("region", { name: "At a glance" });
    expect(
      await within(glance).findByRole("link", { name: "3 to review" }),
    ).toHaveAttribute("href", "/issues?state=proposed&project=project_a");
    expect(
      within(glance).getByRole("link", { name: "1 running" }),
    ).toHaveAttribute("href", "/projects/project_a/audits");
    expect(within(glance).getByText("1 waiting for you")).toBeVisible();
    expect(
      await within(glance).findByRole("link", { name: "2 materials" }),
    ).toHaveAttribute("href", "/projects/project_a/artifacts");
    expect(within(glance).getByText("Source code, API spec")).toBeVisible();
    expect(within(glance).queryByText(/newest checks/)).toBeNull();
  });

  it("says when the count covers only the newest checks", async () => {
    renderAt({ ...overviewServer, moreAudits: true }, "/projects/project_a");
    const glance = await screen.findByRole("region", { name: "At a glance" });
    await within(glance).findByRole("link", { name: "3 to review" });
    expect(within(glance).getByText("In the 50 newest checks")).toBeVisible();
  });

  it("starts a check with the objective and offers check types whose formats match", async () => {
    const { router, user } = renderAt(overviewServer, "/projects/project_a");
    const composer = await screen.findByRole("form", {
      name: "What do you want to check?",
    });
    const suggestion = await within(composer).findByRole("link", {
      name: /OpenAPI · Operation trace/,
    });
    expect(suggestion).toHaveTextContent("Format matches");
    expect(suggestion).toHaveAttribute(
      "href",
      "/checks/new?project=project_a&type=openapi-operation-trace",
    );
    // Active testing needs the live target; a docs input is missing.
    expect(within(composer).queryByRole("link", { name: /WSTG/ })).toBeNull();
    expect(
      within(composer).queryByRole("link", { name: /docs review/ }),
    ).toBeNull();
    expect(
      within(composer).getByRole("link", { name: "All check types" }),
    ).toHaveAttribute("href", "/checks/new?project=project_a");
    await user.type(
      screen.getByRole("textbox", { name: "What do you want to check?" }),
      "Can one customer read another's orders?",
    );
    expect(suggestion).toHaveAttribute(
      "href",
      "/checks/new?project=project_a&objective=Can+one+customer+read+another%27s+orders%3F&type=openapi-operation-trace",
    );
    await user.click(screen.getByRole("button", { name: "Start a check" }));
    // The Start page loads lazily before the navigation settles.
    await waitFor(() =>
      expect(router.state.location.pathname).toBe("/checks/new"),
    );
    const params = new URLSearchParams(router.state.location.search);
    expect(params.get("project")).toBe("project_a");
    expect(params.get("objective")).toBe(
      "Can one customer read another's orders?",
    );
  });

  it("lists checks waiting for a decision with a way to review them", async () => {
    renderAt(overviewServer, "/projects/project_a");
    const attention = await screen.findByRole("region", {
      name: "Needs your attention",
    });
    expect(within(attention).getByText("Source checklist")).toBeVisible();
    expect(
      within(attention).getByRole("link", { name: "Review decisions" }),
    ).toHaveAttribute(
      "href",
      "/projects/project_a/audits/audit_wait/reviews?state=pending",
    );
  });

  it("mixes checks, possible issues, Runs and materials in the timeline and filters it in the URL", async () => {
    const { router, user } = renderAt(overviewServer, "/projects/project_a");
    const timeline = await screen.findByRole("region", { name: "Timeline" });
    await within(timeline).findByRole("link", { name: "Order IDOR" });
    expect(
      within(timeline).getByRole("link", { name: "Order IDOR" }),
    ).toHaveAttribute("href", "/issues/audit_running/finding_1");
    expect(
      within(timeline).getByRole("link", { name: "OpenAPI · Operation trace" }),
    ).toHaveAttribute("href", "/projects/project_a/audits/audit_running");
    expect(
      within(timeline).getByRole("link", { name: "sources/shop" }),
    ).toBeVisible();
    // A check's own Runs are told by the check.
    expect(
      within(timeline).getAllByRole("link", { name: "openapi-from-source@1" }),
    ).toHaveLength(2);
    // A succeeded Run opens its published primary result, and only that.
    expect(
      await within(timeline).findByRole("link", { name: "openapi" }),
    ).toHaveAttribute(
      "href",
      "/runs/run_ok/artifacts/outputs/openapi?revision=r1",
    );
    expect(within(timeline).getAllByText(/Primary result/)).toHaveLength(1);
    expect(within(timeline).getByText("Project created")).toBeVisible();
    await user.click(
      within(timeline).getByRole("button", { name: "Possible issues" }),
    );
    expect(router.state.location.search).toBe("?timeline=issues");
    expect(within(timeline).getAllByText("Possible issue found")).toHaveLength(
      3,
    );
    expect(
      within(timeline).queryByRole("link", {
        name: "OpenAPI · Operation trace",
      }),
    ).toBeNull();
    expect(within(timeline).queryByText("Project created")).toBeNull();
  });

  it("shows the primary result of recent Runs and never guesses a missing one", async () => {
    renderAt(overviewServer, "/projects/project_a");
    const results = await screen.findByRole("region", {
      name: "Recent results",
    });
    expect(
      await within(results).findByRole("link", {
        name: /openapi.*Primary result/,
      }),
    ).toHaveAttribute(
      "href",
      "/runs/run_ok/artifacts/outputs/openapi?revision=r1",
    );
    expect(
      await within(results).findByText(/was not published/),
    ).toHaveTextContent("Primary result openapi was not published.");
    const runs = screen.getByRole("region", { name: "Recent Runs" });
    expect(within(runs).getAllByText("Succeeded")).toHaveLength(2);
    expect(
      within(runs).getByRole("link", { name: "All Runs" }),
    ).toHaveAttribute("href", "/projects/project_a/runs");
  });

  it("offers the materials and the live target as chips", async () => {
    renderAt(overviewServer, "/projects/project_a");
    const chips = await screen.findByRole("list", { name: "Materials" });
    expect(
      await within(chips).findByRole("link", { name: /Source code/ }),
    ).toHaveAttribute("href", "/projects/project_a/artifacts/sources/shop");
    expect(
      within(chips).getByRole("link", { name: /API spec/ }),
    ).toHaveAttribute("href", "/projects/project_a/artifacts/openapi/shop-api");
    expect(
      within(chips).getByRole("link", { name: "Add a live target" }),
    ).toHaveAttribute("href", "/projects/project_a/settings#live-target");
    expect(
      within(chips).getByRole("link", { name: "Add material" }),
    ).toHaveAttribute("href", "/projects/project_a/artifacts?add=artifact");
  });
  it("keeps the possible-issue count while a running check's revision advances", async () => {
    const server: FakeServer = {
      ...overviewServer,
      audits: { project_a: [...overviewServer.audits!.project_a!] },
      findings: { ...overviewServer.findings },
    };
    const { requests, user } = renderAt(server, "/projects/project_a");
    const glance = await screen.findByRole("region", { name: "At a glance" });
    await within(glance).findByRole("link", { name: "3 to review" });
    // The running check moves on: a new revision with one more possible
    // issue, whose read takes a while.
    server.audits!.project_a![0] = {
      ...server.audits!.project_a![0]!,
      revision: 2,
    };
    server.findings!.audit_running = [
      finding("audit_running", "finding_1", "Order IDOR"),
      finding("audit_running", "finding_9", "Price tampering"),
    ];
    let release = () => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    server.holdFindings = (auditId) =>
      auditId === "audit_running" ? held : undefined;
    const issueReads = () =>
      reads(requests, "/v1/audits/audit_running/findings");
    const before = issueReads();
    await user.click(screen.getByRole("button", { name: "Refresh" }));
    await waitFor(() => expect(issueReads()).toBe(before + 1));
    // The previous revision stays counted while the new one loads.
    expect(
      within(glance).getByRole("link", { name: "3 to review" }),
    ).toBeVisible();
    expect(within(glance).queryByText("Counting…")).toBeNull();
    act(() => release());
    expect(
      await within(glance).findByRole("link", { name: "4 to review" }),
    ).toBeVisible();
  });

  it("counts only running checks as running and names stopping and finishing ones", async () => {
    renderAt(
      {
        projects: [projectA],
        audits: {
          project_a: [
            check("project_a", "audit_running", "active"),
            check("project_a", "audit_stopping", "cancelling"),
            check("project_a", "audit_finishing", "finalizing"),
          ],
        },
      },
      "/projects/project_a",
    );
    const glance = await screen.findByRole("region", { name: "At a glance" });
    expect(
      await within(glance).findByRole("link", { name: "1 running" }),
    ).toBeVisible();
    expect(within(glance).getByText("1 finishing · 1 stopping")).toBeVisible();
  });

  it("shows a check paused on the Checks tab when the overview opens again", async () => {
    const checks = [...overviewServer.audits!.project_a!];
    const server: FakeServer = {
      ...overviewServer,
      audits: { project_a: checks },
      mutateAudit: (auditId, action) => {
        const index = checks.findIndex((audit) => audit.auditId === auditId);
        const audit = checks[index];
        if (audit === undefined || action !== "pause")
          throw new Error(`unexpected ${action} of ${auditId}`);
        const paused: Audit = {
          ...audit,
          state: "paused",
          revision: audit.revision + 1,
        };
        checks[index] = paused;
        return json(paused, { headers: { ETag: `"${paused.revision}"` } });
      },
    };
    const { user } = renderAt(server, "/projects/project_a");
    const glance = await screen.findByRole("region", { name: "At a glance" });
    expect(
      await within(glance).findByRole("link", { name: "1 running" }),
    ).toBeVisible();
    const sections = screen.getByRole("navigation", {
      name: "Project sections",
    });
    await user.click(within(sections).getByRole("link", { name: "Checks" }));
    // The section loads lazily, and the overview's timeline links the same
    // check by the same name: look for its row once the overview has left.
    await waitFor(() =>
      expect(screen.queryByRole("region", { name: "At a glance" })).toBeNull(),
    );
    const row = (
      await screen.findByRole("link", { name: "OpenAPI · Operation trace" })
    ).closest("li")!;
    await user.click(
      within(row).getByRole("button", { name: "Pause new work" }),
    );
    expect(await within(row).findByText("Paused")).toBeVisible();
    // Back within the overview's polling interval: the pause refreshed the
    // check page the overview reads, so it does not show the old state.
    await user.click(within(sections).getByRole("link", { name: "Overview" }));
    const after = await screen.findByRole("region", { name: "At a glance" });
    expect(await within(after).findByText("None running")).toBeVisible();
    expect(within(after).queryByRole("link", { name: "1 running" })).toBeNull();
  });

  it("stops listing a check deleted elsewhere once its page finds it gone", async () => {
    const checks = [...overviewServer.audits!.project_a!];
    const { router, user } = renderAt(
      { ...overviewServer, audits: { project_a: checks } },
      "/projects/project_a",
    );
    const glance = await screen.findByRole("region", { name: "At a glance" });
    expect(
      await within(glance).findByRole("link", { name: "1 running" }),
    ).toBeVisible();
    const timeline = screen.getByRole("region", { name: "Timeline" });
    const link = await within(timeline).findByRole("link", {
      name: "OpenAPI · Operation trace",
    });
    // The running check is deleted in another tab; the overview still lists
    // it and links to it.
    checks.splice(
      checks.findIndex((audit) => audit.auditId === "audit_running"),
      1,
    );
    await user.click(link);
    // The check page answers 404 and returns to the project's checks.
    await waitFor(() =>
      expect(router.state.location.pathname).toBe("/projects/project_a/audits"),
    );
    const sections = await screen.findByRole("navigation", {
      name: "Project sections",
    });
    // Back within the overview's polling interval: the 404 refreshed the
    // check page the overview reads, so it neither counts nor links the
    // deleted check.
    await user.click(within(sections).getByRole("link", { name: "Overview" }));
    const after = await screen.findByRole("region", { name: "At a glance" });
    expect(await within(after).findByText("None running")).toBeVisible();
    expect(within(after).queryByRole("link", { name: "1 running" })).toBeNull();
    // Its possible issue ("Order IDOR") is no longer counted either.
    expect(
      await within(after).findByRole("link", { name: "2 to review" }),
    ).toBeVisible();
    expect(
      within(screen.getByRole("region", { name: "Timeline" })).queryByRole(
        "link",
        { name: "OpenAPI · Operation trace" },
      ),
    ).toBeNull();
  });
});

describe("Project settings", () => {
  it("keeps internals behind Technical details and offers deletion in a danger zone", async () => {
    const { user } = renderAt(
      { projects: [projectA] },
      "/projects/project_a/settings",
    );
    await screen.findByRole("region", { name: "Project details" });
    expect(screen.getByRole("region", { name: "Live target" })).toBeVisible();
    await user.click(screen.getByText("Technical details"));
    expect(
      screen.getByRole("link", { name: "Runtime configuration" }),
    ).toHaveAttribute("href", "/operations/configuration");
    expect(screen.getByText("Revision")).toBeVisible();
    await user.click(
      screen.getByRole("button", { name: "Delete this project" }),
    );
    const dialog = screen.getByRole("alertdialog", {
      name: "Delete Project A?",
    });
    expect(
      within(dialog).getByRole("button", { name: "Delete Project" }),
    ).toBeDisabled();
  });

  it("saves project details without waiting for every cross-project list", async () => {
    let saved = false;
    const server: FakeServer = {
      projects: [projectA, projectB],
      // After the write, no project's checks answer (a slow Server).
      holdAudits: () => saved,
      updateProject: async (request) => {
        saved = true;
        const body = (await request.json()) as {
          name: string;
          description: string;
        };
        const updated = { ...projectA, ...body, revision: "2" };
        server.projects = [updated, projectB];
        return json(updated, { headers: { ETag: '"2"' } });
      },
    };
    const { user } = renderAt(server, "/projects/project_a/settings");
    await screen.findByRole("region", { name: "Project details" });
    await user.click(screen.getByRole("button", { name: "Edit metadata" }));
    await user.clear(screen.getByLabelText("Name"));
    await user.type(screen.getByLabelText("Name"), "Shop APIs");
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    expect(
      await screen.findByRole("button", { name: "Edit metadata" }),
    ).toBeVisible();
    expect(
      screen.getByRole("heading", { name: "Shop APIs", level: 2 }),
    ).toBeVisible();
  });

  it("shows the runtime configuration link to operators only", async () => {
    const { user } = renderAt(
      { projects: [projectA], capabilities: ["user"] },
      "/projects/project_a/settings",
    );
    await screen.findByRole("region", { name: "Project details" });
    await user.click(screen.getByText("Technical details"));
    expect(
      screen.queryByRole("link", { name: "Runtime configuration" }),
    ).toBeNull();
  });
  it("opens the live target sheet from the overview's Add chip", async () => {
    const { router, user } = renderAt(
      { projects: [projectA] },
      "/projects/project_a",
    );
    const chips = await screen.findByRole("list", { name: "Materials" });
    await user.click(
      within(chips).getByRole("link", { name: "Add a live target" }),
    );
    const sheet = await screen.findByRole("dialog", {
      name: "Application access",
    });
    expect(router.state.location.pathname).toBe("/projects/project_a/settings");
    expect(router.state.location.hash).toBe("#live-target");
    // Back and reload show Settings without the sheet.
    await waitFor(() =>
      expect(
        (router.state.location.state as Record<string, unknown> | null)
          ?.openTargetSheet,
      ).toBeUndefined(),
    );
    expect(sheet).toBeVisible();
    await user.click(
      within(sheet).getByRole("button", { name: "Close target dialog" }),
    );
    expect(screen.queryByRole("dialog")).toBeNull();
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Configure target" }),
      ).toHaveFocus(),
    );
  });

  it("lands on the Live target for settings#live-target", async () => {
    renderAt(
      {
        projects: [
          { ...projectA, httpTarget: { url: "https://shop.example.test" } },
        ],
      },
      "/projects/project_a/settings#live-target",
    );
    const edit = await screen.findByRole("button", { name: "Edit target" });
    await waitFor(() => expect(edit).toHaveFocus());
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});

describe("Project Runs", () => {
  it("keeps view, state and cursor in the URL and links a check's Runs to the check", async () => {
    const { requests } = renderAt(
      {
        projects: [projectA],
        runs: [run("run_item", "succeeded", { "audit.id": "audit_running" })],
      },
      "/projects/project_a/runs?view=completed&state=succeeded",
    );
    const origin = await screen.findByRole("link", {
      name: "Check · _running",
    });
    expect(origin).toHaveAttribute(
      "href",
      "/projects/project_a/audits/audit_running",
    );
    expect(screen.getByRole("button", { name: "Completed" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByRole("combobox", { name: "State" })).toHaveValue(
      "succeeded",
    );
    const read = requests
      .map((request) => new URL(request.url))
      .find((url) => url.pathname === "/v1/projects/project_a/runs");
    expect(read?.searchParams.get("lifecycle")).toBe("terminal");
    expect(read?.searchParams.get("state")).toBe("succeeded");
    expect(read?.searchParams.get("limit")).toBe("25");
    expect(screen.getByText("Identity & labels (1)")).toBeVisible();
  });
});

describe("Evaluation workspaces", () => {
  const evaluation = projectFixture("evaluation_a", {
    kind: "evaluation",
    name: "Regression eval",
  });

  it("keep their list, sections and wording", async () => {
    const { user } = renderAt({ projects: [evaluation] }, "/evals/legacy");
    expect(await screen.findByRole("heading", { name: "Evals" })).toBeVisible();
    expect(screen.getByRole("button", { name: "New Eval" })).toBeVisible();
    await user.click(
      await screen.findByRole("link", { name: "Regression eval" }),
    );
    expect(
      await screen.findByRole("heading", { name: "Eval Runs" }),
    ).toBeVisible();
    const sections = screen.getByRole("navigation", {
      name: "Project sections",
    });
    expect(
      within(sections)
        .getAllByRole("link")
        .map((link) => link.textContent),
    ).toEqual(["Runs", "Artifacts", "Workspace settings"]);
    expect(screen.getByLabelText("Eval actions")).toBeVisible();
    expect(screen.getByRole("heading", { name: "Artifacts" })).toBeVisible();
  });
});
