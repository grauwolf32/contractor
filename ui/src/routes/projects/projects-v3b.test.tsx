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
      const query = url.searchParams;
      if (path === "/v1/auth/session")
        return json(session(server.capabilities));
      if (path === "/v1/projects") return json(page(server.projects));
      let match = /^\/v1\/projects\/([^/]+)$/.exec(path);
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
        const state = query.get("state");
        const items = (server.audits?.[match[1]!] ?? []).filter(
          (audit) => state === null || audit.state === state,
        );
        return json(page(items, server.moreAudits === true && state === null));
      }
      match = /^\/v1\/audits\/([^/]+)\/(findings|reviews)$/.exec(path);
      if (match !== null) {
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

function renderAt(server: FakeServer, path: string) {
  const { api, requests } = serve(server);
  const router = createMemoryRouter(applicationRoutes(), {
    initialEntries: [path],
  });
  render(<Application api={api} publicAPI={api} router={router} />);
  return { router, requests, user: userEvent.setup() };
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
    renderAt(
      {
        projects: [projectA, projectB, projectC, deleting, quiet],
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
        },
        findings: {
          audit_running: [
            finding("audit_running", "finding_1", "First"),
            finding("audit_running", "finding_2", "Second"),
          ],
          // A finished check's possible issues are counted on its project's
          // overview, not in the list (the list shares the Inbox's reads).
          audit_done: [finding("audit_done", "finding_3", "Third")],
          audit_waiting: [finding("audit_waiting", "finding_4", "Fourth")],
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
        "Check running·2 possible issues to review",
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
    expect(await row("Project D")).toHaveTextContent(/Deleting·requested/);
    expect(await row("Project E")).toHaveTextContent(/Updated/);
    expect(screen.getByText("5 projects, newest first")).toBeVisible();
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
