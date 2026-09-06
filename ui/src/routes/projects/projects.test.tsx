import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";

import type { RuntimeConfig } from "../../config/runtime-config";
import { PublicAPI } from "../../api/client";
import { Application } from "../../app/application";
import { applicationRoutes } from "../../app/router";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

const session = {
  principal: {
    userId: "user_local",
    username: "owner",
    capabilities: ["user", "operations"],
  },
  csrfToken: "a".repeat(43),
  idleExpiresAt: "2026-09-01T20:00:00Z",
  absoluteExpiresAt: "2026-09-02T12:00:00Z",
};

const project = {
  projectId: "project_example",
  kind: "project",
  name: "Payment service",
  description: "Reusable service analysis",
  lifecycle: "active",
  revision: "1",
  createdAt: "2026-09-01T10:00:00Z",
  updatedAt: "2026-09-01T10:00:00Z",
};

function jsonResponse(value: unknown, options: ResponseInit = {}): Response {
  const headers = new Headers(options.headers);
  headers.set("content-type", "application/json");
  headers.set("X-Contractor-API-Version", "contractor.public.v1");
  return new Response(JSON.stringify(value), { ...options, headers });
}

function bytesResponse(body: BodyInit, options: ResponseInit = {}): Response {
  const headers = new Headers(options.headers);
  headers.set("X-Contractor-API-Version", "contractor.public.v1");
  return new Response(body, { ...options, headers });
}

function renderProjectApplication(api: PublicAPI, path: string) {
  const router = createMemoryRouter(applicationRoutes(), {
    initialEntries: [path],
  });
  return {
    ...render(<Application api={api} publicAPI={api} router={router} />),
    router,
  };
}

describe("Project routes", () => {
  it.each(["metadata", "target"] as const)(
    "retains the unsaved %s draft and its exact revision after a Project refresh",
    async (operation) => {
      let currentProject = { ...project };
      let projectReads = 0;
      const writes: Request[] = [];
      const api = new PublicAPI(
        runtimeConfig,
        vi.fn(async (input) => {
          const request = input instanceof Request ? input : new Request(input);
          const url = new URL(request.url);
          if (url.pathname === "/v1/auth/session") return jsonResponse(session);
          if (url.pathname === "/v1/projects/project_example") {
            if (request.method === "PATCH") {
              writes.push(request.clone());
              return jsonResponse(
                {
                  code: "precondition_failed",
                  message: "resource revision precondition failed",
                  retryable: false,
                  requestId: "request-stale-project",
                },
                { status: 412 },
              );
            }
            projectReads += 1;
            return jsonResponse(currentProject, {
              headers: { ETag: `"${currentProject.revision}"` },
            });
          }
          return jsonResponse({ items: [], page: { hasMore: false } });
        }),
      );
      renderProjectApplication(api, "/projects/project_example");
      const user = userEvent.setup();
      await screen.findByRole("heading", { name: "Payment service" });
      await user.click(
        screen.getByRole("button", {
          name: operation === "metadata" ? "Edit metadata" : "Configure target",
        }),
      );
      const label = operation === "metadata" ? "Name" : "Application URL";
      const draft =
        operation === "metadata"
          ? "Unsaved name"
          : "https://draft.example.test/api";
      await user.clear(screen.getByLabelText(label));
      await user.type(screen.getByLabelText(label), draft);
      expect(projectReads).toBe(1);

      currentProject = { ...project, name: "Updated elsewhere", revision: "2" };
      await user.click(
        within(
          screen
            .getByRole("heading", { name: "Payment service" })
            .closest("header")!,
        ).getByRole("button", { name: "Refresh" }),
      );
      await screen.findByRole("heading", { name: "Updated elsewhere" });
      expect(projectReads).toBe(2);
      expect(screen.getByLabelText(label)).toHaveValue(draft);
      await user.click(
        screen.getByRole("button", {
          name: operation === "metadata" ? "Save exact update" : "Save target",
        }),
      );
      await screen.findByText("resource revision precondition failed");
      expect(screen.getByLabelText(label)).toHaveValue(draft);
      expect(writes).toHaveLength(1);
      expect(writes[0]?.headers.get("If-Match")).toBe('"1"');
      expect(writes[0]?.headers.get("X-CSRF-Token")).toBe(session.csrfToken);
      await expect(writes[0]?.json()).resolves.toEqual(
        operation === "metadata"
          ? { name: draft, description: project.description }
          : { httpTarget: { url: draft } },
      );
    },
  );

  it("confirms the exact name, shows durable progress, and reconciles deletion to 404", async () => {
    const deleteRequests: Request[] = [];
    let deleting = false;
    let complete = false;
    const deletingProject = {
      ...project,
      lifecycle: "deleting",
      deletion: {
        phase: "draining",
        requestedAt: "2026-09-05T10:00:00Z",
      },
      revision: "2",
      updatedAt: "2026-09-05T10:00:00Z",
    };
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") {
          return jsonResponse(session);
        }
        if (url.pathname === "/v1/projects/project_example") {
          if (request.method === "DELETE") {
            deleteRequests.push(request.clone());
            deleting = true;
            return jsonResponse(deletingProject, {
              status: 202,
              headers: { ETag: '"2"' },
            });
          }
          if (complete) {
            return jsonResponse(
              {
                code: "not_found",
                message: "resource was not found",
                retryable: false,
                requestId: "request-project-deleted",
              },
              { status: 404 },
            );
          }
          const current = deleting ? deletingProject : project;
          return jsonResponse(current, {
            headers: { ETag: deleting ? '"2"' : '"1"' },
          });
        }
        if (url.pathname === "/v1/projects") {
          return jsonResponse({ items: [], page: { hasMore: false } });
        }
        if (url.pathname.endsWith("/artifacts")) {
          return jsonResponse({ items: [], page: { hasMore: false } });
        }
        if (url.pathname === "/v1/workflows") {
          return jsonResponse({ items: [], page: { hasMore: false } });
        }
        if (url.pathname.endsWith("/runs")) {
          return jsonResponse({ items: [], page: { hasMore: false } });
        }
        throw new Error(`unexpected ${request.method} ${url.pathname}`);
      }),
    );
    const { router } = renderProjectApplication(
      api,
      "/projects/project_example",
    );
    const user = userEvent.setup();

    await screen.findByRole("heading", { name: "Payment service" });
    await user.click(screen.getByRole("button", { name: "Delete Project" }));
    const dialog = screen.getByRole("alertdialog", {
      name: "Delete Payment service?",
    });
    expect(
      within(dialog).getByRole("button", { name: "Cancel" }),
    ).toHaveFocus();
    const confirm = within(dialog).getByRole("button", {
      name: "Delete Project",
    });
    expect(confirm).toBeDisabled();
    await user.type(
      within(dialog).getByLabelText("Type Payment service to confirm"),
      "Payment servic",
    );
    expect(confirm).toBeDisabled();
    await user.type(
      within(dialog).getByLabelText("Type Payment service to confirm"),
      "e",
    );
    await user.click(confirm);

    expect(
      await screen.findByRole("heading", {
        name: "Waiting for Runtime release",
      }),
    ).toBeVisible();
    expect(
      screen.queryByRole("button", { name: "Sources" }),
    ).not.toBeInTheDocument();
    expect(deleteRequests).toHaveLength(1);
    expect(deleteRequests[0]?.headers.get("If-Match")).toBe('"1"');
    expect(deleteRequests[0]?.headers.get("X-CSRF-Token")).toBe(
      session.csrfToken,
    );
    await expect(deleteRequests[0]?.text()).resolves.toBe("");

    complete = true;
    await user.click(screen.getByRole("button", { name: "Refresh" }));
    await vi.waitFor(() =>
      expect(router.state.location.pathname).toBe("/projects"),
    );
  });

  it("creates an owner-scoped Project and opens its dashboard", async () => {
    const requests: Request[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request.clone());
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") {
          return jsonResponse(session);
        }
        if (url.pathname === "/v1/projects" && request.method === "GET") {
          return jsonResponse({ items: [], page: { hasMore: false } });
        }
        if (url.pathname === "/v1/projects" && request.method === "POST") {
          return jsonResponse(project, {
            status: 201,
            headers: { ETag: '"1"' },
          });
        }
        if (url.pathname === "/v1/projects/project_example") {
          return jsonResponse(project, { headers: { ETag: '"1"' } });
        }
        if (url.pathname.endsWith("/artifacts")) {
          return jsonResponse({ items: [], page: { hasMore: false } });
        }
        if (url.pathname === "/v1/workflows") {
          return jsonResponse({ items: [], page: { hasMore: false } });
        }
        if (url.pathname.endsWith("/runs")) {
          return jsonResponse({ items: [], page: { hasMore: false } });
        }
        throw new Error(`unexpected ${request.method} ${url.pathname}`);
      }),
    );
    const { router } = renderProjectApplication(api, "/projects");
    const user = userEvent.setup();

    await screen.findByRole("heading", { name: "Projects" });
    await user.click(
      screen.getAllByRole("button", { name: "New Project" })[0]!,
    );
    await user.type(screen.getByLabelText("Name"), "Payment service");
    await user.type(
      screen.getByLabelText("Description"),
      "Reusable service analysis",
    );
    await user.click(screen.getByRole("button", { name: "Create Project" }));

    await vi.waitFor(() =>
      expect(router.state.location.pathname).toBe("/projects/project_example"),
    );
    expect(
      await screen.findByRole("heading", { name: "Payment service" }),
    ).toBeInTheDocument();
    const create = requests.find((request) => request.method === "POST")!;
    expect(create.headers.get("Idempotency-Key")).toMatch(
      /^create-project-ui-/,
    );
    expect(create.headers.get("X-CSRF-Token")).toBe(session.csrfToken);
    const body = await create.json();
    expect(body).toEqual({
      kind: "project",
      name: "Payment service",
      description: "Reusable service analysis",
    });
    expect(body).not.toHaveProperty("owner_id");
    expect(body).not.toHaveProperty("scope_kind");
  });

  it("drops a ZIP on Sources and shows the authoritative exact binding", async () => {
    const requests: Request[] = [];
    let uploaded = false;
    const artifact = {
      artifact: {
        namespace: "sources",
        name: "payment-service",
        revision: "revision-1",
      },
      mediaType: "application/zip",
      size: 3,
      current: true,
      frozen: false,
      createdAt: "2026-09-01T10:10:00Z",
    };
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") {
          return jsonResponse(session);
        }
        if (url.pathname === "/v1/projects/project_example") {
          return jsonResponse(project, { headers: { ETag: '"1"' } });
        }
        if (
          url.pathname === "/v1/projects/project_example/artifacts" &&
          request.method === "GET"
        ) {
          return jsonResponse({
            items: uploaded ? [artifact] : [],
            page: { hasMore: false },
          });
        }
        if (
          url.pathname ===
            "/v1/projects/project_example/artifacts/sources/payment-service" &&
          request.method === "PUT"
        ) {
          uploaded = true;
          return jsonResponse(
            {
              artifact: artifact.artifact,
              mediaType: artifact.mediaType,
              size: artifact.size,
            },
            { status: 201, headers: { ETag: '"revision-1"' } },
          );
        }
        if (url.pathname === "/v1/workflows") {
          return jsonResponse({ items: [], page: { hasMore: false } });
        }
        if (url.pathname.endsWith("/runs")) {
          return jsonResponse({ items: [], page: { hasMore: false } });
        }
        throw new Error(`unexpected ${request.method} ${url.pathname}`);
      }),
    );
    renderProjectApplication(api, "/projects/project_example");
    const user = userEvent.setup();

    await screen.findByRole("heading", { name: "Payment service" });
    await user.click(screen.getByRole("button", { name: /Sources/ }));
    const dialog = screen.getByRole("dialog", { name: "Sources" });
    const file = new File(["zip"], "payment-service.zip", {
      type: "application/zip",
    });
    await user.upload(within(dialog).getByLabelText("Drop a file here"), file);
    expect(within(dialog).getByLabelText("Namespace")).toHaveValue("sources");
    expect(within(dialog).getByLabelText("Name")).toHaveValue(
      "payment-service",
    );
    await user.click(
      within(dialog).getByRole("button", { name: "Create binding" }),
    );

    expect(
      await screen.findByRole("link", { name: "sources/payment-service" }),
    ).toBeInTheDocument();
    const put = requests.find((request) => request.method === "PUT")!;
    expect(put.headers.get("If-None-Match")).toBe("*");
    expect(put.headers.get("Content-Type")).toBe("application/zip");
    expect(await put.text()).toBe("zip");
    expect(put.url).not.toContain("owner_id");
    expect(put.url).not.toContain("scope_kind");
  });

  it("creates write-only origin auth and attaches only safe Project metadata", async () => {
    const requests: Request[] = [];
    const secret = "project-origin-secret-never-persist";
    let currentProject: Record<string, unknown> = { ...project };
    let credential: Record<string, unknown> | undefined;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request.clone());
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") return jsonResponse(session);
        if (url.pathname === "/v1/projects/project_example") {
          if (request.method === "PATCH") {
            const body = (await request.json()) as Record<string, unknown>;
            currentProject = { ...currentProject, ...body, revision: "2" };
            return jsonResponse(currentProject, { headers: { ETag: '"2"' } });
          }
          return jsonResponse(currentProject, { headers: { ETag: '"1"' } });
        }
        if (url.pathname === "/v1/operations/runtime-credentials") {
          if (request.method === "POST") {
            const body = (await request.json()) as Record<string, unknown>;
            credential = {
              credentialId: body.credentialId,
              kind: body.kind,
              createdBy: "user-1",
              createdAt: "2026-09-01T10:01:00Z",
            };
            return jsonResponse(credential, { status: 201 });
          }
          return jsonResponse({
            items: credential === undefined ? [] : [credential],
            page: { hasMore: false },
          });
        }
        if (url.pathname.endsWith("/artifacts")) {
          return jsonResponse({ items: [], page: { hasMore: false } });
        }
        if (url.pathname === "/v1/workflows") {
          return jsonResponse({ items: [], page: { hasMore: false } });
        }
        if (url.pathname.endsWith("/runs")) {
          return jsonResponse({ items: [], page: { hasMore: false } });
        }
        throw new Error(`unexpected ${request.method} ${url.pathname}`);
      }),
    );
    renderProjectApplication(api, "/projects/project_example");
    const user = userEvent.setup();

    await screen.findByRole("heading", { name: "Payment service" });
    await user.click(screen.getByRole("button", { name: "Configure target" }));
    const dialog = screen.getByRole("dialog", { name: "Application access" });
    await user.type(
      within(dialog).getByLabelText("Application URL"),
      "https://app.example.test/api",
    );
    await user.selectOptions(
      within(dialog).getByLabelText("Authorization"),
      "bearer",
    );
    const token = within(dialog).getByLabelText("Bearer token · write only");
    expect(token).toHaveAttribute("type", "text");
    await user.type(token, secret);
    await user.click(
      within(dialog).getByRole("button", { name: "Save target" }),
    );

    await screen.findByText(/http-origin-bearer@1/);
    const create = requests.find(
      (request) =>
        request.method === "POST" &&
        request.url.includes("runtime-credentials"),
    )!;
    const createdBody = await create.clone().json();
    expect(createdBody.material.token).toBe(secret);
    const patch = requests.find((request) => request.method === "PATCH")!;
    const patchBody = await patch.clone().json();
    expect(JSON.stringify(patchBody)).not.toContain(secret);
    expect(patchBody.httpTarget.credential).toEqual({
      credentialId: createdBody.credentialId,
      kind: "http-origin-bearer@1",
    });
    expect(document.body.textContent).not.toContain(secret);
  });

  it("contains a failed artifact request without crashing Project overview", async () => {
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") {
          return jsonResponse(session);
        }
        if (url.pathname === "/v1/projects/project_example") {
          return jsonResponse(project, { headers: { ETag: '"1"' } });
        }
        if (url.pathname.endsWith("/artifacts")) {
          return jsonResponse(
            {
              code: "artifact_store_unavailable",
              message: "Artifact storage is temporarily unavailable",
              retryable: true,
              requestId: "request-project-artifacts",
            },
            { status: 500 },
          );
        }
        if (url.pathname === "/v1/workflows") {
          return jsonResponse({ items: [], page: { hasMore: false } });
        }
        if (url.pathname.endsWith("/runs")) {
          return jsonResponse({ items: [], page: { hasMore: false } });
        }
        throw new Error(`unexpected ${request.method} ${url.pathname}`);
      }),
    );
    renderProjectApplication(api, "/projects/project_example");

    expect(
      await screen.findByText("Artifact storage is temporarily unavailable"),
    ).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Overview" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Sources" })).toBeEnabled();
  });

  it("opens exact Project Artifact history and bounded preview", async () => {
    const content = "Project documentation\n";
    const artifact = {
      artifact: {
        namespace: "docs",
        name: "readme",
        revision: "revision-2",
      },
      mediaType: "text/plain",
      size: content.length,
      current: true,
      frozen: false,
      createdAt: "2026-09-01T10:10:00Z",
    };
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") {
          return jsonResponse(session);
        }
        if (url.pathname.endsWith("/metadata")) {
          return jsonResponse(artifact);
        }
        if (url.pathname.endsWith("/versions")) {
          return jsonResponse({ items: [artifact], page: { hasMore: false } });
        }
        if (url.pathname.endsWith("/lineage")) {
          return jsonResponse({
            items: [
              {
                kind: "project_output_publish",
                sourceScope: "run",
                source: {
                  namespace: "outputs",
                  name: "docs",
                  revision: "run-revision-1",
                },
                targetScope: "project",
                target: artifact.artifact,
                runId: "run_example",
                createdAt: "2026-09-01T10:10:00Z",
              },
            ],
            page: { hasMore: false },
          });
        }
        if (
          url.pathname === "/v1/projects/project_example/artifacts/docs/readme"
        ) {
          return bytesResponse(content, {
            headers: {
              "content-type": "text/plain",
              "content-length": String(content.length),
            },
          });
        }
        throw new Error(`unexpected ${request.method} ${url.pathname}`);
      }),
    );
    renderProjectApplication(
      api,
      "/projects/project_example/artifacts/docs/readme?revision=revision-2",
    );

    expect(
      (await screen.findAllByText("revision-2", { selector: "code" })).length,
    ).toBeGreaterThan(0);
    expect(await screen.findByText("project output publish")).toBeVisible();
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Load preview" }));
    expect(await screen.findByText("Project documentation")).toBeVisible();
    expect(
      screen.getByRole("button", { name: "Upload exact update" }),
    ).toBeEnabled();
  });

  it("recommends compatible Workflows and launches through the Project endpoint", async () => {
    const requests: Request[] = [];
    const source = {
      artifact: {
        namespace: "sources",
        name: "payment-service",
        revision: "revision-source-1",
      },
      mediaType: "application/zip",
      size: 3,
      current: true,
      frozen: false,
      createdAt: "2026-09-01T10:10:00Z",
    };
    const workflowSummaries = [
      {
        ref: { name: "openapi-from-source", version: "1" },
        entryStage: "analyze",
        parameters: {},
        inputs: {
          source: { required: true, mediaTypes: ["application/zip"] },
        },
        outputs: {
          openapi: {
            required: true,
            mediaTypes: ["application/yaml"],
            primary: true,
          },
        },
      },
      {
        ref: { name: "likec4-from-source", version: "1" },
        entryStage: "analyze",
        parameters: {},
        inputs: {
          source: { required: true, mediaTypes: ["application/zip"] },
        },
        outputs: {
          likec4: {
            required: true,
            mediaTypes: ["text/vnd.likec4"],
            primary: true,
          },
        },
      },
    ];
    const runtimeConfiguration = {
      default: {
        label: "default",
        bindingRevision: "1",
        config: {
          name: "contractor-empty",
          version: "1",
          digest: `sha256:${"0".repeat(64)}`,
        },
      },
      labels: [],
    };
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") {
          return jsonResponse(session);
        }
        if (url.pathname === "/v1/projects/project_example") {
          return jsonResponse(project, { headers: { ETag: '"1"' } });
        }
        if (url.pathname === "/v1/projects/project_example/artifacts") {
          return jsonResponse({ items: [source], page: { hasMore: false } });
        }
        if (
          url.pathname === "/v1/projects/project_example/runs" &&
          request.method === "GET"
        ) {
          return jsonResponse({ items: [], page: { hasMore: false } });
        }
        if (
          url.pathname === "/v1/projects/project_example/runs" &&
          request.method === "POST"
        ) {
          return jsonResponse(
            {
              runId: "run_project_recommended",
              projectId: "project_example",
              state: "initializing",
              runtimeLabels: [],
              labels: {},
              runtimeConfiguration,
            },
            { status: 202 },
          );
        }
        if (url.pathname === "/v1/workflows") {
          return jsonResponse({
            items: workflowSummaries,
            page: { hasMore: false },
          });
        }
        if (url.pathname === "/v1/workflows/openapi-from-source/versions/1") {
          return jsonResponse({
            ...workflowSummaries[0],
            stages: {},
          });
        }
        if (url.pathname === "/v1/runs/run_project_recommended") {
          return jsonResponse({
            runId: "run_project_recommended",
            projectId: "project_example",
            workflow: "openapi-from-source@1",
            state: "initializing",
            runtimeLabels: [],
            labels: {},
            runtimeConfiguration,
            attempts: [],
            transitions: [],
            outputs: {},
            outputPublications: [],
          });
        }
        throw new Error(`unexpected ${request.method} ${url.pathname}`);
      }),
    );
    const { router } = renderProjectApplication(
      api,
      "/projects/project_example",
    );
    const user = userEvent.setup();

    expect(
      await screen.findByRole("button", {
        name: "Run openapi-from-source@1",
      }),
    ).toBeEnabled();
    expect(
      screen.getByRole("button", { name: "Run likec4-from-source@1" }),
    ).toBeEnabled();
    await user.click(
      screen.getByRole("button", { name: "Run openapi-from-source@1" }),
    );
    const dialog = await screen.findByRole("dialog", {
      name: "openapi-from-source@1",
    });
    expect(
      within(dialog).getByRole("combobox", { name: /source required/ }),
    ).toHaveValue("sources/payment-service@revision-source-1");
    await user.click(
      within(dialog).getByRole("button", {
        name: "Start Project Workflow Run",
      }),
    );

    await vi.waitFor(() =>
      expect(router.state.location.pathname).toBe(
        "/runs/run_project_recommended",
      ),
    );
    const create = requests.find(
      (request) =>
        request.method === "POST" &&
        new URL(request.url).pathname === "/v1/projects/project_example/runs",
    )!;
    expect(create.headers.get("Idempotency-Key")).toMatch(/^run-ui-/);
    expect(await create.json()).toMatchObject({
      workflow: "openapi-from-source@1",
      artifacts: {
        source: source.artifact,
      },
    });
  });

  it("suppresses an existing primary result only from Recommended", async () => {
    const source = {
      artifact: {
        namespace: "sources",
        name: "payment-service",
        revision: "revision-source-1",
      },
      mediaType: "application/zip",
      size: 3,
      current: true,
      frozen: false,
      createdAt: "2026-09-01T10:10:00Z",
    };
    const existingOutput = {
      artifact: {
        namespace: "outputs",
        name: "openapi",
        revision: "revision-openapi-1",
      },
      mediaType: "application/yaml",
      size: 30,
      current: true,
      frozen: false,
      createdAt: "2026-09-01T10:11:00Z",
    };
    const workflow = {
      ref: { name: "openapi-from-source", version: "1" },
      entryStage: "analyze",
      parameters: {},
      inputs: {
        source: { required: true, mediaTypes: ["application/zip"] },
      },
      outputs: {
        openapi: {
          required: true,
          mediaTypes: ["application/yaml"],
          primary: true,
        },
      },
    };
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") {
          return jsonResponse(session);
        }
        if (url.pathname === "/v1/projects/project_example") {
          return jsonResponse(project, { headers: { ETag: '"1"' } });
        }
        if (url.pathname === "/v1/projects/project_example/artifacts") {
          return jsonResponse({
            items: [source, existingOutput],
            page: { hasMore: false },
          });
        }
        if (url.pathname === "/v1/projects/project_example/runs") {
          return jsonResponse({ items: [], page: { hasMore: false } });
        }
        if (url.pathname === "/v1/workflows") {
          return jsonResponse({ items: [workflow], page: { hasMore: false } });
        }
        throw new Error(`unexpected ${request.method} ${url.pathname}`);
      }),
    );
    renderProjectApplication(api, "/projects/project_example");
    const user = userEvent.setup();

    expect(
      await screen.findByText("No new Workflow result is recommended."),
    ).toBeVisible();
    expect(
      screen.queryByRole("button", { name: "Run openapi-from-source@1" }),
    ).not.toBeInTheDocument();
    await user.click(screen.getByText("All workflows"));
    expect(
      screen.getByRole("button", {
        name: "Run again openapi-from-source@1",
      }),
    ).toBeEnabled();
  });
});
