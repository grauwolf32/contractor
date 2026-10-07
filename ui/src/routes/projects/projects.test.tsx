import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";

import * as queryClientFactory from "../../app/query-client";
import { queryKeys } from "../../api/query-keys";
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

function renderProjectApplication(
  api: PublicAPI,
  path: string | { pathname: string; search: string; state: unknown },
) {
  const router = createMemoryRouter(applicationRoutes(), {
    initialEntries: [path],
  });
  return {
    ...render(<Application api={api} publicAPI={api} router={router} />),
    router,
  };
}

describe("Project routes", () => {
  it("loads a bounded Overview and preserves section filters through navigation", async () => {
    const requests: URL[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const url = new URL(
          (input instanceof Request ? input : new Request(input)).url,
        );
        requests.push(url);
        if (url.pathname === "/v1/auth/session") return jsonResponse(session);
        if (url.pathname === "/v1/projects/project_example")
          return jsonResponse(project, { headers: { ETag: '"1"' } });
        return jsonResponse({ items: [], page: { hasMore: false } });
      }),
    );
    const { router } = renderProjectApplication(
      api,
      "/projects/project_example",
    );
    const user = userEvent.setup();
    await screen.findByRole("heading", { name: "Timeline" });
    const reads = (suffix: string) =>
      requests.filter(
        (url) => url.pathname === `/v1/projects/project_example/${suffix}`,
      );
    await waitFor(() => expect(reads("artifacts")).toHaveLength(3));
    // Bounded samples only (S06): no complete material or Workflow inventory.
    expect(
      reads("artifacts").map((url) => url.searchParams.get("namespace")),
    ).toEqual(expect.arrayContaining([null, "sources", "openapi"]));
    expect(
      reads("artifacts").every(
        (url) => Number(url.searchParams.get("limit")) <= 5,
      ),
    ).toBe(true);
    expect(
      reads("runs").map((url) => [
        url.searchParams.get("limit"),
        url.searchParams.get("state"),
      ]),
    ).toEqual(
      expect.arrayContaining([
        ["5", null],
        ["3", "succeeded"],
      ]),
    );
    expect(
      reads("audits").every(
        (url) => Number(url.searchParams.get("limit")) <= 50,
      ),
    ).toBe(true);
    expect(requests.some((url) => url.pathname === "/v1/workflows")).toBe(
      false,
    );
    const nav = within(
      screen.getByRole("navigation", { name: "Project sections" }),
    );
    await user.click(nav.getByRole("link", { name: "Runs" }));
    await user.click(await screen.findByRole("button", { name: "Completed" }));
    await user.selectOptions(
      screen.getByRole("combobox", { name: "State" }),
      "failed",
    );
    await waitFor(() =>
      expect(
        requests.some(
          (url) =>
            url.searchParams.get("state") === "failed" &&
            url.searchParams.get("lifecycle") === "terminal" &&
            url.searchParams.get("limit") === "25",
        ),
      ).toBe(true),
    );
    const filtered = router.state.location.search;
    await user.click(nav.getByRole("link", { name: "Settings" }));
    await screen.findByRole("button", { name: "Edit metadata" });
    expect(screen.queryByRole("heading", { name: "Recent Runs" })).toBeNull();
    await user.click(nav.getByRole("link", { name: "Runs" }));
    expect(router.state.location.search).toBe(filtered);
    expect(await screen.findByRole("combobox", { name: "State" })).toHaveValue(
      "failed",
    );
  });

  it("opens Run section links, sends cursors to the filtered API and resets pages when the view changes", async () => {
    const requests: URL[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const url = new URL(
          (input instanceof Request ? input : new Request(input)).url,
        );
        requests.push(url);
        if (url.pathname === "/v1/auth/session") return jsonResponse(session);
        if (url.pathname === "/v1/projects/project_example")
          return jsonResponse(project, { headers: { ETag: '"1"' } });
        return jsonResponse({ items: [], page: { hasMore: false } });
      }),
    );
    const { router } = renderProjectApplication(
      api,
      "/projects/project_example/runs?view=completed&state=succeeded&cursor=page2",
    );
    const user = userEvent.setup();
    await screen.findByRole("button", { name: "All Runs" });
    expect(router.state.location.pathname).toBe(
      "/projects/project_example/runs",
    );
    expect(router.state.location.hash).toBe("");
    await waitFor(() =>
      expect(
        requests.some(
          (url) =>
            url.pathname.endsWith("/runs") &&
            url.searchParams.get("cursor") === "page2" &&
            url.searchParams.get("state") === "succeeded",
        ),
      ).toBe(true),
    );
    expect(
      requests.some(
        (url) =>
          url.pathname.endsWith("/artifacts") ||
          url.pathname.endsWith("/audits"),
      ),
    ).toBe(false);
    await user.click(screen.getByRole("button", { name: "Active" }));
    expect(router.state.location.search).toBe("?view=active");
    await waitFor(() =>
      expect(
        requests.some(
          (url) =>
            url.searchParams.get("lifecycle") === "active" &&
            !url.searchParams.has("cursor") &&
            !url.searchParams.has("state"),
        ),
      ).toBe(true),
    );
  });

  it("polls a Project in deletion in the list and drops it once the Server has removed it", async () => {
    let complete = false;
    const deletingProject = {
      ...project,
      lifecycle: "deleting",
      deletion: { phase: "draining", requestedAt: "2026-09-05T10:00:00Z" },
      revision: "2",
    };
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") return jsonResponse(session);
        if (url.pathname === "/v1/projects") {
          return jsonResponse({
            items: complete ? [] : [deletingProject],
            page: { hasMore: false },
          });
        }
        throw new Error(`unexpected ${request.method} ${url.pathname}`);
      }),
    );
    const { router } = renderProjectApplication(api, "/projects");
    const row = await screen.findByRole("link", { name: project.name });
    expect(row).toHaveAttribute("href", "/projects/project_example");
    expect(row.closest("li")).toHaveTextContent("Deleting");
    // Deletion lives in the project's actions menu and Settings, not in rows.
    expect(screen.queryByRole("button", { name: /Delete/ })).toBeNull();
    complete = true;
    await waitFor(
      () =>
        expect(screen.queryByRole("link", { name: project.name })).toBeNull(),
      { timeout: 3_000 },
    );
    expect(screen.getByText("No projects yet")).toBeVisible();
    expect(router.state.location.pathname).toBe("/projects");
  });

  it("keeps a refused deletion open and confirms again against the re-read Project", async () => {
    let current = project;
    const deleteRequests: Request[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") return jsonResponse(session);
        if (url.pathname === "/v1/projects") {
          return jsonResponse({ items: [current], page: { hasMore: false } });
        }
        if (url.pathname === "/v1/projects/project_example") {
          if (request.method === "DELETE") {
            deleteRequests.push(request.clone());
            current = { ...project, name: "Renamed project", revision: "2" };
            return jsonResponse(
              {
                code: "precondition_failed",
                message: "resource revision precondition failed",
                retryable: false,
              },
              { status: 412 },
            );
          }
          return jsonResponse(current, {
            headers: { ETag: `"${current.revision}"` },
          });
        }
        return jsonResponse({ items: [], page: { hasMore: false } });
      }),
    );
    renderProjectApplication(api, "/projects/project_example");
    const user = userEvent.setup();
    await screen.findByRole("heading", { name: "Payment service" });
    await user.click(screen.getByLabelText("Project actions"));
    await user.click(screen.getByRole("button", { name: "Delete Project" }));
    const dialog = screen.getByRole("alertdialog", {
      name: "Delete Payment service?",
    });
    await user.type(
      within(dialog).getByLabelText("Type Payment service to confirm"),
      project.name,
    );
    await user.click(
      within(dialog).getByRole("button", { name: "Delete Project" }),
    );
    expect(
      await within(dialog).findByText("resource revision precondition failed"),
    ).toBeVisible();
    expect(deleteRequests).toHaveLength(1);
    expect(deleteRequests[0]?.headers.get("If-Match")).toBe('"1"');
    await waitFor(() =>
      expect(
        within(dialog).getByRole("button", { name: "Cancel" }),
      ).toBeEnabled(),
    );
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }));
    // The refusal re-reads the Project: the header and a new confirmation
    // follow its current name and revision.
    expect(
      await screen.findByRole("heading", { name: "Renamed project" }),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Delete Project" }));
    const reopened = screen.getByRole("alertdialog", {
      name: "Delete Renamed project?",
    });
    expect(
      within(reopened).getByLabelText("Type Renamed project to confirm"),
    ).toHaveValue("");
    expect(
      within(reopened).getByRole("button", { name: "Delete Project" }),
    ).toBeDisabled();
    expect(deleteRequests).toHaveLength(1);
  });

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
      const queryClient = queryClientFactory.createApplicationQueryClient();
      const factory = vi
        .spyOn(queryClientFactory, "createApplicationQueryClient")
        .mockReturnValueOnce(queryClient);
      renderProjectApplication(api, "/projects/project_example/settings");
      factory.mockRestore();
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
      // Model an authoritative refresh while the modal makes the background inert.
      await act(async () => {
        await queryClient.invalidateQueries({
          queryKey: queryKeys.projects.detail(project.projectId),
        });
      });
      expect(
        await screen.findByRole("heading", {
          name: "Updated elsewhere",
          hidden: true,
        }),
      ).toBeInTheDocument();
      expect(projectReads).toBe(2);
      expect(screen.getByLabelText(label)).toHaveValue(draft);
      await user.click(
        screen.getByRole("button", {
          name: operation === "metadata" ? "Save changes" : "Save target",
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
    await user.click(screen.getByLabelText("Project actions"));
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
    await user.click(
      screen.getByRole("button", { name: "Refresh project details" }),
    );
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
        if (
          url.pathname.endsWith("/artifacts") ||
          url.pathname.endsWith("/audits") ||
          url.pathname === "/v1/audit-profiles"
        ) {
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
      screen.getAllByRole("button", { name: "New project" })[0]!,
    );
    const dialog = screen.getByRole("dialog", { name: "New project" });
    expect(within(dialog).getByLabelText("Name")).toHaveFocus();
    expect(within(dialog).getByLabelText("Name")).toHaveAttribute(
      "maxlength",
      "160",
    );
    expect(within(dialog).getByLabelText("Description")).toHaveAttribute(
      "maxlength",
      "4096",
    );
    await user.type(within(dialog).getByLabelText("Name"), "Payment service");
    await user.type(
      within(dialog).getByLabelText("Description"),
      "Reusable service analysis",
    );
    await user.click(
      within(dialog).getByRole("button", { name: "Create project" }),
    );

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

  it("opens New project from ?new=1, reuses the key for the same request and clears the parameter on close", async () => {
    const creates: Request[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") return jsonResponse(session);
        if (url.pathname === "/v1/projects" && request.method === "POST") {
          creates.push(request.clone());
          return jsonResponse(
            {
              code: "unavailable",
              message: "Project store unavailable",
              retryable: true,
            },
            { status: 503 },
          );
        }
        return jsonResponse({ items: [], page: { hasMore: false } });
      }),
    );
    const { router } = renderProjectApplication(api, "/projects?new=1");
    const user = userEvent.setup();
    const dialog = await screen.findByRole("dialog", { name: "New project" });
    await user.type(within(dialog).getByLabelText("Name"), "Payment service");
    await user.click(
      within(dialog).getByRole("button", { name: "Create project" }),
    );
    expect(
      await within(dialog).findByText("Project store unavailable"),
    ).toBeVisible();
    await user.click(
      within(dialog).getByRole("button", { name: "Create project" }),
    );
    await waitFor(() => expect(creates).toHaveLength(2));
    // The same request after an unknown outcome carries the same key.
    expect(creates[1]?.headers.get("Idempotency-Key")).toBe(
      creates[0]?.headers.get("Idempotency-Key"),
    );
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(router.state.location.pathname).toBe("/projects");
    expect(router.state.location.search).toBe("");
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
    renderProjectApplication(
      api,
      "/projects/project_example/artifacts?add=artifact",
    );
    const user = userEvent.setup();

    await screen.findByRole("dialog", { name: "Add material" });
    await user.click(screen.getByRole("button", { name: "Source code ZIP" }));
    const dialog = screen.getByRole("dialog", { name: "Source code ZIP" });
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
    renderProjectApplication(api, "/projects/project_example/settings");
    const user = userEvent.setup();

    await screen.findByRole("heading", { name: "Payment service" });
    await user.click(screen.getByRole("button", { name: "Configure target" }));
    let dialog = screen.getByRole("dialog", { name: "Application access" });
    expect(within(dialog).getByLabelText("Application URL")).toHaveFocus();
    await user.keyboard("{Escape}");
    const targetTrigger = screen.getByRole("button", {
      name: "Configure target",
    });
    await waitFor(() => expect(targetTrigger).toHaveFocus());
    await user.click(targetTrigger);
    dialog = screen.getByRole("dialog", { name: "Application access" });
    await user.type(
      within(dialog).getByLabelText("Application URL"),
      "https://app.example.test/api",
    );
    await user.selectOptions(
      within(dialog).getByLabelText("Authorization"),
      "bearer",
    );
    const token = within(dialog).getByLabelText("Bearer token · write only");
    // Secrets stay masked unless the user asks to see them.
    expect(token).toHaveAttribute("type", "password");
    expect(token).toHaveAttribute("autocomplete", "new-password");
    const reveal = within(dialog).getByRole("checkbox", {
      name: "Show secret while entering",
    });
    expect(reveal).not.toBeChecked();
    await user.click(reveal);
    expect(token).toHaveAttribute("type", "text");
    await user.click(reveal);
    expect(token).toHaveAttribute("type", "password");
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
    expect(
      screen.getByRole("navigation", { name: "Project sections" }),
    ).toBeVisible();
    expect(screen.getByRole("link", { name: "Overview" })).toHaveAttribute(
      "aria-current",
      "page",
    );
    await userEvent
      .setup()
      .click(screen.getByRole("link", { name: "Add material" }));
    expect(
      await screen.findByRole("button", { name: "Source code ZIP" }),
    ).toBeEnabled();
    const git = screen.getByRole("button", { name: "Import Git repository" });
    expect(git).toHaveClass("materials-kind");
    expect(git.querySelector("svg")).not.toBeNull();
    await userEvent.setup().click(git);
    expect(
      screen.getByRole("dialog", { name: "Import Git repository" }),
    ).toBeVisible();
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
    expect(
      screen.getByText("Current revision", { selector: ".lede" }),
    ).toBeVisible();
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Versions" }));
    expect(await screen.findByText("project output publish")).toBeVisible();
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Load preview" }));
    expect(await screen.findByText("Project documentation")).toBeVisible();
    await user.click(
      screen.getByText("Upload a new version", { selector: "summary" }),
    );
    expect(
      screen.getByRole("button", { name: "Upload new version" }),
    ).toBeEnabled();
  });

  it.each(["projects", "evals"])(
    "keeps contextual return navigation after a %s Artifact version upload",
    async (scope) => {
      const original = {
        artifact: { namespace: "docs", name: "readme", revision: "revision-1" },
        mediaType: "text/plain",
        size: 4,
        current: true,
        frozen: false,
        createdAt: "2026-09-01T10:10:00Z",
      };
      const latest = {
        ...original,
        artifact: { ...original.artifact, revision: "revision-2" },
      };
      const returnState = {
        returnTo: "/projects/project_example/audits/audit-example#finding-1",
        returnLabel: "Audit",
        returnState: {
          returnTo: "/projects/project_example",
          returnLabel: "Project Overview",
        },
      };
      let written = false;
      const api = new PublicAPI(
        runtimeConfig,
        vi.fn(async (input) => {
          const request = input instanceof Request ? input : new Request(input);
          const url = new URL(request.url);
          if (url.pathname === "/v1/auth/session") return jsonResponse(session);
          if (
            request.method === "PUT" &&
            url.pathname ===
              "/v1/projects/project_example/artifacts/docs/readme"
          ) {
            expect(request.headers.get("If-Match")).toBe('"revision-1"');
            written = true;
            return jsonResponse(
              { artifact: latest.artifact, mediaType: "text/plain", size: 4 },
              { status: 201, headers: { ETag: '"revision-2"' } },
            );
          }
          if (url.pathname.endsWith("/metadata"))
            return jsonResponse(
              url.searchParams.get("revision") === "revision-2"
                ? latest
                : { ...original, current: !written },
            );
          throw new Error(`unexpected ${request.method} ${url}`);
        }),
      );
      const { router } = renderProjectApplication(api, {
        pathname: `/${scope}/project_example/artifacts/docs/readme`,
        search: "?revision=revision-1",
        state: returnState,
      });
      expect(
        await screen.findByText("Current revision", { selector: ".lede" }),
      ).toBeVisible();
      const user = userEvent.setup();
      await user.click(
        screen.getByText("Upload a new version", { selector: "summary" }),
      );
      await user.upload(
        screen.getByLabelText("Drop a file here"),
        new File(["new!"], "readme.txt", { type: "text/plain" }),
      );
      await user.click(
        screen.getByRole("button", { name: "Upload new version" }),
      );
      await waitFor(() =>
        expect(router.state.location.search).toBe("?revision=revision-2"),
      );
      expect(router.state.location.state).toEqual(returnState);
      expect(
        await screen.findByText("Current revision", { selector: ".lede" }),
      ).toBeVisible();
      expect(screen.getByRole("link", { name: "← Audit" })).toHaveAttribute(
        "href",
        returnState.returnTo,
      );
    },
  );

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
        presentation: {
          displayName: "OpenAPI contract",
          description: "Build an API contract from the project sources.",
        },
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
      "/projects/project_example/workflows",
    );
    const user = userEvent.setup();

    expect(
      await screen.findByRole("button", {
        name: "Configure openapi-from-source@1",
      }),
    ).toBeEnabled();
    expect(
      screen.getAllByRole("heading", { name: "OpenAPI contract" }),
    ).toHaveLength(1);
    expect(
      screen.getAllByText("Build an API contract from the project sources."),
    ).toHaveLength(1);
    expect(screen.queryByText("No authored description.")).toBeNull();
    expect(
      screen.getByRole("button", { name: "Configure likec4-from-source@1" }),
    ).toBeEnabled();
    await user.click(
      screen.getByRole("button", { name: "Configure openapi-from-source@1" }),
    );
    const dialog = await screen.findByRole("dialog", {
      name: "Configure Run",
    });
    expect(
      within(dialog).getByRole("combobox", { name: /source required/ }),
    ).toHaveValue("sources/payment-service@revision-source-1");
    expect(within(dialog).getByText("Review 1")).toBeVisible();
    expect(
      within(dialog).getByText(/Suggested only because application\/zip/),
    ).toBeVisible();
    await user.click(
      within(dialog).getByRole("button", {
        name: "Confirm input for source",
      }),
    );
    expect(within(dialog).getByText("Ready", { exact: true })).toBeVisible();
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
    renderProjectApplication(api, "/projects/project_example/workflows");
    const user = userEvent.setup();

    expect(await screen.findByText("No new format matches.")).toBeVisible();
    expect(
      screen.queryByRole("button", { name: "Configure openapi-from-source@1" }),
    ).not.toBeInTheDocument();
    await user.click(screen.getByText("All workflows"));
    expect(
      screen.getByRole("button", {
        name: "Run again openapi-from-source@1",
      }),
    ).toBeEnabled();
  });
});
