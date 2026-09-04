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
  it("creates an owner-scoped Project and opens its dashboard", async () => {
    const requests: Request[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request);
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
});
