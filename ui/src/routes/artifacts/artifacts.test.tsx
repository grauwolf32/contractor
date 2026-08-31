import { render, screen } from "@testing-library/react";
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
  idleExpiresAt: "2026-08-31T20:00:00Z",
  absoluteExpiresAt: "2026-09-01T12:00:00Z",
};

function apiResponse(
  body: BodyInit | null,
  options: ResponseInit = {},
): Response {
  const headers = new Headers(options.headers);
  headers.set("X-Contractor-API-Version", "contractor.public.v1");
  return new Response(body, { ...options, headers });
}

function jsonResponse(value: unknown, options: ResponseInit = {}): Response {
  return apiResponse(JSON.stringify(value), {
    ...options,
    headers: { "content-type": "application/json", ...options.headers },
  });
}

function renderArtifactApplication(api: PublicAPI, path: string) {
  const router = createMemoryRouter(applicationRoutes(), {
    initialEntries: [path],
  });
  return {
    ...render(<Application api={api} publicAPI={api} router={router} />),
    router,
  };
}

describe("Artifact routes", () => {
  it("lists bindings and creates one exact Artifact without optimistic state", async () => {
    const requests: Request[] = [];
    const current = {
      artifact: {
        namespace: "projects",
        name: "existing",
        revision: "revision-1",
      },
      mediaType: "text/plain",
      size: 5,
      current: true,
      frozen: false,
      createdAt: "2026-08-31T12:00:00Z",
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
        if (url.pathname === "/v1/artifacts" && request.method === "GET") {
          if (url.searchParams.get("cursor") === "cursor-next") {
            return jsonResponse({
              items: [
                {
                  ...current,
                  artifact: {
                    ...current.artifact,
                    name: "next-page",
                    revision: "revision-next",
                  },
                },
              ],
              page: { hasMore: false },
            });
          }
          return jsonResponse({
            items: [current],
            page: { hasMore: true, nextCursor: "cursor-next" },
          });
        }
        if (
          url.pathname === "/v1/artifacts/projects/source" &&
          request.method === "PUT"
        ) {
          return jsonResponse(
            {
              artifact: {
                namespace: "projects",
                name: "source",
                revision: "revision-2",
              },
              mediaType: "application/zip",
              size: 3,
            },
            { status: 201, headers: { ETag: '"revision-2"' } },
          );
        }
        return jsonResponse(
          {
            code: "not_found",
            message: "resource was not found",
            retryable: false,
            requestId: "request-test",
          },
          { status: 404 },
        );
      }),
    );
    renderArtifactApplication(api, "/artifacts");
    expect(
      await screen.findByRole("link", { name: "projects/existing" }),
    ).toBeInTheDocument();

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(
      await screen.findByRole("link", { name: "projects/next-page" }),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Previous" }));
    expect(
      await screen.findByRole("link", { name: "projects/existing" }),
    ).toBeInTheDocument();

    await user.type(screen.getByLabelText("Name"), "source");
    const fileInput = screen.getByLabelText(/Local file/);
    await user.upload(
      fileInput,
      new File(["zip"], "source.zip", { type: "application/zip" }),
    );
    await user.click(screen.getByRole("button", { name: "Create binding" }));

    await vi.waitFor(() => {
      expect(requests.some((request) => request.method === "PUT")).toBe(true);
    });
    expect(
      await screen.findByRole("link", {
        name: /projects\/source@revision-2/,
      }),
    ).toBeInTheDocument();
    const put = requests.find((request) => request.method === "PUT");
    expect(put?.headers.get("If-None-Match")).toBe("*");
    expect(put?.headers.get("If-Match")).toBeNull();
    expect(put?.headers.get("X-CSRF-Token")).toBe(session.csrfToken);
    expect(put?.headers.get("Content-Type")).toBe("application/zip");
  });

  it("reconciles a CAS conflict and never retries the unsafe PUT", async () => {
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
        if (url.pathname === "/v1/artifacts" && request.method === "GET") {
          return jsonResponse({ items: [], page: { hasMore: false } });
        }
        if (request.method === "PUT") {
          return jsonResponse(
            {
              code: "conflict",
              message: "resource state changed",
              retryable: true,
              requestId: "request-conflict",
            },
            { status: 409 },
          );
        }
        throw new Error(`unexpected test request ${request.method} ${url}`);
      }),
    );
    renderArtifactApplication(api, "/artifacts");
    await screen.findByText("No Artifact bindings found.");
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Name"), "source");
    await user.upload(
      screen.getByLabelText(/Local file/),
      new File(["next"], "source.txt", { type: "text/plain" }),
    );
    await user.click(screen.getByRole("button", { name: "Create binding" }));

    expect(await screen.findByText(/binding changed/i)).toBeInTheDocument();
    await vi.waitFor(() => {
      expect(
        requests.filter((request) => request.method === "GET").length,
      ).toBeGreaterThanOrEqual(3);
    });
    expect(requests.filter((request) => request.method === "PUT")).toHaveLength(
      1,
    );
  });

  it("renders exact history/lineage and escapes bounded preview text", async () => {
    const preview = "<script>alert('not executable')</script>";
    const selected = {
      artifact: {
        namespace: "projects",
        name: "architecture",
        revision: "revision-2",
      },
      mediaType: "text/vnd.likec4",
      size: preview.length,
      current: false,
      frozen: false,
      createdAt: "2026-08-31T12:00:00Z",
    };
    const current = {
      ...selected,
      artifact: { ...selected.artifact, revision: "revision-3" },
      current: true,
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
          return jsonResponse(selected);
        }
        if (url.pathname.endsWith("/versions")) {
          return jsonResponse({
            items: [current, selected],
            page: { hasMore: false },
          });
        }
        if (url.pathname.endsWith("/lineage")) {
          return jsonResponse({
            items: [
              {
                kind: "input_fork",
                sourceScope: "user",
                source: selected.artifact,
                targetScope: "run",
                target: {
                  namespace: "inputs",
                  name: "architecture",
                  revision: "revision-run-1",
                },
                runId: "run-example",
                stageExecutionId: "stage-example",
                createdAt: "2026-08-31T12:01:00Z",
              },
            ],
            page: { hasMore: false },
          });
        }
        if (url.pathname.endsWith("/projects/architecture")) {
          return apiResponse(preview, {
            headers: {
              "content-type": "text/vnd.likec4",
              "content-length": String(preview.length),
            },
          });
        }
        throw new Error(`unexpected test request ${request.method} ${url}`);
      }),
    );
    renderArtifactApplication(
      api,
      "/artifacts/projects/architecture?revision=revision-2",
    );
    expect(
      (await screen.findAllByText("revision-2", { selector: "code" })).length,
    ).toBeGreaterThan(0);
    expect(await screen.findByText("input fork")).toBeInTheDocument();
    expect(screen.getByText("Run run-example")).toBeInTheDocument();

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Load text preview" }));
    expect(await screen.findByText(preview)).toBeInTheDocument();
    expect(document.querySelector("script")).toBeNull();
    expect(
      screen.getByText("Historical revision is immutable."),
    ).toBeInTheDocument();
  });

  it("does not offer inline preview for binary content", async () => {
    const binary = {
      artifact: {
        namespace: "projects",
        name: "source",
        revision: "revision-1",
      },
      mediaType: "application/zip",
      size: 1024,
      current: true,
      frozen: false,
      createdAt: "2026-08-31T12:00:00Z",
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
          return jsonResponse(binary);
        }
        if (url.pathname.endsWith("/versions")) {
          return jsonResponse({ items: [binary], page: { hasMore: false } });
        }
        if (url.pathname.endsWith("/lineage")) {
          return jsonResponse({ items: [], page: { hasMore: false } });
        }
        throw new Error("binary preview must not fetch Artifact bytes");
      }),
    );
    renderArtifactApplication(api, "/artifacts/projects/source");
    const previewButton = await screen.findByRole("button", {
      name: "Load text preview",
    });
    expect(previewButton).toBeDisabled();
    expect(
      screen.getByText(/Inline preview is unavailable/),
    ).toBeInTheDocument();
  });
});
