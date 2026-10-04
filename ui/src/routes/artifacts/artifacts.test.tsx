import { act, fireEvent, render, screen, within } from "@testing-library/react";
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

function renderArtifactApplication(
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

describe("Artifact routes", () => {
  it("keeps the revision lede neutral while metadata is loading or unavailable", async () => {
    let respond!: (response: Response) => void;
    const metadata = new Promise<Response>((resolve) => {
      respond = resolve;
    });
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const path = new URL(request.url).pathname;
        if (path === "/v1/auth/session") return jsonResponse(session);
        if (path.endsWith("/metadata")) return metadata;
        throw new Error(`unexpected ${path}`);
      }),
    );
    renderArtifactApplication(api, "/artifacts/projects/source");
    expect(await screen.findByText("Loading Artifact metadata…")).toBeVisible();
    expect(
      screen.getByText("Artifact revision", { selector: ".lede" }),
    ).toBeVisible();
    expect(
      screen.queryByText("Current revision", { selector: ".lede" }),
    ).toBeNull();
    await act(async () => {
      respond(
        jsonResponse(
          {
            code: "unavailable",
            message: "Metadata unavailable",
            retryable: false,
          },
          { status: 503 },
        ),
      );
      await metadata;
    });
    expect(
      await screen.findByText("Could not load this Artifact"),
    ).toBeVisible();
    expect(
      screen.getByText("Artifact revision", { selector: ".lede" }),
    ).toBeVisible();
  });

  it("recovers a failed filtered read without submitting a write or dropping filters", async () => {
    let reads = 0;
    const requests: Request[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") return jsonResponse(session);
        if (url.pathname === "/v1/artifacts") {
          expect(url.searchParams.get("namespace")).toBe("projects");
          reads += 1;
          return reads === 1
            ? jsonResponse(
                {
                  code: "unavailable",
                  message: "Storage is temporarily unavailable",
                  retryable: false,
                  requestId: "read-recovery",
                },
                { status: 503 },
              )
            : jsonResponse({ items: [], page: { hasMore: false } });
        }
        throw new Error(`unexpected request ${url.pathname}`);
      }),
    );
    renderArtifactApplication(api, "/artifacts?namespace=projects");
    expect(
      await screen.findByText("Could not load Artifact bindings"),
    ).toBeVisible();
    expect(
      screen.getByText("Storage is temporarily unavailable"),
    ).toBeVisible();
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Try again" }));
    expect(
      await screen.findByText("No Artifact bindings found."),
    ).toBeVisible();
    expect(
      within(
        screen.getByRole("button", { name: "Apply" }).closest("form")!,
      ).getByLabelText("Namespace"),
    ).toHaveValue("projects");
    expect(
      screen.queryByRole("navigation", { name: "Artifact pages" }),
    ).not.toBeInTheDocument();
    expect(reads).toBe(2);
    expect(requests.every((request) => request.method === "GET")).toBe(true);
  });

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
          expect(url.searchParams.get("excludeNamespace")).toBe("skills");
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

    await user.click(screen.getByRole("button", { name: "Upload Artifact" }));
    const file = new File(["zip"], "source.zip", {
      type: "application/zip",
    });
    const dropTarget = screen.getByText("Drop a file here").parentElement;
    expect(dropTarget).not.toBeNull();
    fireEvent.drop(dropTarget!, { dataTransfer: { files: [file] } });
    expect(screen.getByLabelText("Name")).toHaveValue("source");
    expect(screen.getByLabelText("Media type")).toHaveValue("application/zip");
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
    await user.click(screen.getByRole("button", { name: "Upload Artifact" }));
    await user.type(screen.getByLabelText("Name"), "source");
    await user.upload(
      screen.getByLabelText("Drop a file here"),
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

  it("keeps Skill packages on the dedicated Skills surface", async () => {
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
          expect(url.searchParams.get("excludeNamespace")).toBe("skills");
          return jsonResponse({ items: [], page: { hasMore: false } });
        }
        throw new Error(`unexpected test request ${request.method} ${url}`);
      }),
    );
    renderArtifactApplication(api, "/artifacts");
    await screen.findByText("No Artifact bindings found.");

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Upload Artifact" }));
    const uploadForm = screen.getByRole("dialog", { name: "Upload Artifact" });
    const namespace = within(uploadForm).getByLabelText("Namespace");
    await user.clear(namespace);
    await user.type(namespace, "skills");
    await user.upload(
      within(uploadForm).getByLabelText("Drop a file here"),
      new File(["zip"], "reviewed-skill.zip", {
        type: "application/vnd.contractor.agent-skill+zip",
      }),
    );
    await user.click(
      within(uploadForm).getByRole("button", { name: "Create binding" }),
    );

    const alert = await screen.findByRole("alert");
    expect(within(alert).getByRole("link", { name: "Skills" })).toHaveAttribute(
      "href",
      "/catalog/skills",
    );
    expect(requests.some((request) => request.method === "PUT")).toBe(false);
  });

  it("renders exact history/lineage and escapes bounded preview text", async () => {
    const preview = "<script>alert('not executable')</script>";
    const selected = {
      artifact: {
        namespace: "projects",
        name: "architecture",
        revision: "revision-2",
      },
      mediaType: "text/plain",
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
              {
                kind: "project_output_publish",
                sourceScope: "run",
                source: selected.artifact,
                targetScope: "project",
                target: selected.artifact,
                createdAt: "2026-08-31T12:02:00Z",
              },
            ],
            page: { hasMore: false },
          });
        }
        if (url.pathname.endsWith("/projects/architecture")) {
          return apiResponse(preview, {
            headers: {
              "content-type": "text/plain",
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
    expect(
      screen.getByText("Historical revision", { selector: ".lede" }),
    ).toBeVisible();
    expect(screen.queryByText("input fork")).not.toBeInTheDocument();
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Versions" }));
    expect(await screen.findByText("input fork")).toBeInTheDocument();
    expect(screen.getByText("project output publish")).toBeInTheDocument();
    expect(screen.getByText("Run run-example")).toBeInTheDocument();

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Load preview" }));
    expect(await screen.findByText(preview)).toBeInTheDocument();
    expect(document.querySelector("script")).toBeNull();
    expect(
      screen.getByText("Historical revisions are read-only."),
    ).toBeInTheDocument();
  });

  it("keeps contextual return navigation and current status after a user Artifact version upload", async () => {
    const original = {
      artifact: {
        namespace: "projects",
        name: "source",
        revision: "revision-1",
      },
      mediaType: "text/plain",
      size: 4,
      current: true,
      frozen: false,
      createdAt: "2026-09-01T10:00:00Z",
    };
    const latest = {
      ...original,
      artifact: { ...original.artifact, revision: "revision-2" },
    };
    const returnState = {
      returnTo: "/artifacts?namespace=projects&cursor=page-two",
      returnLabel: "Filtered Artifacts",
      returnState: { returnTo: "/catalog/skills", returnLabel: "Skills" },
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
          url.pathname === "/v1/artifacts/projects/source"
        ) {
          expect(request.headers.get("If-Match")).toBe('"revision-1"');
          written = true;
          return jsonResponse(
            { artifact: latest.artifact, mediaType: "text/plain", size: 4 },
            { status: 201, headers: { ETag: '"revision-2"' } },
          );
        }
        if (url.pathname.endsWith("/metadata")) {
          return jsonResponse(
            url.searchParams.get("revision") === "revision-2"
              ? latest
              : { ...original, current: !written },
          );
        }
        if (url.pathname.endsWith("/versions"))
          return jsonResponse({
            items: [latest, { ...original, current: false }],
            page: { hasMore: false },
          });
        if (url.pathname.endsWith("/lineage"))
          return jsonResponse({ items: [], page: { hasMore: false } });
        if (url.pathname === "/v1/artifacts")
          return jsonResponse({ items: [], page: { hasMore: false } });
        throw new Error(`unexpected ${request.method} ${url}`);
      }),
    );
    const { router } = renderArtifactApplication(api, {
      pathname: "/artifacts/projects/source",
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
      new File(["new!"], "source.txt", { type: "text/plain" }),
    );
    await user.click(
      screen.getByRole("button", { name: "Upload new version" }),
    );
    await vi.waitFor(() =>
      expect(router.state.location.search).toBe("?revision=revision-2"),
    );
    expect(router.state.location.state).toEqual(returnState);
    expect(
      await screen.findByText("Current revision", { selector: ".lede" }),
    ).toBeVisible();
    const back = screen.getByRole("link", { name: "← Filtered Artifacts" });
    expect(back).toHaveAttribute("href", returnState.returnTo);
    await user.click(screen.getByRole("button", { name: "Versions" }));
    await user.click(await screen.findByRole("link", { name: /revision-1/ }));
    await vi.waitFor(() =>
      expect(router.state.location.search).toBe("?revision=revision-1"),
    );
    expect(router.state.location.state).toEqual(returnState);
    expect(
      await screen.findByText("Historical revision", { selector: ".lede" }),
    ).toBeVisible();
    await user.click(
      screen.getByRole("link", { name: "← Filtered Artifacts" }),
    );
    await vi.waitFor(() =>
      expect(router.state.location.pathname).toBe("/artifacts"),
    );
    expect(router.state.location.state).toEqual(returnState.returnState);
  });

  it.each([
    { pinned: false, path: { pathname: "/artifacts/projects/source" } },
    {
      pinned: true,
      path: {
        pathname: "/artifacts/projects/source",
        search: "?revision=revision-1",
      },
    },
  ])(
    "keeps a concurrent-writer upload conflict visible with the new current revision (pinned: $pinned)",
    async ({ path }) => {
      const original = {
        artifact: {
          namespace: "projects",
          name: "source",
          revision: "revision-1",
        },
        mediaType: "text/plain",
        size: 4,
        current: true,
        frozen: false,
        createdAt: "2026-09-01T10:00:00Z",
      };
      const concurrent = {
        ...original,
        artifact: { ...original.artifact, revision: "revision-2" },
        size: 6,
        createdAt: "2026-09-01T10:05:00Z",
      };
      const mine = {
        ...original,
        artifact: { ...original.artifact, revision: "revision-3" },
      };
      const returnState = { returnTo: "/artifacts", returnLabel: "Artifacts" };
      // Another writer binds revision-2 while this page shows revision-1.
      let concurrentWriter = false;
      const writes: Request[] = [];
      const api = new PublicAPI(
        runtimeConfig,
        vi.fn(async (input) => {
          const request = input instanceof Request ? input : new Request(input);
          const url = new URL(request.url);
          if (url.pathname === "/v1/auth/session") return jsonResponse(session);
          if (
            request.method === "PUT" &&
            url.pathname === "/v1/artifacts/projects/source"
          ) {
            writes.push(request);
            if (request.headers.get("If-Match") === '"revision-2"') {
              return jsonResponse(
                { artifact: mine.artifact, mediaType: "text/plain", size: 4 },
                { status: 200, headers: { ETag: '"revision-3"' } },
              );
            }
            concurrentWriter = true;
            return jsonResponse(
              {
                code: "conflict",
                message:
                  "resource state changed; retry with the current revision",
                retryable: true,
                requestId: "request-conflict",
              },
              { status: 409 },
            );
          }
          if (url.pathname.endsWith("/metadata")) {
            switch (url.searchParams.get("revision")) {
              case "revision-1":
                return jsonResponse({
                  ...original,
                  current: !concurrentWriter,
                });
              case "revision-3":
                return jsonResponse(mine);
            }
            return jsonResponse(concurrentWriter ? concurrent : original);
          }
          if (url.pathname === "/v1/artifacts")
            return jsonResponse({ items: [], page: { hasMore: false } });
          throw new Error(`unexpected ${request.method} ${url}`);
        }),
      );
      const { router } = renderArtifactApplication(api, {
        search: "",
        ...path,
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
        new File(["mine"], "source.txt", { type: "text/plain" }),
      );
      await user.click(
        screen.getByRole("button", { name: "Upload new version" }),
      );

      expect(
        await screen.findByText('If-Match: "revision-2"'),
      ).toBeInTheDocument();
      expect(
        screen.getByText("Current revision", { selector: ".lede" }),
      ).toBeVisible();
      const notice = screen.getByRole("alert");
      expect(notice).toHaveTextContent(/based on revision-1 was rejected/);
      expect(notice).toHaveTextContent(/binding changed/);
      expect(router.state.location.search).toBe("");
      expect(router.state.location.state).toEqual(returnState);
      expect(writes).toHaveLength(1);
      expect(writes[0]?.headers.get("If-Match")).toBe('"revision-1"');

      // An explicit new upload against the reviewed revision clears it.
      await user.upload(
        screen.getByLabelText("Drop a file here"),
        new File(["mine"], "source.txt", { type: "text/plain" }),
      );
      await user.click(
        screen.getByRole("button", { name: "Upload new version" }),
      );
      await vi.waitFor(() =>
        expect(router.state.location.search).toBe("?revision=revision-3"),
      );
      expect(
        await screen.findByText('If-Match: "revision-3"'),
      ).toBeInTheDocument();
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      expect(writes.map((write) => write.headers.get("If-Match"))).toEqual([
        '"revision-1"',
        '"revision-2"',
      ]);
    },
  );

  it("does not offer inline preview for binary content", async () => {
    const binary = {
      artifact: {
        namespace: "projects",
        name: "source",
        revision: "revision-1",
      },
      mediaType: "application/octet-stream",
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
    renderArtifactApplication(
      api,
      "/artifacts/projects/source?revision=revision-1",
    );
    const previewButton = await screen.findByRole("button", {
      name: "Load preview",
    });
    expect(
      screen.getByText("Current revision", { selector: ".lede" }),
    ).toBeVisible();
    expect(previewButton).toBeDisabled();
    expect(
      screen.getByText(/Inline preview is unavailable/),
    ).toBeInTheDocument();
  });
  it("restores the namespace and cursor after inspecting an Artifact", async () => {
    const binding = {
      artifact: { namespace: "sources", name: "example", revision: "r1" },
      mediaType: "text/plain",
      size: 4,
      current: true,
      frozen: false,
      createdAt: "2026-09-01T10:00:00Z",
    };
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input),
          url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") return jsonResponse(session);
        if (url.pathname.endsWith("/metadata")) return jsonResponse(binding);
        if (url.pathname === "/v1/artifacts") {
          expect(url.searchParams.get("namespace")).toBe("sources");
          expect(url.searchParams.get("cursor")).toBe("page-two");
          return jsonResponse({ items: [binding], page: { hasMore: false } });
        }
        return jsonResponse({ items: [], page: { hasMore: false } });
      }),
    );
    const { router } = renderArtifactApplication(
      api,
      "/artifacts?namespace=sources&cursor=page-two",
    );
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("link", { name: "sources/example" }),
    );
    await user.click(await screen.findByRole("link", { name: "← Artifacts" }));
    expect(router.state.location.search).toBe(
      "?namespace=sources&cursor=page-two",
    );
    expect(screen.getByPlaceholderText("all namespaces")).toHaveValue(
      "sources",
    );
    expect(
      await screen.findByRole("link", { name: "sources/example" }),
    ).toBeVisible();
  });
});
