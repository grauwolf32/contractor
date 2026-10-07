import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter, RouterProvider } from "react-router";
import { describe, expect, it, vi } from "vitest";

import type { ArtifactMetadata } from "../../api/artifacts";
import { PublicAPI } from "../../api/client";
import { PublicAPIProvider } from "../../api/context";
import { ProjectArtifactRegion } from "../projects/artifact-region";

const PROJECT = "project-1";
const LIST = `/v1/projects/${PROJECT}/artifacts`;
const COMMIT = "a".repeat(40);

const gitSource = {
  repositoryUrl: "https://example.test:443/repo.git",
  requestedRef: "main",
  resolvedCommit: COMMIT,
  importedAt: "2026-10-01T10:00:00Z",
};

function material(
  namespace: string,
  name: string,
  mediaType: string,
  extra: Partial<ArtifactMetadata> = {},
): ArtifactMetadata {
  return {
    artifact: { namespace, name, revision: `${name}-r1` },
    mediaType,
    size: 2048,
    current: true,
    frozen: false,
    createdAt: "2026-10-01T10:00:00Z",
    ...extra,
  };
}

function json(value: unknown, status = 200, etag?: string): Response {
  return new Response(status === 204 ? null : JSON.stringify(value), {
    status,
    headers: {
      "content-type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
      ...(etag === undefined ? {} : { ETag: etag }),
    },
  });
}

function renderRegion(
  handle: (request: Request) => Promise<Response> | Response,
  path = `/projects/${PROJECT}/artifacts`,
) {
  const requests: Request[] = [];
  const api = new PublicAPI(
    {
      uiVersion: "0.1.0",
      supportedApiVersions: ["contractor.public.v1"],
      apiBaseUrl: "http://127.0.0.1:8080",
    },
    vi.fn(async (input) => {
      const request = input instanceof Request ? input : new Request(input);
      requests.push(request.clone());
      return handle(request);
    }),
  );
  api.csrf.replace("a".repeat(43));
  const router = createMemoryRouter(
    [
      {
        path: "/projects/:projectId/artifacts",
        element: <ProjectArtifactRegion projectId={PROJECT} />,
      },
      { path: "*", element: <p>Elsewhere</p> },
    ],
    { initialEntries: [path] },
  );
  render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <PublicAPIProvider api={api}>
        <RouterProvider router={router} />
      </PublicAPIProvider>
    </QueryClientProvider>,
  );
  return { router, requests };
}

function listOf(items: ArtifactMetadata[]) {
  return json({ items, page: { hasMore: false } });
}

describe("Project materials", () => {
  it("groups materials by kind with their format, size, Git commit and lock", async () => {
    const { requests } = renderRegion((request) => {
      const url = new URL(request.url);
      if (url.pathname === LIST)
        return listOf([
          material("artifacts", "notes", "text/plain", { frozen: true }),
          material("docs", "readme", "text/markdown"),
          material("sources", "service", "application/zip", { gitSource }),
          material("openapi", "shop", "application/yaml"),
          material("outputs", "report", "text/markdown"),
        ]);
      throw new Error(`unexpected ${request.method} ${url}`);
    });

    const headings = await screen.findAllByRole("heading", { level: 3 });
    expect(headings.map((heading) => heading.textContent)).toEqual([
      "Source code",
      "API spec",
      "Docs",
      "Other",
    ]);
    const docs = within(screen.getByRole("region", { name: "Docs" }));
    expect(docs.getAllByRole("link").map((link) => link.textContent)).toEqual([
      "docs/readme",
      "outputs/report",
    ]);
    expect(docs.getByRole("link", { name: "docs/readme" })).toHaveAttribute(
      "href",
      `/projects/${PROJECT}/artifacts/docs/readme`,
    );
    const source = within(screen.getByRole("region", { name: "Source code" }));
    expect(source.getByText("ZIP")).toBeInTheDocument();
    expect(source.getByText("2.0 KiB")).toBeInTheDocument();
    expect(
      source.getByRole("button", { name: "Copy Git commit" }),
    ).toBeInTheDocument();
    expect(source.getByTitle(COMMIT)).toHaveTextContent("aaaaaaaa…aaaa");
    const other = within(screen.getByRole("region", { name: "Other" }));
    expect(other.getByText("Locked")).toBeInTheDocument();
    expect(new URL(requests[0]!.url).searchParams.has("namespace")).toBeFalsy();
  });

  it("keeps the namespace filter valid and in the URL, and resets the page", async () => {
    const { router, requests } = renderRegion((request) => {
      const url = new URL(request.url);
      if (url.pathname === LIST)
        return url.searchParams.get("namespace") === "docs"
          ? listOf([])
          : json({
              items: [material("docs", "readme", "text/markdown")],
              page: { hasMore: true, nextCursor: "page-2" },
            });
      throw new Error(`unexpected ${request.method} ${url}`);
    });
    const user = userEvent.setup();
    await screen.findByRole("link", { name: "docs/readme" });
    await user.click(screen.getByRole("button", { name: "Next" }));
    await waitFor(() =>
      expect(router.state.location.search).toBe("?artifactsCursor=page-2"),
    );

    const field = screen.getByPlaceholderText("all namespaces");
    await user.type(field, "bad name");
    await user.click(screen.getByRole("button", { name: "Apply" }));
    expect(screen.getByRole("alert")).toHaveTextContent(
      "A namespace uses 1–128 letters",
    );
    expect(router.state.location.search).toBe("?artifactsCursor=page-2");

    await user.clear(field);
    await user.type(field, "docs");
    await user.click(screen.getByRole("button", { name: "Apply" }));
    await waitFor(() =>
      expect(router.state.location.search).toBe("?artifactsNamespace=docs"),
    );
    expect(
      await screen.findByText("Nothing in the docs namespace"),
    ).toBeVisible();
    expect(
      requests.some(
        (request) =>
          new URL(request.url).searchParams.get("namespace") === "docs" &&
          !new URL(request.url).searchParams.has("cursor"),
      ),
    ).toBe(true);

    await user.click(
      screen.getByRole("button", { name: "Show all namespaces" }),
    );
    await waitFor(() => expect(router.state.location.search).toBe(""));
    expect(
      await screen.findByRole("link", { name: "docs/readme" }),
    ).toBeVisible();
  });

  it("opens the Add material sheet from ?add=artifact and keeps other filters", async () => {
    const { router } = renderRegion((request) => {
      if (new URL(request.url).pathname === LIST) return listOf([]);
      throw new Error(`unexpected ${request.url}`);
    }, `/projects/${PROJECT}/artifacts?add=artifact&artifactsNamespace=docs`);
    const user = userEvent.setup();

    const sheet = await screen.findByRole("dialog", { name: "Add material" });
    const kinds = within(
      within(sheet).getByRole("list", { name: "Kinds of material" }),
    ).getAllByRole("button");
    expect(kinds.map((kind) => kind.getAttribute("aria-label"))).toEqual([
      "Import Git repository",
      "Source code ZIP",
      "OpenAPI",
      "LikeC4",
      "Docs",
      "Diffs",
      "Other",
    ]);
    expect(
      within(sheet).getByRole("button", { name: "OpenAPI" }),
    ).toHaveAccessibleDescription("API spec in YAML or JSON");
    expect(
      within(sheet).getByRole("link", { name: "Git key in Settings" }),
    ).toHaveAttribute("href", "/operations/settings#repository-access");

    await user.keyboard("{Escape}");
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    );
    expect(router.state.location.search).toBe("?artifactsNamespace=docs");
    const add = screen.getByRole("button", { name: "Add material" });
    await waitFor(() => expect(add).toHaveFocus());

    await user.click(add);
    expect(router.state.location.search).toBe(
      "?artifactsNamespace=docs&add=artifact",
    );
    await user.click(
      within(screen.getByRole("dialog", { name: "Add material" })).getByRole(
        "button",
        { name: "Close material choices" },
      ),
    );
    await waitFor(() => expect(add).toHaveFocus());
    expect(router.state.location.search).toBe("?artifactsNamespace=docs");
  });

  it("uploads a material with its kind's suggestions and announces the next steps", async () => {
    let stored = false;
    const content = "openapi: 3.1.0";
    const created = material("openapi", "shop", "application/yaml", {
      size: content.length,
    });
    const { router, requests } = renderRegion(async (request) => {
      const url = new URL(request.url);
      if (url.pathname === LIST && request.method === "GET")
        return listOf(stored ? [created] : []);
      if (url.pathname === `${LIST}/openapi/shop` && request.method === "PUT") {
        stored = true;
        return json(
          {
            artifact: created.artifact,
            mediaType: created.mediaType,
            size: created.size,
          },
          201,
          `"${created.artifact.revision}"`,
        );
      }
      throw new Error(`unexpected ${request.method} ${url}`);
    }, `/projects/${PROJECT}/artifacts?add=artifact`);
    const user = userEvent.setup();

    const sheet = await screen.findByRole("dialog", { name: "Add material" });
    await user.click(within(sheet).getByRole("button", { name: "OpenAPI" }));
    const upload = screen.getByRole("dialog", { name: "OpenAPI" });
    expect(
      within(upload).getByRole("button", { name: "Close upload dialog" }),
    ).toHaveFocus();
    expect(within(upload).getByLabelText("Namespace")).toHaveValue("openapi");
    expect(within(upload).getByLabelText("Media type")).toHaveValue(
      "application/yaml",
    );
    await user.upload(
      within(upload).getByLabelText("Drop a file here"),
      new File([content], "shop.yaml", {
        type: "application/x-yaml",
      }),
    );
    expect(within(upload).getByLabelText("Name")).toHaveValue("shop");
    expect(within(upload).getByLabelText("Media type")).toHaveValue(
      "application/yaml",
    );
    await user.click(
      within(upload).getByRole("button", { name: "Create binding" }),
    );

    const notice = await screen.findByRole("status", {
      name: "Material added to this project",
    });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(notice).toHaveTextContent("openapi/shop");
    expect(notice).toHaveTextContent("API spec · YAML · 14 B");
    expect(
      within(notice).getByRole("button", { name: "Copy revision" }),
    ).toBeInTheDocument();
    expect(
      within(notice).getByRole("link", { name: "Open material" }),
    ).toHaveAttribute(
      "href",
      `/projects/${PROJECT}/artifacts/openapi/shop?revision=shop-r1`,
    );
    expect(
      within(notice).getByRole("link", { name: "Start a check" }),
    ).toHaveAttribute("href", `/checks/new?project=${PROJECT}`);
    expect(
      within(notice).getByRole("link", { name: "Run a workflow" }),
    ).toHaveAttribute("href", `/projects/${PROJECT}/workflows`);
    await waitFor(() => expect(notice).toHaveFocus());
    expect(router.state.location.search).toBe("");
    expect(
      await screen.findByRole("link", { name: "openapi/shop" }),
    ).toBeVisible();

    const put = requests.find((request) => request.method === "PUT")!;
    expect(put.headers.get("If-None-Match")).toBe("*");
    expect(put.headers.get("If-Match")).toBeNull();
    expect(put.headers.get("Content-Type")).toBe("application/yaml");

    await user.click(within(notice).getByRole("button", { name: "Dismiss" }));
    expect(screen.queryByRole("status", { name: /Material added/ })).toBeNull();
    expect(screen.getByRole("button", { name: "Add material" })).toHaveFocus();
  });

  it("explains a refused create and never retries it", async () => {
    const { requests } = renderRegion((request) => {
      const url = new URL(request.url);
      if (url.pathname === LIST && request.method === "GET") return listOf([]);
      if (request.method === "PUT")
        return json(
          {
            code: "conflict",
            message: "resource state changed",
            retryable: true,
            requestId: "request-conflict",
          },
          409,
        );
      throw new Error(`unexpected ${request.method} ${url}`);
    }, `/projects/${PROJECT}/artifacts?add=artifact`);
    const user = userEvent.setup();
    const sheet = await screen.findByRole("dialog", { name: "Add material" });
    const docsKind = within(sheet).getByRole("button", { name: "Docs" });
    await user.click(docsKind);
    const upload = screen.getByRole("dialog", { name: "Docs" });
    await user.upload(
      within(upload).getByLabelText("Drop a file here"),
      new File(["# Readme"], "readme.md", { type: "text/markdown" }),
    );
    await user.click(
      within(upload).getByRole("button", { name: "Create binding" }),
    );

    expect(await within(upload).findByRole("alert")).toHaveTextContent(
      /binding changed/,
    );
    expect(upload).toHaveTextContent("Nothing was replaced");
    expect(requests.filter((request) => request.method === "PUT")).toHaveLength(
      1,
    );
    await user.click(
      within(upload).getByRole("button", { name: "Close upload dialog" }),
    );
    await waitFor(() => expect(docsKind).toHaveFocus());
    expect(
      screen.getByRole("dialog", { name: "Add material" }),
    ).toBeInTheDocument();
  });

  it("imports a Git repository and shows the recorded commit", async () => {
    let imported = false;
    const result = material("sources", "source", "application/zip", {
      size: 123,
      gitSource,
    });
    const { requests } = renderRegion(async (request) => {
      const url = new URL(request.url);
      if (url.pathname === LIST) return listOf(imported ? [result] : []);
      if (url.pathname === `${LIST}/sources/source/metadata`)
        return json(
          { code: "not_found", message: "not found", retryable: false },
          404,
        );
      if (url.pathname === `${LIST}/sources/source/git-import`) {
        imported = true;
        return json(
          {
            artifact: result.artifact,
            mediaType: result.mediaType,
            size: result.size,
            gitSource,
          },
          201,
        );
      }
      throw new Error(`unexpected ${request.method} ${url}`);
    }, `/projects/${PROJECT}/artifacts?add=artifact`);
    const user = userEvent.setup();
    const sheet = await screen.findByRole("dialog", { name: "Add material" });
    await user.click(
      within(sheet).getByRole("button", { name: "Import Git repository" }),
    );
    const dialog = screen.getByRole("dialog", {
      name: "Import Git repository",
    });
    expect(within(dialog).getByLabelText("Repository URL")).toHaveFocus();
    await user.type(
      within(dialog).getByLabelText("Repository URL"),
      "https://example.test/repo.git",
    );
    await user.type(
      within(dialog).getByLabelText("Branch or tag (optional)"),
      "main",
    );
    await user.click(
      within(dialog).getByRole("button", { name: "Import snapshot" }),
    );

    const notice = await screen.findByRole("status", {
      name: "Git repository imported into this project",
    });
    expect(notice).toHaveTextContent("sources/source");
    expect(notice).toHaveTextContent("https://example.test:443/repo.git");
    expect(notice).toHaveTextContent("main");
    expect(
      within(notice).getByRole("button", { name: "Copy Git commit" }),
    ).toBeInTheDocument();
    expect(within(notice).getByTitle(COMMIT)).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    const post = requests.find((request) => request.method === "POST")!;
    expect(post.headers.get("If-None-Match")).toBe("*");
    expect(await post.json()).toEqual({
      repositoryUrl: "https://example.test/repo.git",
      ref: "main",
    });
    // The new material lists its commit as well.
    const row = await screen.findByRole("link", { name: "sources/source" });
    expect(
      within(row.closest("li")!).getByRole("button", {
        name: "Copy Git commit",
      }),
    ).toBeInTheDocument();
  });

  it("asks before replacing an existing binding and refuses a locked one", async () => {
    const existing = material("sources", "source", "application/zip", {
      artifact: {
        namespace: "sources",
        name: "source",
        revision: "rev-before",
      },
    });
    const locked = material("sources", "pinned", "application/zip", {
      artifact: {
        namespace: "sources",
        name: "pinned",
        revision: "rev-pinned",
      },
      frozen: true,
    });
    const { requests } = renderRegion((request) => {
      const url = new URL(request.url);
      if (url.pathname === LIST) return listOf([existing, locked]);
      if (url.pathname === `${LIST}/sources/source/metadata`)
        return json(existing);
      if (url.pathname === `${LIST}/sources/pinned/metadata`)
        return json(locked);
      if (url.pathname === `${LIST}/sources/source/git-import`)
        return json({
          artifact: { ...existing.artifact, revision: "rev-after" },
          mediaType: "application/zip",
          size: 123,
          gitSource,
        });
      throw new Error(`unexpected ${request.method} ${url}`);
    }, `/projects/${PROJECT}/artifacts?add=artifact`);
    const user = userEvent.setup();
    const sheet = await screen.findByRole("dialog", { name: "Add material" });
    await user.click(
      within(sheet).getByRole("button", { name: "Import Git repository" }),
    );
    const dialog = screen.getByRole("dialog", {
      name: "Import Git repository",
    });
    await user.type(
      within(dialog).getByLabelText("Repository URL"),
      "https://example.test/repo.git",
    );
    const name = within(dialog).getByLabelText("Name");
    await user.clear(name);
    await user.type(name, "pinned");
    const submit = within(dialog).getByRole("button", {
      name: "Import snapshot",
    });
    await user.click(submit);
    expect(
      await within(dialog).findByText(
        "It is locked and cannot be replaced. Choose another name.",
      ),
    ).toBeVisible();
    expect(within(dialog).getByText("rev-pinned")).toBeVisible();
    expect(within(dialog).queryByLabelText("Replace this revision")).toBeNull();
    expect(submit).toBeDisabled();

    await user.clear(name);
    await user.type(name, "source");
    expect(submit).toBeEnabled();
    await user.click(submit);
    expect(await within(dialog).findByText("rev-before")).toBeVisible();
    expect(submit).toBeDisabled();
    await user.click(within(dialog).getByLabelText("Replace this revision"));
    await user.click(submit);
    await screen.findByRole("status", {
      name: "Git repository imported into this project",
    });
    const posts = requests.filter((request) => request.method === "POST");
    expect(posts).toHaveLength(1);
    expect(posts[0]!.headers.get("If-Match")).toBe('"rev-before"');
    expect(posts[0]!.headers.get("If-None-Match")).toBeNull();
  });

  it("cancels a pending Git import and returns to its kind", async () => {
    let signal: AbortSignal | undefined;
    renderRegion((request) => {
      const url = new URL(request.url);
      if (url.pathname === LIST) return listOf([]);
      if (url.pathname.endsWith("/metadata"))
        return json(
          { code: "not_found", message: "not found", retryable: false },
          404,
        );
      if (url.pathname.endsWith("/git-import")) {
        signal = request.signal;
        return new Promise<Response>((_resolve, reject) =>
          request.signal.addEventListener("abort", () =>
            reject(new DOMException("aborted", "AbortError")),
          ),
        );
      }
      throw new Error(`unexpected ${request.method} ${url}`);
    }, `/projects/${PROJECT}/artifacts?add=artifact`);
    const user = userEvent.setup();
    const sheet = await screen.findByRole("dialog", { name: "Add material" });
    const gitKind = within(sheet).getByRole("button", {
      name: "Import Git repository",
    });
    await user.click(gitKind);
    const dialog = screen.getByRole("dialog", {
      name: "Import Git repository",
    });
    await user.type(
      within(dialog).getByLabelText("Repository URL"),
      "https://example.test/repo.git",
    );
    await user.click(
      within(dialog).getByRole("button", { name: "Import snapshot" }),
    );
    await waitFor(() => expect(signal).toBeDefined());
    await act(async () => {
      await user.click(
        within(dialog).getByRole("button", { name: "Cancel import" }),
      );
    });
    expect(signal?.aborted).toBe(true);
    expect(dialog).not.toBeInTheDocument();
    expect(
      screen.getByRole("dialog", { name: "Add material" }),
    ).toBeInTheDocument();
    await waitFor(() => expect(gitKind).toHaveFocus());
  });
});
