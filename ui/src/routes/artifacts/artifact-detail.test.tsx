import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter, RouterProvider } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ArtifactMetadata } from "../../api/artifacts";
import { PublicAPI } from "../../api/client";
import { PublicAPIProvider } from "../../api/context";
import { ProjectArtifactDetailRoute } from "../projects/artifact-detail";
import { SKILL_ARCHIVE_MEDIA_TYPE } from "./artifact-file";
import { ArtifactDetailView } from "./artifact-detail-view";
import { ArtifactDetailRoute } from "./detail";

const COMMIT = "b".repeat(40);

function metadataOf(
  namespace: string,
  name: string,
  mediaType: string,
  extra: Partial<ArtifactMetadata> = {},
): ArtifactMetadata {
  return {
    artifact: { namespace, name, revision: "rev-2" },
    mediaType,
    size: 64,
    current: true,
    frozen: false,
    createdAt: "2026-10-01T10:00:00Z",
    ...extra,
  };
}

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "content-type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
    },
  });
}

function bytes(body: string, mediaType: string): Response {
  return new Response(body, {
    headers: {
      "content-type": mediaType,
      "content-length": String(new TextEncoder().encode(body).byteLength),
      "X-Contractor-API-Version": "contractor.public.v1",
    },
  });
}

function renderAt(
  path: string,
  handle: (request: Request) => Promise<Response> | Response,
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
      requests.push(request);
      return handle(request);
    }),
  );
  api.csrf.replace("a".repeat(43));
  const router = createMemoryRouter(
    [
      {
        path: "/projects/:projectId/artifacts/:namespace/:name",
        element: <ProjectArtifactDetailRoute />,
      },
      { path: "/artifacts/:namespace/:name", element: <ArtifactDetailRoute /> },
      {
        path: "/runs/:runId/artifacts/:namespace/:name",
        element: (
          <ArtifactDetailView
            scope={{ kind: "run", id: "run-1" }}
            namespace="outputs"
            name="report"
            revision={undefined}
            heading={<a href="/runs/run-1">← Run</a>}
          />
        ),
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

afterEach(() => {
  document.title = "";
});

describe("Material detail", () => {
  it("shows the kind, state, facts with the Git commit, and keeps history behind Technical details", async () => {
    const material = metadataOf("sources", "service", "application/zip", {
      frozen: false,
      gitSource: {
        repositoryUrl: "https://example.test:443/team/service.git",
        requestedRef: null,
        resolvedCommit: COMMIT,
        importedAt: "2026-10-01T09:59:00Z",
      },
    });
    const { requests } = renderAt(
      "/projects/project-1/artifacts/sources/service",
      (request) => {
        const path = new URL(request.url).pathname;
        if (path.endsWith("/metadata")) return json(material);
        if (path.endsWith("/versions"))
          return json({
            items: [
              material,
              {
                ...material,
                artifact: { ...material.artifact, revision: "rev-1" },
                current: false,
              },
            ],
            page: { hasMore: false },
          });
        if (path.endsWith("/lineage"))
          return json({
            items: [
              {
                kind: "input_fork",
                sourceScope: "project",
                source: material.artifact,
                targetScope: "run",
                target: {
                  namespace: "inputs",
                  name: "source",
                  revision: "run-rev",
                },
                runId: "run-7",
                createdAt: "2026-10-01T10:05:00Z",
              },
            ],
            page: { hasMore: false },
          });
        throw new Error(`unexpected ${request.method} ${path}`);
      },
    );

    expect(
      await screen.findByRole("heading", {
        level: 1,
        name: "sources/service",
      }),
    ).toBeVisible();
    expect(
      await screen.findByText("Source code", { selector: ".materials-kicker" }),
    ).toBeVisible();
    expect(
      screen.getByText("Current revision", { selector: ".lede" }),
    ).toBeVisible();
    expect(document.title).toBe("sources/service · Materials · Contractor");
    expect(screen.getByRole("link", { name: "← Materials" })).toHaveAttribute(
      "href",
      "/projects/project-1/artifacts",
    );

    const facts = screen.getByText("Format").closest("dl")!;
    expect(within(facts).getByText("ZIP")).toBeVisible();
    expect(within(facts).getByText("64 B")).toBeVisible();
    expect(within(facts).getByText("no")).toBeVisible();
    expect(
      within(facts).getByText("https://example.test:443/team/service.git"),
    ).toBeVisible();
    expect(within(facts).getByText("Default branch")).toBeVisible();
    expect(
      within(facts).getByRole("button", { name: "Copy Git commit" }),
    ).toBeVisible();
    expect(within(facts).getByTitle(COMMIT)).toHaveTextContent("bbbbbbbb…bbbb");

    expect(
      screen.getByRole("button", { name: "Download this revision" }),
    ).toBeEnabled();
    expect(
      screen.getByText("Upload a new version", { selector: "summary" }),
    ).toBeVisible();
    // ZIP archives open in the archive browser.
    expect(
      screen.getByRole("heading", { level: 2, name: "Archive contents" }),
    ).toBeVisible();
    expect(screen.getByRole("button", { name: "Browse files" })).toBeEnabled();

    // Revisions stay behind Technical details; history loads on demand.
    const revision = screen.getByText("rev-2", {
      selector: ".materials-tech-facts code",
    });
    expect(revision).not.toBeVisible();
    const historyReads = () =>
      requests.filter((request) =>
        /\/(versions|lineage)$/.test(new URL(request.url).pathname),
      );
    expect(historyReads()).toHaveLength(0);
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Versions" }));
    expect(await screen.findByText("input fork")).toBeVisible();
    expect(revision).toBeVisible();
    expect(screen.getByRole("link", { name: "Run run-7" })).toHaveAttribute(
      "href",
      "/runs/run-7",
    );
    // Versions and Lineage sit beside Archive contents in the outline.
    expect(
      screen.getByRole("heading", { level: 2, name: "Lineage" }),
    ).toBeVisible();
    const versions = screen.getByRole("heading", {
      level: 2,
      name: "Versions",
    }).parentElement!;
    expect(
      within(versions).getByRole("link", { name: /rev-2/ }),
    ).toHaveAttribute("aria-current", "page");
    expect(
      within(versions).getByRole("link", { name: /rev-1/ }),
    ).toHaveAttribute(
      "href",
      "/projects/project-1/artifacts/sources/service?revision=rev-1",
    );
    expect(historyReads()).toHaveLength(2);
  });

  it("keeps historical revisions read-only with a way back to the current one", async () => {
    const historical = metadataOf("docs", "readme", "text/markdown", {
      current: false,
    });
    const current = {
      ...historical,
      artifact: { ...historical.artifact, revision: "rev-3" },
      current: true,
    };
    const { router } = renderAt(
      "/projects/project-1/artifacts/docs/readme?revision=rev-2",
      (request) => {
        const url = new URL(request.url);
        if (url.pathname.endsWith("/metadata"))
          return json(
            url.searchParams.get("revision") === "rev-2" ? historical : current,
          );
        throw new Error(`unexpected ${request.method} ${url}`);
      },
    );
    expect(
      await screen.findByText("Historical revision", { selector: ".lede" }),
    ).toBeVisible();
    expect(
      screen.getByText("Historical revisions are read-only."),
    ).toBeVisible();
    expect(
      screen.queryByText("Upload a new version", { selector: "summary" }),
    ).toBeNull();
    await userEvent
      .setup()
      .click(screen.getByRole("link", { name: "Open the current version" }));
    await waitFor(() => expect(router.state.location.search).toBe(""));
    expect(
      await screen.findByText("Current revision", { selector: ".lede" }),
    ).toBeVisible();
    expect(
      screen.getByText("Upload a new version", { selector: "summary" }),
    ).toBeVisible();
  });

  it.each([
    {
      mediaType: "text/markdown",
      source: "# Release notes\n\nAll **good**.",
      renderer: "Markdown",
      check: () => screen.findByRole("heading", { name: "Release notes" }),
    },
    {
      mediaType: "text/x-diff",
      source: "--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-old\n+new\n",
      renderer: "Diff",
      check: () => screen.findByText("1 file changed"),
    },
  ])(
    "previews $mediaType with its renderer and keeps the source one tab away",
    async ({ mediaType, source, renderer, check }) => {
      renderAt("/artifacts/notes/item", (request) => {
        const url = new URL(request.url);
        if (url.pathname.endsWith("/metadata"))
          return json(
            metadataOf("notes", "item", mediaType, {
              size: new TextEncoder().encode(source).byteLength,
            }),
          );
        if (url.pathname === "/v1/artifacts/notes/item")
          return bytes(source, mediaType);
        throw new Error(`unexpected ${request.method} ${url}`);
      });
      const user = userEvent.setup();
      await user.click(
        await screen.findByRole("button", { name: "Load preview" }),
      );
      await check();
      const tabs = screen.getByRole("tablist", { name: "Preview mode" });
      expect(tabs).toHaveTextContent(renderer);
      expect(
        within(tabs).getByRole("tab", { name: "Rendered" }),
      ).toHaveAttribute("aria-selected", "true");
      await user.click(within(tabs).getByRole("tab", { name: "Source" }));
      expect(
        screen.getByRole("tabpanel", { name: "Source" }),
      ).toHaveTextContent(source.split("\n")[0]!);
    },
  );

  it("previews plain text and scanner reports without tabs", async () => {
    const report = JSON.stringify({
      schemaVersion: 1,
      tool: "scan_ffuf",
      inputDigest: `sha256:${"c".repeat(64)}`,
      inputArtifacts: {},
      observation: {
        status: "completed",
        exitCode: 0,
        errorCode: null,
        stdoutTruncated: false,
        stderrTruncated: false,
        outputLimitExceeded: false,
        scanComplete: true,
        results: [],
      },
    });
    renderAt("/artifacts/scans/ffuf", (request) => {
      const url = new URL(request.url);
      if (url.pathname.endsWith("/metadata"))
        return json(
          metadataOf("scans", "ffuf", "application/json", {
            size: report.length,
          }),
        );
      if (url.pathname === "/v1/artifacts/scans/ffuf")
        return bytes(report, "application/json");
      throw new Error(`unexpected ${request.method} ${url}`);
    });
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Load preview" }));
    expect(
      await screen.findByRole("region", { name: "Scanner execution" }),
    ).toHaveTextContent("Completed");
    expect(screen.queryByRole("tablist")).toBeNull();
    expect(document.querySelector("pre.artifact-preview")).toHaveTextContent(
      '"tool":"scan_ffuf"',
    );
  });

  it("returns Skill packages to Skills and names them", async () => {
    renderAt("/artifacts/skills/review", (request) => {
      const url = new URL(request.url);
      if (url.pathname.endsWith("/metadata"))
        return json(metadataOf("skills", "review", SKILL_ARCHIVE_MEDIA_TYPE));
      throw new Error(`unexpected ${request.method} ${url}`);
    });
    expect(
      await screen.findByText("Skill package", {
        selector: ".materials-kicker",
      }),
    ).toBeVisible();
    expect(screen.getByRole("link", { name: "← Skills" })).toHaveAttribute(
      "href",
      "/catalog/skills",
    );
    expect(document.title).toBe("skills/review · Skills · Contractor");
    expect(screen.getByRole("button", { name: "Browse files" })).toBeEnabled();
  });

  it("downloads the exact revision and explains a failed download", async () => {
    let fail = false;
    const file = metadataOf("projects", "notes", "text/plain", { size: 5 });
    const { requests } = renderAt("/artifacts/projects/notes", (request) => {
      const url = new URL(request.url);
      if (url.pathname.endsWith("/metadata")) return json(file);
      if (url.pathname === "/v1/artifacts/projects/notes")
        return fail
          ? json(
              {
                code: "artifact_unavailable",
                message: "Artifact bytes are unavailable",
                retryable: true,
              },
              503,
            )
          : bytes("hello", "text/plain");
      throw new Error(`unexpected ${request.method} ${url}`);
    });
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});
    const originalCreate = URL.createObjectURL;
    const originalRevoke = URL.revokeObjectURL;
    URL.createObjectURL = vi.fn(() => "blob:notes");
    URL.revokeObjectURL = vi.fn();
    try {
      const user = userEvent.setup();
      await user.click(
        await screen.findByRole("button", { name: "Download this revision" }),
      );
      await waitFor(() => expect(click).toHaveBeenCalledOnce());
      const download = requests.find(
        (request) =>
          new URL(request.url).pathname === "/v1/artifacts/projects/notes",
      )!;
      expect(new URL(download.url).searchParams.get("revision")).toBe("rev-2");

      fail = true;
      await user.click(
        screen.getByRole("button", { name: "Download this revision" }),
      );
      expect(await screen.findByRole("alert")).toHaveTextContent(
        "The download did not finish",
      );
    } finally {
      click.mockRestore();
      URL.createObjectURL = originalCreate;
      URL.revokeObjectURL = originalRevoke;
    }
  });

  it("keeps Run files read-only", async () => {
    renderAt("/runs/run-1/artifacts/outputs/report", (request) => {
      const url = new URL(request.url);
      if (url.pathname.endsWith("/metadata"))
        return json(
          metadataOf("outputs", "report", "text/plain", { frozen: true }),
        );
      throw new Error(`unexpected ${request.method} ${url}`);
    });
    expect(
      await screen.findByRole("heading", { level: 1, name: "outputs/report" }),
    ).toBeVisible();
    expect(await screen.findByText("yes")).toBeVisible();
    expect(
      screen.queryByText("Upload a new version", { selector: "summary" }),
    ).toBeNull();
    expect(
      screen.queryByText("Historical revisions are read-only."),
    ).toBeNull();
    // Run files show no kind kicker unless the page asks for it.
    expect(document.querySelector(".materials-kicker")).toBeNull();
  });
});
