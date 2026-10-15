import { screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ArtifactMetadata } from "../../api/artifacts";
import { artifactMetadata, conforming } from "../../test/builders";
import {
  bytes,
  failure,
  fakeServer,
  json,
  page,
  type Routes,
} from "../../test/fake-server";
import { renderWithAPI } from "../../test/render";
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
  return artifactMetadata(namespace, name, {
    artifact: { namespace, name, revision: "rev-2" },
    mediaType,
    createdAt: "2026-10-01T10:00:00Z",
    ...extra,
  });
}

function renderAt(path: string, routes: Routes) {
  return renderWithAPI(null, {
    path,
    server: fakeServer(routes),
    routes: [
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
    ],
  });
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
    const { requests, user } = renderAt(
      "/projects/project-1/artifacts/sources/service",
      {
        "GET /v1/projects/{projectId}/artifacts/{namespace}/{name}/metadata":
          () => json(material),
        "GET /v1/projects/{projectId}/artifacts/{namespace}/{name}/versions":
          () =>
            json(
              page([
                material,
                {
                  ...material,
                  artifact: { ...material.artifact, revision: "rev-1" },
                  current: false,
                },
              ]),
            ),
        "GET /v1/projects/{projectId}/artifacts/{namespace}/{name}/lineage":
          () =>
            json(
              page([
                conforming("ArtifactLineageEdge", {
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
                }),
              ]),
            ),
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
    await user.click(screen.getByRole("button", { name: "Versions" }));
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
    const { router, user } = renderAt(
      "/projects/project-1/artifacts/docs/readme?revision=rev-2",
      {
        "GET /v1/projects/{projectId}/artifacts/{namespace}/{name}/metadata": ({
          url,
        }) =>
          json(
            url.searchParams.get("revision") === "rev-2" ? historical : current,
          ),
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
    await user.click(
      screen.getByRole("link", { name: "Open the current version" }),
    );
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
      const { user } = renderAt("/artifacts/notes/item", {
        "GET /v1/artifacts/{namespace}/{name}/metadata": () =>
          json(
            metadataOf("notes", "item", mediaType, {
              size: new TextEncoder().encode(source).byteLength,
            }),
          ),
        "GET /v1/artifacts/{namespace}/{name}": () => bytes(source, mediaType),
      });
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
    const { user } = renderAt("/artifacts/scans/ffuf", {
      "GET /v1/artifacts/{namespace}/{name}/metadata": () =>
        json(
          metadataOf("scans", "ffuf", "application/json", {
            size: report.length,
          }),
        ),
      "GET /v1/artifacts/{namespace}/{name}": () =>
        bytes(report, "application/json"),
    });
    await user.click(
      await screen.findByRole("button", { name: "Load preview" }),
    );
    expect(
      await screen.findByRole("region", { name: "Scanner execution" }),
    ).toHaveTextContent("Completed");
    expect(screen.queryByRole("tablist")).toBeNull();
    expect(document.querySelector("pre.artifact-preview")).toHaveTextContent(
      '"tool":"scan_ffuf"',
    );
  });

  it("returns Skill packages to Skills and names them", async () => {
    renderAt("/artifacts/skills/review", {
      "GET /v1/artifacts/{namespace}/{name}/metadata": () =>
        json(metadataOf("skills", "review", SKILL_ARCHIVE_MEDIA_TYPE)),
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
    const { sent, user } = renderAt("/artifacts/projects/notes", {
      "GET /v1/artifacts/{namespace}/{name}/metadata": () => json(file),
      "GET /v1/artifacts/{namespace}/{name}": () =>
        fail
          ? failure(
              503,
              "artifact_unavailable",
              "Artifact bytes are unavailable",
              true,
            )
          : bytes("hello", "text/plain"),
    });
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});
    const originalCreate = URL.createObjectURL;
    const originalRevoke = URL.revokeObjectURL;
    URL.createObjectURL = vi.fn(() => "blob:notes");
    URL.revokeObjectURL = vi.fn();
    try {
      await user.click(
        await screen.findByRole("button", { name: "Download this revision" }),
      );
      await waitFor(() => expect(click).toHaveBeenCalledOnce());
      const download = sent("GET", "/v1/artifacts/projects/notes")[0]!;
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
    renderAt("/runs/run-1/artifacts/outputs/report", {
      "GET /v1/runs/{runId}/artifacts/{namespace}/{name}/metadata": () =>
        json(metadataOf("outputs", "report", "text/plain", { frozen: true })),
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
