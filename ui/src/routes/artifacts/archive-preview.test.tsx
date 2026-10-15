import { screen, within } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import { conforming } from "../../test/builders";
import { fakeServer, json, type FakeCall } from "../../test/fake-server";
import { renderWithAPI } from "../../test/render";
import { ArtifactPreviewPanel } from "./preview";
import { artifactMetadata } from "./test-support";

const metadata = artifactMetadata("skills", "example", {
  mediaType: "application/vnd.contractor.agent-skill+zip",
  size: 500000,
  current: false,
  frozen: true,
});

const files: Record<string, string> = {
  "SKILL.md":
    "# Skill guide\n\n![tracker](https://invalid.example/track)\n\n<script>alert(1)</script>",
  "assets/page.html":
    '<img src="https://invalid.example/x" onerror="alert(1)">',
};

function exactRef({ url }: FakeCall) {
  return { ...metadata.artifact, revision: url.searchParams.get("revision")! };
}

/** The panel at r1, and a button that moves it to the revision r2. */
function Panel({ textPreview }: { textPreview: () => Promise<string> }) {
  const [revision, setRevision] = useState("r1");
  return (
    <>
      <button type="button" onClick={() => setRevision("r2")}>
        Show r2
      </button>
      <ArtifactPreviewPanel
        metadata={{ ...metadata, artifact: { ...metadata.artifact, revision } }}
        archiveScope={{ kind: "user" }}
        loadPreview={textPreview}
        unavailableCopy="Download only"
      />
    </>
  );
}

describe("Archive preview", () => {
  it("loads on demand, opens SKILL.md, browses folders and treats active files as text", async () => {
    const server = fakeServer({
      "GET /v1/artifacts/{namespace}/{name}/archive": (call) =>
        json(
          conforming("ArtifactArchive", {
            artifact: exactRef(call),
            entries: [
              {
                path: "SKILL.md",
                kind: "file",
                size: files["SKILL.md"]!.length,
                previewable: true,
              },
              {
                path: "assets",
                kind: "directory",
                size: 0,
                previewable: false,
              },
              {
                path: "assets/page.html",
                kind: "file",
                size: files["assets/page.html"]!.length,
                previewable: true,
              },
              {
                path: "assets/large.txt",
                kind: "file",
                size: 300000,
                previewable: false,
              },
            ],
          }),
          200,
          { ETag: `"${exactRef(call).revision}"` },
        ),
      "GET /v1/artifacts/{namespace}/{name}/archive/file": (call) => {
        const path = call.url.searchParams.get("path")!;
        const text = files[path] ?? "# New revision";
        return json(
          { artifact: exactRef(call), path, size: text.length, text },
          200,
          { ETag: `"${exactRef(call).revision}"` },
        );
      },
    });
    const textPreview = vi.fn();
    const { user, sent } = renderWithAPI(<Panel textPreview={textPreview} />, {
      server,
    });
    const reads = () => [
      ...sent("GET", "/archive"),
      ...sent("GET", "/archive/file"),
    ];
    expect(reads()).toHaveLength(0);
    await user.click(screen.getByRole("button", { name: "Browse files" }));
    expect(
      await screen.findByRole("heading", { name: "Skill guide" }),
    ).toBeInTheDocument();
    expect(document.querySelector(".archive-browser img")).toBeNull();
    expect(document.querySelector(".archive-browser script")).toBeNull();
    const nav = screen.getByRole("navigation", { name: "Archive files" });
    await user.click(within(nav).getByText("assets/"));
    await user.click(within(nav).getByRole("button", { name: /page.html/ }));
    expect(
      await screen.findByText(files["assets/page.html"]!),
    ).toBeInTheDocument();
    expect(document.querySelector(".archive-browser img")).toBeNull();
    await user.click(within(nav).getByRole("button", { name: /large.txt/ }));
    expect(screen.getByText(/This file exceeds/)).toBeInTheDocument();
    const urls = () => reads().map((request) => new URL(request.url));
    expect(
      urls().some((url) => url.searchParams.get("path") === "assets/large.txt"),
    ).toBe(false);
    expect(
      urls().every((url) => url.searchParams.get("revision") === "r1"),
    ).toBe(true);
    expect(textPreview).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Show r2" }));
    expect(
      screen.queryByRole("navigation", { name: "Archive files" }),
    ).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Browse files" }));
    await screen.findByRole("heading", { name: "Skill guide" });
    expect(
      new URL(server.requests.at(-1)!.url).searchParams.get("revision"),
    ).toBe("r2");
  });
});
