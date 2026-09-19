import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { PublicAPI } from "../../api/client";
import { PublicAPIProvider } from "../../api/context";
import type { ArtifactMetadata } from "../../api/artifacts";
import { ArtifactPreviewPanel } from "./preview";

const metadata: ArtifactMetadata = {
  artifact: { namespace: "skills", name: "example", revision: "r1" },
  mediaType: "application/vnd.contractor.agent-skill+zip",
  size: 500000,
  current: false,
  frozen: true,
  createdAt: "2026-09-19T10:00:00Z",
};

describe("Archive preview", () => {
  it("loads on demand, opens SKILL.md, browses folders and treats active files as text", async () => {
    const paths: URL[] = [];
    const files: Record<string, string> = {
      "SKILL.md":
        "# Skill guide\n\n![tracker](https://invalid.example/track)\n\n<script>alert(1)</script>",
      "assets/page.html":
        '<img src="https://invalid.example/x" onerror="alert(1)">',
    };
    const api = new PublicAPI(
      {
        uiVersion: "0.1.0",
        supportedApiVersions: ["contractor.public.v1"],
        apiBaseUrl: "http://127.0.0.1:8080",
      },
      vi.fn(async (input) => {
        const url = new URL((input as Request).url);
        paths.push(url);
        const path = url.searchParams.get("path");
        const text = files[path ?? ""] ?? "# New revision";
        const artifact = {
          ...metadata.artifact,
          revision: url.searchParams.get("revision")!,
        };
        return new Response(
          JSON.stringify(
            path === null
              ? {
                  artifact,
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
                }
              : { artifact, path, size: text.length, text },
          ),
          {
            headers: {
              "Content-Type": "application/json",
              ETag: `"${artifact.revision}"`,
              "X-Contractor-API-Version": "contractor.public.v1",
            },
          },
        );
      }),
    );
    const client = new QueryClient();
    const textPreview = vi.fn();
    const panel = (revision = "r1") => (
      <QueryClientProvider client={client}>
        <PublicAPIProvider api={api}>
          <ArtifactPreviewPanel
            metadata={{
              ...metadata,
              artifact: { ...metadata.artifact, revision },
            }}
            archiveScope={{ kind: "user" }}
            loadPreview={textPreview}
            unavailableCopy="Download only"
          />
        </PublicAPIProvider>
      </QueryClientProvider>
    );
    const view = render(panel());
    const user = userEvent.setup();
    expect(paths).toHaveLength(0);
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
    expect(
      paths.some((url) => url.searchParams.get("path") === "assets/large.txt"),
    ).toBe(false);
    expect(
      paths.every((url) => url.searchParams.get("revision") === "r1"),
    ).toBe(true);
    expect(textPreview).not.toHaveBeenCalled();

    view.rerender(panel("r2"));
    expect(
      screen.queryByRole("navigation", { name: "Archive files" }),
    ).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Browse files" }));
    await screen.findByRole("heading", { name: "Skill guide" });
    expect(paths.at(-1)?.searchParams.get("revision")).toBe("r2");
  });
});
