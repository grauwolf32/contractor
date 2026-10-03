import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter, MemoryRouter, RouterProvider } from "react-router";
import { describe, expect, it, vi } from "vitest";

import { PublicAPI } from "../../api/client";
import { PublicAPIProvider } from "../../api/context";
import {
  MAXIMUM_SKILL_ARCHIVE_BYTES,
  SKILL_ARCHIVE_MEDIA_TYPE,
} from "./artifact-file";
import { ArtifactWriteForm } from "./common";
import { ArtifactDetailRoute } from "./detail";

const runtimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

function jsonResponse(value: unknown, status = 200, etag?: string): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "content-type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
      ...(etag === undefined ? {} : { ETag: etag }),
    },
  });
}

describe("Artifact version media types", () => {
  it.each(["user", "project"])(
    "keeps the %s binding's semantic type when a file has a generic browser MIME type",
    async (scope) => {
      const mediaType = "application/vnd.contractor.workspace-overlay+json";
      const artifact = {
        namespace: "overlays",
        name: "baseline",
        revision: "revision-2",
      };
      const writes: Request[] = [];
      const onWritten = vi.fn();
      const api = new PublicAPI(runtimeConfig, async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        writes.push(request);
        return jsonResponse(
          { artifact, mediaType, size: 2 },
          201,
          '"revision-2"',
        );
      });
      api.csrf.replace("a".repeat(43));
      render(
        <QueryClientProvider client={new QueryClient()}>
          <PublicAPIProvider api={api}>
            <MemoryRouter>
              {scope === "user" ? (
                <ArtifactWriteForm
                  fixedIdentity={{ namespace: "overlays", name: "baseline" }}
                  initialMediaType={mediaType}
                  expectedRevision="revision-1"
                  onWritten={onWritten}
                />
              ) : (
                <ArtifactWriteForm
                  scope={{ kind: "project", id: "project_example" }}
                  fixedIdentity={{ namespace: "overlays", name: "baseline" }}
                  initialMediaType={mediaType}
                  expectedRevision="revision-1"
                  onWritten={onWritten}
                />
              )}
            </MemoryRouter>
          </PublicAPIProvider>
        </QueryClientProvider>,
      );
      const user = userEvent.setup();
      expect(screen.getByLabelText("Media type")).toHaveValue(mediaType);
      await user.upload(
        screen.getByLabelText("Drop a file here"),
        new File(["{}"], "baseline.json", { type: "application/json" }),
      );
      expect(screen.getByLabelText("Media type")).toHaveValue(mediaType);
      await user.click(
        screen.getByRole("button", { name: "Upload new version" }),
      );
      await waitFor(() => expect(onWritten).toHaveBeenCalledOnce());
      expect(writes).toHaveLength(1);
      expect(writes[0]?.headers.get("Content-Type")).toBe(mediaType);
      expect(writes[0]?.headers.get("If-Match")).toBe('"revision-1"');
    },
  );

  it.each(["user", "project"])(
    "preserves a newly selected %s format when a file is chosen later",
    async (scope) => {
      const api = new PublicAPI(runtimeConfig, async () => {
        throw new Error("no upload expected");
      });
      render(
        <QueryClientProvider client={new QueryClient()}>
          <PublicAPIProvider api={api}>
            <MemoryRouter>
              {scope === "user" ? (
                <ArtifactWriteForm onWritten={vi.fn()} />
              ) : (
                <ArtifactWriteForm
                  scope={{ kind: "project", id: "project_example" }}
                  onWritten={vi.fn()}
                />
              )}
            </MemoryRouter>
          </PublicAPIProvider>
        </QueryClientProvider>,
      );
      const user = userEvent.setup();
      await user.selectOptions(
        screen.getByLabelText("File format"),
        "application/json",
      );
      await user.upload(
        screen.getByLabelText("Drop a file here"),
        new File(["notes"], "notes.txt", { type: "text/plain" }),
      );
      expect(screen.getByLabelText("Media type")).toHaveValue(
        "application/json",
      );
    },
  );

  it("updates a Skill from its detail page with the canonical type and 16 MiB limit", async () => {
    const artifact = {
      namespace: "skills",
      name: "review",
      revision: "revision-1",
    };
    const metadata = {
      artifact,
      mediaType: SKILL_ARCHIVE_MEDIA_TYPE,
      size: 3,
      current: true,
      frozen: false,
      createdAt: "2026-10-02T10:00:00Z",
    };
    const writes: Request[] = [];
    const api = new PublicAPI(runtimeConfig, async (input) => {
      const request = input instanceof Request ? input : new Request(input);
      const url = new URL(request.url);
      if (request.method === "PUT") {
        writes.push(request);
        return jsonResponse(
          {
            artifact: { ...artifact, revision: "revision-2" },
            mediaType: SKILL_ARCHIVE_MEDIA_TYPE,
            size: 3,
          },
          201,
          '"revision-2"',
        );
      }
      if (url.pathname.endsWith("/metadata")) return jsonResponse(metadata);
      return jsonResponse({ items: [], page: { hasMore: false } });
    });
    api.csrf.replace("a".repeat(43));
    const router = createMemoryRouter(
      [
        {
          path: "/artifacts/:namespace/:name",
          element: <ArtifactDetailRoute />,
        },
      ],
      { initialEntries: ["/artifacts/skills/review"] },
    );
    render(
      <QueryClientProvider client={new QueryClient()}>
        <PublicAPIProvider api={api}>
          <RouterProvider router={router} />
        </PublicAPIProvider>
      </QueryClientProvider>,
    );
    const user = userEvent.setup();
    await user.click(
      await screen.findByText("Upload a new version", { selector: "summary" }),
    );
    expect(screen.getByLabelText("Media type")).toHaveValue(
      SKILL_ARCHIVE_MEDIA_TYPE,
    );
    expect(screen.getByLabelText("Media type")).toBeDisabled();
    const oversized = new File(["zip"], "review.zip", {
      type: "application/zip",
    });
    Object.defineProperty(oversized, "size", {
      value: MAXIMUM_SKILL_ARCHIVE_BYTES + 1,
    });
    await user.upload(screen.getByLabelText("Drop a file here"), oversized);
    await user.click(
      screen.getByRole("button", { name: "Upload new version" }),
    );
    expect(screen.getByRole("alert")).toHaveTextContent("16 MiB upload limit");
    expect(writes).toHaveLength(0);

    await user.upload(
      screen.getByLabelText("review.zip"),
      new File(["zip"], "review.zip", { type: "application/zip" }),
    );
    await user.click(
      screen.getByRole("button", { name: "Upload new version" }),
    );
    await waitFor(() => expect(writes).toHaveLength(1));
    expect(writes[0]?.headers.get("Content-Type")).toBe(
      SKILL_ARCHIVE_MEDIA_TYPE,
    );
    expect(writes[0]?.headers.get("If-Match")).toBe('"revision-1"');
  });
});
