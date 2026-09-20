import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { PublicAPI } from "../../api/client";
import { PublicAPIProvider } from "../../api/context";
import type { RuntimeCredentialMetadata } from "../../api/operations";
import type { Project } from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import { ProjectHTTPTargetEditor } from "./http-target-editor";

const project: Project = {
  projectId: "project_example",
  kind: "project",
  name: "Payment service",
  description: "",
  lifecycle: "active",
  revision: "1",
  createdAt: "2026-09-20T10:00:00Z",
  updatedAt: "2026-09-20T10:00:00Z",
};

function credential(
  credentialId: string,
  kind: RuntimeCredentialMetadata["kind"] = "http-origin-bearer@1",
): RuntimeCredentialMetadata {
  return {
    credentialId,
    kind,
    createdBy: "user_local",
    createdAt: "2026-09-20T10:00:00Z",
  };
}

function response(value: unknown, status = 200, headers: HeadersInit = {}) {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "Content-Type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
      ...headers,
    },
  });
}

function fixture(
  fetcher: (request: Request) => Promise<Response>,
  currentProject = project,
) {
  const api = new PublicAPI(
    {
      uiVersion: "0.1.0",
      supportedApiVersions: ["contractor.public.v1"],
      apiBaseUrl: "http://127.0.0.1:8080",
    },
    vi.fn(async (input) => fetcher(input as Request)),
  );
  api.csrf.replace("a".repeat(43));
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const pageSnapshot = {
    items: [credential("cached-page")],
    page: { hasMore: false },
  };
  queryClient.setQueryData(
    queryKeys.operations.runtimeCredentials.list(),
    pageSnapshot,
  );
  render(
    <QueryClientProvider client={queryClient}>
      <PublicAPIProvider api={api}>
        <ProjectHTTPTargetEditor project={currentProject} />
      </PublicAPIProvider>
    </QueryClientProvider>,
  );
  return { user: userEvent.setup(), queryClient, pageSnapshot };
}

describe("Project HTTP credential picker", () => {
  it("selects a later-page origin credential without replacing the page cache", async () => {
    const cursors: Array<string | null> = [];
    let update: unknown;
    const { user, queryClient, pageSnapshot } = fixture(async (request) => {
      const url = new URL(request.url);
      if (request.method === "PATCH") {
        update = await request.json();
        return response(
          { ...project, ...(update as object), revision: "2" },
          200,
          { ETag: '"2"' },
        );
      }
      cursors.push(url.searchParams.get("cursor"));
      return response(
        url.searchParams.has("cursor")
          ? {
              items: [credential("later-origin")],
              page: { hasMore: false },
            }
          : {
              items: Array.from({ length: 50 }, (_, index) =>
                credential(`proxy-${index}`, "http-proxy-basic@1"),
              ),
              page: { hasMore: true, nextCursor: "page-2" },
            },
      );
    });
    await user.click(screen.getByRole("button", { name: "Configure target" }));
    await user.selectOptions(
      screen.getByLabelText("Authorization"),
      "existing",
    );
    await screen.findByRole("option", {
      name: "later-origin · http-origin-bearer@1",
    });
    expect(cursors).toEqual([null, "page-2"]);
    expect(
      queryClient.getQueryData(queryKeys.operations.runtimeCredentials.list()),
    ).toEqual(pageSnapshot);
    await user.type(
      screen.getByLabelText("Application URL"),
      "https://app.example.test",
    );
    await user.selectOptions(
      screen.getByLabelText("Active HTTP origin credential"),
      "later-origin",
    );
    await user.click(screen.getByRole("button", { name: "Save target" }));
    await waitFor(() =>
      expect(update).toEqual({
        httpTarget: {
          url: "https://app.example.test",
          credential: {
            credentialId: "later-origin",
            kind: "http-origin-bearer@1",
          },
        },
      }),
    );
  });

  it("reports a later-page failure and reloads the whole inventory on retry", async () => {
    let fail = true;
    const cursors: Array<string | null> = [];
    const { user } = fixture(async (request) => {
      const cursor = new URL(request.url).searchParams.get("cursor");
      cursors.push(cursor);
      if (cursor !== null)
        return fail
          ? response(
              {
                code: "unavailable",
                message: "Later credentials unavailable",
                retryable: true,
              },
              503,
            )
          : response({
              items: [credential("later-origin")],
              page: { hasMore: false },
            });
      return response({
        items: [credential("first-origin")],
        page: { hasMore: true, nextCursor: "page-2" },
      });
    });
    await user.click(screen.getByRole("button", { name: "Configure target" }));
    await user.selectOptions(
      screen.getByLabelText("Authorization"),
      "existing",
    );
    await screen.findByText("Later credentials unavailable");
    expect(screen.queryByRole("option", { name: /first-origin/ })).toBeNull();
    expect(screen.getByRole("button", { name: "Save target" })).toBeDisabled();
    fail = false;
    await user.click(screen.getByRole("button", { name: "Retry credentials" }));
    await screen.findByRole("option", { name: /later-origin/ });
    expect(screen.getByRole("option", { name: /first-origin/ })).toBeVisible();
    expect(cursors).toEqual([null, "page-2", null, "page-2"]);
  });

  it("keeps the Project's current credential visible when it is absent from the inventory", async () => {
    const { user } = fixture(
      async () => response({ items: [], page: { hasMore: false } }),
      {
        ...project,
        httpTarget: {
          url: "https://app.example.test/",
          credential: {
            credentialId: "current-origin",
            kind: "http-origin-bearer@1",
          },
        },
      },
    );
    await user.click(screen.getByRole("button", { name: "Edit target" }));
    expect(
      await screen.findByRole("option", { name: /current-origin/ }),
    ).toBeVisible();
    expect(screen.getByLabelText("Active HTTP origin credential")).toHaveValue(
      "current-origin",
    );
  });

  it("rejects cyclic pagination instead of treating the first page as complete", async () => {
    const reads = vi.fn(async () =>
      response({
        items: [credential("first-origin")],
        page: { hasMore: true, nextCursor: "repeated" },
      }),
    );
    const { user } = fixture(reads);
    await user.click(screen.getByRole("button", { name: "Configure target" }));
    await user.selectOptions(
      screen.getByLabelText("Authorization"),
      "existing",
    );
    await screen.findByText(
      "Runtime credentials could not be fully loaded. Retry before choosing a credential.",
    );
    expect(reads).toHaveBeenCalledTimes(2);
    expect(screen.getByRole("button", { name: "Save target" })).toBeDisabled();
  });
});
