import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";

import { PublicAPI } from "../../api/client";
import { PublicAPIProvider } from "../../api/context";
import type { Project } from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import { useProjectDeletion } from "./use-project-deletion";

const project: Project = {
  projectId: "project_example",
  kind: "evaluation",
  name: "Router evals",
  description: "",
  lifecycle: "active",
  revision: "1",
  createdAt: "2026-09-01T10:00:00Z",
  updatedAt: "2026-09-01T10:00:00Z",
};

describe("useProjectDeletion", () => {
  it("invalidates every Project list page and the Evals Project list", async () => {
    const deleting = {
      ...project,
      lifecycle: "deleting",
      revision: "2",
      deletion: { phase: "draining", requestedAt: "2026-09-05T10:00:00Z" },
    };
    const api = new PublicAPI(
      {
        uiVersion: "0.1.0",
        supportedApiVersions: ["contractor.public.v1"],
        apiBaseUrl: "http://127.0.0.1:8080",
      },
      vi.fn(
        async () =>
          new Response(JSON.stringify(deleting), {
            status: 202,
            headers: {
              "content-type": "application/json",
              ETag: '"2"',
              "X-Contractor-API-Version": "contractor.public.v1",
            },
          }),
      ),
    );
    api.csrf.replace("a".repeat(43));
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const listed = { items: [], page: { hasMore: false } };
    queryClient.setQueryData(queryKeys.projects.list("evaluation"), listed);
    queryClient.setQueryData(
      queryKeys.projects.list("evaluation", "page-2"),
      listed,
    );
    queryClient.setQueryData(["evals", "projects"], []);
    const wrapper = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={queryClient}>
        <PublicAPIProvider api={api}>
          <MemoryRouter>{children}</MemoryRouter>
        </PublicAPIProvider>
      </QueryClientProvider>
    );
    const { result } = renderHook(
      () => useProjectDeletion({ data: project, error: null }, "/evals"),
      { wrapper },
    );
    act(() => result.current.deletion.mutate(project));
    await waitFor(() => expect(result.current.deletion.isSuccess).toBe(true));
    const invalidated = (queryKey: readonly unknown[]) =>
      queryClient.getQueryState(queryKey)?.isInvalidated;
    expect(invalidated(queryKeys.projects.list("evaluation"))).toBe(true);
    expect(invalidated(queryKeys.projects.list("evaluation", "page-2"))).toBe(
      true,
    );
    expect(invalidated(["evals", "projects"])).toBe(true);
  });
});
