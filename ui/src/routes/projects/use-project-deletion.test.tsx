import { act, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { queryKeys } from "../../api/query-keys";
import * as build from "../../test/builders";
import { fakeServer } from "../../test/fake-server";
import { json, page } from "../../test/responses";
import { renderWithAPI } from "../../test/render";
import {
  useWorkspaceProject,
  type WorkspaceProject,
} from "./use-project-deletion";

const project = build.project("project_example", {
  kind: "evaluation",
  name: "Router evals",
});

describe("useWorkspaceProject", () => {
  it("invalidates every Project list page and the Evals Project list", async () => {
    const deleting = build.project("project_example", {
      kind: "evaluation",
      name: "Router evals",
      lifecycle: "deleting",
      revision: "2",
      deletion: { phase: "draining", requestedAt: "2026-09-05T10:00:00Z" },
    });
    let workspace: WorkspaceProject | undefined;
    function Probe() {
      workspace = useWorkspaceProject(project.projectId, "evaluation");
      return null;
    }
    const server = fakeServer({
      "GET /v1/projects/{projectId}": () =>
        json(workspace?.deletion.data ?? project, 200, {
          ETag: `"${(workspace?.deletion.data ?? project).revision}"`,
        }),
      "DELETE /v1/projects/{projectId}": () =>
        json(deleting, 202, { ETag: '"2"' }),
    });
    const { queryClient } = renderWithAPI(<Probe />, { server });
    const listed = page([]);
    queryClient.setQueryData(queryKeys.projects.list("evaluation"), listed);
    queryClient.setQueryData(
      queryKeys.projects.list("evaluation", "page-2"),
      listed,
    );
    queryClient.setQueryData(queryKeys.evals.projects, []);
    await waitFor(() => expect(workspace?.activeProject).toEqual(project));

    act(() => workspace?.deletion.mutate(project));

    await waitFor(() => expect(workspace?.deletion.isSuccess).toBe(true));
    const invalidated = (queryKey: readonly unknown[]) =>
      queryClient.getQueryState(queryKey)?.isInvalidated;
    expect(invalidated(queryKeys.projects.list("evaluation"))).toBe(true);
    expect(invalidated(queryKeys.projects.list("evaluation", "page-2"))).toBe(
      true,
    );
    expect(invalidated(queryKeys.evals.projects)).toBe(true);
  });
});
