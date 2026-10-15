import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import { usePublicAPI } from "../../api/context";
import { invalidateCrossProject } from "../../api/cross-project";
import { PublicAPIError } from "../../api/error";
import {
  deleteProject,
  type Project,
  type ProjectPage,
} from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import { pollInterval } from "../../app/polling";

const isDeleting = (project: Project) => project.lifecycle === "deleting";

/**
 * A project being deleted changes on its own: its query polls each second
 * until the deletion finishes, which the route observes as a 404.
 */
export const pollWhileDeleting = pollInterval<Project>({
  interval: 1_000,
  settled: (project) => !isDeleting(project),
});

/** A page of projects polls each second while one of them is being deleted. */
export const pollWhileAnyDeleting = pollInterval<ProjectPage>({
  interval: 1_000,
  settled: (page) => !page.items.some(isDeleting),
});

// The route owns the sole Project query. Deletion updates that cache entry and
// observes its polling result to reconcile the eventual 404 exactly once.
export function useProjectDeletion(
  project: { data: Project | undefined; error: Error | null },
  destination: "/projects" | "/evals",
) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const [deleteOpen, setDeleteOpen] = useState(false);
  const deletion = useMutation({
    mutationFn: (target: Project) =>
      deleteProject(api, {
        projectId: target.projectId,
        expectedRevision: target.revision,
      }),
    onSuccess: (deleting) => {
      setDeleteOpen(false);
      queryClient.setQueryData(
        queryKeys.projects.detail(deleting.projectId),
        deleting,
      );
      void queryClient.invalidateQueries({
        queryKey: queryKeys.projects.lists(deleting.kind),
      });
      void queryClient.invalidateQueries({
        queryKey: queryKeys.evals.projects,
      });
      // The cross-project lists leave out projects that are being deleted.
      void invalidateCrossProject(queryClient);
    },
    // A refused deletion (412: the project changed) is never retried; the
    // project is read again so a new confirmation shows its current name and
    // carries its current revision.
    onError: (_error, target) =>
      queryClient.invalidateQueries({
        queryKey: queryKeys.projects.detail(target.projectId),
        exact: true,
      }),
  });
  const deletionObserved =
    deletion.data?.lifecycle === "deleting" ||
    project.data?.lifecycle === "deleting";

  useEffect(() => {
    if (
      deletionObserved &&
      project.error instanceof PublicAPIError &&
      project.error.status === 404
    ) {
      void queryClient.invalidateQueries({ queryKey: queryKeys.projects.all });
      void navigate(destination, { replace: true });
    }
  }, [deletionObserved, destination, navigate, project.error, queryClient]);

  return { deletion, deletionObserved, deleteOpen, setDeleteOpen };
}
