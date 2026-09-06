import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import { usePublicAPI } from "../../api/context";
import { PublicAPIError } from "../../api/error";
import { deleteProject, type Project } from "../../api/projects";
import { queryKeys } from "../../api/query-keys";

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
        queryKey: queryKeys.projects.list(deleting.kind),
      });
    },
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
