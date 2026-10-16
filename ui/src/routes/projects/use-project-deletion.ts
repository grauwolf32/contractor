import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useState } from "react";

import { usePublicAPI } from "../../api/context";
import { invalidateCrossProject } from "../../api/cross-project";
import { PublicAPIError } from "../../api/error";
import {
  deleteProject,
  getProject,
  PROJECT_ID_PATTERN,
  type Project,
  type ProjectKind,
  type ProjectPage,
} from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import { useGuardedNavigate } from "../../app/page-location";
import { pollInterval } from "../../app/polling";

const isDeleting = (project: Project) => project.lifecycle === "deleting";

/**
 * A project being deleted changes on its own: its query polls each second
 * until the deletion finishes, which the route observes as a 404.
 */
const pollWhileDeleting = pollInterval<Project>({
  interval: 1_000,
  settled: (project) => !isDeleting(project),
});

/** A page of projects polls each second while one of them is being deleted. */
export const pollWhileAnyDeleting = pollInterval<ProjectPage>({
  interval: 1_000,
  settled: (page) => !page.items.some(isDeleting),
});

const DESTINATION: Readonly<Record<ProjectKind, "/projects" | "/evals">> = {
  project: "/projects",
  evaluation: "/evals",
};

/**
 * The project a workspace page shows (the Projects detail pane, or a legacy
 * evaluation workspace at /evals/:projectId) and its deletion flow. The page
 * owns the sole Project query: it polls while the project is being deleted,
 * and once the Server answers 404 after a deletion was observed the page
 * returns to its list (`/projects` or `/evals`), unless the user already
 * left. A refused deletion (412: the project changed) is never retried; the
 * project is read again so a new confirmation shows its current name and
 * carries its current revision.
 */
export function useWorkspaceProject(projectId: string, kind: ProjectKind) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const navigate = useGuardedNavigate();
  const valid = PROJECT_ID_PATTERN.test(projectId);
  const project = useQuery({
    queryKey: queryKeys.projects.detail(projectId),
    queryFn: () => getProject(api, projectId),
    enabled: valid,
    refetchInterval: pollWhileDeleting,
    retry: (failureCount, error) =>
      !(error instanceof PublicAPIError && error.status === 404) &&
      failureCount < 2,
  });
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
    onError: (_error, target) =>
      queryClient.invalidateQueries({
        queryKey: queryKeys.projects.detail(target.projectId),
        exact: true,
      }),
  });
  const gone =
    project.error instanceof PublicAPIError && project.error.status === 404;
  const deletionObserved =
    deletion.data?.lifecycle === "deleting" ||
    project.data?.lifecycle === "deleting";
  const destination = DESTINATION[kind];

  useEffect(() => {
    if (deletionObserved && gone) {
      void queryClient.invalidateQueries({ queryKey: queryKeys.projects.all });
      void navigate(destination, { replace: true });
    }
  }, [deletionObserved, destination, gone, navigate, queryClient]);

  const activeProject =
    project.data?.kind === kind && project.data.lifecycle === "active"
      ? project.data
      : undefined;
  // The mutation's reset is stable, so callers can memoize on this.
  const resetDeletion = deletion.reset;
  const requestDeletion = useCallback(() => {
    resetDeletion();
    setDeleteOpen(true);
  }, [resetDeletion]);
  const cancelDeletion = () => {
    if (deletion.isPending) return;
    setDeleteOpen(false);
    deletion.reset();
  };

  return {
    valid,
    project,
    /** The project while it is active and of this page's kind. */
    activeProject,
    /** The Server answered 404 for the project. */
    gone,
    /** A deletion was sent or seen, so a 404 means it finished. */
    deletionObserved,
    deletion,
    deleteOpen,
    requestDeletion,
    cancelDeletion,
  };
}

export type WorkspaceProject = ReturnType<typeof useWorkspaceProject>;
