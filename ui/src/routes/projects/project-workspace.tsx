import { useQuery } from "@tanstack/react-query";
import {
  type ReactNode,
  useCallback,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Link, Outlet } from "react-router";

import { usePublicAPI } from "../../api/context";
import { PublicAPIError } from "../../api/error";
import { getProject, PROJECT_ID_PATTERN } from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import { ActionMenu } from "../../app/action-menu";
import { useDocumentTitle } from "../../app/document-title";
import { ErrorNotice } from "../../app/error-notice";
import { StaleDataWarning } from "../../app/query-view";
import { DetailHeader, DetailPane, IdChip } from "../../ui";
import { DeleteProjectDialog, ProjectDeletionProgress } from "./deletion";
import { ProjectNavigation } from "./navigation";
import { ProjectSectionActionsContext } from "./section-actions-context";
import { useProjectDeletion } from "./use-project-deletion";
import type { ProjectWorkspaceContext } from "./workspace-context";

import "../primary-actions.css";

/**
 * The selected project in the Projects detail pane: header with the
 * project's name, description, ID and actions; the section tabs; and the
 * section (Overview, Materials, Checks, Issues, Settings, Runs, Workflows)
 * through the router Outlet. While the project is being deleted the pane
 * shows the deletion progress instead and polls until the Server answers 404,
 * then returns to /projects.
 */
export function ProjectWorkspace({ projectId }: { projectId: string }) {
  const api = usePublicAPI();
  const validProject = PROJECT_ID_PATTERN.test(projectId);
  const project = useQuery({
    queryKey: queryKeys.projects.detail(projectId),
    queryFn: () => getProject(api, projectId),
    enabled: validProject,
    refetchInterval: (query) =>
      query.state.data?.lifecycle === "deleting" ? 1_000 : false,
    retry: (failureCount, error) =>
      !(error instanceof PublicAPIError && error.status === 404) &&
      failureCount < 2,
  });
  const { deletion, deletionObserved, deleteOpen, setDeleteOpen } =
    useProjectDeletion(project, "/projects");
  const activeProject =
    project.data?.kind === "project" && project.data.lifecycle === "active"
      ? project.data
      : undefined;
  const [sectionActions, setSectionActions] = useState<HTMLDivElement | null>(
    null,
  );
  useDocumentTitle(project.data?.name ?? "Project");
  // Another project opens at its top, also when the detail pane had been
  // scrolled for the previous one (the pane itself stays mounted).
  const workspace = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const pane = workspace.current?.closest<HTMLElement>(".ui-panes-detail");
    if (pane !== null && pane !== undefined) pane.scrollTop = 0;
  }, []);
  // The mutation's reset is stable, so the sections' context only changes
  // with the project.
  const resetDeletion = deletion.reset;
  const requestDeletion = useCallback(() => {
    resetDeletion();
    setDeleteOpen(true);
  }, [resetDeletion, setDeleteOpen]);
  const context = useMemo<ProjectWorkspaceContext | undefined>(
    () =>
      activeProject === undefined
        ? undefined
        : { project: activeProject, requestDeletion },
    [activeProject, requestDeletion],
  );

  if (!validProject) {
    return (
      <DetailPane>
        <ErrorNotice error={new Error("Project route is invalid")} />
        <Link to="/projects">Return to Projects</Link>
      </DetailPane>
    );
  }

  const name = project.data?.name ?? projectId;
  const header = (
    <DetailHeader
      breadcrumb={[{ label: "Projects", to: "/projects" }, { label: name }]}
      title={name}
      meta={
        <>
          {project.data === undefined ||
          project.data.description === "" ? null : (
            <p
              className="projects-description"
              title={project.data.description}
            >
              {project.data.description}
            </p>
          )}
          <span className="projects-id">
            Project ID <IdChip value={projectId} label="project ID" />
          </span>
        </>
      }
      actions={
        <ActionMenu label="Project actions">
          <button
            className="ui-btn"
            data-variant="ghost"
            data-size="sm"
            type="button"
            disabled={project.isFetching}
            onClick={() => void project.refetch()}
          >
            {project.isFetching ? "Refreshing…" : "Refresh project details"}
          </button>
          {activeProject === undefined ? null : (
            <button
              className="ui-btn"
              data-variant="danger"
              data-size="sm"
              type="button"
              onClick={requestDeletion}
            >
              Delete Project
            </button>
          )}
        </ActionMenu>
      }
    />
  );

  const data = project.data;
  const gone =
    project.error instanceof PublicAPIError && project.error.status === 404;
  // Without a body the section shows; a failed refresh of a project that is
  // still there keeps the section and says the header may be stale.
  let body: ReactNode = null;
  if (project.isPending) {
    body = (
      <p className="loading-copy" role="status">
        Loading Project…
      </p>
    );
  } else if (deletionObserved && gone) {
    body = (
      <p className="loading-copy" role="status">
        Project deleted. Returning to Projects…
      </p>
    );
  } else if (data === undefined || gone) {
    body = (
      <ErrorNotice
        error={project.error ?? new Error("Project could not be loaded")}
        onRetry={() => void project.refetch()}
        retryPending={project.isFetching}
      />
    );
  } else if (data.kind !== "project") {
    body = (
      <>
        <ErrorNotice
          error={new Error("Evaluation workspaces are available in Evals.")}
        />
        <Link to={`/evals/${encodeURIComponent(projectId)}`}>
          Open this workspace in Evals
        </Link>
      </>
    );
  } else if (data.lifecycle === "deleting") {
    body = <ProjectDeletionProgress project={data} />;
  }

  return (
    <ProjectSectionActionsContext value={sectionActions}>
      <div className="projects-workspace" ref={workspace}>
        <DetailPane
          header={
            <>
              {header}
              {context === undefined ? null : (
                <ProjectNavigation
                  projectId={projectId}
                  actionsRef={setSectionActions}
                />
              )}
            </>
          }
        >
          {body ?? (
            <>
              {project.error === null ? null : (
                <StaleDataWarning
                  error={project.error}
                  onRetry={() => void project.refetch()}
                  retryPending={project.isFetching}
                />
              )}
              <Outlet context={context} />
            </>
          )}
        </DetailPane>
      </div>
      {deleteOpen && activeProject !== undefined ? (
        <DeleteProjectDialog
          project={activeProject}
          pending={deletion.isPending}
          error={deletion.error}
          onCancel={() => {
            if (!deletion.isPending) {
              setDeleteOpen(false);
              deletion.reset();
            }
          }}
          onConfirm={() => deletion.mutate(activeProject)}
        />
      ) : null}
    </ProjectSectionActionsContext>
  );
}
