import {
  type ReactNode,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Link, Outlet } from "react-router";

import { useDocumentTitle } from "../../app/document-title";
import { ErrorNotice } from "../../app/error-notice";
import { StaleDataWarning } from "../../app/query-view";
import { DetailHeader, DetailPane, IdChip } from "../../ui";
import {
  DeleteProjectDialog,
  ProjectActions,
  ProjectDeletionProgress,
} from "./deletion";
import { ProjectNavigation } from "./navigation";
import { ProjectSectionActionsContext } from "./section-actions-context";
import { useWorkspaceProject } from "./use-project-deletion";
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
  const workspace = useWorkspaceProject(projectId, "project");
  const { project, activeProject, requestDeletion } = workspace;
  const [sectionActions, setSectionActions] = useState<HTMLDivElement | null>(
    null,
  );
  useDocumentTitle(project.data?.name ?? "Project");
  // Another project opens at its top, also when the detail pane had been
  // scrolled for the previous one (the pane itself stays mounted).
  const frame = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const pane = frame.current?.closest<HTMLElement>(".ui-panes-detail");
    if (pane !== null && pane !== undefined) pane.scrollTop = 0;
  }, []);
  const context = useMemo<ProjectWorkspaceContext | undefined>(
    () =>
      activeProject === undefined
        ? undefined
        : { project: activeProject, requestDeletion },
    [activeProject, requestDeletion],
  );

  if (!workspace.valid) {
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
      actions={<ProjectActions workspace={workspace} noun="Project" />}
    />
  );

  const data = project.data;
  const { gone } = workspace;
  // Without a body the section shows; a failed refresh of a project that is
  // still there keeps the section and says the header may be stale.
  let body: ReactNode = null;
  if (project.isPending) {
    body = (
      <p className="loading-copy" role="status">
        Loading Project…
      </p>
    );
  } else if (workspace.deletionObserved && gone) {
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
      <div className="projects-workspace" ref={frame}>
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
      <DeleteProjectDialog workspace={workspace} />
    </ProjectSectionActionsContext>
  );
}
