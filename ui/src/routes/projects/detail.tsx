import { useQuery } from "@tanstack/react-query";
import { useId } from "react";
import { Link, useParams } from "react-router";

import { usePublicAPI } from "../../api/context";
import { PublicAPIError } from "../../api/error";
import { getProject, PROJECT_ID_PATTERN } from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import { ActionMenu } from "../../app/action-menu";
import { useDocumentTitle } from "../../app/document-title";
import { ErrorNotice } from "../../app/error-notice";
import { AuditAnchor } from "./audits/shared";
import { ProjectArtifactBindings, ProjectRegion } from "./common";
import { DeleteProjectDialog, ProjectDeletionProgress } from "./deletion";
import { ProjectHTTPTargetEditor } from "./http-target-editor";
import { ProjectFacts, ProjectMetadataEditor } from "./metadata-editor";
import { ProjectsRoute } from "./projects-route";
import { ProjectRunsRegion } from "./runs-region";
import { useProjectDeletion } from "./use-project-deletion";

import "../primary-actions.css";
import "./collection.css";
import "./projects.css";

/**
 * /projects/:projectId and its sections. The same component serves
 * /projects, so choosing a project keeps the list pane mounted (see
 * ProjectsRoute).
 */
export const ProjectDetailRoute = ProjectsRoute;

/**
 * A legacy evaluation workspace at /evals/:projectId. It keeps its layout
 * (S06: "Eval workspaces retain their existing layout and scope, including
 * their in-page anchors"): Runs, read-only Artifacts and Workspace settings.
 */
export function EvaluationDetailRoute() {
  const api = usePublicAPI();
  const { projectId = "" } = useParams();
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
    useProjectDeletion(project, "/evals");
  const activeProject =
    project.data?.kind === "evaluation" && project.data.lifecycle === "active"
      ? project.data
      : undefined;
  const description =
    project.data?.description ||
    "Legacy evaluation workspace: execution history and recorded inputs.";
  useDocumentTitle(project.data?.name ?? "Eval");
  const targetHeading = useId();

  if (!validProject) {
    return (
      <section className="route-page">
        <ErrorNotice error={new Error("Eval route is invalid")} />
        <Link to="/evals">Return to Evals</Link>
      </section>
    );
  }

  return (
    <section className="route-page projects-page project-detail-page eval-detail-page">
      <header className="route-header-row">
        <div>
          <Link className="back-link" to="/evals">
            ← All Evals
          </Link>
          <h2>{project.data?.name ?? projectId}</h2>
          <p className="lede" title={description}>
            {description}
          </p>
        </div>
        <ActionMenu label="Eval actions">
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
              onClick={() => {
                deletion.reset();
                setDeleteOpen(true);
              }}
            >
              Delete Eval
            </button>
          )}
        </ActionMenu>
      </header>

      {project.isPending ? (
        <p className="loading-copy" role="status">
          Loading Project…
        </p>
      ) : deletionObserved &&
        project.error instanceof PublicAPIError &&
        project.error.status === 404 ? (
        <p className="loading-copy" role="status">
          Project deleted. Returning to Evals…
        </p>
      ) : project.error !== null ? (
        <ErrorNotice
          error={project.error}
          onRetry={() => void project.refetch()}
          retryPending={project.isFetching}
        />
      ) : project.data.kind !== "evaluation" ? (
        <ErrorNotice
          error={new Error("This workspace is available in Projects.")}
        />
      ) : project.data.lifecycle === "deleting" ? (
        <ProjectDeletionProgress project={project.data} />
      ) : (
        <>
          <nav
            className="project-local-navigation section-navigation"
            aria-label="Project sections"
          >
            <a href="#project-runs">Runs</a>
            <a href="#project-artifacts">Artifacts</a>
            <a href="#project-overview">Workspace settings</a>
          </nav>
          <AuditAnchor />
          <ProjectRunsRegion projectId={project.data.projectId} evaluation />
          <ProjectArtifactBindings
            projectId={project.data.projectId}
            detailRoot="/evals"
          />
          <details className="eval-workspace-settings" id="project-overview">
            <summary>Workspace settings</summary>
            <ProjectRegion
              eyebrow="Evaluation metadata"
              title="Overview"
              id="eval-metadata"
            >
              <ProjectMetadataEditor
                key={project.data.projectId}
                project={project.data}
              />
              <ProjectFacts project={project.data} />
              <section
                className="eval-workspace-target"
                aria-labelledby={targetHeading}
              >
                <h4 id={targetHeading}>Live target</h4>
                <ProjectHTTPTargetEditor
                  key={`target-${project.data.projectId}`}
                  project={project.data}
                />
              </section>
            </ProjectRegion>
          </details>
        </>
      )}
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
    </section>
  );
}
