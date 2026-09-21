import { useDocumentTitle } from "../../app/document-title";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link, Outlet, useParams } from "react-router";
import { ActionMenu } from "../../app/action-menu";
import { usePublicAPI } from "../../api/context";
import { PublicAPIError } from "../../api/error";
import { getProject, PROJECT_ID_PATTERN } from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import { ErrorNotice } from "../artifacts/common";
import { ProjectRegion } from "./common";
import { ProjectMetadataEditor } from "./metadata-editor";
import { ProjectHTTPTargetEditor } from "./http-target-editor";
import { ProjectArtifactRegion } from "./artifact-region";
import { ProjectRunsRegion } from "./runs-region";
import { DeleteProjectDialog, ProjectDeletionProgress } from "./deletion";
import { useProjectDeletion } from "./use-project-deletion";
import { ProjectWorkflowRecommendations } from "./workflow-recommendations";

import { ProjectNavigation } from "./navigation";
import { ProjectSectionActionsContext } from "./section-actions-context";
import { AuditAnchor } from "./audits/shared";

import "../primary-actions.css";
import "./workspace.css";

function ProjectWorkspaceRoute({
  expectedKind,
}: {
  expectedKind: "project" | "evaluation";
}) {
  const api = usePublicAPI();
  const { projectId = "" } = useParams();
  const validProject = PROJECT_ID_PATTERN.test(projectId);
  const destination = expectedKind === "evaluation" ? "/evals" : "/projects";
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
    useProjectDeletion(project, destination);
  const activeProject =
    project.data?.kind === expectedKind && project.data.lifecycle === "active"
      ? project.data
      : undefined;
  const deleteLabel =
    expectedKind === "evaluation" ? "Delete Eval" : "Delete Project";
  const [sectionActions, setSectionActions] = useState<HTMLDivElement | null>(
    null,
  );
  const description =
    expectedKind === "evaluation"
      ? "Each sample remains an ordinary isolated Workflow Run; eval labels group it without changing execution semantics."
      : project.data?.description ||
        "Sources, audits and results in one workspace.";
  useDocumentTitle(
    project.data?.name ?? (expectedKind === "evaluation" ? "Eval" : "Project"),
  );

  if (!validProject) {
    return (
      <section className="route-page">
        <ErrorNotice
          error={
            new Error(
              expectedKind === "evaluation"
                ? "Eval route is invalid"
                : "Project route is invalid",
            )
          }
        />
        <Link to={expectedKind === "evaluation" ? "/evals" : "/projects"}>
          Return to {expectedKind === "evaluation" ? "Evals" : "Projects"}
        </Link>
      </section>
    );
  }

  return (
    <section
      className={`route-page projects-page project-detail-page ${expectedKind === "evaluation" ? "eval-detail-page" : "project-section-layout"}`}
    >
      <header className="route-header-row">
        <div>
          <Link className="back-link" to={destination}>
            ← All {expectedKind === "evaluation" ? "Evals" : "Projects"}
          </Link>
          <h2>{project.data?.name ?? projectId}</h2>
          <p className="lede" title={description}>
            {description}
          </p>
        </div>
        <ActionMenu
          label={
            expectedKind === "evaluation" ? "Eval actions" : "Project actions"
          }
        >
          <button
            className="secondary-button"
            type="button"
            disabled={project.isFetching}
            onClick={() => void project.refetch()}
          >
            {project.isFetching ? "Refreshing…" : "Refresh project details"}
          </button>
          {activeProject === undefined ? null : (
            <button
              className="danger-button"
              type="button"
              onClick={() => {
                deletion.reset();
                setDeleteOpen(true);
              }}
            >
              {deleteLabel}
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
          Project deleted. Returning to{" "}
          {expectedKind === "evaluation" ? "Evals" : "Projects"}…
        </p>
      ) : project.error !== null ? (
        <ErrorNotice
          error={project.error}
          onRetry={() => void project.refetch()}
          retryPending={project.isFetching}
        />
      ) : project.data.kind !== expectedKind ? (
        <ErrorNotice
          error={
            new Error(
              expectedKind === "evaluation"
                ? "This workspace is available in Projects."
                : "Evaluation workspaces are available in Evals.",
            )
          }
        />
      ) : project.data.lifecycle === "deleting" ? (
        <ProjectDeletionProgress project={project.data} />
      ) : expectedKind === "project" ? (
        <ProjectSectionActionsContext value={sectionActions}>
          <ProjectNavigation
            projectId={projectId}
            actionsRef={setSectionActions}
          />
          <Outlet context={project.data} />
        </ProjectSectionActionsContext>
      ) : (
        <>
          <nav
            className="project-local-navigation section-navigation"
            aria-label="Project sections"
          >
            <a href="#project-runs">Runs</a>
            <a href="#project-artifacts">Artifacts</a>
            <a href="#project-workflows">Workflows</a>
            <a href="#project-overview">Workspace settings</a>
          </nav>
          <AuditAnchor />
          <ProjectRunsRegion projectId={project.data.projectId} evaluation />
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
              <ProjectHTTPTargetEditor
                key={`target-${project.data.projectId}`}
                project={project.data}
              />
            </ProjectRegion>
          </details>
          <ProjectArtifactRegion
            projectId={project.data.projectId}
            detailRoot="/evals"
          />
          <ProjectWorkflowRecommendations projectId={project.data.projectId} />
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

export function ProjectDetailRoute() {
  return <ProjectWorkspaceRoute expectedKind="project" />;
}

export function EvaluationDetailRoute() {
  return <ProjectWorkspaceRoute expectedKind="evaluation" />;
}
