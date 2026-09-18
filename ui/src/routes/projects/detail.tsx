import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router";
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

import { ProjectAuditWorkspace } from "./audits/list";
import { AuditAnchor } from "./audits/shared";

import "../primary-actions.css";

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
      className={`route-page projects-page project-detail-page ${expectedKind === "evaluation" ? "eval-detail-page" : ""}`}
    >
      <header className="route-header-row">
        <div>
          <Link className="back-link" to={destination}>
            ← All {expectedKind === "evaluation" ? "Evals" : "Projects"}
          </Link>
          <p className="eyebrow">
            {expectedKind === "evaluation"
              ? "Evaluation workspace"
              : "Project workspace"}
          </p>
          <h2>{project.data?.name ?? projectId}</h2>
          <p className="lede">
            {expectedKind === "evaluation"
              ? "Each sample remains an ordinary isolated Workflow Run; eval labels group it without changing execution semantics."
              : "Sources, audits and results in one workspace."}
          </p>
        </div>
        <details className="additional-actions project-actions-menu">
          <summary aria-label="Project actions" title="Project actions">
            ⋯
          </summary>
          <div className="additional-actions-menu">
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
          </div>
        </details>
      </header>

      {project.isPending ? (
        <p className="loading-copy" aria-live="polite">
          Loading Project…
        </p>
      ) : deletionObserved &&
        project.error instanceof PublicAPIError &&
        project.error.status === 404 ? (
        <p className="loading-copy" aria-live="polite">
          Project deleted. Returning to{" "}
          {expectedKind === "evaluation" ? "Evals" : "Projects"}…
        </p>
      ) : project.error !== null ? (
        <ErrorNotice error={project.error} />
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
      ) : (
        <>
          <nav
            className="project-local-navigation"
            aria-label="Project sections"
          >
            {expectedKind === "project" ? (
              <a href="#project-audits">Audits</a>
            ) : null}
            <a href="#project-overview">Overview</a>
            <a href="#project-artifacts">Artifacts</a>
            <a href="#project-workflows">Workflows</a>
            {expectedKind === "project" ? (
              <>
                <Link
                  to={`/projects/${encodeURIComponent(projectId)}/findings`}
                >
                  Findings
                </Link>
              </>
            ) : null}
            <a href="#project-runs">Runs</a>
          </nav>
          <AuditAnchor />
          {expectedKind === "project" ? (
            <ProjectRegion
              eyebrow="Project security"
              title="Audits"
              id="project-audits"
              action={
                <Link
                  to={`/projects/${encodeURIComponent(projectId)}/findings`}
                >
                  All project findings →
                </Link>
              }
            >
              <ProjectAuditWorkspace
                key={project.data.projectId}
                projectId={project.data.projectId}
                projectName={project.data.name}
              />
            </ProjectRegion>
          ) : null}
          <ProjectRegion
            eyebrow={
              expectedKind === "evaluation"
                ? "Evaluation metadata"
                : "Project metadata"
            }
            title="Overview"
            id="project-overview"
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
          <ProjectArtifactRegion
            projectId={project.data.projectId}
            detailRoot={expectedKind === "evaluation" ? "/evals" : "/projects"}
          />
          <ProjectWorkflowRecommendations projectId={project.data.projectId} />
          <ProjectRunsRegion
            projectId={project.data.projectId}
            evaluation={expectedKind === "evaluation"}
          />
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
