import { useId } from "react";
import { Link, useParams } from "react-router";

import { useDocumentTitle } from "../../app/document-title";
import { ErrorNotice } from "../../app/error-notice";
import { AuditAnchor } from "./audits/shared";
import { ProjectArtifactBindings, ProjectRegion } from "./common";
import {
  DeleteProjectDialog,
  ProjectActions,
  ProjectDeletionProgress,
} from "./deletion";
import { ProjectHTTPTargetEditor } from "./http-target-editor";
import { ProjectFacts, ProjectMetadataEditor } from "./metadata-editor";
import { ProjectsRoute } from "./projects-route";
import { ProjectRunsRegion } from "./runs-region";
import { useWorkspaceProject } from "./use-project-deletion";

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
  const { projectId = "" } = useParams();
  const workspace = useWorkspaceProject(projectId, "evaluation");
  const { project } = workspace;
  const description =
    project.data?.description ||
    "Legacy evaluation workspace: execution history and recorded inputs.";
  useDocumentTitle(project.data?.name ?? "Eval");
  const targetHeading = useId();

  if (!workspace.valid) {
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
        <ProjectActions workspace={workspace} noun="Eval" />
      </header>

      {project.isPending ? (
        <p className="loading-copy" role="status">
          Loading Project…
        </p>
      ) : workspace.deletionObserved && workspace.gone ? (
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
      <DeleteProjectDialog workspace={workspace} />
    </section>
  );
}
