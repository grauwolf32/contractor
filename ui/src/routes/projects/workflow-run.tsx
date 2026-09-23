import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router";

import { usePublicAPI } from "../../api/context";
import { useDocumentTitle } from "../../app/document-title";
import { getProject, PROJECT_ID_PATTERN } from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import {
  CONFIG_ID_PATTERN,
  CONFIG_VERSION_PATTERN,
  getWorkflow,
} from "../../api/workflows";
import { ErrorNotice } from "../artifacts/common";
import { WorkflowRunForm } from "../workflows/run-form";

export function ProjectWorkflowRunRoute() {
  const api = usePublicAPI();
  const { projectId = "", name = "", version = "" } = useParams();
  const valid =
    PROJECT_ID_PATTERN.test(projectId) &&
    CONFIG_ID_PATTERN.test(name) &&
    CONFIG_VERSION_PATTERN.test(version);
  const project = useQuery({
    queryKey: queryKeys.projects.detail(projectId),
    queryFn: () => getProject(api, projectId),
    enabled: valid,
  });
  const workflow = useQuery({
    queryKey: queryKeys.workflows.detail(name, version),
    queryFn: () => getWorkflow(api, name, version),
    enabled: valid,
  });
  useDocumentTitle(
    valid ? `Configure Run · ${name}@${version}` : "Configure Run",
  );
  const backPath =
    project.data?.kind === "evaluation"
      ? `/evals/${encodeURIComponent(projectId)}`
      : `/projects/${encodeURIComponent(projectId)}/workflows`;

  if (!valid) {
    return (
      <section className="route-page">
        <ErrorNotice
          error={new Error("Project Workflow Run route is invalid")}
        />
        <Link to="/projects">Return to Projects</Link>
      </section>
    );
  }
  return (
    <section className="route-page projects-page project-workflow-run-page">
      <header className="route-header-row">
        <div>
          <Link className="back-link" to={backPath}>
            ← Back to {project.data?.kind === "evaluation" ? "Eval" : "Project"}
          </Link>
          <p className="eyebrow">Project Workflow</p>
          <h2>Configure Run</h2>
          <p className="lede">
            Inputs stay in this Project until you submit the Run.
          </p>
        </div>
      </header>
      {project.isPending || workflow.isPending ? (
        <p className="loading-copy" role="status">
          Loading Project and Workflow contracts…
        </p>
      ) : project.error !== null || workflow.error !== null ? (
        <ErrorNotice error={project.error ?? workflow.error} />
      ) : project.data.lifecycle !== "active" ? (
        <div className="notice notice-warning" role="alert">
          <strong>This Project is deleting.</strong>
          <p>A new Run cannot be submitted from this Project.</p>
        </div>
      ) : (
        <WorkflowRunForm
          workflow={workflow.data}
          projectId={project.data.projectId}
        />
      )}
    </section>
  );
}
