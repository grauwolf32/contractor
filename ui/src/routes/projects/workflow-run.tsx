import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router";

import { usePublicAPI } from "../../api/context";
import { getProject, PROJECT_ID_PATTERN } from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import {
  CONFIG_ID_PATTERN,
  CONFIG_VERSION_PATTERN,
  getWorkflow,
} from "../../api/workflows";
import { useDocumentTitle } from "../../app/document-title";
import { ErrorNotice } from "../../app/error-notice";
import { DetailHeader, IdChip } from "../../ui";
import { WorkflowRunForm } from "../workflows/run-form";

import "./projects.css";

/**
 * /projects/:projectId/workflows/:name/:version/run: the Run form of one
 * exact Workflow version in this project. The `name@version` stays visible
 * before the Run starts (UUS:62-63).
 */
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
  const evaluation = project.data?.kind === "evaluation";
  const backPath = evaluation
    ? `/evals/${encodeURIComponent(projectId)}`
    : `/projects/${encodeURIComponent(projectId)}/workflows`;

  if (!valid) {
    return (
      <section className="route-page projects-run-page">
        <ErrorNotice
          error={new Error("Project Workflow Run route is invalid")}
        />
        <Link to="/projects">Return to Projects</Link>
      </section>
    );
  }
  const selector = `${name}@${version}`;
  return (
    <section className="route-page projects-page projects-run-page project-workflow-run-page">
      <DetailHeader
        breadcrumb={
          evaluation
            ? [
                { label: "Evals", to: "/evals" },
                { label: project.data?.name ?? projectId, to: backPath },
                { label: "Configure Run" },
              ]
            : [
                { label: "Projects", to: "/projects" },
                {
                  label: project.data?.name ?? projectId,
                  to: `/projects/${encodeURIComponent(projectId)}`,
                },
                { label: "Workflows", to: backPath },
                { label: "Configure Run" },
              ]
        }
        title="Configure Run"
        titleAs="h2"
        meta={
          <>
            <IdChip
              value={selector}
              display={selector}
              label="workflow version"
            />
            <span>Inputs stay in this Project until you submit the Run.</span>
          </>
        }
        actions={
          <Link className="ui-btn" data-size="sm" to={backPath}>
            ← Back to {evaluation ? "Eval" : "Project"}
          </Link>
        }
      />
      <div className="projects-run-body">
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
      </div>
    </section>
  );
}
