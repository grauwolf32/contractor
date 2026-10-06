import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useNavigate } from "react-router";

import { usePublicAPI } from "../../api/context";
import { listProjects } from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import { compactId } from "../../app/format";
import { CursorControls } from "../../app/cursor-controls";
import { useDocumentTitle } from "../../app/document-title";
import { useCursorStack } from "../../app/pagination";
import { QueryView } from "../../app/query-view";
import { RecordedTime } from "../../app/recorded-time";
import { RefreshButton } from "../../app/refresh-button";
import { EvaluationActivity } from "./evaluation-activity";
import { NewProjectDialog } from "./new-project-dialog";
import { ProjectsRoute } from "./projects-route";

import "./collection.css";

/**
 * /projects. The same component serves /projects/:projectId, so choosing a
 * project keeps the list pane mounted (see ProjectsRoute).
 */
export const ProjectListRoute = ProjectsRoute;

/**
 * Legacy evaluation workspaces at /evals/legacy: Projects of kind
 * `evaluation` that hold experiments and datasets. They keep their own
 * layout and wording (S06: "Eval workspaces retain their existing layout").
 */
export function EvaluationListRoute() {
  useDocumentTitle("Evals");
  const api = usePublicAPI();
  const navigate = useNavigate();
  const pages = useCursorStack();
  const [createOpen, setCreateOpen] = useState(false);
  const cursor = pages.cursor;
  const query = useQuery({
    queryKey: queryKeys.projects.list("evaluation", cursor),
    queryFn: () =>
      listProjects(api, {
        kind: "evaluation",
        ...(cursor === undefined ? {} : { cursor }),
      }),
    refetchInterval: (current) =>
      current.state.data?.items.some(
        (project) => project.lifecycle === "deleting",
      )
        ? 1_000
        : false,
  });

  return (
    <section className="route-page projects-page evals-page">
      <header className="route-header-row">
        <div>
          <p className="eyebrow">Evaluation workspaces</p>
          <h2>Evals</h2>
          <p className="lede">
            Compare execution history across evaluation workspaces. A succeeded
            Run is an execution status; review outputs for the evaluation
            result.
          </p>
        </div>
        <button
          type="button"
          className="ui-btn"
          data-variant="primary"
          onClick={() => setCreateOpen(true)}
        >
          New Eval
        </button>
      </header>

      {createOpen ? (
        <NewProjectDialog
          kind="evaluation"
          wording={{
            title: "New Eval",
            closeLabel: "Close New Eval form",
            submitLabel: "Create Eval",
          }}
          onClose={() => setCreateOpen(false)}
          onCreated={(project) => {
            setCreateOpen(false);
            void navigate(`/evals/${encodeURIComponent(project.projectId)}`);
          }}
        />
      ) : null}

      <div className="project-list-heading">
        <div>
          <p className="eyebrow">Project kind</p>
          <h3>Evaluation workspaces</h3>
        </div>
        <RefreshButton
          isFetching={query.isFetching}
          onRefresh={() => void query.refetch()}
          label="Refresh"
        />
      </div>

      <QueryView
        query={query}
        loading={
          <p className="loading-copy" role="status">
            Loading Evals…
          </p>
        }
        errorContext="Could not load Eval workspaces"
        onRetry={() => void query.refetch()}
        isEmpty={(data) => data.items.length === 0}
        empty={
          <div className="empty-state panel">
            <p className="eyebrow">Nothing here yet</p>
            <h3>Create your first Eval</h3>
            <p>
              An Eval is a Project workspace whose ordinary Runs use
              purpose=eval and eval.* metadata labels.
            </p>
            <button
              type="button"
              className="ui-btn"
              data-variant="primary"
              onClick={() => setCreateOpen(true)}
            >
              New Eval
            </button>
          </div>
        }
      >
        {(data) => (
          <div className="project-card-grid">
            {data.items.map((project) => {
              const path = `/evals/${encodeURIComponent(project.projectId)}`;
              return (
                <article
                  className={`panel project-card ${project.lifecycle === "deleting" ? "is-deleting" : ""}`}
                  key={project.projectId}
                >
                  <div>
                    <p className="eyebrow project-card-eyebrow">
                      {project.lifecycle === "deleting" ? (
                        <span>Deletion in progress</span>
                      ) : null}
                    </p>
                    <h3>
                      <Link to={path}>{project.name}</Link>
                    </h3>
                    <p className="project-card-description">
                      {project.description === ""
                        ? "No description provided."
                        : project.description}
                    </p>
                  </div>
                  <EvaluationActivity projectId={project.projectId} />
                  <dl className="project-card-meta">
                    <div>
                      <dt>Updated</dt>
                      <dd>
                        <RecordedTime value={project.updatedAt} />
                      </dd>
                    </div>
                    <div>
                      <dt>ID</dt>
                      <dd>
                        <code title={project.projectId}>
                          {compactId(project.projectId)}
                        </code>
                      </dd>
                    </div>
                  </dl>
                  <div className="project-card-actions">
                    <Link className="project-card-open" to={path}>
                      {project.lifecycle === "deleting"
                        ? "View deletion status →"
                        : "Open Eval →"}
                    </Link>
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </QueryView>

      <CursorControls
        label="Evals pages"
        {...pages.controls(query.data?.page)}
      />
    </section>
  );
}
