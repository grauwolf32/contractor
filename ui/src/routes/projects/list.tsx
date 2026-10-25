import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router";

import { usePublicAPI } from "../../api/context";
import { listProjects } from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import { compactId } from "../../app/format";
import { CursorControls } from "../../app/cursor-controls";
import { useDocumentTitle } from "../../app/document-title";
import { useGuardedNavigate } from "../../app/page-location";
import { useCursorStack } from "../../app/pagination";
import { QueryView } from "../../app/query-view";
import { RecordedTime } from "../../app/recorded-time";
import { RefreshButton } from "../../app/refresh-button";
import { Button } from "../../ui";
import { EvaluationActivity } from "./evaluation-activity";
import { EvaluationWorkspaceDialog } from "../evals/workspace-dialog";
import { ProjectsRoute } from "./projects-route";
import { pollWhileAnyDeleting } from "./use-project-deletion";

import "./collection.css";

/**
 * /projects. The same component serves /projects/:projectId, so choosing a
 * project keeps the list pane mounted (see ProjectsRoute).
 */
export const ProjectListRoute = ProjectsRoute;

/**
 * Evaluation workspaces at /evals/workspaces: Projects of kind
 * `evaluation` that hold experiments and datasets. They keep their own
 * layout and wording (S06: "Eval workspaces retain their existing layout").
 */
export function EvaluationListRoute() {
  useDocumentTitle("Evaluation workspaces");
  const api = usePublicAPI();
  const navigate = useGuardedNavigate();
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
    refetchInterval: pollWhileAnyDeleting,
  });

  return (
    <section className="route-page projects-page evals-page">
      <header className="route-header-row">
        <div>
          <p className="eyebrow">Evaluation workspaces</p>
          <h2>Evaluation workspaces</h2>
          <p className="lede">
            Manage the workspaces that hold experiments, datasets and recorded
            execution history.
          </p>
        </div>
        <Button variant="primary" onClick={() => setCreateOpen(true)}>
          New workspace
        </Button>
      </header>

      {createOpen ? (
        <EvaluationWorkspaceDialog
          onClose={() => setCreateOpen(false)}
          onCreated={(project) => {
            setCreateOpen(false);
            void navigate(
              `/evals/workspaces/${encodeURIComponent(project.projectId)}`,
            );
          }}
        />
      ) : null}

      <div className="project-list-heading">
        <div>
          <p className="eyebrow">Project kind</p>
          <h3>Workspaces</h3>
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
            Loading workspaces…
          </p>
        }
        errorContext="Could not load evaluation workspaces"
        onRetry={() => void query.refetch()}
        isEmpty={(data) => data.items.length === 0}
        empty={
          <div className="empty-state ui-panel">
            <p className="eyebrow">Nothing here yet</p>
            <h3>Create your first evaluation workspace</h3>
            <p>
              A workspace holds datasets and experiments. Create an experiment
              inside it to compare two variants.
            </p>
            <Button variant="primary" onClick={() => setCreateOpen(true)}>
              New workspace
            </Button>
          </div>
        }
      >
        {(data) => (
          <div className="project-card-grid">
            {data.items.map((project) => {
              const path = `/evals/workspaces/${encodeURIComponent(project.projectId)}`;
              return (
                <article
                  className={`ui-panel project-card ${project.lifecycle === "deleting" ? "is-deleting" : ""}`}
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
                        : "Open workspace →"}
                    </Link>
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </QueryView>

      <CursorControls
        label="Workspace pages"
        {...pages.controls(query.data?.page)}
      />
    </section>
  );
}
