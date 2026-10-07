import { useQuery } from "@tanstack/react-query";
import { Link, useLocation, useSearchParams } from "react-router";

import { AUDIT_ID_PATTERN } from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import { listProjectRuns } from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import {
  isTerminalRunState,
  RUN_STATES,
  type RunSummary,
  type WorkflowRunState,
} from "../../api/runs";
import { ContextLink } from "../../app/context-navigation";
import { CursorControls } from "../../app/cursor-controls";
import { compactId, formatTimestamp } from "../../app/format";
import { useURLCursorStack } from "../../app/pagination";
import { QueryView } from "../../app/query-view";
import { RefreshButton } from "../../app/refresh-button";
import { EmptyState, FilterChips, IdChip, StatusGlyph } from "../../ui";
import { RunMetadataLabelChips } from "../runs/components";
import { ProjectSectionActions } from "./navigation";
import { projectPath } from "./project-sections";
import { runStateLabel } from "./run-state";

import "./projects.css";

type RunView = "all" | "active" | "completed";

const RUN_VIEWS: readonly { value: RunView; label: string }[] = [
  { value: "all", label: "All Runs" },
  { value: "active", label: "Active" },
  { value: "completed", label: "Completed" },
];

/** A Run's Workflow (linking to the Run) and its shortened ID. */
export function ProjectRunIdentity({
  run,
  returnLabel = "Project Runs",
}: {
  run: RunSummary;
  returnLabel?: string;
}) {
  return (
    <div className="project-run-identity">
      <ContextLink
        returnLabel={returnLabel}
        to={`/runs/${encodeURIComponent(run.runId)}`}
        title={run.workflow}
      >
        {run.workflow}
      </ContextLink>
      <code title={run.runId}>{compactId(run.runId)}</code>
    </div>
  );
}

/**
 * The project's Runs, 25 per page with Server-side lifecycle and state
 * filters. The view, state and cursor live in the URL; active views poll.
 */
export function ProjectRunHistory({ projectId }: { projectId: string }) {
  const api = usePublicAPI();
  const [filters, setFilters] = useSearchParams();
  const location = useLocation();
  const view: RunView =
    RUN_VIEWS.find((option) => option.value === filters.get("view"))?.value ??
    "all";
  const states = RUN_STATES.filter(
    (state) =>
      view === "all" || (view === "completed") === isTerminalRunState(state),
  );
  const state = states.find((candidate) => candidate === filters.get("state"));
  const pages = useURLCursorStack({
    navigateOptions: { state: location.state },
    onChange: () =>
      document
        .getElementById("project-runs")
        ?.scrollIntoView?.({ block: "start" }),
  });
  const cursor = pages.cursor;
  const options = {
    limit: 25,
    ...(view === "all"
      ? {}
      : {
          lifecycle:
            view === "active" ? ("active" as const) : ("terminal" as const),
        }),
    ...(state === undefined ? {} : { state }),
    ...(cursor === undefined ? {} : { cursor }),
  };
  const runs = useQuery({
    queryKey: queryKeys.projects.runView(projectId, options),
    queryFn: () => listProjectRuns(api, { projectId, ...options }),
    refetchInterval: (query) =>
      view === "active" ||
      query.state.data?.items.some((run) => !isTerminalRunState(run.state))
        ? 5_000
        : false,
  });
  function chooseView(value: RunView) {
    const next = new URLSearchParams(filters);
    next.delete("cursor");
    next.delete("state");
    if (value === "all") next.delete("view");
    else next.set("view", value);
    setFilters(next, { state: location.state });
  }
  function chooseState(value: WorkflowRunState | "") {
    const next = new URLSearchParams(filters);
    next.delete("cursor");
    if (value === "") next.delete("state");
    else next.set("state", value);
    setFilters(next, { state: location.state });
  }
  return (
    <section
      className="projects-runs"
      id="project-runs"
      aria-label="Project Runs"
    >
      <ProjectSectionActions>
        <Link
          className="ui-btn"
          data-size="sm"
          to={projectPath(projectId, "workflows")}
        >
          Choose workflow
        </Link>
      </ProjectSectionActions>
      <div className="projects-toolbar">
        <FilterChips
          label="Run views"
          options={RUN_VIEWS}
          value={view}
          onChange={chooseView}
        />
        <label className="projects-inline-field">
          <span>State</span>
          <select
            value={state ?? ""}
            onChange={(event) =>
              chooseState(event.target.value as WorkflowRunState | "")
            }
          >
            <option value="">All states</option>
            {states.map((value) => (
              <option key={value} value={value}>
                {runStateLabel(value).label}
              </option>
            ))}
          </select>
        </label>
        <RefreshButton
          isFetching={runs.isFetching}
          onRefresh={() => void runs.refetch()}
        />
      </div>
      <QueryView
        query={runs}
        loading={
          <p className="loading-copy" role="status">
            Loading Project Runs…
          </p>
        }
        onRetry={() => void runs.refetch()}
        isEmpty={(runsData) => runsData.items.length === 0}
        empty={
          <EmptyState title="No Runs in this view">
            {view === "all" && !state
              ? "Choose a Workflow to start a Project Run."
              : "Choose another state or view to inspect the rest of the history."}
          </EmptyState>
        }
      >
        {(runsData) => (
          <ul role="list" className="projects-run-list">
            {runsData.items.map((run) => {
              const auditId = run.labels["audit.id"];
              const runState = runStateLabel(run.state);
              const labelCount = Object.keys(run.labels).length;
              return (
                <li className="projects-run" key={run.runId}>
                  <StatusGlyph tone={runState.tone} />
                  <div className="projects-run-main">
                    <ContextLink
                      className="projects-run-title"
                      returnLabel="Project Runs"
                      to={`/runs/${encodeURIComponent(run.runId)}`}
                      title={run.workflow}
                    >
                      {run.workflow}
                    </ContextLink>
                    <div className="projects-row-meta">
                      <strong
                        className="projects-row-status"
                        data-tone={runState.tone}
                      >
                        {runState.label}
                      </strong>
                      <span>
                        {auditId && AUDIT_ID_PATTERN.test(auditId) ? (
                          <ContextLink
                            returnLabel="Project Runs"
                            to={`${projectPath(projectId, "audits")}/${encodeURIComponent(auditId)}`}
                          >
                            Check · {auditId.slice(-8)}
                          </ContextLink>
                        ) : (
                          "Workflow Run"
                        )}
                      </span>
                      <span>
                        Updated{" "}
                        <time dateTime={run.updatedAt}>
                          {formatTimestamp(run.updatedAt)}
                        </time>
                      </span>
                    </div>
                    <details className="project-run-labels">
                      <summary>
                        Identity & labels
                        {labelCount ? ` (${labelCount})` : ""}
                      </summary>
                      <div className="projects-run-identity">
                        <IdChip
                          value={run.runId}
                          display={run.runId}
                          label="Run ID"
                        />
                        <RunMetadataLabelChips labels={run.labels} />
                      </div>
                    </details>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </QueryView>
      <CursorControls
        label="Project Run pages"
        {...pages.controls(runs.data?.page)}
      />
    </section>
  );
}
