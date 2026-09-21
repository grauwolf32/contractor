import { useQuery } from "@tanstack/react-query";
import { useLocation, Link, useSearchParams } from "react-router";
import { usePublicAPI } from "../../api/context";
import { listProjectRuns } from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import {
  RUN_STATES,
  TERMINAL_RUN_STATES,
  type RunSummary,
  type WorkflowRunState,
} from "../../api/runs";
import { AUDIT_ID_PATTERN } from "../../api/audits";
import { ContextLink } from "../../app/context-navigation";
import {
  CursorControls,
  ErrorNotice,
  formatTimestamp,
} from "../artifacts/common";
import { RunMetadataLabelChips, StateBadge } from "../runs/components";
import { RefreshButton } from "../../app/refresh-button";
import { ProjectSectionActions } from "./navigation";

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
      <code title={run.runId}>
        {run.runId.length > 24
          ? `${run.runId.slice(0, 8)}…${run.runId.slice(-8)}`
          : run.runId}
      </code>
    </div>
  );
}

export function ProjectRunHistory({ projectId }: { projectId: string }) {
  const api = usePublicAPI();
  const [filters, setFilters] = useSearchParams();
  const location = useLocation();
  const view = ["active", "completed"].includes(filters.get("view") ?? "")
    ? filters.get("view")!
    : "all";
  const states = RUN_STATES.filter(
    (state) =>
      view === "all" ||
      (view === "completed") ===
        TERMINAL_RUN_STATES.includes(
          state as (typeof TERMINAL_RUN_STATES)[number],
        ),
  );
  const state = states.find((candidate) => candidate === filters.get("state"));
  const cursors = filters.getAll("cursor");
  const cursor = cursors.at(-1);
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
      query.state.data?.items.some(
        (run) =>
          !TERMINAL_RUN_STATES.includes(
            run.state as (typeof TERMINAL_RUN_STATES)[number],
          ),
      )
        ? 5_000
        : false,
  });
  function chooseView(value: string) {
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
  function changePage(nextCursors: string[]) {
    const next = new URLSearchParams(filters);
    next.delete("cursor");
    for (const value of nextCursors) next.append("cursor", value);
    setFilters(next, { state: location.state });
    document
      .getElementById("project-runs")
      ?.scrollIntoView?.({ block: "start" });
  }
  return (
    <section className="project-run-history" id="project-runs">
      <ProjectSectionActions>
        <Link
          className="audit-open-link"
          to={`/projects/${encodeURIComponent(projectId)}/workflows`}
        >
          Choose workflow →
        </Link>
      </ProjectSectionActions>
      <div className="project-section-toolbar">
        <div className="workflow-filter-tabs" aria-label="Run views">
          {[
            ["all", "All Runs"],
            ["active", "Active"],
            ["completed", "Completed"],
          ].map(([value, label]) => (
            <button
              key={value}
              className="secondary-button"
              type="button"
              aria-pressed={view === value}
              onClick={() => chooseView(value!)}
            >
              {label}
            </button>
          ))}
        </div>
        <label>
          State
          <select
            value={state ?? ""}
            onChange={(event) =>
              chooseState(event.target.value as WorkflowRunState | "")
            }
          >
            <option value="">All states</option>
            {states.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
        <RefreshButton
          isFetching={runs.isFetching}
          onRefresh={() => void runs.refetch()}
        />
      </div>
      {runs.isPending ? (
        <p className="loading-copy" role="status">
          Loading Project Runs…
        </p>
      ) : runs.error ? (
        <ErrorNotice error={runs.error} />
      ) : runs.data.items.length === 0 ? (
        <div className="panel compact-empty">
          <h3>No Runs in this view</h3>
          <p>
            {view === "all" && !state
              ? "Choose a Workflow to start a Project Run."
              : "Choose another state or view to inspect the rest of the history."}
          </p>
        </div>
      ) : (
        <div className="table-scroll project-run-history-table">
          <table className="responsive-table">
            <thead>
              <tr>
                <th>Workflow / Run</th>
                <th>State</th>
                <th>Origin</th>
                <th>Updated</th>
              </tr>
            </thead>
            <tbody>
              {runs.data.items.map((run) => {
                const auditId = run.labels["audit.id"];
                return (
                  <tr key={run.runId}>
                    <td data-label="Workflow / Run">
                      <ProjectRunIdentity run={run} />
                      <details className="project-run-labels">
                        <summary>
                          Identity & labels
                          {Object.keys(run.labels).length
                            ? ` (${Object.keys(run.labels).length})`
                            : ""}
                        </summary>
                        <code>{run.runId}</code>
                        <RunMetadataLabelChips labels={run.labels} />
                      </details>
                    </td>
                    <td data-label="State">
                      <StateBadge state={run.state} />
                    </td>
                    <td data-label="Origin">
                      {auditId && AUDIT_ID_PATTERN.test(auditId) ? (
                        <ContextLink
                          returnLabel="Project Runs"
                          to={`/projects/${encodeURIComponent(projectId)}/audits/${encodeURIComponent(auditId)}`}
                        >
                          Audit · {auditId.slice(-8)} ↗
                        </ContextLink>
                      ) : (
                        "Workflow Run"
                      )}
                    </td>
                    <td data-label="Updated">
                      <time dateTime={run.updatedAt}>
                        {formatTimestamp(run.updatedAt)}
                      </time>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      <CursorControls
        label="Project Run pages"
        canGoBack={cursors.length > 0}
        {...(runs.data?.page.hasMore && runs.data.page.nextCursor
          ? { nextCursor: runs.data.page.nextCursor }
          : {})}
        onBack={() => changePage(cursors.slice(0, -1))}
        onNext={(next) => changePage([...cursors, next])}
      />
    </section>
  );
}
