import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useSearchParams } from "react-router";

import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import { listRuns, RUN_STATES, type WorkflowRunState } from "../../api/runs";
import {
  CursorControls,
  ErrorNotice,
  formatTimestamp,
} from "../artifacts/common";
import { StateBadge } from "./components";

export function RunListRoute() {
  const api = usePublicAPI();
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedState = searchParams.get("state");
  const state = RUN_STATES.find((candidate) => candidate === requestedState) as
    WorkflowRunState | undefined;
  const [cursors, setCursors] = useState<Array<string | undefined>>([
    undefined,
  ]);
  const cursor = cursors.at(-1);
  const query = useQuery({
    queryKey: queryKeys.runs.list(state, cursor),
    queryFn: () =>
      listRuns(api, {
        ...(state === undefined ? {} : { state }),
        ...(cursor === undefined ? {} : { cursor }),
      }),
  });

  return (
    <section className="route-page runs-page">
      <header className="route-header-row">
        <div>
          <p className="eyebrow">Authoritative execution history</p>
          <h2>Runs</h2>
          <p className="lede">
            Lifecycle state comes only from Go Server snapshots. Open a Run to
            inspect its ordered Stage attempts and live typed Planner plan.
          </p>
        </div>
        <button
          className="secondary-button"
          type="button"
          disabled={query.isFetching}
          onClick={() => void query.refetch()}
        >
          {query.isFetching ? "Refreshing…" : "Refresh"}
        </button>
      </header>

      <div className="panel run-library">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Owner scope</p>
            <h3>Workflow Runs</h3>
          </div>
          <label className="compact-select">
            State
            <select
              value={state ?? ""}
              onChange={(event) => {
                const selected = event.target.value as WorkflowRunState | "";
                const next = new URLSearchParams(searchParams);
                if (selected === "") {
                  next.delete("state");
                } else {
                  next.set("state", selected);
                }
                setSearchParams(next, { replace: true });
                setCursors([undefined]);
              }}
            >
              <option value="">All states</option>
              {RUN_STATES.map((candidate) => (
                <option key={candidate} value={candidate}>
                  {candidate}
                </option>
              ))}
            </select>
          </label>
        </div>
        {query.isPending ? (
          <p className="loading-copy" aria-live="polite">
            Loading Runs…
          </p>
        ) : query.error !== null ? (
          <ErrorNotice error={query.error} />
        ) : query.data.items.length === 0 ? (
          <div className="compact-empty">
            <strong>No Runs match this view.</strong>
            <p>Create one from an exact published Workflow.</p>
          </div>
        ) : (
          <div className="table-scroll">
            <table className="responsive-table">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Workflow</th>
                  <th>State</th>
                  <th>Created</th>
                  <th>Updated</th>
                  <th>Finished</th>
                </tr>
              </thead>
              <tbody>
                {query.data.items.map((run) => (
                  <tr key={run.runId}>
                    <td data-label="Run">
                      <Link to={`/runs/${encodeURIComponent(run.runId)}`}>
                        {run.runId}
                      </Link>
                    </td>
                    <td data-label="Workflow">
                      <code>{run.workflow}</code>
                    </td>
                    <td data-label="State">
                      <StateBadge state={run.state} />
                    </td>
                    <td data-label="Created">
                      {formatTimestamp(run.createdAt)}
                    </td>
                    <td data-label="Updated">
                      {formatTimestamp(run.updatedAt)}
                    </td>
                    <td data-label="Finished">
                      {run.finishedAt === undefined
                        ? "—"
                        : formatTimestamp(run.finishedAt)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <CursorControls
          label="Run pages"
          canGoBack={cursors.length > 1}
          {...(query.data?.page.hasMore === true &&
          query.data.page.nextCursor !== undefined
            ? { nextCursor: query.data.page.nextCursor }
            : {})}
          onBack={() =>
            setCursors((current) =>
              current.slice(0, Math.max(1, current.length - 1)),
            )
          }
          onNext={(next) => setCursors((current) => [...current, next])}
        />
      </div>
    </section>
  );
}
