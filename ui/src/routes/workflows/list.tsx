import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router";

import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import { listWorkflows } from "../../api/workflows";
import { CursorControls, ErrorNotice } from "../artifacts/common";

function slotSummary(slots: Record<string, { required: boolean }>): string {
  const values = Object.values(slots);
  const required = values.filter((slot) => slot.required).length;
  return `${values.length} total · ${required} required`;
}

export function WorkflowListRoute() {
  const api = usePublicAPI();
  const [cursors, setCursors] = useState<Array<string | undefined>>([
    undefined,
  ]);
  const cursor = cursors.at(-1);
  const query = useQuery({
    queryKey: queryKeys.workflows.list(cursor),
    queryFn: () => listWorkflows(api, cursor === undefined ? {} : { cursor }),
  });

  return (
    <section className="route-page workflow-page">
      <header className="route-header-row">
        <div>
          <p className="eyebrow">Published contracts</p>
          <h2>Workflows</h2>
          <p className="lede">
            Choose a Workflow version, provide its inputs, and start a Run.
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

      <div className="panel workflow-library">
        {query.isPending ? (
          <p className="loading-copy" aria-live="polite">
            Loading published Workflows…
          </p>
        ) : query.error !== null ? (
          <ErrorNotice error={query.error} />
        ) : query.data.items.length === 0 ? (
          <div className="compact-empty">
            <strong>No published Workflow versions found.</strong>
            <p>Publish YAML configuration before creating a Run.</p>
          </div>
        ) : (
          <div className="table-scroll">
            <table className="responsive-table">
              <thead>
                <tr>
                  <th>Exact Workflow</th>
                  <th>Entry Stage</th>
                  <th>Parameters</th>
                  <th>Inputs</th>
                  <th>Outputs</th>
                </tr>
              </thead>
              <tbody>
                {query.data.items.map((workflow) => (
                  <tr key={`${workflow.ref.name}@${workflow.ref.version}`}>
                    <td data-label="Workflow">
                      <Link
                        to={`/workflows/${encodeURIComponent(workflow.ref.name)}/${encodeURIComponent(workflow.ref.version)}`}
                      >
                        {workflow.ref.name}@{workflow.ref.version}
                      </Link>
                    </td>
                    <td data-label="Entry stage">
                      <code>{workflow.entryStage}</code>
                    </td>
                    <td data-label="Parameters">
                      {slotSummary(workflow.parameters)}
                    </td>
                    <td data-label="Inputs">{slotSummary(workflow.inputs)}</td>
                    <td data-label="Outputs">
                      {slotSummary(workflow.outputs)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <CursorControls
          label="Workflow pages"
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
