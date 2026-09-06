import { useQuery } from "@tanstack/react-query";
import { Link, useLocation } from "react-router";

import { usePublicAPI } from "../../api/context";
import { listWorkflows, type WorkflowSummary } from "../../api/workflows";
import { CursorControls, ErrorNotice } from "../artifacts/common";
import { locationDestination } from "../catalog/navigation";
import { useCatalogQueryState } from "../catalog/query-state";
import {
  workflowDescription,
  workflowDisplayName,
  workflowSelector,
} from "./presentation";

function ArtifactSlots({
  empty,
  slots,
}: {
  empty: string;
  slots: WorkflowSummary["inputs"] | WorkflowSummary["outputs"];
}) {
  const entries = Object.entries(slots).sort(([left], [right]) =>
    left.localeCompare(right),
  );
  if (entries.length === 0) return <span className="muted-copy">{empty}</span>;
  return (
    <ul className="catalog-slot-list">
      {entries.map(([name, slot]) => (
        <li key={name}>
          <code>{name}</code>
          <span>{slot.required ? "required" : "optional"}</span>
          <small>{slot.mediaTypes.join(", ")}</small>
        </li>
      ))}
    </ul>
  );
}

export function WorkflowListRoute() {
  const api = usePublicAPI();
  const location = useLocation();
  const state = useCatalogQueryState();
  const query = useQuery({
    queryKey: [
      "catalog",
      "workflows",
      state.committedSearch,
      state.cursor ?? null,
    ],
    queryFn: ({ signal }) =>
      listWorkflows(api, {
        ...(state.committedSearch === "" ? {} : { q: state.committedSearch }),
        ...(state.cursor === undefined ? {} : { cursor: state.cursor }),
        signal,
      }),
  });
  const returnTo = locationDestination(location);

  return (
    <section className="route-page workflow-page">
      <header className="route-header-row catalog-discovery-header">
        <div>
          <p className="eyebrow">Published contracts</p>
          <h2>Workflows</h2>
          <p className="lede">
            Find a purpose, inspect its required materials and choose one exact
            published version.
          </p>
        </div>
        <div className="catalog-discovery-actions">
          <label className="catalog-search">
            Search workflows
            <input
              type="search"
              value={state.draftSearch}
              placeholder="Name, version or authored description"
              onChange={(event) => state.changeDraftSearch(event.target.value)}
            />
          </label>
          <button
            className="secondary-button"
            type="button"
            disabled={query.isFetching}
            onClick={() => void query.refetch()}
          >
            {query.isFetching ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </header>

      <div className="catalog-result-summary" aria-live="polite">
        <span>Page {state.page}</span>
        {state.committedSearch === "" ? null : (
          <span>
            Results for <strong>{state.committedSearch}</strong>
          </span>
        )}
      </div>

      {query.isPending ? (
        <p className="loading-copy" role="status">
          Searching published Workflows…
        </p>
      ) : query.error !== null ? (
        <ErrorNotice error={query.error} />
      ) : query.data.items.length === 0 ? (
        <div className="panel compact-empty">
          <strong>
            {state.committedSearch === ""
              ? "No published Workflow versions found."
              : "No Workflows match this search."}
          </strong>
          <p>
            {state.committedSearch === ""
              ? "Publish YAML configuration before creating a Run."
              : "Try a different literal name, version or description."}
          </p>
        </div>
      ) : (
        <div className="catalog-workflow-grid">
          {query.data.items.map((workflow) => {
            const selector = workflowSelector(workflow);
            return (
              <article className="panel catalog-workflow-card" key={selector}>
                <header>
                  <div>
                    <p className="eyebrow">Exact Workflow</p>
                    <h3>
                      <Link
                        to={`/catalog/workflows/${encodeURIComponent(workflow.ref.name)}/${encodeURIComponent(workflow.ref.version)}`}
                        state={{
                          returnTo,
                          returnLabel: "Workflow search",
                        }}
                      >
                        {workflowDisplayName(workflow)}
                      </Link>
                    </h3>
                    <code>{selector}</code>
                  </div>
                  <span className="state-badge">
                    entry {workflow.entryStage}
                  </span>
                </header>
                <p>{workflowDescription(workflow)}</p>
                <div className="catalog-workflow-contracts">
                  <section>
                    <h4>Artifact inputs</h4>
                    <ArtifactSlots
                      slots={workflow.inputs}
                      empty="No Artifact inputs required."
                    />
                  </section>
                  <section>
                    <h4>Declared outputs</h4>
                    <ArtifactSlots
                      slots={workflow.outputs}
                      empty="No Artifact outputs declared."
                    />
                  </section>
                </div>
                <small className="muted-copy">
                  {Object.keys(workflow.parameters).length} string parameter
                  {Object.keys(workflow.parameters).length === 1 ? "" : "s"}
                </small>
              </article>
            );
          })}
        </div>
      )}

      {query.data === undefined ? null : (
        <CursorControls
          label="Workflow pages"
          canGoBack={state.canGoBack}
          {...(query.data.page.hasMore &&
          query.data.page.nextCursor !== undefined
            ? { nextCursor: query.data.page.nextCursor }
            : {})}
          onBack={state.previousPage}
          onNext={state.nextPage}
        />
      )}
    </section>
  );
}
