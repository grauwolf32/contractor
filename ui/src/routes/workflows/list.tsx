import { useLocation } from "react-router";
import { ErrorNotice } from "../artifacts/common";
import { locationDestination } from "../catalog/navigation";
import { useCatalogQueryState } from "../catalog/query-state";
import { WorkflowCard } from "./card";
import { useWorkflowFamilies } from "./families";
import { useWorkflowInventory } from "./inventory";

export function WorkflowListRoute() {
  const location = useLocation();
  const state = useCatalogQueryState();
  const query = useWorkflowInventory();
  const { families, selectVersion } = useWorkflowFamilies(query.data ?? []);
  const term = state.committedSearch.toLowerCase();
  const filtered = families.filter((family) =>
    family.versions.some((w) =>
      `${w.ref.name} ${w.ref.version} ${w.presentation?.displayName ?? ""} ${w.presentation?.description ?? ""}`
        .toLowerCase()
        .includes(term),
    ),
  );
  return (
    <section className="route-page workflow-page">
      <header className="route-header-row">
        <div>
          <h2>Workflows</h2>
          <p className="lede">
            Explore required materials, results and exact published versions.
          </p>
        </div>
      </header>
      <div className="workflow-discovery-toolbar">
        <p className="muted-copy" role="status">
          {query.data
            ? `${filtered.length} workflows · ${query.data.length} published versions`
            : "Loading published versions…"}
        </p>
        <div className="catalog-discovery-actions">
          <label className="catalog-search">
            Search workflows
            <input
              type="search"
              value={state.draftSearch}
              placeholder="Name, version or description"
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
      </div>
      {query.error ? (
        <ErrorNotice error={query.error} />
      ) : query.isPending ? (
        <p className="loading-copy" role="status">
          Loading the complete Workflow inventory…
        </p>
      ) : filtered.length === 0 ? (
        <div className="panel compact-empty">
          <strong>
            {term
              ? "No Workflows match this search."
              : "No published Workflows found."}
          </strong>
          <p>
            {term
              ? "Try a different name, version or description."
              : "Publish a Workflow to make it available here."}
          </p>
          {term ? (
            <button
              type="button"
              className="secondary-button"
              onClick={() => state.changeDraftSearch("")}
            >
              Clear search
            </button>
          ) : null}
        </div>
      ) : (
        <div className="workflow-family-grid">
          {filtered.map(({ name, versions, workflow }) => (
            <WorkflowCard
              key={name}
              workflow={workflow}
              versions={versions}
              onVersion={(version) => selectVersion(name, version)}
              returnTo={locationDestination(location)}
              returnState={location.state}
            />
          ))}
        </div>
      )}
    </section>
  );
}
