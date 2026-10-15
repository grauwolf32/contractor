import { useId } from "react";
import { useLocation } from "react-router";

import { plural } from "../../app/format";
import { QueryView } from "../../app/query-view";
import { RefreshButton } from "../../app/refresh-button";
import { Button, EmptyState } from "../../ui";
import { LibrarySearch, LibrarySectionHeader } from "../catalog/library-parts";
import { locationDestination } from "../catalog/navigation";
import { useCatalogQueryState } from "../catalog/query-state";
import { WorkflowCard } from "./card";
import { useWorkflowFamilies } from "./families";
import { useWorkflowInventory } from "./inventory";

/** Library → Workflows: every published Workflow family and its versions. */
export function WorkflowListRoute() {
  const location = useLocation();
  const heading = useId();
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
    <section
      className="library-section workflow-page"
      aria-labelledby={heading}
    >
      <LibrarySectionHeader
        id={heading}
        title="Workflows"
        description="Explore required inputs, results and published versions."
        actions={
          <>
            <LibrarySearch
              label="Search workflows"
              placeholder="Name, version or description"
              value={state.draftSearch}
              onChange={state.changeDraftSearch}
            />
            <RefreshButton
              isFetching={query.isFetching}
              onRefresh={() => void query.refetch()}
              label="Refresh"
            />
          </>
        }
      />
      <p className="library-count" role="status">
        {query.data
          ? `${plural(filtered.length, "workflow")} · ${plural(query.data.length, "published version")}`
          : "Loading published versions…"}
      </p>
      <QueryView
        query={query}
        loading={
          <p className="library-muted" role="status">
            Loading the complete Workflow inventory…
          </p>
        }
      >
        {() =>
          filtered.length === 0 ? (
            <div className="library-empty">
              <EmptyState
                title={
                  term
                    ? "No Workflows match this search."
                    : "No published Workflows found."
                }
                action={
                  term ? (
                    <Button
                      size="sm"
                      onClick={() => state.changeDraftSearch("")}
                    >
                      Clear search
                    </Button>
                  ) : undefined
                }
              >
                {term
                  ? "Try a different name, version or description."
                  : "Publish a Workflow to make it available here."}
              </EmptyState>
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
          )
        }
      </QueryView>
    </section>
  );
}
