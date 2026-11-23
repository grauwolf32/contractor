import { libraryReadableName } from "../catalog/presentation";
import { useLibraryBrowser } from "../catalog/browser-state";
import { useId } from "react";
import { useParams, useSearchParams } from "react-router";
import { plural } from "../../app/format";
import { QueryView } from "../../app/query-view";
import { RefreshButton } from "../../app/refresh-button";
import { Button, EmptyState } from "../../ui";
import {
  LibraryBrowser,
  LibraryFilters,
  LibraryInventory,
  LibraryTag,
} from "../catalog/browser";
import { LibrarySearch, LibrarySectionHeader } from "../catalog/library-parts";
import { useCatalogQueryState } from "../catalog/query-state";
import { WorkflowDetailRoute } from "./detail";
import { groupWorkflowVersions } from "./families";
import { useWorkflowInventory } from "./inventory";

const BASE = "/catalog/workflows";
export function WorkflowListRoute() {
  const { name, version } = useParams();
  const heading = useId();
  const state = useCatalogQueryState(BASE);
  const browser = useLibraryBrowser(BASE);
  const query = useWorkflowInventory();
  const [params, setParams] = useSearchParams();
  const mode = params.get("inputs") ?? "all";
  const term = state.committedSearch.toLowerCase();
  const shown = groupWorkflowVersions(query.data ?? []).flatMap((family) => {
    const versions = family.versions.filter(
      (w) =>
        `${w.ref.name} ${w.ref.version} ${w.presentation?.displayName ?? ""} ${w.presentation?.description ?? ""}`
          .toLowerCase()
          .includes(term) &&
        (mode === "all" ||
          (mode === "files"
            ? Object.keys(w.inputs).length > 0
            : Object.keys(w.inputs).length === 0)),
    );
    const workflow =
      versions.find(
        (w) => w.ref.version === (name === family.name ? version : undefined),
      ) ?? versions[0];
    return workflow ? [workflow] : [];
  });
  const first = shown[0];
  const selectedName = name ?? first?.ref.name;
  const selectedVersion =
    name !== undefined ? (version ?? "") : first?.ref.version;
  return (
    <LibraryBrowser
      base={BASE}
      title="Workflows"
      selected={name !== undefined}
      list={
        <div className="library-inventory">
          <LibrarySectionHeader
            id={heading}
            title="Workflows"
            description="From materials to results."
            actions={
              <RefreshButton
                isFetching={query.isFetching}
                onRefresh={() => void query.refetch()}
                label="Refresh"
              />
            }
          />
          <LibrarySearch
            label="Search workflows"
            placeholder="Search workflows…"
            value={state.draftSearch}
            onChange={state.changeDraftSearch}
          />
          <LibraryFilters
            label="Workflow input filters"
            value={mode}
            options={[
              ["all", "All"],
              ["files", "With files"],
              ["none", "No files"],
            ]}
            onChange={(value) => {
              const next = new URLSearchParams(params);
              if (value === "all") next.delete("inputs");
              else next.set("inputs", value);
              setParams(next, { replace: true, state: browser.location.state });
            }}
          />
          <p className="library-inventory-count" role="status">
            {query.data
              ? `${plural(shown.length, "workflow")} · ${plural(query.data.length, "published version")}`
              : "Loading published versions…"}
          </p>
          <QueryView
            query={query}
            loading={
              <p role="status">Loading the complete Workflow inventory…</p>
            }
          >
            {() =>
              shown.length === 0 ? (
                <EmptyState
                  title="No Workflows match this search."
                  action={
                    <Button
                      size="sm"
                      onClick={() => state.changeDraftSearch("")}
                    >
                      Clear search
                    </Button>
                  }
                >
                  Try another name or input filter.
                </EmptyState>
              ) : (
                <LibraryInventory
                  selected={selectedName}
                  state={browser.rowState}
                  items={shown.map((w) => ({
                    key: w.ref.name,
                    to: `${BASE}/${encodeURIComponent(w.ref.name)}/${encodeURIComponent(w.ref.version)}${browser.search}`,
                    title:
                      w.presentation?.displayName ??
                      libraryReadableName(w.ref.name),
                    description: w.presentation?.description ?? "",
                    kind: "Workflow",
                    icon: "workflow",
                    meta: (
                      <>
                        <LibraryTag tone="blue">v{w.ref.version}</LibraryTag>
                        <span>
                          {plural(
                            Object.values(w.inputs).filter((i) => i.required)
                              .length,
                            "required input",
                          )}{" "}
                          · {plural(Object.keys(w.outputs).length, "output")}
                        </span>
                      </>
                    ),
                  }))}
                />
              )
            }
          </QueryView>
        </div>
      }
      detail={
        selectedName !== undefined && selectedVersion !== undefined ? (
          <WorkflowDetailRoute name={selectedName} version={selectedVersion} />
        ) : (
          <EmptyState title="Choose a Workflow">
            Its inputs, stages and results will appear here.
          </EmptyState>
        )
      }
    />
  );
}
