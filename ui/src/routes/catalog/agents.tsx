import { libraryReadableName } from "./presentation";
import { useLibraryBrowser } from "./browser-state";
import { useQuery } from "@tanstack/react-query";
import { useId } from "react";
import { useParams } from "react-router";
import { agentPath } from "../../api/agents";
import { usePublicAPI } from "../../api/context";
import {
  listConfigurations,
  type AgentTemplateBody,
  type ConfigurationResource,
} from "../../api/operations";
import { queryKeys } from "../../api/query-keys";
import { CursorControls } from "../../app/cursor-controls";
import { compareVersions, plural } from "../../app/format";
import { QueryView } from "../../app/query-view";
import { EmptyState } from "../../ui";
import { AgentDetailRoute } from "./agent-detail";
import { LibraryBrowser, LibraryInventory, LibraryTag } from "./browser";
import { LibrarySearch, LibrarySectionHeader } from "./library-parts";
import { useCatalogQueryState } from "./query-state";

const BASE = "/catalog/agents";
export function AgentListRoute() {
  const api = usePublicAPI();
  const heading = useId();
  const { name, version } = useParams();
  const state = useCatalogQueryState(BASE);
  const browser = useLibraryBrowser(BASE);
  const query = useQuery({
    queryKey: queryKeys.catalog.agentTemplates(
      state.committedSearch,
      state.cursor,
    ),
    queryFn: ({ signal }) =>
      listConfigurations(api, "agent-templates", {
        ...(state.committedSearch === "" ? {} : { q: state.committedSearch }),
        ...(state.cursor === undefined ? {} : { cursor: state.cursor }),
        signal,
      }),
  });
  const families = new Map<string, ConfigurationResource[]>();
  for (const item of query.data?.items ?? []) {
    const values = families.get(item.ref.name) ?? [];
    values.push(item);
    families.set(item.ref.name, values);
  }
  const shown = [...families.values()].map((items) => {
    items.sort((a, b) => compareVersions(b.ref.version, a.ref.version));
    return (
      items.find((i) => i.ref.name === name && i.ref.version === version) ??
      items[0]!
    );
  });
  const first = shown[0];
  const selectedName = name ?? first?.ref.name;
  const selectedVersion =
    name !== undefined ? (version ?? "") : first?.ref.version;
  const toolCount = shown.filter(
    (i) => (i.body as AgentTemplateBody).runtime === "tool@1",
  ).length;
  return (
    <LibraryBrowser
      base={BASE}
      title="Agents"
      selected={name !== undefined}
      list={
        <div className="library-inventory">
          <LibrarySectionHeader
            id={heading}
            title="Agents"
            description="Instructions, tools and their purpose."
            actions={<span className="library-count-pill">{shown.length}</span>}
          />
          <LibrarySearch
            label="Search agents"
            placeholder="Search agents…"
            value={state.draftSearch}
            onChange={state.changeDraftSearch}
          />
          <div
            className="library-runtime-composition"
            aria-label="Agent types on this page"
          >
            <LibraryTag tone="blue">
              Model based {shown.length - toolCount}
            </LibraryTag>
            <LibraryTag tone="amber">Tool only {toolCount}</LibraryTag>
          </div>
          <p className="library-inventory-count">
            <span>Page {state.page}</span> · {plural(shown.length, "Agent")} on
            this page
          </p>
          <QueryView
            query={query}
            loading={<p role="status">Searching published Agents…</p>}
            isEmpty={(data) => data.items.length === 0}
            empty={
              <EmptyState
                title={
                  state.committedSearch
                    ? "No Agents match this search."
                    : "No Agents published."
                }
              >
                Try a name, version or description.
              </EmptyState>
            }
          >
            {() => (
              <LibraryInventory
                selected={selectedName}
                state={browser.rowState}
                items={shown.map((item) => {
                  const body = item.body as AgentTemplateBody;
                  const tool = body.runtime === "tool@1";
                  const tools = body.toolsets.reduce(
                    (n, t) => n + t.tools.length,
                    0,
                  );
                  return {
                    key: item.ref.name,
                    to:
                      agentPath(item.ref.name, item.ref.version) +
                      browser.search,
                    title: libraryReadableName(item.ref.name),
                    description: body.description || "No authored description.",
                    kind: tool ? "Tool agent" : "Model agent",
                    icon: tool ? "tool" : "agent",
                    tone: tool ? "amber" : "blue",
                    meta: (
                      <>
                        <LibraryTag tone={tool ? "amber" : "blue"}>
                          {tool ? "No model" : "Model based"}
                        </LibraryTag>
                        <span>
                          {plural(tools, "tool")} ·{" "}
                          {plural(body.skills?.length ?? 0, "Skill")}
                        </span>
                      </>
                    ),
                  };
                })}
              />
            )}
          </QueryView>
          {query.data ? (
            <CursorControls
              size="sm"
              label="Agent pages"
              {...state.controls(query.data.page)}
            />
          ) : null}
        </div>
      }
      detail={
        selectedName !== undefined && selectedVersion !== undefined ? (
          <AgentDetailRoute name={selectedName} version={selectedVersion} />
        ) : (
          <EmptyState title="Choose an Agent">
            Its instructions and tools will appear here.
          </EmptyState>
        )
      }
    />
  );
}
