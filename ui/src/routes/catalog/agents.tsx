import { useState } from "react";
import { compareWorkflowVersions } from "../workflows/families";
import { useQuery } from "@tanstack/react-query";
import { Link, useLocation } from "react-router";

import { agentPath } from "../../api/agents";
import { usePublicAPI } from "../../api/context";
import {
  listConfigurations,
  type AgentTemplateBody,
} from "../../api/operations";
import { CursorControls } from "../../app/cursor-controls";
import { locationDestination } from "./navigation";
import { useCatalogQueryState } from "./query-state";
import { QueryView } from "../../app/query-view";
import { queryKeys } from "../../api/query-keys";

export function AgentListRoute() {
  const api = usePublicAPI();
  const location = useLocation();
  const state = useCatalogQueryState();
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
  const returnTo = locationDestination(location);
  const [choices, setChoices] = useState<Record<string, string>>({});
  const families = new Map<string, NonNullable<typeof query.data>["items"]>();
  for (const item of query.data?.items ?? []) {
    const family = families.get(item.ref.name) ?? [];
    family.push(item);
    families.set(item.ref.name, family);
  }

  return (
    <section className="catalog-agents">
      <header className="route-header-row catalog-discovery-header">
        <div>
          <h2>Agents</h2>
          <p className="lede">
            Explore Agent versions, instructions, Skills, tools and Workflow
            usage.
          </p>
        </div>
        <label className="catalog-search">
          Search agents
          <input
            type="search"
            value={state.draftSearch}
            placeholder="Name, version or description"
            onChange={(event) => state.changeDraftSearch(event.target.value)}
          />
        </label>
      </header>

      <div className="catalog-result-summary" aria-live="polite">
        <span>Page {state.page}</span>
        {state.committedSearch === "" ? null : (
          <span>
            Results for <strong>{state.committedSearch}</strong>
          </span>
        )}
      </div>

      <QueryView
        query={query}
        loading={<p role="status">Searching published Agents…</p>}
        onRetry={() => void query.refetch()}
        isEmpty={(data) => data.items.length === 0}
        empty={
          <div className="panel compact-empty">
            <strong>
              {state.committedSearch === ""
                ? "No Agents published."
                : "No Agents match this search."}
            </strong>
            {state.committedSearch === "" ? null : (
              <p>Try a different literal name, version or description.</p>
            )}
          </div>
        }
      >
        {() => (
          <div className="catalog-agent-grid">
            {[...families].map(([name, versions]) => {
              versions.sort((a, b) =>
                compareWorkflowVersions(b.ref.version, a.ref.version),
              );
              const item =
                versions.find(
                  (version) => version.ref.version === choices[name],
                ) ?? versions[0]!;
              const body = item.body as AgentTemplateBody;
              return (
                <article className="panel catalog-agent-family" key={name}>
                  <label className="catalog-agent-version">
                    Version
                    <select
                      aria-label={`Version of ${name}`}
                      value={item.ref.version}
                      onChange={(event) =>
                        setChoices((previous) => ({
                          ...previous,
                          [name]: event.target.value,
                        }))
                      }
                    >
                      {versions.map((version) => (
                        <option
                          key={version.ref.version}
                          value={version.ref.version}
                        >
                          {version.ref.version}
                        </option>
                      ))}
                    </select>
                    <small>
                      {versions.length} matching{" "}
                      {versions.length === 1 ? "version" : "versions"} on this
                      page
                    </small>
                  </label>
                  <Link
                    className="catalog-agent-card"
                    to={agentPath(item.ref.name, item.ref.version)}
                    state={{
                      returnTo,
                      returnLabel: "Agent search",
                      returnState: location.state,
                    }}
                  >
                    <div className="catalog-agent-identity">
                      <strong>{item.ref.name}</strong>
                      <span className="state-badge">{item.ref.version}</span>
                    </div>
                    <p>{body.description || "No authored description."}</p>
                    <span className="muted-copy">
                      {body.runtime} · {body.skills?.length ?? 0} skills ·{" "}
                      {body.toolsets?.reduce(
                        (total, toolset) => total + toolset.tools.length,
                        0,
                      ) ?? 0}{" "}
                      tools
                    </span>
                    <code className="catalog-exact-selector">
                      {item.ref.name}@{item.ref.version}
                    </code>
                    <span className="catalog-agent-open">
                      Inspect version →
                    </span>
                  </Link>
                </article>
              );
            })}
          </div>
        )}
      </QueryView>

      {query.data === undefined ? null : (
        <CursorControls
          label="Agent pages"
          {...state.controls(query.data.page)}
        />
      )}
    </section>
  );
}
