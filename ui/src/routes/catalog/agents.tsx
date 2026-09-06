import { useQuery } from "@tanstack/react-query";
import { Link, useLocation } from "react-router";

import { agentPath } from "../../api/agents";
import { usePublicAPI } from "../../api/context";
import {
  listConfigurations,
  type AgentTemplateBody,
} from "../../api/operations";
import { CursorControls, ErrorNotice } from "../artifacts/common";
import { locationDestination } from "./navigation";
import { useCatalogQueryState } from "./query-state";

export function AgentListRoute() {
  const api = usePublicAPI();
  const location = useLocation();
  const state = useCatalogQueryState();
  const query = useQuery({
    queryKey: [
      "catalog",
      "agent-templates",
      state.committedSearch,
      state.cursor ?? null,
    ],
    queryFn: ({ signal }) =>
      listConfigurations(api, "agent-templates", {
        ...(state.committedSearch === "" ? {} : { q: state.committedSearch }),
        ...(state.cursor === undefined ? {} : { cursor: state.cursor }),
        signal,
      }),
  });
  const returnTo = locationDestination(location);

  return (
    <section className="catalog-agents">
      <header className="route-header-row catalog-discovery-header">
        <div>
          <h3>Agents</h3>
          <p className="lede">
            Explore exact Agent versions, instructions, Skills, tools and
            Workflow usage.
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

      {query.isPending ? (
        <p role="status">Searching published Agents…</p>
      ) : query.error ? (
        <ErrorNotice error={query.error} />
      ) : query.data.items.length === 0 ? (
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
      ) : (
        <div className="catalog-agent-grid">
          {query.data.items.map((item) => {
            const body = item.body as AgentTemplateBody;
            return (
              <Link
                className="panel catalog-agent-card"
                key={`${item.ref.name}@${item.ref.version}:${item.ref.digest}`}
                to={agentPath(item.ref.name, item.ref.version)}
                state={{ returnTo, returnLabel: "Agent search" }}
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
                  Inspect exact version →
                </span>
              </Link>
            );
          })}
        </div>
      )}

      {query.data === undefined ? null : (
        <CursorControls
          label="Agent pages"
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
