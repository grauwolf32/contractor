import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router";

import { agentPath } from "../../api/agents";
import { usePublicAPI } from "../../api/context";
import {
  listConfigurations,
  type AgentTemplateBody,
} from "../../api/operations";
import { queryKeys } from "../../api/query-keys";
import { CursorControls, ErrorNotice } from "../artifacts/common";

export function AgentListRoute() {
  const api = usePublicAPI();
  const [cursors, setCursors] = useState<Array<string | undefined>>([
    undefined,
  ]);
  const cursor = cursors.at(-1);
  const [search, setSearch] = useState("");
  const query = useQuery({
    queryKey: queryKeys.configurations.list("agent-templates", cursor),
    queryFn: () =>
      listConfigurations(
        api,
        "agent-templates",
        cursor === undefined ? {} : { cursor },
      ),
  });
  const items = (query.data?.items ?? []).filter((item) =>
    `${item.ref.name} ${item.ref.version} ${(item.body as AgentTemplateBody).description ?? ""}`
      .toLowerCase()
      .includes(search.trim().toLowerCase()),
  );
  return (
    <section className="catalog-agents">
      <header className="route-header-row">
        <div>
          <h3>Agents</h3>
          <p className="lede">
            Explore agent instructions, skills and tools by version.
          </p>
        </div>
        <label className="catalog-search">
          Search this page
          <input
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </label>
      </header>
      {query.isPending ? (
        <p role="status">Loading agents…</p>
      ) : query.error ? (
        <ErrorNotice error={query.error} />
      ) : items.length === 0 ? (
        <p className="panel compact-empty">
          {search ? "No matching agents on this page." : "No agents published."}
        </p>
      ) : (
        <div className="catalog-agent-grid">
          {items.map((item) => {
            const body = item.body as AgentTemplateBody;
            return (
              <Link
                className="panel catalog-agent-card"
                key={`${item.ref.name}@${item.ref.version}`}
                to={agentPath(item.ref.name, item.ref.version)}
              >
                <div className="catalog-agent-identity">
                  <strong>{item.ref.name}</strong>
                  <span className="state-badge">{item.ref.version}</span>
                </div>
                <p>{body.description}</p>
                <span className="muted-copy">
                  {body.runtime} · {body.skills?.length ?? 0} skills ·{" "}
                  {body.toolsets?.reduce(
                    (total, item) => total + item.tools.length,
                    0,
                  ) ?? 0}{" "}
                  tools
                </span>
                <span className="catalog-agent-open">View prompt →</span>
              </Link>
            );
          })}
        </div>
      )}
      <CursorControls
        label="Agent pages"
        canGoBack={cursors.length > 1}
        {...(query.data?.page.nextCursor === undefined
          ? {}
          : { nextCursor: query.data.page.nextCursor })}
        onBack={() => {
          setSearch("");
          setCursors((value) => value.slice(0, -1));
        }}
        onNext={(next) => {
          setSearch("");
          setCursors((value) => [...value, next]);
        }}
      />
    </section>
  );
}
