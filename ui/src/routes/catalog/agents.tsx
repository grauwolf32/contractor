import { useQuery } from "@tanstack/react-query";
import { useId, useState } from "react";
import { Link, useLocation } from "react-router";

import { agentPath } from "../../api/agents";
import { usePublicAPI } from "../../api/context";
import {
  listConfigurations,
  type AgentTemplateBody,
  type ConfigurationResource,
} from "../../api/operations";
import { queryKeys } from "../../api/query-keys";
import { CursorControls } from "../../app/cursor-controls";
import { QueryView } from "../../app/query-view";
import { EmptyState } from "../../ui";
import { compareWorkflowVersions } from "../workflows/families";
import { LibrarySearch, LibrarySectionHeader } from "./library-parts";
import { locationDestination } from "./navigation";
import { useCatalogQueryState } from "./query-state";

function plural(count: number, singular: string, many: string): string {
  return `${count} ${count === 1 ? singular : many}`;
}

function AgentCard({
  name,
  versions,
  item,
  onVersion,
}: {
  name: string;
  versions: ConfigurationResource[];
  item: ConfigurationResource;
  onVersion: (version: string) => void;
}) {
  const location = useLocation();
  const body = item.body as AgentTemplateBody;
  const tools =
    body.toolsets?.reduce(
      (total, toolset) => total + toolset.tools.length,
      0,
    ) ?? 0;
  return (
    <article className="library-card library-agent-card">
      <div className="library-card-top">
        <label className="library-version">
          <span>Version</span>
          <select
            aria-label={`Version of ${name}`}
            value={item.ref.version}
            onChange={(event) => onVersion(event.target.value)}
          >
            {versions.map((version) => (
              <option key={version.ref.version} value={version.ref.version}>
                {version.ref.version}
              </option>
            ))}
          </select>
        </label>
        <small className="library-muted">
          {plural(versions.length, "matching version", "matching versions")} on
          this page
        </small>
      </div>
      <Link
        className="library-agent-link"
        to={agentPath(item.ref.name, item.ref.version)}
        state={{
          returnTo: locationDestination(location),
          returnLabel: "Agent search",
          returnState: location.state,
        }}
      >
        <span className="library-agent-identity">
          <strong>{item.ref.name}</strong>
          <span className="library-tag">{item.ref.version}</span>
        </span>
        <span className="library-card-text">
          {body.description || "No authored description."}
        </span>
        <span className="library-agent-facts">
          {body.runtime} · {plural(body.skills?.length ?? 0, "Skill", "Skills")}{" "}
          · {plural(tools, "tool", "tools")}
        </span>
        <code className="library-selector">
          {item.ref.name}@{item.ref.version}
        </code>
        <span className="library-agent-open">
          Inspect version <span aria-hidden="true">→</span>
        </span>
      </Link>
    </article>
  );
}

/** Library → Agents: published Agent templates, searched on the server. */
export function AgentListRoute() {
  const api = usePublicAPI();
  const heading = useId();
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
  const [choices, setChoices] = useState<Record<string, string>>({});
  const families = new Map<string, ConfigurationResource[]>();
  for (const item of query.data?.items ?? []) {
    const family = families.get(item.ref.name) ?? [];
    family.push(item);
    families.set(item.ref.name, family);
  }

  return (
    <section className="library-section" aria-labelledby={heading}>
      <LibrarySectionHeader
        id={heading}
        title="Agents"
        description="Agent versions with their instructions, Skills, tools and the Workflows that use them."
        actions={
          <LibrarySearch
            label="Search agents"
            placeholder="Name, version or description"
            value={state.draftSearch}
            onChange={state.changeDraftSearch}
          />
        }
      />

      <p className="library-count" aria-live="polite">
        <span>Page {state.page}</span>
        {state.committedSearch === "" ? null : (
          <span>
            Results for <strong>{state.committedSearch}</strong>
          </span>
        )}
      </p>

      <QueryView
        query={query}
        loading={<p role="status">Searching published Agents…</p>}
        onRetry={() => void query.refetch()}
        isEmpty={(data) => data.items.length === 0}
        empty={
          <div className="library-empty">
            <EmptyState
              title={
                state.committedSearch === ""
                  ? "No Agents published."
                  : "No Agents match this search."
              }
            >
              {state.committedSearch === ""
                ? null
                : "Try a different literal name, version or description."}
            </EmptyState>
          </div>
        }
      >
        {() => (
          <div className="library-grid">
            {[...families].map(([name, versions]) => {
              versions.sort((a, b) =>
                compareWorkflowVersions(b.ref.version, a.ref.version),
              );
              const item =
                versions.find(
                  (version) => version.ref.version === choices[name],
                ) ?? versions[0]!;
              return (
                <AgentCard
                  key={name}
                  name={name}
                  versions={versions}
                  item={item}
                  onVersion={(version) =>
                    setChoices((previous) => ({ ...previous, [name]: version }))
                  }
                />
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
