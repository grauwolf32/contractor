import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { lazy, Suspense, useId, useMemo, useState } from "react";
import { Link, useLocation, useNavigate, useParams } from "react-router";

import {
  agentPath,
  getAgentInstructions,
  listAgentTemplateWorkflowBindings,
} from "../../api/agents";
import { usePublicAPI } from "../../api/context";
import {
  getConfiguration,
  listConfigurations,
  type AgentTemplateBody,
  type ConfigurationResource,
} from "../../api/operations";
import { queryKeys } from "../../api/query-keys";
import { CONFIG_ID_PATTERN, CONFIG_VERSION_PATTERN } from "../../api/workflows";
import { CursorControls } from "../../app/cursor-controls";
import { ErrorNotice } from "../../app/error-notice";
import { LoadMoreButton } from "../../app/load-more";
import { nextPageCursor } from "../../app/pagination";
import { QueryView } from "../../app/query-view";
import { DetailHeader, DetailPane } from "../../ui";
import { ConfigurationBodyView } from "../operations/llm-configurations/body";
import {
  useCatalogCursorState,
  withoutCatalogPagination,
} from "./cursor-state";
import { LibraryBackLink } from "./library-parts";
import { locationDestination } from "./navigation";

const MarkdownPreview = lazy(() => import("../artifacts/previews/markdown"));
const INITIAL_CURSOR = null;

function AgentVersionSelector({
  resource,
}: {
  resource: ConfigurationResource;
}) {
  const api = usePublicAPI();
  const navigate = useNavigate();
  const location = useLocation();
  const query = useInfiniteQuery({
    queryKey: queryKeys.catalog.agentVersions(resource.ref.name),
    initialPageParam: INITIAL_CURSOR as string | null,
    queryFn: ({ pageParam, signal }) =>
      listConfigurations(api, "agent-templates", {
        name: resource.ref.name,
        ...(pageParam === null ? {} : { cursor: pageParam }),
        signal,
      }),
    getNextPageParam: (page) => nextPageCursor(page.page),
  });
  const versions = useMemo(() => {
    const byVersion = new Map([[resource.ref.version, resource]]);
    for (const page of query.data?.pages ?? []) {
      for (const item of page.items) {
        if (item.ref.name === resource.ref.name) {
          byVersion.set(item.ref.version, item);
        }
      }
    }
    return [...byVersion.values()].sort((left, right) =>
      left.ref.version.localeCompare(right.ref.version),
    );
  }, [query.data, resource]);

  return (
    <section className="library-agent-versions" aria-label="Agent versions">
      <label className="library-field">
        <span>Published version</span>
        <select
          value={resource.ref.version}
          onChange={(event) =>
            void navigate(agentPath(resource.ref.name, event.target.value), {
              state: withoutCatalogPagination(location.state),
            })
          }
        >
          {versions.map((item) => (
            <option
              key={`${item.ref.version}:${item.ref.digest}`}
              value={item.ref.version}
            >
              {item.ref.version}
            </option>
          ))}
        </select>
      </label>
      <span className="library-muted">
        {versions.length} version{versions.length === 1 ? "" : "s"} loaded
      </span>
      <LoadMoreButton
        query={query}
        label="Load more versions"
        pendingLabel="Loading versions…"
      />
      {query.isPending ? (
        <p className="library-muted" role="status">
          Loading published versions…
        </p>
      ) : null}
      {query.error ? <ErrorNotice error={query.error} /> : null}
    </section>
  );
}

function AgentPrompt({ resource }: { resource: ConfigurationResource }) {
  const api = usePublicAPI();
  const heading = useId();
  const [view, setView] = useState<"preview" | "source">("preview");
  const [copyStatus, setCopyStatus] = useState("");
  const query = useQuery({
    queryKey: queryKeys.catalog.agentInstructions(resource.ref),
    queryFn: async ({ signal }) => {
      const result = await getAgentInstructions(
        api,
        resource.ref.name,
        resource.ref.version,
        signal,
      );
      const body = resource.body as AgentTemplateBody;
      if (
        body.instructions === undefined ||
        result.template.digest !== resource.ref.digest ||
        result.instructions.digest !== body.instructions.digest ||
        result.instructions.ref !== body.instructions.ref
      ) {
        throw new Error(
          "Agent instructions do not match this version. Refresh to reload the Library.",
        );
      }
      return result;
    },
  });
  async function copyPrompt() {
    if (query.data === undefined) return;
    try {
      await navigator.clipboard.writeText(query.data.instructions.text);
      setCopyStatus("Prompt copied.");
    } catch {
      setCopyStatus(
        "Could not copy. Select the text in Source to copy it manually.",
      );
    }
  }
  return (
    <div className="library-agent-grid">
      <section
        className="library-panel library-prompt"
        aria-labelledby={heading}
      >
        <header className="library-panel-header">
          <div>
            <h3 id={heading} className="library-block-title">
              Base prompt
            </h3>
            <p className="library-muted">
              Task context and Skill instructions are added during execution.
            </p>
          </div>
          <div className="library-panel-actions">
            <div
              className="library-segmented"
              role="group"
              aria-label="Prompt view"
            >
              <button
                type="button"
                aria-pressed={view === "preview"}
                onClick={() => setView("preview")}
              >
                Preview
              </button>
              <button
                type="button"
                aria-pressed={view === "source"}
                onClick={() => setView("source")}
              >
                Source
              </button>
            </div>
            <button
              type="button"
              className="ui-btn"
              data-size="sm"
              disabled={query.data === undefined || query.error !== null}
              onClick={() => void copyPrompt()}
            >
              Copy
            </button>
          </div>
        </header>
        {copyStatus && (
          <p className="library-muted" role="status">
            {copyStatus}
          </p>
        )}
        <QueryView
          query={query}
          loading={<p role="status">Loading prompt…</p>}
          onRetry={() => void query.refetch()}
        >
          {(data) =>
            view === "source" ? (
              <pre className="catalog-prompt-source" tabIndex={0}>
                {data.instructions.text}
              </pre>
            ) : (
              <div className="library-prompt-preview">
                <Suspense fallback={<p role="status">Loading preview…</p>}>
                  <MarkdownPreview source={data.instructions.text} />
                </Suspense>
              </div>
            )
          }
        </QueryView>
      </section>
      <AgentConfiguration resource={resource} />
    </div>
  );
}

/** The template's declared settings: runtime, model policy, tools, Skills. */
function AgentConfiguration({ resource }: { resource: ConfigurationResource }) {
  return (
    <aside
      className="library-panel library-agent-configuration"
      aria-label="Agent configuration"
    >
      <h3 className="library-block-title">Configuration</h3>
      <ConfigurationBodyView resource={resource} />
    </aside>
  );
}

function AgentUsage({ resource }: { resource: ConfigurationResource }) {
  const api = usePublicAPI();
  const location = useLocation();
  const heading = useId();
  const pagination = useCatalogCursorState("usageCursor", "usagePage");
  const { cursor, page } = pagination;
  const query = useQuery({
    queryKey: queryKeys.catalog.agentUsage(resource.ref, cursor),
    queryFn: ({ signal }) =>
      listAgentTemplateWorkflowBindings(
        api,
        resource.ref.name,
        resource.ref.version,
        { ...(cursor === undefined ? {} : { cursor }), signal },
      ),
  });

  return (
    <section className="library-block" aria-labelledby={heading}>
      <header className="library-block-header">
        <h3 id={heading} className="library-block-title">
          Where used
        </h3>
        <span className="library-muted">Page {page}</span>
      </header>
      <QueryView
        query={query}
        loading={<p role="status">Loading Workflow bindings…</p>}
        onRetry={() => void query.refetch()}
        isEmpty={(data) => data.items.length === 0}
        empty={
          <p className="library-muted">
            This Agent version is not referenced by a published Workflow.
          </p>
        }
      >
        {(data) => (
          <ul className="library-rows library-usage">
            {data.items.map((binding) => {
              const workflow = `${binding.workflow.name}@${binding.workflow.version}`;
              return (
                <li
                  key={`${workflow}:${binding.stage}:${binding.logicalWorker}`}
                >
                  <Link
                    className="library-row-name"
                    to={`/catalog/workflows/${encodeURIComponent(binding.workflow.name)}/${encodeURIComponent(binding.workflow.version)}`}
                    state={{
                      returnTo: locationDestination(location),
                      returnLabel: `${resource.ref.name}@${resource.ref.version}`,
                      returnState: location.state,
                    }}
                  >
                    {workflow}
                  </Link>
                  <span className="library-row-detail">
                    Stage <code>{binding.stage}</code>
                  </span>
                  <span className="library-row-detail">
                    Worker <code>{binding.logicalWorker}</code>
                  </span>
                </li>
              );
            })}
          </ul>
        )}
      </QueryView>
      {query.data === undefined ? null : (
        <CursorControls
          label="Agent usage pages"
          {...pagination.controls(query.data.page)}
        />
      )}
    </section>
  );
}

/** Library → Agents → one Agent template at an exact version. */
export function AgentDetailRoute() {
  const api = usePublicAPI();
  const titleId = useId();
  const { name = "", version = "" } = useParams();
  const valid =
    CONFIG_ID_PATTERN.test(name) && CONFIG_VERSION_PATTERN.test(version);
  const query = useQuery({
    queryKey: queryKeys.configurations.detail("agent-templates", name, version),
    queryFn: ({ signal }) =>
      getConfiguration(api, "agent-templates", name, version, signal),
    enabled: valid,
  });
  const body = query.data?.body as AgentTemplateBody | undefined;
  return (
    <div className="library-detail">
      <LibraryBackLink
        fallback={{ returnTo: "/catalog/agents", returnLabel: "All agents" }}
      />
      <article className="library-sheet" aria-labelledby={titleId}>
        <DetailPane
          header={
            <DetailHeader
              actions={
                <Link
                  className="ui-btn"
                  to="/catalog/studio?kind=AgentTemplate"
                >
                  Open Node Studio
                </Link>
              }
              title={
                <span id={titleId}>
                  {name}
                  <span className="library-title-version">@{version}</span>
                </span>
              }
              meta={
                <>
                  <span>Agent template</span>
                  {body === undefined ? null : <span>{body.runtime}</span>}
                </>
              }
            />
          }
        >
          {!valid ? (
            <ErrorNotice error={new Error("Agent version is invalid")} />
          ) : (
            <QueryView
              query={query}
              loading={
                <p className="library-muted" role="status">
                  Loading Agent version…
                </p>
              }
              onRetry={() => void query.refetch()}
            >
              {(data) => (
                <>
                  <AgentVersionSelector resource={data} />
                  {(data.body as AgentTemplateBody).runtime === "tool@1" ? (
                    <div className="library-agent-tool">
                      <p className="library-note">
                        This agent runs its configured tool directly without a
                        model or prompt.
                      </p>
                      <AgentConfiguration resource={data} />
                    </div>
                  ) : (
                    <AgentPrompt
                      key={`prompt:${data.ref.digest}`}
                      resource={data}
                    />
                  )}
                  <AgentUsage
                    key={`usage:${data.ref.digest}`}
                    resource={data}
                  />
                </>
              )}
            </QueryView>
          )}
        </DetailPane>
      </article>
    </div>
  );
}
