import { libraryReadableName } from "./presentation";
import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { lazy, Suspense, useId, useMemo, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router";

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
import {
  CONFIG_NAME_PATTERN,
  CONFIG_VERSION_PATTERN,
} from "../../api/workflows";
import { CursorControls } from "../../app/cursor-controls";
import { ErrorNotice } from "../../app/error-notice";
import { LoadMoreButton } from "../../app/load-more";
import { nextPageCursor } from "../../app/pagination";
import { compareVersions, plural } from "../../app/format";
import { QueryView } from "../../app/query-view";
import { Button, ButtonLink, TechnicalDetails } from "../../ui";
import { ConfigurationBodyView } from "../operations/llm-configurations/body";
import { useCatalogCursorState } from "./cursor-state";
import { LibraryDetailHeading, LibraryTag } from "./browser";
import { Icon } from "../../app/icon";
import { artifactDetailPath } from "../../app/artifact-path";
import { locationDestination } from "./navigation";

const MarkdownPreview = lazy(
  () => import("../../widgets/artifact-preview/previews/markdown"),
);
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
      compareVersions(right.ref.version, left.ref.version),
    );
  }, [query.data, resource]);

  return (
    <section className="library-agent-versions" aria-label="Agent versions">
      <label className="library-version">
        <span>Version</span>
        <select
          className="ui-control"
          data-size="xs"
          aria-label="Published version"
          value={resource.ref.version}
          onChange={(event) => {
            const params = new URLSearchParams(location.search);
            params.delete("usageCursor");
            params.delete("usagePage");
            const state = {
              ...(location.state as Record<string, unknown> | null),
            };
            delete state.catalogUsagePagination;
            void navigate(
              agentPath(resource.ref.name, event.target.value) +
                (params.size ? `?${params}` : ""),
              { state },
            );
          }}
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
      {query.hasNextPage ? (
        <span className="ui-caption">
          {plural(versions.length, "version")} loaded
        </span>
      ) : null}
      <LoadMoreButton
        size="sm"
        query={query}
        label="Load more versions"
        pendingLabel="Loading versions…"
      />
      {query.isPending ? (
        <p className="ui-caption" role="status">
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
    <>
      <section className="library-prompt-panel" aria-labelledby={heading}>
        <header className="library-prompt-header">
          <div>
            <h3 id={heading} className="library-block-title">
              Base instructions
            </h3>
          </div>
          <div className="ui-toolbar">
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
            <Button
              size="sm"
              disabled={query.data === undefined || query.error !== null}
              onClick={() => void copyPrompt()}
              aria-label="Copy instructions"
              title="Copy instructions"
            >
              <Icon name="copy" />
            </Button>
          </div>
        </header>
        {copyStatus && (
          <p className="ui-caption" role="status">
            {copyStatus}
          </p>
        )}
        <QueryView query={query} loading={<p role="status">Loading prompt…</p>}>
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
        <footer>
          Base instructions only. Task context and Skill instructions are added
          during execution.
        </footer>
      </section>
    </>
  );
}

function AgentCapabilities({ resource }: { resource: ConfigurationResource }) {
  const body = resource.body as AgentTemplateBody;
  const count = body.toolsets.reduce((n, set) => n + set.tools.length, 0);
  return (
    <aside
      className="library-context-card library-capabilities"
      aria-label="Agent capabilities"
    >
      <span className="library-eyebrow">Capabilities</span>
      <h3>Tools & Skills</h3>
      <dl className="library-context-facts">
        <div>
          <dt>Runtime</dt>
          <dd>
            <code>{body.runtime}</code>
          </dd>
        </div>
        <div>
          <dt>Sandbox</dt>
          <dd>
            <code>{body.sandboxProfile}</code>
          </dd>
        </div>
      </dl>
      <h4>
        Tools <span>{count}</span>
      </h4>
      {body.toolsets.length === 0 ? (
        <p>No model-visible tools.</p>
      ) : (
        body.toolsets.map((set) => (
          <details className="library-toolset" key={set.ref}>
            <summary>
              <Icon name="tool" />
              <span>{set.ref}</span>
              <small>{set.tools.length}</small>
            </summary>
            <ul>
              {set.tools.map((tool) => (
                <li key={tool}>
                  <code>{tool}</code>
                </li>
              ))}
            </ul>
          </details>
        ))
      )}
      <h4>
        Skills <span>{body.skills?.length ?? 0}</span>
      </h4>
      {body.skills?.length ? (
        <ul className="library-skill-links">
          {body.skills.map((skill) => (
            <li key={`${skill.namespace}/${skill.name}`}>
              <Link to={artifactDetailPath({ kind: "user" }, skill)}>
                <Icon name="docs" />
                {skill.namespace}/{skill.name}
                <span aria-hidden="true">↗</span>
              </Link>
            </li>
          ))}
        </ul>
      ) : (
        <p>No global Skills declared.</p>
      )}
    </aside>
  );
}

function ToolExecution({ body }: { body: AgentTemplateBody }) {
  const execution = body.execution;
  return (
    <section className="library-tool-panel" aria-label="Direct tool execution">
      <header>
        <Icon name="tool" />
        <div>
          <strong>Direct tool call</strong>
          <p>
            This agent runs its configured tool directly without a model or
            prompt.
          </p>
        </div>
      </header>
      {execution ? (
        <div className="library-tool-call">
          <code className="library-tool-name">{execution.tool}</code>
          <dl>
            {Object.entries(execution.arguments).map(([name, arg]) => (
              <div key={name}>
                <dt>{name}</dt>
                <dd>
                  {arg.source === "literal" ? (
                    <>
                      Literal · <code>{JSON.stringify(arg.value)}</code>
                    </>
                  ) : (
                    <>
                      {arg.source} · <code>{arg.name}</code>
                      {arg.optional ? " · optional" : " · required"}
                      {"parse" in arg ? ` · ${arg.parse}` : ""}
                      {"default" in arg
                        ? ` · default: ${JSON.stringify(arg.default)}`
                        : ""}
                    </>
                  )}
                </dd>
              </div>
            ))}
            <div>
              <dt>Result artifact</dt>
              <dd>
                <code>{execution.resultArtifact}</code>
              </dd>
            </div>
            <div>
              <dt>Timeout</dt>
              <dd>{execution.timeoutSeconds}s</dd>
            </div>
          </dl>
        </div>
      ) : null}
    </section>
  );
}

function AgentUsage({ resource }: { resource: ConfigurationResource }) {
  const api = usePublicAPI();
  const location = useLocation();
  const heading = useId();
  const pagination = useCatalogCursorState("usageCursor", "usagePage", {
    stateKey: "catalogUsagePagination",
    path: agentPath(resource.ref.name, resource.ref.version),
    ignoreParams: ["q", "cursor", "page"],
  });
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
    <section className="library-design-section" aria-labelledby={heading}>
      <header className="library-design-section-header">
        <h3 id={heading} className="library-block-title">
          Where used
        </h3>
        <span className="ui-caption">Page {page}</span>
      </header>
      <QueryView
        query={query}
        loading={<p role="status">Loading Workflow bindings…</p>}
        isEmpty={(data) => data.items.length === 0}
        empty={
          <p className="ui-caption">
            This Agent version is not referenced by a published Workflow.
          </p>
        }
      >
        {(data) => (
          <ul className="library-usage-links">
            {data.items.map((binding) => {
              const workflow = `${binding.workflow.name}@${binding.workflow.version}`;
              return (
                <li
                  key={`${workflow}:${binding.stage}:${binding.logicalWorker}`}
                >
                  <Link
                    className="library-usage-link"
                    to={`/catalog/workflows/${encodeURIComponent(binding.workflow.name)}/${encodeURIComponent(binding.workflow.version)}`}
                    state={{
                      returnTo: locationDestination({
                        ...location,
                        pathname: agentPath(
                          resource.ref.name,
                          resource.ref.version,
                        ),
                      }),
                      returnLabel: `${resource.ref.name}@${resource.ref.version}`,
                      returnState: location.state,
                    }}
                  >
                    <Icon name="workflow" />
                    <span>
                      <strong>{workflow}</strong>
                      <small>
                        Stage {binding.stage} · Worker {binding.logicalWorker}
                      </small>
                    </span>
                    <span aria-hidden="true">↗</span>
                  </Link>
                </li>
              );
            })}
          </ul>
        )}
      </QueryView>
      {query.data === undefined ? null : (
        <CursorControls
          size="sm"
          label="Agent usage pages"
          {...pagination.controls(query.data.page)}
        />
      )}
    </section>
  );
}

/** Library → Agents → one Agent template at an exact version. */
export function AgentDetailRoute({
  name,
  version,
}: {
  name: string;
  version: string;
}) {
  const api = usePublicAPI();
  const valid =
    CONFIG_NAME_PATTERN.test(name) && CONFIG_VERSION_PATTERN.test(version);
  const query = useQuery({
    queryKey: queryKeys.configurations.detail("agent-templates", name, version),
    queryFn: ({ signal }) =>
      getConfiguration(api, "agent-templates", name, version, signal),
    enabled: valid,
  });
  return !valid ? (
    <ErrorNotice error={new Error("Agent version is invalid")} />
  ) : (
    <QueryView
      query={query}
      loading={<p role="status">Loading Agent version…</p>}
    >
      {(data) => {
        const body = data.body as AgentTemplateBody;
        const tool = body.runtime === "tool@1";
        return (
          <article className="library-design-detail library-agent-detail">
            <LibraryDetailHeading
              title={libraryReadableName(name)}
              kind={tool ? "Tool Agent" : "Model Agent"}
              description={body.description}
              selector={`${name}@${version}`}
              identityLabel="agent version"
              icon={tool ? "tool" : "agent"}
              tone={tool ? "amber" : "blue"}
              version={<AgentVersionSelector resource={data} />}
              meta={
                <LibraryTag tone={tool ? "amber" : "blue"}>
                  {tool ? "No model" : "Model based"}
                </LibraryTag>
              }
            />
            <div className="library-detail-grid">
              <div className="library-detail-main">
                {tool ? (
                  <ToolExecution body={body} />
                ) : (
                  <AgentPrompt
                    key={`prompt:${data.ref.digest}`}
                    resource={data}
                  />
                )}
                <AgentUsage key={`usage:${data.ref.digest}`} resource={data} />
                <TechnicalDetails
                  className="library-design-technical"
                  description="Exact published configuration"
                >
                  <ConfigurationBodyView resource={data} />
                  <dl className="library-exact-contract">
                    <div>
                      <dt>Digest</dt>
                      <dd>
                        <code>{data.ref.digest}</code>
                      </dd>
                    </div>
                    {body.summarizer ? (
                      <div>
                        <dt>Summarizer</dt>
                        <dd>
                          <pre>{JSON.stringify(body.summarizer, null, 2)}</pre>
                        </dd>
                      </div>
                    ) : null}
                  </dl>
                  <ButtonLink size="sm" to="/catalog/studio?kind=AgentTemplate">
                    Open Node Studio
                  </ButtonLink>
                </TechnicalDetails>
              </div>
              <AgentCapabilities resource={data} />
            </div>
          </article>
        );
      }}
    </QueryView>
  );
}
