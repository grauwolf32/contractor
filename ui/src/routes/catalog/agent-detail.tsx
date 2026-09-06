import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { lazy, Suspense, useMemo, useState } from "react";
import {
  Link,
  useLocation,
  useNavigate,
  useParams,
  useSearchParams,
} from "react-router";

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
import { CursorControls, ErrorNotice } from "../artifacts/common";
import { ConfigurationBodyView } from "../operations/llm-configurations/body";
import { catalogReturnState, locationDestination } from "./navigation";

const MarkdownPreview = lazy(() => import("../artifacts/previews/markdown"));
const INITIAL_CURSOR = null;

function cursorPage(raw: string | null): number {
  if (raw === null || !/^[1-9][0-9]*$/.test(raw)) return 1;
  const parsed = Number(raw);
  return Number.isSafeInteger(parsed) ? parsed : 1;
}

function AgentVersionSelector({
  resource,
}: {
  resource: ConfigurationResource;
}) {
  const api = usePublicAPI();
  const navigate = useNavigate();
  const location = useLocation();
  const query = useInfiniteQuery({
    queryKey: ["catalog", "agent-versions", resource.ref.name],
    initialPageParam: INITIAL_CURSOR as string | null,
    queryFn: ({ pageParam, signal }) =>
      listConfigurations(api, "agent-templates", {
        name: resource.ref.name,
        ...(pageParam === null ? {} : { cursor: pageParam }),
        signal,
      }),
    getNextPageParam: (page) =>
      page.page.hasMore ? page.page.nextCursor : undefined,
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
    <section className="catalog-version-picker" aria-label="Agent versions">
      <label>
        Published version
        <select
          value={resource.ref.version}
          onChange={(event) =>
            void navigate(agentPath(resource.ref.name, event.target.value), {
              state: location.state,
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
      <div>
        <span className="muted-copy">
          {versions.length} exact version{versions.length === 1 ? "" : "s"}{" "}
          loaded
        </span>
        {query.hasNextPage ? (
          <button
            type="button"
            className="secondary-button"
            disabled={query.isFetchingNextPage}
            onClick={() => void query.fetchNextPage()}
          >
            {query.isFetchingNextPage
              ? "Loading versions…"
              : "Load more versions"}
          </button>
        ) : null}
      </div>
      {query.isPending ? (
        <p role="status">Loading published versions…</p>
      ) : null}
      {query.error ? <ErrorNotice error={query.error} /> : null}
    </section>
  );
}

function AgentPrompt({ resource }: { resource: ConfigurationResource }) {
  const api = usePublicAPI();
  const [view, setView] = useState<"preview" | "source">("preview");
  const [copyStatus, setCopyStatus] = useState("");
  const query = useQuery({
    queryKey: [
      "catalog",
      "agent-instructions",
      resource.ref.name,
      resource.ref.version,
      resource.ref.digest,
    ],
    queryFn: async ({ signal }) => {
      const result = await getAgentInstructions(
        api,
        resource.ref.name,
        resource.ref.version,
        signal,
      );
      const body = resource.body as AgentTemplateBody;
      if (
        result.template.digest !== resource.ref.digest ||
        result.instructions.digest !== body.instructions.digest ||
        result.instructions.ref !== body.instructions.ref
      ) {
        throw new Error(
          "Agent instructions do not match this version. Refresh to reload the catalog.",
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
    <div className="catalog-agent-detail-grid">
      <section className="panel catalog-prompt" aria-label="Base prompt">
        <div className="catalog-prompt-toolbar">
          <div>
            <h3>Base prompt</h3>
            <p className="muted-copy">
              Task context and Skill instructions are added during execution.
            </p>
          </div>
          <button
            type="button"
            className="secondary-button"
            disabled={query.data === undefined || query.error !== null}
            onClick={() => void copyPrompt()}
          >
            Copy
          </button>
        </div>
        <div
          className="artifact-preview-tabs"
          role="group"
          aria-label="Prompt view"
        >
          <button
            type="button"
            aria-pressed={view === "preview"}
            className={view === "preview" ? "selected" : ""}
            onClick={() => setView("preview")}
          >
            Preview
          </button>
          <button
            type="button"
            aria-pressed={view === "source"}
            className={view === "source" ? "selected" : ""}
            onClick={() => setView("source")}
          >
            Source
          </button>
        </div>
        {copyStatus && <p role="status">{copyStatus}</p>}
        {query.isPending ? (
          <p role="status">Loading prompt…</p>
        ) : query.error ? (
          <ErrorNotice error={query.error} />
        ) : view === "source" ? (
          <pre className="catalog-prompt-source" tabIndex={0}>
            {query.data.instructions.text}
          </pre>
        ) : (
          <Suspense fallback={<p role="status">Loading preview…</p>}>
            <MarkdownPreview source={query.data.instructions.text} />
          </Suspense>
        )}
      </section>
      <aside
        className="panel catalog-agent-configuration"
        aria-label="Agent configuration"
      >
        <h3>Configuration</h3>
        <ConfigurationBodyView resource={resource} />
      </aside>
    </div>
  );
}

function AgentUsage({ resource }: { resource: ConfigurationResource }) {
  const api = usePublicAPI();
  const navigate = useNavigate();
  const location = useLocation();
  const [searchParams, setSearchParams] = useSearchParams();
  const cursor = searchParams.get("usageCursor") ?? undefined;
  const page = cursorPage(searchParams.get("usagePage"));
  const query = useQuery({
    queryKey: [
      "catalog",
      "agent-usage",
      resource.ref.name,
      resource.ref.version,
      resource.ref.digest,
      cursor ?? null,
    ],
    queryFn: ({ signal }) =>
      listAgentTemplateWorkflowBindings(
        api,
        resource.ref.name,
        resource.ref.version,
        { ...(cursor === undefined ? {} : { cursor }), signal },
      ),
  });

  function nextPage(nextCursor: string): void {
    const next = new URLSearchParams(searchParams);
    next.set("usageCursor", nextCursor);
    next.set("usagePage", String(page + 1));
    setSearchParams(next, { state: location.state });
  }

  return (
    <section
      className="panel catalog-agent-usage"
      aria-labelledby="agent-usage-title"
    >
      <div className="section-heading">
        <div>
          <p className="eyebrow">Exact reverse references</p>
          <h3 id="agent-usage-title">Where used</h3>
        </div>
        <span>Page {page}</span>
      </div>
      {query.isPending ? (
        <p role="status">Loading Workflow bindings…</p>
      ) : query.error ? (
        <ErrorNotice error={query.error} />
      ) : query.data.items.length === 0 ? (
        <p className="compact-empty">
          This exact Agent version is not referenced by a published Workflow.
        </p>
      ) : (
        <ul className="catalog-usage-list">
          {query.data.items.map((binding) => {
            const workflow = `${binding.workflow.name}@${binding.workflow.version}`;
            return (
              <li key={`${workflow}:${binding.stage}:${binding.logicalWorker}`}>
                <Link
                  to={`/catalog/workflows/${encodeURIComponent(binding.workflow.name)}/${encodeURIComponent(binding.workflow.version)}`}
                  state={{
                    returnTo: locationDestination(location),
                    returnLabel: `${resource.ref.name}@${resource.ref.version}`,
                    returnState: location.state,
                  }}
                >
                  {workflow}
                </Link>
                <span>
                  Stage <code>{binding.stage}</code>
                </span>
                <span>
                  Worker <code>{binding.logicalWorker}</code>
                </span>
              </li>
            );
          })}
        </ul>
      )}
      {query.data === undefined ? null : (
        <CursorControls
          label="Agent usage pages"
          canGoBack={cursor !== undefined && page > 1}
          {...(query.data.page.hasMore &&
          query.data.page.nextCursor !== undefined
            ? { nextCursor: query.data.page.nextCursor }
            : {})}
          onBack={() => void navigate(-1)}
          onNext={nextPage}
        />
      )}
    </section>
  );
}

export function AgentDetailRoute() {
  const api = usePublicAPI();
  const location = useLocation();
  const { name = "", version = "" } = useParams();
  const valid =
    CONFIG_ID_PATTERN.test(name) && CONFIG_VERSION_PATTERN.test(version);
  const query = useQuery({
    queryKey: queryKeys.configurations.detail("agent-templates", name, version),
    queryFn: ({ signal }) =>
      getConfiguration(api, "agent-templates", name, version, signal),
    enabled: valid,
  });
  const back = catalogReturnState(location.state, {
    returnTo: "/catalog/agents",
    returnLabel: "All agents",
  });
  return (
    <section className="catalog-agent-detail">
      <Link className="back-link" to={back.returnTo} state={back.returnState}>
        ← {back.returnLabel}
      </Link>
      <header>
        <p className="eyebrow">Exact Agent template</p>
        <h2>
          {name}
          <span className="catalog-version-label">@{version}</span>
        </h2>
      </header>
      {!valid ? (
        <ErrorNotice error={new Error("Agent version is invalid")} />
      ) : query.isPending ? (
        <p role="status">Loading exact Agent version…</p>
      ) : query.error ? (
        <ErrorNotice error={query.error} />
      ) : (
        <>
          <AgentVersionSelector resource={query.data} />
          <AgentPrompt
            key={`prompt:${query.data.ref.digest}`}
            resource={query.data}
          />
          <AgentUsage
            key={`usage:${query.data.ref.digest}`}
            resource={query.data}
          />
        </>
      )}
    </section>
  );
}
