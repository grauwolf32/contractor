import { useQuery } from "@tanstack/react-query";
import { lazy, Suspense, useState } from "react";
import { Link, useNavigate, useParams } from "react-router";

import { agentPath, getAgentInstructions } from "../../api/agents";
import { usePublicAPI } from "../../api/context";
import {
  getConfiguration,
  type AgentTemplateBody,
  type ConfigurationResource,
} from "../../api/operations";
import { queryKeys } from "../../api/query-keys";
import { CONFIG_ID_PATTERN, CONFIG_VERSION_PATTERN } from "../../api/workflows";
import { ErrorNotice } from "../artifacts/common";
import { ConfigurationBodyView } from "../operations/llm-configurations/body";

const MarkdownPreview = lazy(() => import("../artifacts/previews/markdown"));

function AgentPrompt({ resource }: { resource: ConfigurationResource }) {
  const api = usePublicAPI();
  const navigate = useNavigate();
  const [version, setVersion] = useState(resource.ref.version);
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
    <>
      <form
        className="catalog-version-form"
        onSubmit={(event) => {
          event.preventDefault();
          if (CONFIG_VERSION_PATTERN.test(version))
            void navigate(agentPath(resource.ref.name, version));
        }}
      >
        <label>
          Version
          <input
            value={version}
            required
            maxLength={128}
            onChange={(event) => setVersion(event.target.value)}
          />
        </label>
        <button
          type="submit"
          className="secondary-button"
          disabled={!CONFIG_VERSION_PATTERN.test(version)}
        >
          Open version
        </button>
      </form>
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
    </>
  );
}

export function AgentDetailRoute() {
  const api = usePublicAPI();
  const { name = "", version = "" } = useParams();
  const valid =
    CONFIG_ID_PATTERN.test(name) && CONFIG_VERSION_PATTERN.test(version);
  const query = useQuery({
    queryKey: queryKeys.configurations.detail("agent-templates", name, version),
    queryFn: () => getConfiguration(api, "agent-templates", name, version),
    enabled: valid,
  });
  return (
    <section className="catalog-agent-detail">
      <Link className="back-link" to="/catalog/agents">
        ← All agents
      </Link>
      <header>
        <h2>
          {name}
          <span className="catalog-version-label">@{version}</span>
        </h2>
        <p className="muted-copy">Agent template</p>
      </header>
      {!valid ? (
        <ErrorNotice error={new Error("Agent version is invalid")} />
      ) : query.isPending ? (
        <p role="status">Loading agent…</p>
      ) : query.error ? (
        <ErrorNotice error={query.error} />
      ) : (
        <AgentPrompt key={query.data.ref.digest} resource={query.data} />
      )}
    </section>
  );
}
