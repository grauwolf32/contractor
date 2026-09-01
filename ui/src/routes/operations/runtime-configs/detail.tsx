import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router";

import { usePublicAPI } from "../../../api/context";
import { getRuntimeConfig } from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import { ErrorNotice, formatTimestamp } from "../../artifacts/common";

function OptionalBlock({ title, value }: { title: string; value: unknown }) {
  if (value === undefined) return null;
  const record = value as Record<string, unknown> | null;
  return (
    <article className="panel runtime-config-inspector">
      <h3>{title}</h3>
      {record === null ? (
        <p>Explicitly clears the lower-precedence block.</p>
      ) : (
        <dl className="key-value-list">
          {Object.entries(record).map(([key, item]) => (
            <div key={key}>
              <dt>{key}</dt>
              <dd>
                {key === "caBundlePem" ? (
                  <span>present · content hidden from this view</span>
                ) : Array.isArray(item) ? (
                  item.join(", ")
                ) : typeof item === "object" && item !== null ? (
                  <code>{Object.values(item).join(" · ")}</code>
                ) : (
                  String(item)
                )}
              </dd>
            </div>
          ))}
        </dl>
      )}
    </article>
  );
}

export function RuntimeConfigDetailRoute() {
  const api = usePublicAPI();
  const { name = "", version = "" } = useParams();
  const query = useQuery({
    queryKey: queryKeys.operations.runtimeConfigs.detail(name, version),
    queryFn: () => getRuntimeConfig(api, name, version),
  });
  return (
    <div className="configuration-detail">
      <Link className="back-link" to="/operations/runtime-configs">
        ← Runtime configuration
      </Link>
      {query.isPending ? (
        <p className="loading-copy">Loading exact RuntimeConfig…</p>
      ) : query.error !== null ? (
        <ErrorNotice error={query.error} />
      ) : (
        <>
          <div className="panel">
            <p className="eyebrow">Immutable typed infrastructure patch</p>
            <h3>
              {query.data.ref.name}@{query.data.ref.version}
            </h3>
            <dl className="key-value-list">
              <div>
                <dt>Digest</dt>
                <dd>
                  <code>{query.data.ref.digest}</code>
                </dd>
              </div>
              <div>
                <dt>Source</dt>
                <dd>
                  {query.data.builtIn ? "built-in" : query.data.createdBy}
                </dd>
              </div>
              <div>
                <dt>Created</dt>
                <dd>{formatTimestamp(query.data.createdAt)}</dd>
              </div>
            </dl>
          </div>
          <div className="runtime-config-detail-grid">
            <OptionalBlock
              title="Worker LLM Gateway"
              value={query.data.document.spec.worker?.llmGateway}
            />
            <OptionalBlock
              title="Worker telemetry"
              value={query.data.document.spec.worker?.telemetry}
            />
            <OptionalBlock
              title="Worker HTTP proxy"
              value={query.data.document.spec.worker?.httpProxy}
            />
            <OptionalBlock
              title="Planner telemetry"
              value={query.data.document.spec.planner?.telemetry}
            />
          </div>
        </>
      )}
    </div>
  );
}
