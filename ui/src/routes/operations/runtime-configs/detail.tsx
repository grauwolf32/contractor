import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router";

import { usePublicAPI } from "../../../api/context";
import { getRuntimeConfig } from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import { RUNTIME_CONFIGURATION_PATH } from "../../../app/navigation";
import {
  ErrorNotice,
  formatBytes,
  formatTimestamp,
} from "../../artifacts/common";

const FIELD_LABELS: Record<string, string> = {
  gatewayId: "Gateway ID",
  batchSizeBytes: "Batch size",
  maxPendingBytes: "Maximum pending bytes",
  maxPendingSpans: "Maximum pending spans",
  maxAttempts: "Maximum attempts",
  initialBackoffMilliseconds: "Initial backoff",
  maxBackoffMilliseconds: "Maximum backoff",
};

function fieldLabel(key: string): string {
  return (
    FIELD_LABELS[key] ??
    key
      .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
      .replace(/^./, (first) => first.toUpperCase())
  );
}

function ConfigValue({ name, value }: { name: string; value: unknown }) {
  if (name === "caBundlePem") {
    return <span>present · content hidden from this view</span>;
  }
  if (value === null) return <span>Explicitly cleared</span>;
  if (Array.isArray(value))
    return value.length === 0 ? "None" : value.join(", ");
  if (typeof value === "object") {
    return <ConfigFields record={value as Record<string, unknown>} />;
  }
  if (typeof value === "number") {
    if (name === "batchSizeBytes" || name === "maxPendingBytes") {
      return formatBytes(value);
    }
    if (name.endsWith("BackoffMilliseconds")) return `${value} ms`;
  }
  return String(value);
}

function ConfigFields({ record }: { record: Record<string, unknown> }) {
  return (
    <dl className="key-value-list">
      {Object.entries(record).map(([key, value]) => (
        <div key={key}>
          <dt>{fieldLabel(key)}</dt>
          <dd>
            <ConfigValue name={key} value={value} />
          </dd>
        </div>
      ))}
    </dl>
  );
}

function OptionalBlock({ title, value }: { title: string; value: unknown }) {
  if (value === undefined) return null;
  return (
    <article className="panel runtime-config-inspector">
      <h3>{title}</h3>
      {value === null ? (
        <p>Explicitly clears the lower-precedence block.</p>
      ) : (
        <ConfigFields record={value as Record<string, unknown>} />
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
      <Link className="back-link" to={RUNTIME_CONFIGURATION_PATH}>
        ← Runtime configuration
      </Link>
      {query.isPending ? (
        <p className="loading-copy" role="status">
          Loading RuntimeConfig…
        </p>
      ) : query.error !== null ? (
        <ErrorNotice error={query.error} />
      ) : (
        <>
          <div className="panel">
            <p className="eyebrow">Runtime configuration</p>
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
                <dd>
                  {query.data.builtIn
                    ? "Built-in"
                    : formatTimestamp(query.data.createdAt)}
                </dd>
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
              title="Worker Caido"
              value={query.data.document.spec.worker?.caido}
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
