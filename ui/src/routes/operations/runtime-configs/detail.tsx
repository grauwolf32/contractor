import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router";

import { usePublicAPI } from "../../../api/context";
import { getRuntimeConfig } from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import { RUNTIME_CONFIGURATION_PATH } from "../../../app/navigation";
import {
  compactDigest,
  formatBytes,
  formatTimestamp,
} from "../../../app/format";
import { QueryView } from "../../../app/query-view";
import { DetailHeader, IdChip } from "../../../ui";
import { Glance, ScopeChip } from "../common";

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

/**
 * The fields of a block. A nested block repeats inside its value; each list
 * is laid out by the width it gets (a block card is narrow, a nested value
 * narrower still).
 */
function ConfigFields({ record }: { record: Record<string, unknown> }) {
  return (
    <div className="ops-facts-frame">
      <dl className="ops-facts">
        {Object.entries(record).map(([key, value]) => (
          <div key={key}>
            <dt>{fieldLabel(key)}</dt>
            <dd>
              <ConfigValue name={key} value={value} />
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function OptionalBlock({ title, value }: { title: string; value: unknown }) {
  if (value === undefined) return null;
  return (
    <article className="ops-panel">
      <h3 className="ops-panel-title">{title}</h3>
      {value === null ? (
        <p className="ops-note">
          Explicitly clears the lower-precedence block.
        </p>
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
    <div className="ops-stack ops-detail">
      <DetailHeader
        breadcrumb={[
          { label: "Runtime configuration", to: RUNTIME_CONFIGURATION_PATH },
          { label: `${name}@${version}` },
        ]}
        title={`${name}@${version}`}
        status={
          query.data === undefined ? undefined : (
            <ScopeChip>
              {query.data.builtIn ? "Built-in" : "Published"}
            </ScopeChip>
          )
        }
      />
      <QueryView
        query={query}
        loading={
          <p className="ops-loading" role="status">
            Loading RuntimeConfig…
          </p>
        }
        onRetry={() => void query.refetch()}
      >
        {(data) => (
          <>
            <Glance
              label="Version identity"
              items={[
                [
                  "Digest",
                  <IdChip
                    key="digest"
                    value={data.ref.digest}
                    display={compactDigest(data.ref.digest)}
                    label="RuntimeConfig digest"
                  />,
                ],
                ["Source", data.builtIn ? "built-in" : data.createdBy],
                [
                  "Created",
                  data.builtIn ? "Built-in" : formatTimestamp(data.createdAt),
                ],
              ]}
            />
            <div className="ops-block-grid">
              <OptionalBlock
                title="Worker LLM Gateway"
                value={data.document.spec.worker?.llmGateway}
              />
              <OptionalBlock
                title="Worker telemetry"
                value={data.document.spec.worker?.telemetry}
              />
              <OptionalBlock
                title="Worker HTTP proxy"
                value={data.document.spec.worker?.httpProxy}
              />
              <OptionalBlock
                title="Worker Caido"
                value={data.document.spec.worker?.caido}
              />
              <OptionalBlock
                title="Planner telemetry"
                value={data.document.spec.planner?.telemetry}
              />
            </div>
          </>
        )}
      </QueryView>
    </div>
  );
}
