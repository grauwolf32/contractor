import { Link } from "react-router";

import type { AllocationObservation } from "../../api/operations";
import { formatTimestamp } from "../artifacts/common";
import { StateBadge } from "../runs/components";
import type { ExactConfigurationRef } from "./references";

export function ConfigurationRefLink({
  value,
}: {
  value: ExactConfigurationRef;
}) {
  return (
    <span className="exact-config-ref">
      <Link
        to={`/operations/configurations/${value.kind}/${encodeURIComponent(value.name)}/${encodeURIComponent(value.version)}`}
      >
        {value.name}@{value.version}
      </Link>
      <code title={value.digest}>{value.digest.slice(0, 18)}…</code>
    </span>
  );
}

export function OptionalTimestamp({ value }: { value: string | undefined }) {
  return value === undefined ? (
    <span className="muted-copy">Not observed</span>
  ) : (
    formatTimestamp(value)
  );
}

export function SafeReason({
  reason,
}: {
  reason: { code: string; retryable: boolean } | undefined;
}) {
  if (reason === undefined) {
    return <span className="muted-copy">None</span>;
  }
  return (
    <span className="reason-record">
      <code>{reason.code}</code>
      <span>{reason.retryable ? "retryable" : "not retryable"}</span>
    </span>
  );
}

export function OperationsState({ state }: { state: string }) {
  return <StateBadge state={state} />;
}

export function MetricsSummary({
  metrics,
}: {
  metrics: AllocationObservation["metrics"];
}) {
  const values = [
    ["Model calls", metrics.modelCalls],
    ["Tool calls", metrics.toolCalls],
    ["Tool failures", metrics.toolFailures],
    ["Input tokens", metrics.inputTokens],
    ["Output tokens", metrics.outputTokens],
    ["Total tokens", metrics.totalTokens],
    ["Errors", metrics.errorCount],
  ] as const;
  return (
    <dl className="metrics-grid operations-metrics">
      {values.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{value.toLocaleString()}</dd>
        </div>
      ))}
      <div>
        <dt>Reports</dt>
        <dd>{metrics.reportsComplete ? "complete" : "incomplete"}</dd>
      </div>
      <div>
        <dt>Truncated</dt>
        <dd>{metrics.truncated ? "yes" : "no"}</dd>
      </div>
    </dl>
  );
}
