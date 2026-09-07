import { Link } from "react-router";

import type { AllocationResourceSummary } from "../../../api/performance";
import { formatBytes, formatTimestamp } from "../../artifacts/common";
import { OperationsState } from "../common";

function unavailable(
  value: number | undefined,
  format: (value: number) => string,
) {
  return value === undefined ? (
    <span className="muted-copy">Unavailable</span>
  ) : (
    format(value)
  );
}

function seconds(value: number): string {
  return `${value.toLocaleString(undefined, { maximumFractionDigits: 3 })} s`;
}

function resourceReason(item: AllocationResourceSummary): string {
  if (item.reason !== undefined) return item.reason.replaceAll("_", " ");
  switch (item.status) {
    case "disabled":
      return "collection disabled by the Server for this allocation";
    case "unsupported":
      return "the selected Runtime did not advertise resource metrics v1";
    case "pending":
      return "the terminal allocation report has not been released yet";
    default:
      return "no bounded diagnostic reason was reported";
  }
}

function averageCores(item: AllocationResourceSummary): number | undefined {
  const resources = item.resources;
  if (
    resources?.durationSeconds === undefined ||
    resources.durationSeconds === 0 ||
    resources.cpuUserSeconds === undefined ||
    resources.cpuSystemSeconds === undefined
  ) {
    return undefined;
  }
  return (
    (resources.cpuUserSeconds + resources.cpuSystemSeconds) /
    resources.durationSeconds
  );
}

export function AllocationResourceList({
  items,
  showRunLink = true,
}: {
  items: readonly AllocationResourceSummary[];
  showRunLink?: boolean;
}) {
  return (
    <div className="allocation-resource-list">
      {items.map((item) => {
        const resources = item.resources;
        const cores = averageCores(item);
        return (
          <article className="allocation-resource-card" key={item.allocationId}>
            <header>
              <div>
                <strong>{item.logicalAgent}</strong>
                <code title={item.allocationId}>{item.allocationId}</code>
              </div>
              <span>
                <OperationsState state={item.outcome} />
                <OperationsState state={item.status} />
              </span>
            </header>
            <dl className="allocation-resource-identity">
              {showRunLink ? (
                <div>
                  <dt>Run</dt>
                  <dd>
                    <Link to={`/runs/${encodeURIComponent(item.runId)}`}>
                      {item.runId}
                    </Link>
                  </dd>
                </div>
              ) : null}
              <div>
                <dt>Stage</dt>
                <dd>
                  <code>{item.stage}</code>
                </dd>
              </div>
              <div>
                <dt>Finished</dt>
                <dd>{formatTimestamp(item.finishedAt)}</dd>
              </div>
              <div>
                <dt>Collection policy</dt>
                <dd>{item.collectionPolicy}</dd>
              </div>
            </dl>
            <dl className="metrics-grid allocation-resource-metrics">
              <div>
                <dt>Measured interval</dt>
                <dd>{unavailable(resources?.durationSeconds, seconds)}</dd>
              </div>
              <div>
                <dt>CPU user</dt>
                <dd>{unavailable(resources?.cpuUserSeconds, seconds)}</dd>
              </div>
              <div>
                <dt>CPU system</dt>
                <dd>{unavailable(resources?.cpuSystemSeconds, seconds)}</dd>
              </div>
              <div>
                <dt>Average used cores</dt>
                <dd>
                  {unavailable(cores, (value) =>
                    value.toLocaleString(undefined, {
                      minimumFractionDigits: 2,
                      maximumFractionDigits: 2,
                    }),
                  )}
                </dd>
              </div>
              <div>
                <dt>Observed RSS peak</dt>
                <dd>
                  {unavailable(resources?.rssPeakObservedBytes, formatBytes)}
                </dd>
              </div>
              <div>
                <dt>RSS samples</dt>
                <dd>
                  {unavailable(resources?.rssSampleCount, (value) =>
                    value.toLocaleString(),
                  )}
                </dd>
              </div>
            </dl>
            <p className="allocation-resource-note">
              {resources === undefined
                ? resourceReason(item)
                : `Runtime process scope · ${resources.status}${resources.reason === undefined ? "" : ` · ${resources.reason.replaceAll("_", " ")}`}`}
            </p>
            <small>
              RSS is an observed Runtime-process peak. It excludes child
              processes and containers and is not guaranteed to be the true
              allocation peak.
            </small>
          </article>
        );
      })}
    </div>
  );
}
