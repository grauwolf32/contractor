// Runs and project pages show resource usage without the Operations layout.
import "../operations.css";
import "./reading.css";
import { useId, useState } from "react";
import { Link } from "react-router";

import type { AllocationResourceSummary } from "../../../api/performance";
import {
  compactId,
  formatBytes,
  formatTimestamp,
  formatSeconds,
} from "../../../app/format";
import { Facts } from "../../../ui";
import { DisclosureChevron, OperationsState } from "../common";
import { RecordedTime } from "../../../app/recorded-time";

function unavailable(
  value: number | undefined,
  format: (value: number) => string,
) {
  return value === undefined ? (
    <span className="ops-muted">Unavailable</span>
  ) : (
    format(value)
  );
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

function CompletedAllocationRow({ item }: { item: AllocationResourceSummary }) {
  const [expanded, setExpanded] = useState(false);
  const detailsId = useId();

  return (
    <>
      <tr className={expanded ? "ops-row-expanded" : undefined}>
        <td data-label="Stage / agent">
          <div className="ops-history-stage">
            <strong>{item.stage}</strong>
            <small>{item.logicalAgent}</small>
            <button
              className="ops-metrics-toggle"
              type="button"
              aria-label={`Metrics for ${item.allocationId}`}
              aria-expanded={expanded}
              aria-controls={detailsId}
              onClick={() => setExpanded((current) => !current)}
            >
              <DisclosureChevron size={12} />
              <span>Metrics and identity</span>
            </button>
          </div>
        </td>
        <td data-label="Run">
          <Link
            className="ui-mono"
            to={`/runs/${encodeURIComponent(item.runId)}`}
            title={item.runId}
            aria-label={item.runId}
          >
            {compactId(item.runId)}
          </Link>
        </td>
        <td data-label="Outcome">
          <OperationsState state={item.outcome} />
        </td>
        <td data-label="Measurements">
          <span>{item.status.replaceAll("_", " ")}</span>
        </td>
        <td data-label="Finished">
          <RecordedTime value={item.finishedAt} />
        </td>
      </tr>
      {expanded ? (
        <tr className="ops-metrics-row">
          <td colSpan={5} data-label="">
            <section
              id={detailsId}
              aria-label={`Metrics for ${item.allocationId}`}
            >
              <AllocationResourceList items={[item]} showRunLink={false} />
            </section>
          </td>
        </tr>
      ) : null}
    </>
  );
}

export function AllocationResourceList({
  items,
  showRunLink = true,
  compact = false,
}: {
  items: readonly AllocationResourceSummary[];
  showRunLink?: boolean;
  compact?: boolean;
}) {
  if (compact)
    return (
      <div className="ops-table-wrap">
        <table className="ops-table" data-stack="">
          <thead>
            <tr>
              <th>Stage / agent</th>
              <th>Run</th>
              <th>Outcome</th>
              <th>Measurements</th>
              <th>Finished</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => (
              <CompletedAllocationRow key={item.allocationId} item={item} />
            ))}
          </tbody>
        </table>
      </div>
    );
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
            <Facts
              className="allocation-resource-identity"
              items={[
                {
                  term: "Run",
                  value: showRunLink && (
                    <Link to={`/runs/${encodeURIComponent(item.runId)}`}>
                      {item.runId}
                    </Link>
                  ),
                },
                { term: "Stage", value: <code>{item.stage}</code> },
                { term: "Finished", value: formatTimestamp(item.finishedAt) },
                { term: "Collection policy", value: item.collectionPolicy },
              ]}
            />
            <Facts
              className="allocation-resource-metrics"
              items={[
                {
                  term: "Measured interval",
                  value: unavailable(resources?.durationSeconds, formatSeconds),
                },
                {
                  term: "CPU user",
                  value: unavailable(resources?.cpuUserSeconds, formatSeconds),
                },
                {
                  term: "CPU system",
                  value: unavailable(
                    resources?.cpuSystemSeconds,
                    formatSeconds,
                  ),
                },
                {
                  term: "Average used cores",
                  value: unavailable(cores, (value) =>
                    value.toLocaleString(undefined, {
                      minimumFractionDigits: 2,
                      maximumFractionDigits: 2,
                    }),
                  ),
                },
                {
                  term: "Observed RSS peak",
                  value: unavailable(
                    resources?.rssPeakObservedBytes,
                    formatBytes,
                  ),
                },
                {
                  term: "RSS samples",
                  value: unavailable(resources?.rssSampleCount, (value) =>
                    value.toLocaleString(),
                  ),
                },
              ]}
            />
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
