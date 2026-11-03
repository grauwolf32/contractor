// Runs and project pages show resource usage without the Operations layout.
import "../styles";
import "./reading.css";
import { useId, useState } from "react";
import { Link } from "react-router";

import type { AllocationResourceSummary } from "../../../api/performance";
import {
  formatBytes,
  formatNumber,
  formatSeconds,
  formatTimestamp,
} from "../../../app/format";
import { Button, Facts } from "../../../ui";
import { EntryHeading, OperationsState } from "../common";

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
    <article className="ops-resource-row ops-completed-row">
      <Button
        className="ops-entry-button"
        variant="ghost"
        aria-label={`Metrics for ${item.allocationId}`}
        aria-expanded={expanded}
        aria-controls={detailsId}
        onClick={() => setExpanded((current) => !current)}
      >
        <EntryHeading
          icon="history"
          title={<strong>{item.logicalAgent}</strong>}
          subtitle={item.allocationId}
        >
          <OperationsState state={item.outcome} />
          <OperationsState
            state={item.status}
            tone={
              item.status === "unavailable" || item.status === "pending"
                ? "neutral"
                : undefined
            }
          />
        </EntryHeading>
      </Button>
      {expanded ? (
        <section
          className="ops-resource-body"
          id={detailsId}
          aria-label={`Metrics for ${item.allocationId}`}
        >
          <AllocationResourceList items={[item]} showRunLink />
        </section>
      ) : null}
    </article>
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
      <div className="ops-resource-list">
        {items.map((item) => (
          <CompletedAllocationRow key={item.allocationId} item={item} />
        ))}
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
                <OperationsState
                  state={item.status}
                  tone={
                    item.status === "unavailable" || item.status === "pending"
                      ? "neutral"
                      : undefined
                  }
                />
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
                    formatNumber(value, 2, 2),
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
                    formatNumber(value),
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
