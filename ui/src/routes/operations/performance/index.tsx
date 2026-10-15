import { useQuery } from "@tanstack/react-query";
import "./reading.css";
import { useMemo, useState, type ReactNode } from "react";

import { usePublicAPI } from "../../../api/context";
import {
  getPerformance,
  getPerformanceHistory,
  histogramQuantileUpperBound,
  PERFORMANCE_RANGES,
  performanceHistoryWindow,
  type PerformanceFreshness,
  type PerformanceHistogram,
  type PerformanceHistory,
  type PerformanceHistoryPoint,
  type PerformanceRange,
  type PerformanceSnapshot,
} from "../../../api/performance";
import { queryKeys } from "../../../api/query-keys";
import {
  formatBytes,
  formatTimestamp,
  formatSeconds,
} from "../../../app/format";
import { Facts, Field, TechnicalDetails } from "../../../ui";
import { OperationsState, OpsSection } from "../common";
import { type ChartDatum, MetricChart } from "./chart";
import { performanceFreshnessState } from "./freshness";
import { GPUCurrentMetrics, GPUHistoryCharts } from "./gpu";
import { historyGPUDevices, useGPUColors, type GPUColors } from "./gpu-colors";
import { RefreshButton } from "../../../app/refresh-button";
import { QueryView } from "../../../app/query-view";
import { pollInterval } from "../../../app/polling";

interface HTTPOverview {
  requestRate?: number;
  errorRate?: number;
  inFlight?: number;
  histogram?: PerformanceHistogram;
}

function mergeHistograms(
  values: readonly PerformanceHistogram[],
): PerformanceHistogram | undefined {
  if (values.length === 0) return undefined;
  const buckets = values[0]!.buckets.map(() => 0);
  let count = 0;
  let sumSeconds = 0;
  for (const histogram of values) {
    count += histogram.count;
    sumSeconds += histogram.sumSeconds;
    histogram.buckets.forEach((value, index) => {
      buckets[index] = (buckets[index] ?? 0) + value;
    });
  }
  return { count, sumSeconds, buckets };
}

function currentHTTP(snapshot: PerformanceSnapshot): HTTPOverview {
  const http = snapshot.current?.http;
  if (http === undefined) return {};
  const duration = http.freshness.coverage.durationSeconds;
  let errors = 0;
  for (const surface of http.surfaces) {
    for (const row of surface.counts) {
      errors += (row[4] ?? 0) + (row[5] ?? 0);
    }
  }
  const count = http.surfaces.reduce(
    (total, surface) => total + surface.duration.count,
    0,
  );
  const histogram = mergeHistograms(
    http.surfaces.map((surface) => surface.duration),
  );
  return {
    ...(duration > 0 ? { requestRate: count / duration } : {}),
    ...(duration > 0 ? { errorRate: errors / duration } : {}),
    inFlight: http.surfaces.reduce(
      (total, surface) => total + surface.inFlight,
      0,
    ),
    ...(histogram === undefined ? {} : { histogram }),
  };
}

function pointTime(point: PerformanceHistoryPoint): string {
  return point.kind === "sample" ? point.observedAt : point.minuteStart;
}

function pointHTTPRate(point: PerformanceHistoryPoint): number | undefined {
  const surfaces = point.kind === "sample" ? point.http?.surfaces : point.http;
  if (surfaces === undefined) return undefined;
  const duration =
    point.kind === "sample"
      ? point.http?.freshness.coverage.durationSeconds
      : point.coverageSeconds;
  if (duration === undefined || duration <= 0) return undefined;
  return (
    surfaces.reduce((total, surface) => total + surface.duration.count, 0) /
    duration
  );
}

function chartData(history: PerformanceHistory | undefined): {
  cpu: ChartDatum[];
  rss: ChartDatum[];
  requests: ChartDatum[];
} {
  const cpu: ChartDatum[] = [];
  const rss: ChartDatum[] = [];
  const requests: ChartDatum[] = [];
  for (const point of history?.points ?? []) {
    const common = {
      observedAt: pointTime(point),
      generation: point.generation,
    };
    const cpuValue =
      point.kind === "sample" ? point.process?.cpuCores : point.cpu?.cores;
    const rssValue =
      point.kind === "sample"
        ? point.process?.rssBytes
        : point.process.rssBytes?.last;
    const requestValue = pointHTTPRate(point);
    cpu.push({
      ...common,
      ...(cpuValue === undefined ? {} : { value: cpuValue }),
    });
    rss.push({
      ...common,
      ...(rssValue === undefined ? {} : { value: rssValue }),
    });
    requests.push({
      ...common,
      ...(requestValue === undefined ? {} : { value: requestValue }),
    });
  }
  return { cpu, rss, requests };
}

function metric(
  value: number | undefined,
  formatter: (current: number) => string,
): string {
  return value === undefined ? "Unavailable" : formatter(value);
}

// V385: needs shared number formatter (en-US, at most two fraction digits,
// as plural() counts); every value here is an integer or a rate.
function decimal(value: number): string {
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

const cores = (value: number) => `${decimal(value)} cores`;
const perSecond = (value: number) => `${decimal(value)}/s`;

function quantile(
  histogram: PerformanceHistogram | undefined,
  value: number,
): string {
  const bound = histogramQuantileUpperBound(histogram, value);
  return bound === undefined
    ? "Unavailable"
    : bound === "overflow"
      ? "> 60 s"
      : `≤ ${formatSeconds(bound)}`;
}

function MetricCard({
  title,
  state,
  facts,
  children,
}: {
  title: string;
  state: string;
  facts: readonly { term: string; value: string }[];
  children: ReactNode;
}) {
  return (
    <article className="performance-metric-card" data-freshness={state}>
      <header>
        <h3>{title}</h3>
        <OperationsState state={state} />
      </header>
      <Facts className="ops-glance" items={facts} />
      {children}
    </article>
  );
}

function GroupCard({
  title,
  freshness,
  readAt,
  facts,
  note,
}: {
  title: string;
  freshness: PerformanceFreshness | undefined;
  readAt: string;
  facts: readonly { term: string; value: string }[];
  note?: string;
}) {
  return (
    <MetricCard
      title={title}
      state={performanceFreshnessState(freshness, readAt)}
      facts={facts}
    >
      <small>
        {freshness?.observedAt === undefined
          ? "No successful observation"
          : `Observed ${formatTimestamp(freshness.observedAt)}`}
        {freshness?.reason === undefined
          ? ""
          : ` · ${freshness.reason.replaceAll("_", " ")}`}
      </small>
      {note === undefined ? null : <p className="ui-caption">{note}</p>}
    </MetricCard>
  );
}

function CurrentResourceSummary({
  snapshot,
  gpuColors,
}: {
  snapshot: PerformanceSnapshot;
  gpuColors: GPUColors;
}) {
  const sample = snapshot.current;
  return (
    <div className="ops-metric-grid">
      <GroupCard
        title="Server process"
        freshness={sample?.process?.freshness}
        readAt={snapshot.observedAt}
        facts={[
          { term: "CPU", value: metric(sample?.process?.cpuCores, cores) },
          {
            term: "Resident memory",
            value: metric(sample?.process?.rssBytes, formatBytes),
          },
        ]}
      />
      <GPUCurrentMetrics
        gpu={sample?.gpu}
        readAt={snapshot.observedAt}
        colors={gpuColors}
        compact
      />
      <GroupCard
        title="PostgreSQL"
        freshness={sample?.database?.freshness}
        readAt={snapshot.observedAt}
        facts={[
          {
            term: "Active connections",
            value: metric(sample?.database?.activeConnections, decimal),
          },
          {
            term: "Lock waiting",
            value: metric(sample?.database?.lockWaitingConnections, decimal),
          },
        ]}
      />
    </div>
  );
}

function CurrentMetrics({
  snapshot,
  gpuColors,
}: {
  snapshot: PerformanceSnapshot;
  gpuColors: GPUColors;
}) {
  const sample = snapshot.current;
  const process = sample?.process;
  const pool = sample?.pool;
  const database = sample?.database;
  const databaseSize = sample?.databaseSize;
  const diagnostics = snapshot.diagnostics;
  const http = currentHTTP(snapshot);
  const acquisitionMean =
    pool?.acquireCount !== undefined &&
    pool.acquireCount > 0 &&
    pool.acquireDurationSeconds !== undefined
      ? pool.acquireDurationSeconds / pool.acquireCount
      : undefined;
  return (
    <div className="ops-metric-grid">
      <GroupCard
        title="Server process"
        freshness={process?.freshness}
        readAt={snapshot.observedAt}
        facts={[
          { term: "CPU", value: metric(process?.cpuCores, cores) },
          { term: "RSS", value: metric(process?.rssBytes, formatBytes) },
          {
            term: "Go live heap",
            value: metric(process?.heapLiveBytes, formatBytes),
          },
          { term: "Goroutines", value: metric(process?.goroutines, decimal) },
          { term: "GC cycles", value: metric(process?.gcCycles, decimal) },
          {
            term: "GC pause delta",
            value: metric(process?.gcPauseSeconds, formatSeconds),
          },
        ]}
      />
      <GPUCurrentMetrics
        gpu={sample?.gpu}
        readAt={snapshot.observedAt}
        colors={gpuColors}
      />
      <GroupCard
        title="HTTP surfaces"
        freshness={sample?.http?.freshness}
        readAt={snapshot.observedAt}
        note="Latency values are upper bounds from merged Server histogram buckets."
        facts={[
          { term: "Request rate", value: metric(http.requestRate, perSecond) },
          {
            term: "5xx / no-response rate",
            value: metric(http.errorRate, perSecond),
          },
          { term: "In flight", value: metric(http.inFlight, decimal) },
          { term: "p50 latency", value: quantile(http.histogram, 0.5) },
          { term: "p95 latency", value: quantile(http.histogram, 0.95) },
          { term: "p99 latency", value: quantile(http.histogram, 0.99) },
        ]}
      />
      <GroupCard
        title="Working connection pool"
        freshness={pool?.freshness}
        readAt={snapshot.observedAt}
        note="Acquisition duration is a successful-acquisition mean, not p95."
        facts={[
          {
            term: "Acquired",
            value: metric(pool?.acquiredConnections, decimal),
          },
          { term: "Idle", value: metric(pool?.idleConnections, decimal) },
          {
            term: "Total / maximum",
            value:
              pool?.totalConnections === undefined ||
              pool.maxConnections === undefined
                ? "Unavailable"
                : `${decimal(pool.totalConnections)} / ${decimal(pool.maxConnections)}`,
          },
          {
            term: "Acquire mean",
            value: metric(acquisitionMean, formatSeconds),
          },
          {
            term: "Empty-pool wait",
            value: metric(pool?.emptyAcquireWaitSeconds, formatSeconds),
          },
          {
            term: "Cancelled acquires",
            value: metric(pool?.canceledAcquireCount, decimal),
          },
        ]}
      />
      <GroupCard
        title="PostgreSQL"
        freshness={database?.freshness}
        readAt={snapshot.observedAt}
        note="Dead tuples are planner estimates, not a bloat verdict."
        facts={[
          {
            term: "Active connections",
            value: metric(database?.activeConnections, decimal),
          },
          {
            term: "Lock waiting",
            value: metric(database?.lockWaitingConnections, decimal),
          },
          {
            term: "Longest transaction",
            value: metric(database?.longestTransactionSeconds, formatSeconds),
          },
          {
            term: "Commits / s",
            value: metric(database?.rates?.commitsPerSecond, decimal),
          },
          {
            term: "Buffer hit ratio",
            value: metric(
              database?.rates?.bufferHitRatio,
              (value) => `${(value * 100).toFixed(1)}%`,
            ),
          },
          {
            term: "Estimated dead tuples",
            value: metric(database?.estimatedDeadTuples, decimal),
          },
        ]}
      />
      <GroupCard
        title="Database storage"
        freshness={databaseSize?.freshness}
        readAt={snapshot.observedAt}
        note="Database size is allocated database content, not free disk space."
        facts={[
          {
            term: "Database size",
            value: metric(databaseSize?.sizeBytes, formatBytes),
          },
          {
            term: "Vacuum count",
            value: metric(database?.vacuumCount, decimal),
          },
          {
            term: "Autovacuum count",
            value: metric(database?.autovacuumCount, decimal),
          },
        ]}
      />
      <MetricCard
        title="Collection diagnostics"
        state={
          diagnostics.collectorReason === undefined &&
          diagnostics.writerReason === undefined &&
          diagnostics.droppedMinutes === 0
            ? "ok"
            : "partial"
        }
        facts={[
          {
            term: "Skipped samples",
            value: decimal(diagnostics.skippedSamples),
          },
          {
            term: "Rejected samples",
            value: decimal(diagnostics.rejectedSamples),
          },
          {
            term: "Skipped minutes",
            value: decimal(diagnostics.skippedMinutes),
          },
          {
            term: "Dropped minutes",
            value: decimal(diagnostics.droppedMinutes),
          },
          {
            term: "Pending minutes",
            value: decimal(diagnostics.pendingMinutes),
          },
        ]}
      >
        <small>
          {diagnostics.collectorReason ?? "collector healthy"} ·{" "}
          {diagnostics.writerReason ?? "writer healthy"}
        </small>
      </MetricCard>
    </div>
  );
}

export function OperationsPerformanceRoute() {
  const api = usePublicAPI();
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [range, setRange] = useState<PerformanceRange>("1h");
  const current = useQuery({
    queryKey: queryKeys.operations.performance.current,
    queryFn: ({ signal }) => getPerformance(api, signal),
    refetchInterval: pollInterval<PerformanceSnapshot>({ interval: 15_000 }),
    refetchIntervalInBackground: false,
  });
  const history = useQuery({
    queryKey: queryKeys.operations.performance.history(range),
    queryFn: ({ signal }) =>
      getPerformanceHistory(api, performanceHistoryWindow(range), signal),
    refetchInterval: pollInterval<PerformanceHistory>({
      interval: PERFORMANCE_RANGES[range].step === "15s" ? 15_000 : 60_000,
    }),
    refetchIntervalInBackground: false,
  });
  const series = useMemo(() => chartData(history.data), [history.data]);
  const gpuColors = useGPUColors([
    ...(current.data?.current?.gpu?.devices.map((device) => device.id) ?? []),
    ...historyGPUDevices(history.data).keys(),
  ]);
  const stepSeconds = (
    { "15s": 15, "1m": 60, "5m": 300, "1h": 3_600 } as const
  )[PERFORMANCE_RANGES[range].step];
  const generations = new Set(
    history.data?.points.map((point) => point.generation) ?? [],
  );
  return (
    <div className="ops-stack">
      <OpsSection
        id="performance-heading"
        title="Server performance"
        description="Server process, PostgreSQL and available host GPU observations. History preserves collection gaps and Server restarts."
        actions={
          <RefreshButton
            isFetching={current.isFetching || history.isFetching}
            onRefresh={() =>
              void Promise.all([current.refetch(), history.refetch()])
            }
            label="Refresh"
          />
        }
      >
        <QueryView
          query={current}
          loading={
            <p className="ops-loading" role="status">
              Loading current performance…
            </p>
          }
        >
          {(data) =>
            !data.enabled ? (
              <div className="notice notice-warning" role="status">
                <strong>Performance collection is disabled.</strong>
                <p>
                  Retained history remains available below until its seven-day
                  expiry.
                </p>
              </div>
            ) : data.current === undefined ? (
              <p className="ops-empty" role="status">
                Collection is enabled and waiting for its first bounded sample.
              </p>
            ) : (
              <>
                <p className="ops-observed-at">
                  Read {formatTimestamp(data.observedAt)} · generation{" "}
                  <code>{data.generation}</code>
                </p>
                <CurrentResourceSummary snapshot={data} gpuColors={gpuColors} />
              </>
            )
          }
        </QueryView>
      </OpsSection>

      <section
        className="ops-section"
        aria-labelledby="performance-history-heading"
      >
        <header className="ops-history-head">
          <div className="ops-section-heading">
            <h2 id="performance-history-heading" className="ops-section-title">
              History
            </h2>
            <p className="ops-section-description">
              Lines are not joined across gaps or Server restarts.
            </p>
          </div>
          <Field label="Time range" className="ops-range-field">
            {(control) => (
              <select
                {...control}
                value={range}
                onChange={(event) =>
                  setRange(event.target.value as PerformanceRange)
                }
              >
                {(
                  Object.entries(PERFORMANCE_RANGES) as Array<
                    [
                      PerformanceRange,
                      (typeof PERFORMANCE_RANGES)[PerformanceRange],
                    ]
                  >
                ).map(([value, option]) => (
                  <option key={value} value={value}>
                    {option.label} · {option.step}
                  </option>
                ))}
              </select>
            )}
          </Field>
        </header>
        <QueryView
          query={history}
          loading={
            <p className="ops-loading" role="status">
              Loading bounded history…
            </p>
          }
        >
          {(historyData) => (
            <>
              {generations.size > 1 ? (
                <div className="notice notice-warning" role="status">
                  This range crosses {generations.size} Server generations.
                  Lines are intentionally not joined across restarts.
                </div>
              ) : null}
              <div className="ops-chart-grid">
                <MetricChart
                  title="CPU usage"
                  description="Average Server process CPU expressed as used cores"
                  data={series.cpu}
                  expectedStepSeconds={stepSeconds}
                  unit="cores"
                />
                <MetricChart
                  title="Resident memory"
                  description="Observed Server process resident set size"
                  data={series.rss.map((datum) => ({
                    ...datum,
                    ...(datum.value === undefined
                      ? {}
                      : { value: datum.value / (1024 * 1024) }),
                  }))}
                  expectedStepSeconds={stepSeconds}
                  unit="MiB"
                />
                <GPUHistoryCharts
                  colors={gpuColors}
                  history={historyData}
                  stepSeconds={stepSeconds}
                />
                <MetricChart
                  title="HTTP request rate"
                  description="Public and private completed requests per second"
                  data={series.requests}
                  expectedStepSeconds={stepSeconds}
                  unit="req/s"
                />
              </div>
            </>
          )}
        </QueryView>
      </section>
      {current.data?.current === undefined ? null : (
        <TechnicalDetails
          className="ops-counters"
          summary="Detailed counters and collection diagnostics"
          description="HTTP surfaces, connection pool, PostgreSQL, storage and the collector itself."
          onToggle={setDetailsOpen}
        >
          {detailsOpen ? (
            <CurrentMetrics snapshot={current.data} gpuColors={gpuColors} />
          ) : null}
        </TechnicalDetails>
      )}
    </div>
  );
}
