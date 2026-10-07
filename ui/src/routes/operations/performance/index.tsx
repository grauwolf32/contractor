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
import { formatBytes, formatTimestamp } from "../../../app/format";
import { OperationsState, OpsSection } from "../common";
import { type ChartDatum, MetricChart } from "./chart";
import { performanceFreshnessState } from "./freshness";
import { GPUCurrentMetrics, GPUHistoryCharts } from "./gpu";
import { historyGPUDevices, useGPUColors, type GPUColors } from "./gpu-colors";
import { RefreshButton } from "../../../app/refresh-button";
import { QueryView } from "../../../app/query-view";

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

function decimal(value: number): string {
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function seconds(value: number): string {
  return `${value.toLocaleString(undefined, { maximumFractionDigits: 3 })} s`;
}

function quantile(
  histogram: PerformanceHistogram | undefined,
  value: number,
): string {
  const bound = histogramQuantileUpperBound(histogram, value);
  return bound === undefined
    ? "Unavailable"
    : bound === "overflow"
      ? "> 60 s"
      : `≤ ${seconds(bound)}`;
}

function GroupCard({
  title,
  freshness,
  readAt,
  children,
  note,
}: {
  title: string;
  freshness: PerformanceFreshness | undefined;
  readAt: string;
  children: ReactNode;
  note?: string;
}) {
  const state = performanceFreshnessState(freshness, readAt);
  return (
    <article className="performance-metric-card" data-freshness={state}>
      <header>
        <h3>{title}</h3>
        <OperationsState state={state} />
      </header>
      {children}
      <small>
        {freshness?.observedAt === undefined
          ? "No successful observation"
          : `Observed ${formatTimestamp(freshness.observedAt)}`}
        {freshness?.reason === undefined
          ? ""
          : ` · ${freshness.reason.replaceAll("_", " ")}`}
      </small>
      {note === undefined ? null : <p className="ops-note">{note}</p>}
    </article>
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
      >
        <dl className="ops-glance">
          <div>
            <dt>CPU</dt>
            <dd>
              {metric(
                sample?.process?.cpuCores,
                (value) => `${decimal(value)} cores`,
              )}
            </dd>
          </div>
          <div>
            <dt>Resident memory</dt>
            <dd>{metric(sample?.process?.rssBytes, formatBytes)}</dd>
          </div>
        </dl>
      </GroupCard>
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
      >
        <dl className="ops-glance">
          <div>
            <dt>Active connections</dt>
            <dd>{metric(sample?.database?.activeConnections, decimal)}</dd>
          </div>
          <div>
            <dt>Lock waiting</dt>
            <dd>{metric(sample?.database?.lockWaitingConnections, decimal)}</dd>
          </div>
        </dl>
      </GroupCard>
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
      >
        <dl className="ops-glance">
          <div>
            <dt>CPU</dt>
            <dd>
              {metric(process?.cpuCores, (value) => `${decimal(value)} cores`)}
            </dd>
          </div>
          <div>
            <dt>RSS</dt>
            <dd>{metric(process?.rssBytes, formatBytes)}</dd>
          </div>
          <div>
            <dt>Go live heap</dt>
            <dd>{metric(process?.heapLiveBytes, formatBytes)}</dd>
          </div>
          <div>
            <dt>Goroutines</dt>
            <dd>
              {metric(process?.goroutines, (value) => value.toLocaleString())}
            </dd>
          </div>
          <div>
            <dt>GC cycles</dt>
            <dd>
              {metric(process?.gcCycles, (value) => value.toLocaleString())}
            </dd>
          </div>
          <div>
            <dt>GC pause delta</dt>
            <dd>{metric(process?.gcPauseSeconds, seconds)}</dd>
          </div>
        </dl>
      </GroupCard>
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
      >
        <dl className="ops-glance">
          <div>
            <dt>Request rate</dt>
            <dd>
              {metric(http.requestRate, (value) => `${decimal(value)}/s`)}
            </dd>
          </div>
          <div>
            <dt>5xx / no-response rate</dt>
            <dd>{metric(http.errorRate, (value) => `${decimal(value)}/s`)}</dd>
          </div>
          <div>
            <dt>In flight</dt>
            <dd>{metric(http.inFlight, (value) => value.toLocaleString())}</dd>
          </div>
          <div>
            <dt>p50 latency</dt>
            <dd>{quantile(http.histogram, 0.5)}</dd>
          </div>
          <div>
            <dt>p95 latency</dt>
            <dd>{quantile(http.histogram, 0.95)}</dd>
          </div>
          <div>
            <dt>p99 latency</dt>
            <dd>{quantile(http.histogram, 0.99)}</dd>
          </div>
        </dl>
      </GroupCard>
      <GroupCard
        title="Working connection pool"
        freshness={pool?.freshness}
        readAt={snapshot.observedAt}
        note="Acquisition duration is a successful-acquisition mean, not p95."
      >
        <dl className="ops-glance">
          <div>
            <dt>Acquired</dt>
            <dd>
              {metric(pool?.acquiredConnections, (value) =>
                value.toLocaleString(),
              )}
            </dd>
          </div>
          <div>
            <dt>Idle</dt>
            <dd>
              {metric(pool?.idleConnections, (value) => value.toLocaleString())}
            </dd>
          </div>
          <div>
            <dt>Total / maximum</dt>
            <dd>
              {pool?.totalConnections === undefined ||
              pool.maxConnections === undefined
                ? "Unavailable"
                : `${pool.totalConnections.toLocaleString()} / ${pool.maxConnections.toLocaleString()}`}
            </dd>
          </div>
          <div>
            <dt>Acquire mean</dt>
            <dd>{metric(acquisitionMean, seconds)}</dd>
          </div>
          <div>
            <dt>Empty-pool wait</dt>
            <dd>{metric(pool?.emptyAcquireWaitSeconds, seconds)}</dd>
          </div>
          <div>
            <dt>Cancelled acquires</dt>
            <dd>
              {metric(pool?.canceledAcquireCount, (value) =>
                value.toLocaleString(),
              )}
            </dd>
          </div>
        </dl>
      </GroupCard>
      <GroupCard
        title="PostgreSQL"
        freshness={database?.freshness}
        readAt={snapshot.observedAt}
        note="Dead tuples are planner estimates, not a bloat verdict."
      >
        <dl className="ops-glance">
          <div>
            <dt>Active connections</dt>
            <dd>
              {metric(database?.activeConnections, (value) =>
                value.toLocaleString(),
              )}
            </dd>
          </div>
          <div>
            <dt>Lock waiting</dt>
            <dd>
              {metric(database?.lockWaitingConnections, (value) =>
                value.toLocaleString(),
              )}
            </dd>
          </div>
          <div>
            <dt>Longest transaction</dt>
            <dd>{metric(database?.longestTransactionSeconds, seconds)}</dd>
          </div>
          <div>
            <dt>Commits / s</dt>
            <dd>{metric(database?.rates?.commitsPerSecond, decimal)}</dd>
          </div>
          <div>
            <dt>Buffer hit ratio</dt>
            <dd>
              {metric(
                database?.rates?.bufferHitRatio,
                (value) => `${(value * 100).toFixed(1)}%`,
              )}
            </dd>
          </div>
          <div>
            <dt>Estimated dead tuples</dt>
            <dd>
              {metric(database?.estimatedDeadTuples, (value) =>
                value.toLocaleString(),
              )}
            </dd>
          </div>
        </dl>
      </GroupCard>
      <GroupCard
        title="Database storage"
        freshness={databaseSize?.freshness}
        readAt={snapshot.observedAt}
        note="Database size is allocated database content, not free disk space."
      >
        <dl className="ops-glance">
          <div>
            <dt>Database size</dt>
            <dd>{metric(databaseSize?.sizeBytes, formatBytes)}</dd>
          </div>
          <div>
            <dt>Vacuum count</dt>
            <dd>
              {metric(database?.vacuumCount, (value) => value.toLocaleString())}
            </dd>
          </div>
          <div>
            <dt>Autovacuum count</dt>
            <dd>
              {metric(database?.autovacuumCount, (value) =>
                value.toLocaleString(),
              )}
            </dd>
          </div>
        </dl>
      </GroupCard>
      <article className="performance-metric-card">
        <header>
          <h3>Collection diagnostics</h3>
          <OperationsState
            state={
              snapshot.diagnostics.collectorReason === undefined &&
              snapshot.diagnostics.writerReason === undefined &&
              snapshot.diagnostics.droppedMinutes === 0
                ? "ok"
                : "partial"
            }
          />
        </header>
        <dl className="ops-glance">
          <div>
            <dt>Skipped samples</dt>
            <dd>{snapshot.diagnostics.skippedSamples.toLocaleString()}</dd>
          </div>
          <div>
            <dt>Rejected samples</dt>
            <dd>{snapshot.diagnostics.rejectedSamples.toLocaleString()}</dd>
          </div>
          <div>
            <dt>Skipped minutes</dt>
            <dd>{snapshot.diagnostics.skippedMinutes.toLocaleString()}</dd>
          </div>
          <div>
            <dt>Dropped minutes</dt>
            <dd>{snapshot.diagnostics.droppedMinutes.toLocaleString()}</dd>
          </div>
          <div>
            <dt>Pending minutes</dt>
            <dd>{snapshot.diagnostics.pendingMinutes.toLocaleString()}</dd>
          </div>
        </dl>
        <small>
          {snapshot.diagnostics.collectorReason ?? "collector healthy"} ·{" "}
          {snapshot.diagnostics.writerReason ?? "writer healthy"}
        </small>
      </article>
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
    refetchInterval: 15_000,
    refetchIntervalInBackground: false,
  });
  const history = useQuery({
    queryKey: queryKeys.operations.performance.history(range),
    queryFn: ({ signal }) =>
      getPerformanceHistory(api, performanceHistoryWindow(range), signal),
    refetchInterval: PERFORMANCE_RANGES[range].step === "15s" ? 15_000 : 60_000,
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
          onRetry={() => void current.refetch()}
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
          <label className="ops-field ops-range-field">
            <span>Time range</span>
            <select
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
          </label>
        </header>
        <QueryView
          query={history}
          loading={
            <p className="ops-loading" role="status">
              Loading bounded history…
            </p>
          }
          onRetry={() => void history.refetch()}
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
        <details
          className="ui-tech ops-counters"
          onToggle={(event) => setDetailsOpen(event.currentTarget.open)}
        >
          <summary>
            <span className="ui-tech-label">
              <svg
                className="ui-tech-chevron"
                width="13"
                height="13"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
                focusable="false"
              >
                <path d="M9.5 6l6 6-6 6" />
              </svg>
              Detailed counters and collection diagnostics
            </span>
            <span className="ui-tech-description">
              HTTP surfaces, connection pool, PostgreSQL, storage and the
              collector itself.
            </span>
          </summary>
          <div className="ui-tech-body">
            {detailsOpen ? (
              <CurrentMetrics snapshot={current.data} gpuColors={gpuColors} />
            ) : null}
          </div>
        </details>
      )}
    </div>
  );
}
