import type { PublicAPI } from "./client";
import { invalidAPIResponse, requireData } from "./error";
import type { components } from "./generated/public";

export const ALLOCATION_HISTORY_PAGE_SIZE = 50;
export const PERFORMANCE_RANGES = {
  "15m": { label: "15 minutes", durationMs: 15 * 60_000, step: "15s" },
  "1h": { label: "1 hour", durationMs: 60 * 60_000, step: "15s" },
  "6h": { label: "6 hours", durationMs: 6 * 60 * 60_000, step: "1m" },
  "24h": { label: "24 hours", durationMs: 24 * 60 * 60_000, step: "5m" },
  "7d": { label: "7 days", durationMs: 7 * 24 * 60 * 60_000, step: "1h" },
} as const;

export type PerformanceRange = keyof typeof PERFORMANCE_RANGES;
export type PerformanceStep =
  components["schemas"]["PerformanceHistory"]["step"];
export type PerformanceSnapshot = components["schemas"]["PerformanceSnapshot"];
export type PerformanceHistory = components["schemas"]["PerformanceHistory"];
export type PerformanceHistoryPoint = PerformanceHistory["points"][number];
export type PerformanceHistogram =
  components["schemas"]["PerformanceHistogram"];
export type PerformanceFreshness =
  components["schemas"]["PerformanceFreshness"];
export type AllocationResourceSummary =
  components["schemas"]["AllocationResourceSummary"];
export type AllocationResourcePage =
  components["schemas"]["AllocationResourcePage"];

export interface PerformanceHistoryRequest {
  from: string;
  to: string;
  step: PerformanceStep;
}

export interface AllocationResourcePageRequest {
  runId?: string;
  cursor?: string;
}

const RESOURCE_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$/;
const HISTOGRAM_BOUNDS_SECONDS = [
  0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60,
] as const;
const AGGREGATE_STEP_SECONDS = { "1m": 60, "5m": 300, "1h": 3_600 } as const;

type Sample = components["schemas"]["PerformanceSample"];
type Freshness = PerformanceFreshness | undefined;
type GPU =
  | components["schemas"]["PerformanceGPU"]
  | components["schemas"]["PerformanceGPUAggregate"];

/** Buckets are cumulative: nondecreasing, at most and ending at the count. */
function monotonicHistogram(value: PerformanceHistogram): boolean {
  let previous = 0;
  for (const bucket of value.buckets) {
    if (bucket < previous || bucket > value.count) return false;
    previous = bucket;
  }
  return previous === value.count;
}

/** A window never observes more samples than it expects. */
function consistentFreshness(value: Freshness): boolean {
  return (
    value === undefined ||
    value.coverage.observedSamples <= value.coverage.expectedSamples
  );
}

function consistentSurfaces(
  surfaces: components["schemas"]["PerformanceHTTPSurface"][] | undefined,
): boolean {
  return (surfaces ?? []).every((surface) =>
    monotonicHistogram(surface.duration),
  );
}

/**
 * GPU devices are distinct; a current reading lists devices exactly when it
 * is available and never uses more memory than the device has; a gauge
 * summary keeps its last value between its minimum and maximum.
 */
function consistentGPU(value: GPU | undefined, aggregate: boolean): boolean {
  if (value === undefined) return true;
  const devices: { id: string }[] = value.devices;
  if (
    !consistentFreshness(value.freshness) ||
    new Set(devices.map((device) => device.id)).size !== devices.length
  ) {
    return false;
  }
  if (aggregate) {
    const gauges =
      value.devices as components["schemas"]["PerformanceGPUDeviceGauges"][];
    return gauges.every((device) =>
      Object.values(device).every(
        (gauge) =>
          typeof gauge !== "object" ||
          (gauge.min <= gauge.last && gauge.last <= gauge.max),
      ),
    );
  }
  const current =
    value.devices as components["schemas"]["PerformanceGPUDevice"][];
  return (
    (value.freshness.status === "unavailable") === (current.length === 0) &&
    current.every(
      (device) =>
        device.memoryUsedBytes === undefined ||
        device.memoryTotalBytes === undefined ||
        device.memoryUsedBytes <= device.memoryTotalBytes,
    )
  );
}

function consistentSample(value: Sample): boolean {
  return (
    [
      value.http?.freshness,
      value.process?.freshness,
      value.pool?.freshness,
      value.database?.freshness,
      value.databaseSize?.freshness,
    ].every(consistentFreshness) &&
    consistentSurfaces(value.http?.surfaces) &&
    consistentGPU(value.gpu, false)
  );
}

function requireSnapshot(
  value: PerformanceSnapshot,
  status: number,
): PerformanceSnapshot {
  // The current sample belongs to the snapshot's collector generation.
  if (
    value.current !== undefined &&
    (value.current.generation !== value.generation ||
      !consistentSample(value.current))
  ) {
    throw invalidAPIResponse(
      status,
      "Server returned invalid performance data",
    );
  }
  return value;
}

function pointTimestamp(point: PerformanceHistoryPoint): string {
  return point.kind === "sample" ? point.observedAt : point.minuteStart;
}

/**
 * History answers the requested window and step: fine steps carry samples,
 * coarse steps aggregates of that step, ordered by time inside the window.
 */
function requireHistory(
  value: PerformanceHistory,
  request: PerformanceHistoryRequest,
  status: number,
): PerformanceHistory {
  const requestedFrom = Date.parse(request.from);
  const requestedTo = Date.parse(request.to);
  const invalid = () =>
    invalidAPIResponse(status, "Server returned invalid performance history");
  if (
    Date.parse(value.from) !== requestedFrom ||
    Date.parse(value.to) !== requestedTo ||
    value.step !== request.step
  ) {
    throw invalid();
  }
  let previous = Number.NEGATIVE_INFINITY;
  for (const point of value.points) {
    const timestamp = Date.parse(pointTimestamp(point));
    const consistent =
      point.kind === "sample"
        ? request.step === "15s" &&
          consistentSample(point) &&
          timestamp >= requestedFrom
        : request.step !== "15s" &&
          point.stepSeconds === AGGREGATE_STEP_SECONDS[request.step] &&
          consistentSurfaces(point.http) &&
          consistentGPU(point.gpu, true) &&
          timestamp + point.stepSeconds * 1_000 > requestedFrom;
    if (!consistent || timestamp >= requestedTo || timestamp < previous) {
      throw invalid();
    }
    previous = timestamp;
  }
  return value;
}

/**
 * Invariants of a resource summary that the schema cannot express: the
 * status follows the collection policy, an available or partial summary
 * carries resources of the same completeness, and RSS boundaries never
 * exceed the observed peak that sampling recorded.
 */
export function consistentAllocationResourceSummary(
  value: AllocationResourceSummary,
): boolean {
  const resource = value.resources;
  const policyMatches =
    value.collectionPolicy === "requested"
      ? !["disabled", "unsupported"].includes(value.status)
      : value.status === value.collectionPolicy;
  if (
    !policyMatches ||
    (value.status === "available" && resource?.status !== "complete") ||
    (value.status === "partial" && resource?.status !== "partial")
  ) {
    return false;
  }
  if (resource === undefined) return true;
  const hasSamples = (resource.rssSampleCount ?? 0) > 0;
  const boundaries = [resource.rssStartBytes, resource.rssEndBytes].filter(
    (boundary) => boundary !== undefined,
  );
  return (
    (resource.maxSampleGapSeconds === undefined ||
      resource.durationSeconds === undefined ||
      resource.maxSampleGapSeconds <= resource.durationSeconds) &&
    hasSamples === (resource.rssPeakObservedBytes !== undefined) &&
    boundaries.every(
      (boundary) =>
        hasSamples && boundary <= (resource.rssPeakObservedBytes ?? -1),
    ) &&
    (!hasSamples || (resource.rssSampleCount ?? 0) >= boundaries.length)
  );
}

export function performanceHistoryWindow(
  range: PerformanceRange,
  now = new Date(),
): PerformanceHistoryRequest {
  if (!Number.isFinite(now.valueOf())) {
    throw new TypeError("Performance history clock is invalid");
  }
  const selection = PERFORMANCE_RANGES[range];
  return {
    from: new Date(now.valueOf() - selection.durationMs).toISOString(),
    to: now.toISOString(),
    step: selection.step,
  };
}

export function histogramQuantileUpperBound(
  histogram: PerformanceHistogram | undefined,
  quantile: number,
): number | "overflow" | undefined {
  if (
    histogram === undefined ||
    !monotonicHistogram(histogram) ||
    !Number.isFinite(quantile) ||
    quantile <= 0 ||
    quantile > 1 ||
    histogram.count === 0
  ) {
    return undefined;
  }
  const target = Math.ceil(histogram.count * quantile);
  const index = histogram.buckets.findIndex((count) => count >= target);
  return index < HISTOGRAM_BOUNDS_SECONDS.length
    ? HISTOGRAM_BOUNDS_SECONDS[index]
    : "overflow";
}

export async function getPerformance(
  api: PublicAPI,
  signal?: AbortSignal,
): Promise<PerformanceSnapshot> {
  const result = await api.request((client) =>
    client.GET("/v1/operations/performance", {
      ...(signal === undefined ? {} : { signal }),
    }),
  );
  return requireSnapshot(requireData(result), result.response.status);
}

export async function getPerformanceHistory(
  api: PublicAPI,
  request: PerformanceHistoryRequest,
  signal?: AbortSignal,
): Promise<PerformanceHistory> {
  const result = await api.request((client) =>
    client.GET("/v1/operations/performance/history", {
      params: { query: request },
      ...(signal === undefined ? {} : { signal }),
    }),
  );
  return requireHistory(requireData(result), request, result.response.status);
}

export async function listAllocationResourceHistory(
  api: PublicAPI,
  request: AllocationResourcePageRequest = {},
  signal?: AbortSignal,
): Promise<AllocationResourcePage> {
  if (request.runId !== undefined && !RESOURCE_ID_PATTERN.test(request.runId)) {
    throw new TypeError("Run ID is invalid");
  }
  const result = await api.request((client) =>
    client.GET("/v1/operations/allocation-history", {
      params: {
        query: {
          limit: ALLOCATION_HISTORY_PAGE_SIZE,
          ...(request.runId === undefined ? {} : { runId: request.runId }),
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
        },
      },
      ...(signal === undefined ? {} : { signal }),
    }),
  );
  const page = requireData(result);
  if (
    // At most the requested page size, continued exactly when more remain.
    page.items.length > ALLOCATION_HISTORY_PAGE_SIZE ||
    page.page.hasMore !== (page.page.nextCursor !== undefined) ||
    // One summary per allocation, all of the requested Run.
    new Set(page.items.map((item) => item.allocationId)).size !==
      page.items.length ||
    (request.runId !== undefined &&
      page.items.some((item) => item.runId !== request.runId)) ||
    !page.items.every(consistentAllocationResourceSummary)
  ) {
    throw invalidAPIResponse(
      result.response.status,
      "Server returned an invalid allocation resource page",
    );
  }
  return page;
}
