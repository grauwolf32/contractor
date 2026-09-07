import type { PublicAPI } from "./client";
import { PublicAPIError, publicAPIError } from "./error";
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
const CONFIG_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/;
const TERMINAL_STAGE_STATES = new Set([
  "succeeded",
  "failed",
  "interrupted",
  "cancelled",
]);
const PERFORMANCE_REASONS = new Set([
  "unsupported_platform",
  "read_failed",
  "sampling_gap",
  "counter_reset",
  "missing_baseline",
  "permission_denied",
  "statistics_disabled",
  "database_unavailable",
  "budget_exceeded",
  "record_limit",
]);
const RESOURCE_REASONS = new Set([
  "unsupported_platform",
  "read_failed",
  "sampling_gap",
  "counter_reset",
  "invalid_report",
  "legacy",
  "report_missing",
]);
const RUNTIME_RESOURCE_REASONS = new Set([
  "unsupported_platform",
  "read_failed",
  "sampling_gap",
  "counter_reset",
  "invalid_report",
]);
const HISTOGRAM_BOUNDS_SECONDS = [
  0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60,
] as const;

function invalidResponse(status: number, message: string): PublicAPIError {
  return new PublicAPIError({
    status,
    code: "invalid_api_response",
    message,
  });
}

function requireData<T>(result: {
  data?: T;
  error?: unknown;
  response: Response;
}): T {
  if (result.data === undefined) {
    throw publicAPIError(result.response.status, result.error);
  }
  return result.data;
}

function validTimestamp(value: unknown): value is string {
  return (
    typeof value === "string" &&
    value.length <= 64 &&
    !Number.isNaN(Date.parse(value))
  );
}

function finiteNonnegative(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

function unsignedInteger(value: unknown): value is number {
  return finiteNonnegative(value) && Number.isSafeInteger(value);
}

function exactKeys(
  value: object | null | undefined,
  required: readonly string[],
  optional: readonly string[] = [],
): boolean {
  if (value === null || value === undefined || Array.isArray(value)) {
    return false;
  }
  const keys = Object.keys(value);
  const allowed = new Set([...required, ...optional]);
  return (
    required.every((key) => keys.includes(key)) &&
    keys.every((key) => allowed.has(key))
  );
}

function boundedJSONTree(value: unknown, depth = 0): boolean {
  if (value === null || depth > 12) return false;
  if (typeof value === "number") return finiteNonnegative(value);
  if (typeof value === "string") return value.length <= 1_024;
  if (typeof value === "boolean") return true;
  if (Array.isArray(value)) {
    return (
      value.length <= 1_024 &&
      value.every((entry) => boundedJSONTree(entry, depth + 1))
    );
  }
  if (typeof value !== "object") return false;
  const entries = Object.entries(value);
  return (
    entries.length <= 64 &&
    entries.every(
      ([key, entry]) => key.length <= 128 && boundedJSONTree(entry, depth + 1),
    )
  );
}

function validFreshness(value: PerformanceFreshness): boolean {
  return (
    exactKeys(
      value,
      ["status", "lastAttemptAt", "intervalSeconds", "coverage"],
      ["reason", "observedAt"],
    ) &&
    ["ok", "partial", "unavailable"].includes(value.status) &&
    (value.reason === undefined || PERFORMANCE_REASONS.has(value.reason)) &&
    validTimestamp(value.lastAttemptAt) &&
    (value.observedAt === undefined || validTimestamp(value.observedAt)) &&
    [15, 60, 300].includes(value.intervalSeconds) &&
    exactKeys(value.coverage, [
      "startedAt",
      "endedAt",
      "durationSeconds",
      "expectedSamples",
      "observedSamples",
    ]) &&
    validTimestamp(value.coverage.startedAt) &&
    validTimestamp(value.coverage.endedAt) &&
    finiteNonnegative(value.coverage.durationSeconds) &&
    unsignedInteger(value.coverage.expectedSamples) &&
    unsignedInteger(value.coverage.observedSamples) &&
    value.coverage.observedSamples <= value.coverage.expectedSamples
  );
}

function validHistogram(value: PerformanceHistogram): boolean {
  if (
    !exactKeys(value, ["count", "sumSeconds", "buckets"]) ||
    !unsignedInteger(value.count) ||
    !finiteNonnegative(value.sumSeconds) ||
    value.buckets.length !== HISTOGRAM_BOUNDS_SECONDS.length + 1
  ) {
    return false;
  }
  let previous = 0;
  for (const bucket of value.buckets) {
    if (!unsignedInteger(bucket) || bucket < previous || bucket > value.count) {
      return false;
    }
    previous = bucket;
  }
  return previous === value.count;
}

function validHTTPSurface(value: unknown): boolean {
  if (typeof value !== "object" || value === null) return false;
  const surface = value as components["schemas"]["PerformanceHTTPSurface"];
  return (
    exactKeys(surface, ["surface", "inFlight", "counts", "duration"]) &&
    (surface.surface === "public" || surface.surface === "private") &&
    unsignedInteger(surface.inFlight) &&
    surface.counts.length === 10 &&
    surface.counts.every(
      (row) => row.length === 6 && row.every(unsignedInteger),
    ) &&
    validHistogram(surface.duration)
  );
}

function validSample(
  value: components["schemas"]["PerformanceSample"],
  historyPoint = false,
): boolean {
  if (
    !exactKeys(
      value,
      ["version", "generation", "observedAt"],
      [
        "http",
        "process",
        "pool",
        "database",
        "databaseSize",
        ...(historyPoint ? ["kind"] : []),
      ],
    ) ||
    (historyPoint &&
      (value as components["schemas"]["PerformanceFineHistoryPoint"]).kind !==
        "sample") ||
    value.version !== 1 ||
    typeof value.generation !== "string" ||
    value.generation.length === 0 ||
    value.generation.length > 128 ||
    !validTimestamp(value.observedAt) ||
    !boundedJSONTree(value)
  ) {
    return false;
  }
  if (
    value.http !== undefined &&
    (!exactKeys(value.http, ["freshness", "surfaces"]) ||
      !validFreshness(value.http.freshness) ||
      value.http.surfaces.length !== 2 ||
      !value.http.surfaces.every(validHTTPSurface))
  ) {
    return false;
  }
  for (const group of [
    value.process,
    value.pool,
    value.database,
    value.databaseSize,
  ]) {
    if (group !== undefined && !validFreshness(group.freshness)) return false;
  }
  if (
    value.database?.rates?.bufferHitRatio !== undefined &&
    value.database.rates.bufferHitRatio > 1
  ) {
    return false;
  }
  return true;
}

function safeSnapshot(
  value: PerformanceSnapshot,
  status: number,
): PerformanceSnapshot {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw invalidResponse(status, "Server returned invalid performance data");
  }
  const diagnostics = value.diagnostics;
  if (
    !exactKeys(
      value,
      [
        "enabled",
        "generation",
        "observedAt",
        "sampleIntervalSeconds",
        "databaseIntervalSeconds",
        "databaseSizeIntervalSeconds",
        "diagnostics",
      ],
      ["current"],
    ) ||
    typeof value.enabled !== "boolean" ||
    typeof value.generation !== "string" ||
    value.generation.length === 0 ||
    value.generation.length > 128 ||
    !validTimestamp(value.observedAt) ||
    value.sampleIntervalSeconds !== 15 ||
    value.databaseIntervalSeconds !== 60 ||
    value.databaseSizeIntervalSeconds !== 300 ||
    !exactKeys(
      diagnostics,
      [
        "skippedSamples",
        "rejectedSamples",
        "skippedMinutes",
        "droppedMinutes",
        "pendingMinutes",
      ],
      ["collectorReason", "writerReason"],
    ) ||
    ![
      diagnostics.skippedSamples,
      diagnostics.rejectedSamples,
      diagnostics.skippedMinutes,
      diagnostics.droppedMinutes,
      diagnostics.pendingMinutes,
    ].every(unsignedInteger) ||
    diagnostics.pendingMinutes > 10 ||
    (diagnostics.collectorReason !== undefined &&
      !PERFORMANCE_REASONS.has(diagnostics.collectorReason)) ||
    (diagnostics.writerReason !== undefined &&
      !PERFORMANCE_REASONS.has(diagnostics.writerReason)) ||
    (!value.enabled && value.current !== undefined) ||
    (value.current !== undefined &&
      (!validSample(value.current) ||
        value.current.generation !== value.generation))
  ) {
    throw invalidResponse(status, "Server returned invalid performance data");
  }
  return {
    ...value,
    diagnostics: { ...diagnostics },
    ...(value.current === undefined ? {} : { current: { ...value.current } }),
  };
}

function pointTimestamp(point: PerformanceHistoryPoint): string {
  return point.kind === "sample" ? point.observedAt : point.minuteStart;
}

function safeHistory(
  value: PerformanceHistory,
  request: PerformanceHistoryRequest,
  status: number,
): PerformanceHistory {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw invalidResponse(
      status,
      "Server returned invalid performance history",
    );
  }
  const from = Date.parse(value.from);
  const to = Date.parse(value.to);
  const requestedFrom = Date.parse(request.from);
  const requestedTo = Date.parse(request.to);
  let previous = Number.NEGATIVE_INFINITY;
  if (
    !exactKeys(value, ["from", "to", "step", "points"]) ||
    !validTimestamp(value.from) ||
    !validTimestamp(value.to) ||
    from !== requestedFrom ||
    to !== requestedTo ||
    value.step !== request.step ||
    !Array.isArray(value.points) ||
    value.points.length > 1_000
  ) {
    throw invalidResponse(
      status,
      "Server returned invalid performance history",
    );
  }
  for (const point of value.points) {
    if (typeof point !== "object" || point === null || Array.isArray(point)) {
      throw invalidResponse(
        status,
        "Server returned invalid performance history points",
      );
    }
    const timestamp = Date.parse(pointTimestamp(point));
    const kindMatches =
      request.step === "15s"
        ? point.kind === "sample"
        : point.kind === "aggregate";
    const structurallyValid =
      point.kind === "sample"
        ? validSample(point, true)
        : exactKeys(
            point,
            [
              "kind",
              "version",
              "generation",
              "minuteStart",
              "status",
              "omittedWindows",
              "coverageSeconds",
              "process",
              "pool",
              "droppedMinutes",
              "stepSeconds",
              "observedMinutes",
              "expectedMinutes",
            ],
            [
              "http",
              "cpu",
              "poolLast",
              "gcPausesLast",
              "database",
              "databaseSize",
            ],
          ) &&
          point.version === 1 &&
          typeof point.generation === "string" &&
          point.generation.length > 0 &&
          point.generation.length <= 128 &&
          validTimestamp(point.minuteStart) &&
          ["ok", "partial", "unavailable"].includes(point.status) &&
          point.stepSeconds ===
            ({ "1m": 60, "5m": 300, "1h": 3_600 } as const)[
              request.step as "1m" | "5m" | "1h"
            ] &&
          Array.isArray(point.http ?? []) &&
          (point.http === undefined ||
            (point.http.length === 2 && point.http.every(validHTTPSurface))) &&
          boundedJSONTree(point);
    if (
      !kindMatches ||
      !structurallyValid ||
      !Number.isFinite(timestamp) ||
      timestamp < requestedFrom ||
      timestamp >= requestedTo ||
      timestamp < previous
    ) {
      throw invalidResponse(
        status,
        "Server returned invalid performance history points",
      );
    }
    previous = timestamp;
  }
  return { ...value, points: value.points.map((point) => ({ ...point })) };
}

export function safeAllocationResourceSummary(
  value: AllocationResourceSummary,
  status = 200,
): AllocationResourceSummary {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw invalidResponse(
      status,
      "Server returned an invalid allocation resource summary",
    );
  }
  const allowedStatuses = [
    "disabled",
    "unsupported",
    "pending",
    "available",
    "partial",
    "unavailable",
  ];
  const resource = value.resources;
  const resourceFloats = [
    resource?.durationSeconds,
    resource?.cpuUserSeconds,
    resource?.cpuSystemSeconds,
    resource?.maxSampleGapSeconds,
  ];
  const resourceIntegers = [
    resource?.rssStartBytes,
    resource?.rssEndBytes,
    resource?.rssPeakObservedBytes,
    resource?.rssSampleCount,
  ];
  const hasSamples =
    resource?.rssSampleCount !== undefined && resource.rssSampleCount > 0;
  const hasPeak = resource?.rssPeakObservedBytes !== undefined;
  const boundaries = [resource?.rssStartBytes, resource?.rssEndBytes].filter(
    (boundary) => boundary !== undefined,
  );
  const resourceValid =
    resource === undefined ||
    (exactKeys(
      resource,
      ["version", "scope", "status"],
      [
        "reason",
        "durationSeconds",
        "cpuUserSeconds",
        "cpuSystemSeconds",
        "rssStartBytes",
        "rssEndBytes",
        "rssPeakObservedBytes",
        "rssSampleCount",
        "maxSampleGapSeconds",
      ],
    ) &&
      resource.version === 1 &&
      resource.scope === "runtime_process" &&
      ["complete", "partial", "unavailable"].includes(resource.status) &&
      (resource.reason === undefined ||
        RUNTIME_RESOURCE_REASONS.has(resource.reason)) &&
      resourceFloats.every(
        (measurement) =>
          measurement === undefined || finiteNonnegative(measurement),
      ) &&
      resourceIntegers.every(
        (measurement) =>
          measurement === undefined || unsignedInteger(measurement),
      ) &&
      (resource.maxSampleGapSeconds === undefined ||
        resource.durationSeconds === undefined ||
        resource.maxSampleGapSeconds <= resource.durationSeconds) &&
      hasSamples === hasPeak &&
      boundaries.every(
        (boundary) =>
          hasSamples && boundary <= (resource.rssPeakObservedBytes ?? -1),
      ) &&
      (!hasSamples || (resource.rssSampleCount ?? 0) >= boundaries.length) &&
      (resource.status !== "complete" ||
        (resource.reason === undefined &&
          resource.durationSeconds !== undefined &&
          resource.cpuUserSeconds !== undefined &&
          resource.cpuSystemSeconds !== undefined &&
          boundaries.length === 2 &&
          resource.maxSampleGapSeconds !== undefined &&
          resource.maxSampleGapSeconds <= 30)) &&
      boundedJSONTree(resource));
  const policyMatches =
    (value.collectionPolicy === "disabled" && value.status === "disabled") ||
    (value.collectionPolicy === "unsupported" &&
      value.status === "unsupported") ||
    (value.collectionPolicy === "legacy" &&
      value.status === "unavailable" &&
      value.reason === "legacy") ||
    (value.collectionPolicy === "requested" &&
      ["pending", "available", "partial", "unavailable"].includes(
        value.status,
      ));
  if (
    !exactKeys(
      value,
      [
        "allocationId",
        "runId",
        "stageExecutionId",
        "stage",
        "logicalAgent",
        "outcome",
        "finishedAt",
        "collectionPolicy",
        "status",
      ],
      ["reason", "resources"],
    ) ||
    typeof value.allocationId !== "string" ||
    typeof value.runId !== "string" ||
    typeof value.stageExecutionId !== "string" ||
    typeof value.stage !== "string" ||
    typeof value.logicalAgent !== "string" ||
    !RESOURCE_ID_PATTERN.test(value.allocationId) ||
    !RESOURCE_ID_PATTERN.test(value.runId) ||
    !RESOURCE_ID_PATTERN.test(value.stageExecutionId) ||
    !CONFIG_ID_PATTERN.test(value.stage) ||
    !CONFIG_ID_PATTERN.test(value.logicalAgent) ||
    !TERMINAL_STAGE_STATES.has(value.outcome) ||
    !validTimestamp(value.finishedAt) ||
    !allowedStatuses.includes(value.status) ||
    (value.reason !== undefined && !RESOURCE_REASONS.has(value.reason)) ||
    !resourceValid ||
    !policyMatches ||
    ((value.status === "available" || value.status === "partial") &&
      resource === undefined) ||
    (value.status === "available" && resource?.status !== "complete") ||
    (value.status === "partial" && resource?.status !== "partial")
  ) {
    throw invalidResponse(
      status,
      "Server returned an invalid allocation resource summary",
    );
  }
  return {
    ...value,
    ...(resource === undefined ? {} : { resources: { ...resource } }),
  };
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
    !validHistogram(histogram) ||
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
  return safeSnapshot(requireData(result), result.response.status);
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
  return safeHistory(requireData(result), request, result.response.status);
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
    typeof page !== "object" ||
    page === null ||
    Array.isArray(page) ||
    !exactKeys(page, ["items", "page"]) ||
    !Array.isArray(page.items) ||
    page.items.length > ALLOCATION_HISTORY_PAGE_SIZE ||
    typeof page.page !== "object" ||
    page.page === null ||
    Array.isArray(page.page) ||
    !exactKeys(page.page, ["hasMore"], ["nextCursor"]) ||
    typeof page.page.hasMore !== "boolean" ||
    page.page.hasMore !== (page.page.nextCursor !== undefined) ||
    (page.page.nextCursor !== undefined &&
      (page.page.nextCursor.length === 0 || page.page.nextCursor.length > 512))
  ) {
    throw invalidResponse(
      result.response.status,
      "Server returned an invalid allocation resource page",
    );
  }
  const items = page.items.map((item) =>
    safeAllocationResourceSummary(item, result.response.status),
  );
  if (
    new Set(items.map((item) => item.allocationId)).size !== items.length ||
    (request.runId !== undefined &&
      items.some((item) => item.runId !== request.runId))
  ) {
    throw invalidResponse(
      result.response.status,
      "Server returned mixed allocation resource identities",
    );
  }
  return {
    items,
    page: {
      hasMore: page.page.hasMore,
      ...(page.page.nextCursor === undefined
        ? {}
        : { nextCursor: page.page.nextCursor }),
    },
  };
}
