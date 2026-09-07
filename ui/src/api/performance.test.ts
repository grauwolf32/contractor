import { describe, expect, it, vi } from "vitest";

import type { RuntimeConfig } from "../config/runtime-config";
import { PublicAPI } from "./client";
import {
  getPerformance,
  getPerformanceHistory,
  histogramQuantileUpperBound,
  listAllocationResourceHistory,
  PERFORMANCE_RANGES,
  performanceHistoryWindow,
} from "./performance";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

function response(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "content-type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
    },
  });
}

const disabledSnapshot = {
  enabled: false,
  generation: "generation-1",
  observedAt: "2026-09-06T12:00:00Z",
  sampleIntervalSeconds: 15,
  databaseIntervalSeconds: 60,
  databaseSizeIntervalSeconds: 300,
  diagnostics: {
    skippedSamples: 0,
    rejectedSamples: 0,
    skippedMinutes: 0,
    droppedMinutes: 0,
    pendingMinutes: 0,
  },
};

describe("performance API", () => {
  it("reads the independent current snapshot and rejects fabricated disabled data", async () => {
    const requests: Request[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request);
        return response(disabledSnapshot);
      }),
    );
    const result = await getPerformance(api);
    expect(result.enabled).toBe(false);
    expect(result.current).toBeUndefined();
    expect(requests[0]?.url).toBe(
      "http://127.0.0.1:8080/v1/operations/performance",
    );

    const invalid = new PublicAPI(
      runtimeConfig,
      vi.fn(async () =>
        response({
          ...disabledSnapshot,
          current: {
            version: 1,
            generation: "generation-1",
            observedAt: "2026-09-06T12:00:00Z",
          },
        }),
      ),
    );
    await expect(getPerformance(invalid)).rejects.toMatchObject({
      code: "invalid_api_response",
    });

    const nullResponse = new PublicAPI(
      runtimeConfig,
      vi.fn(async () => response(null)),
    );
    await expect(getPerformance(nullResponse)).rejects.toMatchObject({
      code: "invalid_api_response",
    });
  });

  it("builds bounded range requests and preserves discriminated history points", async () => {
    const now = new Date("2026-09-06T12:00:00Z");
    for (const range of Object.keys(PERFORMANCE_RANGES) as Array<
      keyof typeof PERFORMANCE_RANGES
    >) {
      const request = performanceHistoryWindow(range, now);
      const points =
        (Date.parse(request.to) - Date.parse(request.from)) /
        ({ "15s": 15_000, "1m": 60_000, "5m": 300_000, "1h": 3_600_000 }[
          request.step
        ] ?? 1);
      expect(points).toBeLessThanOrEqual(1_000);
    }
    const request = performanceHistoryWindow("1h", now);
    const requests: Request[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const incoming = input instanceof Request ? input : new Request(input);
        requests.push(incoming);
        return response({
          ...request,
          points: [
            {
              kind: "sample",
              version: 1,
              generation: "generation-1",
              observedAt: "2026-09-06T11:59:45Z",
            },
          ],
        });
      }),
    );
    const result = await getPerformanceHistory(api, request);
    expect(result.points[0]?.kind).toBe("sample");
    expect(new URL(requests[0]!.url).searchParams).toEqual(
      new URLSearchParams({
        from: request.from,
        to: request.to,
        step: request.step,
      }),
    );

    const malformed = new PublicAPI(
      runtimeConfig,
      vi.fn(async () =>
        response({
          ...request,
          points: [
            {
              kind: "aggregate",
              version: 1,
              generation: "generation-1",
              minuteStart: "2026-09-06T11:59:00Z",
              status: "partial",
              omittedWindows: 0,
              coverageSeconds: 30,
              process: {},
              pool: {},
              droppedMinutes: 0,
              stepSeconds: 60,
              observedMinutes: 1,
              expectedMinutes: 1,
            },
          ],
        }),
      ),
    );
    await expect(
      getPerformanceHistory(malformed, request),
    ).rejects.toMatchObject({ code: "invalid_api_response" });
  });

  it("derives quantile bucket bounds without inventing an empty or overflow value", () => {
    const histogram = {
      count: 4,
      sumSeconds: 61,
      buckets: [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4],
    };
    expect(histogramQuantileUpperBound(histogram, 0.5)).toBe(30);
    expect(histogramQuantileUpperBound(histogram, 0.99)).toBe("overflow");
    expect(
      histogramQuantileUpperBound(
        {
          ...histogram,
          count: 0,
          sumSeconds: 0,
          buckets: histogram.buckets.map(() => 0),
        },
        0.5,
      ),
    ).toBeUndefined();
  });

  it("pages only safe terminal allocation summaries for the requested Run", async () => {
    const requests: Request[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request);
        return response({
          items: [
            {
              allocationId: "allocation-1",
              runId: "run-1",
              stageExecutionId: "stage-1",
              stage: "analyze",
              logicalAgent: "builder",
              outcome: "succeeded",
              finishedAt: "2026-09-06T12:00:00Z",
              collectionPolicy: "requested",
              status: "available",
              resources: {
                version: 1,
                scope: "runtime_process",
                status: "complete",
                durationSeconds: 30,
                cpuUserSeconds: 4,
                cpuSystemSeconds: 1,
                rssStartBytes: 100,
                rssEndBytes: 120,
                rssPeakObservedBytes: 140,
                rssSampleCount: 4,
                maxSampleGapSeconds: 15,
              },
            },
          ],
          page: { hasMore: true, nextCursor: "next-page" },
        });
      }),
    );
    const page = await listAllocationResourceHistory(api, {
      runId: "run-1",
      cursor: "previous-page",
    });
    expect(page.items[0]?.resources?.rssPeakObservedBytes).toBe(140);
    expect(new URL(requests[0]!.url).searchParams).toEqual(
      new URLSearchParams({
        limit: "50",
        runId: "run-1",
        cursor: "previous-page",
      }),
    );

    const invalid = new PublicAPI(
      runtimeConfig,
      vi.fn(async () =>
        response({
          items: [
            {
              allocationId: "allocation-1",
              runId: "run-2",
              stageExecutionId: "stage-1",
              stage: "analyze",
              logicalAgent: "builder",
              outcome: "running",
              finishedAt: "2026-09-06T12:00:00Z",
              collectionPolicy: "disabled",
              status: "disabled",
            },
          ],
          page: { hasMore: false },
        }),
      ),
    );
    await expect(
      listAllocationResourceHistory(invalid, { runId: "run-1" }),
    ).rejects.toMatchObject({ code: "invalid_api_response" });

    const inconsistentResources = new PublicAPI(
      runtimeConfig,
      vi.fn(async () =>
        response({
          items: [
            {
              allocationId: "allocation-1",
              runId: "run-1",
              stageExecutionId: "stage-1",
              stage: "analyze",
              logicalAgent: "builder",
              outcome: "succeeded",
              finishedAt: "2026-09-06T12:00:00Z",
              collectionPolicy: "requested",
              status: "available",
              resources: {
                version: 1,
                scope: "runtime_process",
                status: "complete",
                durationSeconds: 30,
                cpuUserSeconds: 4,
                cpuSystemSeconds: 1,
                rssPeakObservedBytes: 100,
                rssSampleCount: 1,
                maxSampleGapSeconds: 15,
              },
            },
          ],
          page: { hasMore: false },
        }),
      ),
    );
    await expect(
      listAllocationResourceHistory(inconsistentResources),
    ).rejects.toMatchObject({ code: "invalid_api_response" });
  });
});
