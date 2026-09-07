import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { PublicAPI } from "../../../api/client";
import { Application } from "../../../app/application";
import { applicationRoutes } from "../../../app/router";
import type { RuntimeConfig } from "../../../config/runtime-config";
import { RunEventsManager } from "../../../events/run-events";
import { performanceFreshnessState } from "./freshness";
import { metricSeriesSegments } from "./series";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

const session = {
  principal: {
    userId: "user_local",
    username: "owner",
    capabilities: ["user", "operations"] as const,
  },
  csrfToken: "a".repeat(43),
  idleExpiresAt: "2099-09-06T20:00:00Z",
  absoluteExpiresAt: "2099-09-07T12:00:00Z",
};

function apiResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "content-type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
    },
  });
}

class UnexpectedOperationsWebSocket {
  static instances = 0;

  constructor() {
    UnexpectedOperationsWebSocket.instances += 1;
  }
}

function renderRoute(api: PublicAPI, path: string) {
  const router = createMemoryRouter(applicationRoutes(), {
    initialEntries: [path],
  });
  const events = new RunEventsManager(runtimeConfig.apiBaseUrl, {
    WebSocketImplementation:
      UnexpectedOperationsWebSocket as unknown as typeof WebSocket,
  });
  return render(
    <Application
      api={api}
      publicAPI={api}
      runEvents={events}
      router={router}
    />,
  );
}

function freshness(timestamp: string, intervalSeconds: 15 | 60 | 300 = 15) {
  return {
    status: "ok",
    observedAt: timestamp,
    lastAttemptAt: timestamp,
    intervalSeconds,
    coverage: {
      startedAt: new Date(
        Date.parse(timestamp) - intervalSeconds * 1_000,
      ).toISOString(),
      endedAt: timestamp,
      durationSeconds: intervalSeconds,
      expectedSamples: 1,
      observedSamples: 1,
    },
  } as const;
}

function histogram(count: number) {
  return {
    count,
    sumSeconds: count * 0.1,
    buckets: Array.from({ length: 14 }, (_, index) => (index < 4 ? 0 : count)),
  };
}

function surface(kind: "public" | "private", count: number) {
  const counts = Array.from({ length: 10 }, () => [0, 0, 0, 0, 0, 0]);
  counts[0]![1] = count;
  return {
    surface: kind,
    inFlight: kind === "public" ? 2 : 0,
    counts,
    duration: histogram(count),
  };
}

function sample(timestamp: string) {
  return {
    version: 1,
    generation: "performance-generation-1",
    observedAt: timestamp,
    http: {
      freshness: freshness(timestamp),
      surfaces: [surface("public", 4), surface("private", 1)],
    },
    process: {
      freshness: freshness(timestamp),
      cpuCores: 1.25,
      rssBytes: 256 * 1024 * 1024,
      heapLiveBytes: 96 * 1024 * 1024,
      goroutines: 42,
      gcCycles: 7,
      gcPauseSeconds: 0.003,
    },
    pool: {
      freshness: freshness(timestamp),
      acquiredConnections: 3,
      idleConnections: 2,
      totalConnections: 5,
      maxConnections: 20,
      acquireCount: 5,
      acquireDurationSeconds: 0.25,
      emptyAcquireWaitSeconds: 0.01,
      canceledAcquireCount: 1,
    },
    database: {
      freshness: freshness(timestamp, 60),
      activeConnections: 2,
      lockWaitingConnections: 0,
      longestTransactionSeconds: 0.4,
      estimatedDeadTuples: 12,
      vacuumCount: 4,
      autovacuumCount: 3,
      rates: {
        intervalSeconds: 60,
        commitsPerSecond: 2,
        rollbacksPerSecond: 0,
        deadlocksPerSecond: 0,
        tempFilesPerSecond: 0,
        tempBytesPerSecond: 0,
        blocksReadPerSecond: 1,
        blocksHitPerSecond: 9,
        bufferHitRatio: 0.9,
      },
    },
    databaseSize: {
      freshness: freshness(timestamp, 300),
      sizeBytes: 2 * 1024 * 1024 * 1024,
    },
  };
}

function snapshot(timestamp = "2026-09-06T12:00:00Z") {
  return {
    enabled: true,
    generation: "performance-generation-1",
    observedAt: timestamp,
    sampleIntervalSeconds: 15,
    databaseIntervalSeconds: 60,
    databaseSizeIntervalSeconds: 300,
    current: sample(timestamp),
    diagnostics: {
      skippedSamples: 0,
      rejectedSamples: 0,
      skippedMinutes: 0,
      droppedMinutes: 0,
      pendingMinutes: 0,
    },
  };
}

function sessionResponse(request: Request): Response | undefined {
  return new URL(request.url).pathname === "/v1/auth/session"
    ? apiResponse(session)
    : undefined;
}

beforeEach(() => {
  UnexpectedOperationsWebSocket.instances = 0;
});

describe("Operations performance views", () => {
  it("renders truthful current/history metrics without reading the registry", async () => {
    let snapshotReads = 0;
    const historySteps: string[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const authenticated = sessionResponse(request);
        if (authenticated !== undefined) return authenticated;
        const url = new URL(request.url);
        if (url.pathname === "/v1/operations/snapshot") {
          snapshotReads += 1;
          throw new Error("performance must not read the registry snapshot");
        }
        if (url.pathname === "/v1/operations/performance") {
          return apiResponse(snapshot());
        }
        if (url.pathname === "/v1/operations/performance/history") {
          const from = url.searchParams.get("from")!;
          const to = url.searchParams.get("to")!;
          const step = url.searchParams.get("step")!;
          historySteps.push(step);
          const observedAt = new Date(Date.parse(to) - 15_000).toISOString();
          return apiResponse({
            from,
            to,
            step,
            points:
              step === "15s" ? [{ kind: "sample", ...sample(observedAt) }] : [],
          });
        }
        throw new Error(`unexpected ${request.method} ${request.url}`);
      }),
    );

    renderRoute(api, "/operations/performance");
    expect(
      await screen.findByRole("heading", { name: "Server performance" }),
    ).toBeInTheDocument();
    expect(
      await screen.findByRole("img", { name: "CPU usage" }),
    ).toBeInTheDocument();
    expect(
      screen.getByLabelText("CPU usage numeric summary"),
    ).toHaveTextContent("Observed points1");
    expect(screen.getAllByText("1.25 cores").length).toBeGreaterThan(0);
    expect(
      screen.getByText(/successful-acquisition mean, not p95/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/not free disk space/i)).toBeInTheDocument();
    expect(screen.getByText(/not a bloat verdict/i)).toBeInTheDocument();
    expect(snapshotReads).toBe(0);
    expect(UnexpectedOperationsWebSocket.instances).toBe(0);

    await userEvent.selectOptions(screen.getByLabelText("Time range"), "7d");
    await waitFor(() => expect(historySteps).toContain("1h"));
    expect(snapshotReads).toBe(0);
  });

  it("aborts obsolete history when the selected range changes", async () => {
    let fineSignal: AbortSignal | undefined;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const authenticated = sessionResponse(request);
        if (authenticated !== undefined) return authenticated;
        const url = new URL(request.url);
        if (url.pathname === "/v1/operations/performance") {
          return apiResponse({
            ...snapshot(),
            enabled: false,
            current: undefined,
          });
        }
        if (url.pathname === "/v1/operations/performance/history") {
          const from = url.searchParams.get("from")!;
          const to = url.searchParams.get("to")!;
          const step = url.searchParams.get("step")!;
          if (step === "15s") {
            fineSignal = request.signal;
            return new Promise<Response>((_resolve, reject) => {
              request.signal.addEventListener("abort", () =>
                reject(new DOMException("aborted", "AbortError")),
              );
            });
          }
          return apiResponse({ from, to, step, points: [] });
        }
        throw new Error(`unexpected ${request.method} ${request.url}`);
      }),
    );

    renderRoute(api, "/operations/performance");
    expect(
      await screen.findByText("Performance collection is disabled."),
    ).toBeInTheDocument();
    await waitFor(() => expect(fineSignal).toBeDefined());
    await userEvent.selectOptions(screen.getByLabelText("Time range"), "7d");
    await waitFor(() => expect(fineSignal?.aborted).toBe(true));
    expect(
      await screen.findAllByText("No observations in this range."),
    ).toHaveLength(3);
    expect(UnexpectedOperationsWebSocket.instances).toBe(0);
  });

  it("shows durable completed summaries without lifecycle controls", async () => {
    let snapshotReads = 0;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const authenticated = sessionResponse(request);
        if (authenticated !== undefined) return authenticated;
        const url = new URL(request.url);
        if (url.pathname === "/v1/operations/snapshot") {
          snapshotReads += 1;
          throw new Error("completed history must not read the registry");
        }
        if (url.pathname === "/v1/operations/allocation-history") {
          return apiResponse({
            items: [
              {
                allocationId: "allocation-complete",
                runId: "run-complete",
                stageExecutionId: "stage-execution-complete",
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
                  durationSeconds: 20,
                  cpuUserSeconds: 3,
                  cpuSystemSeconds: 1,
                  rssStartBytes: 100_000_000,
                  rssEndBytes: 120_000_000,
                  rssPeakObservedBytes: 134_217_728,
                  rssSampleCount: 3,
                  maxSampleGapSeconds: 10,
                },
              },
              {
                allocationId: "allocation-missing",
                runId: "run-missing",
                stageExecutionId: "stage-execution-missing",
                stage: "document",
                logicalAgent: "writer",
                outcome: "failed",
                finishedAt: "2026-09-06T11:00:00Z",
                collectionPolicy: "requested",
                status: "unavailable",
                reason: "report_missing",
              },
            ],
            page: { hasMore: false },
          });
        }
        throw new Error(`unexpected ${request.method} ${request.url}`);
      }),
    );

    renderRoute(api, "/operations/allocations/completed");
    expect(
      await screen.findByRole("heading", {
        name: "Completed allocation resources",
      }),
    ).toBeInTheDocument();
    expect(
      await screen.findByRole("link", { name: "run-complete" }),
    ).toHaveAttribute("href", "/runs/run-complete");
    expect(screen.getByText("128.0 MiB")).toBeInTheDocument();
    expect(screen.getByText("report missing")).toBeInTheDocument();
    expect(screen.getAllByText("Unavailable").length).toBeGreaterThan(0);
    expect(
      screen.queryByRole("button", { name: /force|idle|reassign|release/i }),
    ).not.toBeInTheDocument();
    expect(snapshotReads).toBe(0);
    expect(UnexpectedOperationsWebSocket.instances).toBe(0);
  });

  it("splits charts at missing values, gaps and generation changes", () => {
    const data = [
      { observedAt: "2026-09-06T12:00:00Z", generation: "one", value: 1 },
      { observedAt: "2026-09-06T12:00:15Z", generation: "one" },
      { observedAt: "2026-09-06T12:00:30Z", generation: "one", value: 2 },
      { observedAt: "2026-09-06T12:02:00Z", generation: "one", value: 3 },
      { observedAt: "2026-09-06T12:02:15Z", generation: "two", value: 4 },
    ];
    expect(
      metricSeriesSegments(data, 15).map((part) => part.values.length),
    ).toEqual([1, 1, 1, 1]);
    expect(
      performanceFreshnessState(
        freshness("2026-09-06T12:00:00Z"),
        "2026-09-06T12:00:31Z",
      ),
    ).toBe("stale");
  });
});
