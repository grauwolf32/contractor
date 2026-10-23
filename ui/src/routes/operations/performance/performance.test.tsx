import { screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { components } from "../../../api/generated/public";
import { FakeEventSocket, renderApplication } from "../../../test/application";
import { conforming } from "../../../test/builders";
import type { FakeCall } from "../../../test/fake-server";
import { json, page } from "../../../test/responses";
import { performanceFreshnessState } from "./freshness";
import { metricSeriesSegments } from "./series";

type Schemas = components["schemas"];

function freshness(
  timestamp: string,
  intervalSeconds: 15 | 60 | 300 = 15,
): Schemas["PerformanceFreshness"] {
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
  };
}

function histogram(count: number): Schemas["PerformanceHistogram"] {
  return {
    count,
    sumSeconds: count * 0.1,
    buckets: Array.from({ length: 14 }, (_, index) => (index < 4 ? 0 : count)),
  };
}

function surface(
  kind: "public" | "private",
  count: number,
): Schemas["PerformanceHTTPSurface"] {
  const counts = Array.from({ length: 10 }, () => [0, 0, 0, 0, 0, 0]);
  counts[0]![1] = count;
  return {
    surface: kind,
    inFlight: kind === "public" ? 2 : 0,
    counts,
    duration: histogram(count),
  };
}

function sample(timestamp: string): Schemas["PerformanceSample"] {
  return conforming("PerformanceSample", {
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
  });
}

const diagnostics: Schemas["PerformanceDiagnostics"] = {
  skippedSamples: 0,
  rejectedSamples: 0,
  skippedMinutes: 0,
  droppedMinutes: 0,
  pendingMinutes: 0,
};

function snapshot(
  timestamp = "2026-09-06T12:00:00Z",
): Schemas["PerformanceSnapshot"] {
  return conforming("PerformanceSnapshot", {
    enabled: true,
    generation: "performance-generation-1",
    observedAt: timestamp,
    sampleIntervalSeconds: 15,
    databaseIntervalSeconds: 60,
    databaseSizeIntervalSeconds: 300,
    current: sample(timestamp),
    diagnostics,
  });
}

const disabledSnapshot = () =>
  conforming("PerformanceSnapshot", {
    enabled: false,
    generation: "performance-generation-1",
    observedAt: "2026-09-06T12:00:00Z",
    sampleIntervalSeconds: 15,
    databaseIntervalSeconds: 60,
    databaseSizeIntervalSeconds: 300,
    diagnostics,
  });

/** The history request's window, as the page asked for it. */
function window({ url }: FakeCall) {
  return {
    from: url.searchParams.get("from")!,
    to: url.searchParams.get("to")!,
    step: url.searchParams.get("step") as Schemas["PerformanceHistory"]["step"],
  };
}

describe("Operations performance views", () => {
  it.each(["24h", "7d"] as const)(
    "renders the partial first interval for %s",
    async (range) => {
      vi.useFakeTimers({ toFake: ["Date"] });
      vi.setSystemTime(new Date("2026-09-06T12:01:30Z"));
      try {
        const { user } = renderApplication("/operations/performance", {
          server: {
            "GET /v1/operations/performance": () => json(snapshot()),
            "GET /v1/operations/performance/history": (call) => {
              const { from, to, step } = window(call);
              const stepSeconds = step === "5m" ? 300 : 3600;
              const first =
                Math.floor(Date.parse(from) / (stepSeconds * 1000)) *
                stepSeconds *
                1000;
              return json({
                from,
                to,
                step,
                points:
                  step === "15s"
                    ? []
                    : [
                        {
                          kind: "aggregate",
                          version: 1,
                          generation: "performance-generation-1",
                          minuteStart: new Date(first).toISOString(),
                          status: "partial",
                          omittedWindows: 0,
                          coverageSeconds: 30,
                          process: {},
                          pool: {},
                          droppedMinutes: 0,
                          stepSeconds,
                          observedMinutes: 1,
                          expectedMinutes: stepSeconds / 60,
                          cpu: {
                            userSeconds: 30,
                            systemSeconds: 0,
                            durationSeconds: 30,
                            cores: 1,
                          },
                        },
                      ],
              });
            },
          },
        });
        await user.selectOptions(
          await screen.findByLabelText("Time range"),
          range,
        );
        expect(
          await screen.findByRole("img", { name: "CPU usage" }),
        ).toBeVisible();
        expect(
          screen.getByLabelText("CPU usage numeric summary"),
        ).toHaveTextContent("Observed points1");
        expect(screen.getByLabelText("Time range")).toHaveValue(range);
        expect(
          screen.queryByText(/Server returned invalid performance history/),
        ).toBeNull();
      } finally {
        vi.useRealTimers();
      }
    },
  );

  it("renders truthful current/history metrics without reading the registry", async () => {
    const historySteps: string[] = [];
    const { user, sent } = renderApplication("/operations/performance", {
      server: {
        "GET /v1/operations/performance": () => json(snapshot()),
        "GET /v1/operations/performance/history": (call) => {
          const { from, to, step } = window(call);
          historySteps.push(step);
          const observedAt = new Date(Date.parse(to) - 15_000).toISOString();
          return json({
            from,
            to,
            step,
            points:
              step === "15s" ? [{ kind: "sample", ...sample(observedAt) }] : [],
          });
        },
      },
    });

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
    await user.click(
      screen.getByText("Detailed counters and collection diagnostics"),
    );
    expect(
      screen.getByText(/successful-acquisition mean, not p95/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/not free disk space/i)).toBeInTheDocument();
    expect(screen.getByText(/not a bloat verdict/i)).toBeInTheDocument();
    expect(sent("GET", "/v1/operations/snapshot")).toHaveLength(0);
    expect(FakeEventSocket.instances).toHaveLength(0);

    await user.selectOptions(screen.getByLabelText("Time range"), "7d");
    await waitFor(() => expect(historySteps).toContain("1h"));
    expect(sent("GET", "/v1/operations/snapshot")).toHaveLength(0);
  });

  it("aborts obsolete history when the selected range changes", async () => {
    let fineSignal: AbortSignal | undefined;
    const { user } = renderApplication("/operations/performance", {
      server: {
        "GET /v1/operations/performance": () => json(disabledSnapshot()),
        "GET /v1/operations/performance/history": (call) => {
          const { from, to, step } = window(call);
          if (step !== "15s") return json({ from, to, step, points: [] });
          const signal = call.request.signal;
          fineSignal = signal;
          return new Promise<Response>((_resolve, reject) => {
            signal.addEventListener("abort", () =>
              reject(new DOMException("aborted", "AbortError")),
            );
          });
        },
      },
    });

    expect(
      await screen.findByText("Performance collection is disabled."),
    ).toBeInTheDocument();
    await waitFor(() => expect(fineSignal).toBeDefined());
    await user.selectOptions(screen.getByLabelText("Time range"), "7d");
    await waitFor(() => expect(fineSignal?.aborted).toBe(true));
    expect(
      await screen.findAllByText("No observations in this range."),
    ).toHaveLength(3);
    expect(FakeEventSocket.instances).toHaveLength(0);
  });

  it("shows durable completed summaries without lifecycle controls", async () => {
    const { user, sent } = renderApplication(
      "/operations/allocations/completed",
      {
        server: {
          "GET /v1/operations/allocation-history": () =>
            json(
              page([
                conforming("AllocationResourceSummary", {
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
                }),
                conforming("AllocationResourceSummary", {
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
                }),
                conforming("AllocationResourceSummary", {
                  allocationId: "allocation-expired",
                  runId: "run-expired",
                  stageExecutionId: "stage-execution-expired",
                  stage: "document",
                  logicalAgent: "writer",
                  outcome: "succeeded",
                  finishedAt: "2026-08-01T11:00:00Z",
                  collectionPolicy: "requested",
                  status: "unavailable",
                  reason: "report_expired",
                }),
              ]),
            ),
        },
      },
    );

    expect(
      await screen.findByRole("heading", {
        name: "Completed allocation resources",
      }),
    ).toBeInTheDocument();
    expect(
      await screen.findByRole("link", { name: "run-complete" }),
    ).toHaveAttribute("href", "/runs/run-complete");
    const metrics = screen.getByRole("button", {
      name: "Metrics for allocation-complete",
    });
    expect(metrics).toHaveAttribute("aria-expanded", "false");
    await user.click(metrics);
    expect(metrics).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("128.0 MiB")).toBeVisible();
    await user.click(
      screen.getByRole("button", { name: "Metrics for allocation-missing" }),
    );
    expect(screen.getAllByText("report missing").length).toBeGreaterThan(0);
    await user.click(
      screen.getByRole("button", { name: "Metrics for allocation-expired" }),
    );
    expect(screen.getAllByText("report expired").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Unavailable").length).toBeGreaterThan(0);
    expect(
      screen.queryByRole("button", { name: /force|idle|reassign|release/i }),
    ).not.toBeInTheDocument();
    expect(sent("GET", "/v1/operations/snapshot")).toHaveLength(0);
    expect(FakeEventSocket.instances).toHaveLength(0);
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
