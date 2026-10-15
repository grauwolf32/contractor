import { QueryClient, QueryObserver } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { PublicAPIError } from "../api/error";
import { pollInterval } from "./polling";

interface Job {
  state: "running" | "done";
}

/**
 * Observes one polled query on fake timers and records when each read
 * happened, so the intervals between reads can be checked.
 */
function observe(
  read: () => Promise<Job>,
  options: Parameters<typeof pollInterval<Job>>[0],
) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const reads: number[] = [];
  const observer = new QueryObserver<Job>(client, {
    queryKey: ["job"],
    queryFn: () => {
      reads.push(Date.now());
      return read();
    },
    refetchInterval: pollInterval<Job>(options),
    refetchIntervalInBackground: true,
  });
  const unsubscribe = observer.subscribe(() => undefined);
  return {
    reads,
    /** Milliseconds between consecutive reads. */
    gaps: () => reads.slice(1).map((time, index) => time - reads[index]!),
    stop: () => {
      unsubscribe();
      client.clear();
    },
  };
}

describe("pollInterval", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("polls at the base interval until the data settles, then stops", async () => {
    const states: Job["state"][] = ["running", "running", "done"];
    const polled = observe(async () => ({ state: states.shift() ?? "done" }), {
      interval: 1_000,
      settled: (job) => job.state === "done",
    });
    await vi.advanceTimersByTimeAsync(10_000);
    expect(polled.gaps()).toEqual([1_000, 1_000]);
    polled.stop();
  });

  it("keeps a settled object on a slow poll when asked to", async () => {
    const polled = observe(async () => ({ state: "done" }), {
      interval: 1_000,
      settled: (job) => job.state === "done",
      settledInterval: 30_000,
    });
    await vi.advanceTimersByTimeAsync(65_000);
    expect(polled.gaps()).toEqual([30_000, 30_000]);
    polled.stop();
  });

  it("backs off exponentially while reads fail and returns to the base after a success", async () => {
    // Reads at 0 (ok), then four failures, then successes.
    const outcomes = [true, false, false, false, false, true, true];
    const polled = observe(
      async () => {
        if (outcomes.shift() === false) throw new Error("Server unavailable");
        return { state: "running" };
      },
      { interval: 1_000, settled: (job) => job.state === "done" },
    );
    await vi.advanceTimersByTimeAsync(40_000);
    // 1 s to the first failure, then 2, 4, 8 and 16 s after each failure,
    // then the base interval again once a read succeeded.
    expect(polled.gaps().slice(0, 6)).toEqual([
      1_000, 2_000, 4_000, 8_000, 16_000, 1_000,
    ]);
    polled.stop();
  });

  it("caps the backoff at the maximum interval", async () => {
    const polled = observe(
      async () => {
        throw new Error("Server unavailable");
      },
      { interval: 1_000, maxInterval: 5_000 },
    );
    await vi.advanceTimersByTimeAsync(30_000);
    expect(polled.gaps().slice(0, 5)).toEqual([
      2_000, 4_000, 5_000, 5_000, 5_000,
    ]);
    polled.stop();
  });

  it("stops after a refusal that reading again cannot change", async () => {
    const outcomes = [true, false];
    const polled = observe(
      async () => {
        if (outcomes.shift() === false)
          throw new PublicAPIError({
            status: 404,
            code: "not_found",
            message: "Project not found",
          });
        return { state: "running" };
      },
      { interval: 1_000 },
    );
    await vi.advanceTimersByTimeAsync(30_000);
    expect(polled.gaps()).toEqual([1_000]);
    polled.stop();
  });
});
