import { afterEach, describe, expect, it, vi } from "vitest";

import type { OperationsSnapshotCursor } from "../../events/run-events";
import { OperationsLiveRefresh } from "./live-refresh";

const generation = "operations-generation-1";
const cursor = (revision: number): OperationsSnapshotCursor => ({
  generation,
  revision: String(revision),
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((complete) => {
    resolve = complete;
  });
  return { promise, resolve };
}

afterEach(() => vi.useRealTimers());

describe("Operations live REST refreshes", () => {
  it("coalesces 50 changes into one applied snapshot read", async () => {
    vi.useFakeTimers();
    const refreshSnapshot = vi.fn(async () => cursor(57));
    const updates = new OperationsLiveRefresh({
      initial: cursor(7),
      refreshSnapshot,
      refreshPrincipals: vi.fn(async () => undefined),
      resume: vi.fn(),
    });
    for (let revision = 8; revision <= 57; revision++) {
      updates.event(cursor(revision), "allocation");
    }
    await vi.advanceTimersByTimeAsync(999);
    expect(refreshSnapshot).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(refreshSnapshot).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(5_000);
    expect(refreshSnapshot).toHaveBeenCalledTimes(1);
    updates.dispose();
  });

  it("keeps an in-flight read and follows up only for a newer revision", async () => {
    vi.useFakeTimers();
    const first = deferred<OperationsSnapshotCursor>();
    const refreshSnapshot = vi
      .fn<() => Promise<OperationsSnapshotCursor>>()
      .mockImplementationOnce(() => first.promise)
      .mockResolvedValueOnce(cursor(9));
    const updates = new OperationsLiveRefresh({
      initial: cursor(7),
      refreshSnapshot,
      refreshPrincipals: vi.fn(async () => undefined),
      resume: vi.fn(),
    });
    updates.event(cursor(8), "allocation");
    await vi.advanceTimersByTimeAsync(1_000);
    updates.event(cursor(9), "allocation");
    expect(refreshSnapshot).toHaveBeenCalledTimes(1);
    first.resolve(cursor(8));
    await vi.advanceTimersByTimeAsync(999);
    expect(refreshSnapshot).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(refreshSnapshot).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(5_000);
    expect(refreshSnapshot).toHaveBeenCalledTimes(2);
    updates.dispose();
  });

  it("avoids a follow-up when the in-flight read includes the later event", async () => {
    vi.useFakeTimers();
    const first = deferred<OperationsSnapshotCursor>();
    const refreshSnapshot = vi.fn(() => first.promise);
    const updates = new OperationsLiveRefresh({
      initial: cursor(7),
      refreshSnapshot,
      refreshPrincipals: vi.fn(async () => undefined),
      resume: vi.fn(),
    });
    updates.event(cursor(8), "allocation");
    await vi.advanceTimersByTimeAsync(1_000);
    updates.event(cursor(9), "allocation");
    first.resolve(cursor(9));
    await vi.advanceTimersByTimeAsync(5_000);
    expect(refreshSnapshot).toHaveBeenCalledTimes(1);
    updates.dispose();
  });

  it("throttles principal reads under continuous heartbeat changes", async () => {
    vi.useFakeTimers();
    const first = deferred<void>();
    const refreshPrincipals = vi
      .fn<() => Promise<void>>()
      .mockImplementationOnce(() => first.promise)
      .mockResolvedValue(undefined);
    const updates = new OperationsLiveRefresh({
      initial: cursor(7),
      refreshSnapshot: vi.fn(async () => cursor(100)),
      refreshPrincipals,
      resume: vi.fn(),
    });
    for (let revision = 8; revision <= 57; revision++) {
      updates.event(cursor(revision), "runtimeAgent");
    }
    await vi.advanceTimersByTimeAsync(1_000);
    expect(refreshPrincipals).toHaveBeenCalledTimes(1);
    updates.event(cursor(58), "runtimeAgent");
    await vi.advanceTimersByTimeAsync(1_000);
    expect(refreshPrincipals).toHaveBeenCalledTimes(1);
    first.resolve();
    await vi.advanceTimersByTimeAsync(999);
    expect(refreshPrincipals).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(refreshPrincipals).toHaveBeenCalledTimes(2);
    updates.dispose();
  });

  it("backs off repeated generation resyncs and resumes after each baseline", async () => {
    vi.useFakeTimers();
    const refreshSnapshot = vi.fn(async () => cursor(7));
    const resume = vi.fn();
    const updates = new OperationsLiveRefresh({
      initial: cursor(7),
      refreshSnapshot,
      refreshPrincipals: vi.fn(async () => undefined),
      resume,
      random: () => 0,
    });
    for (const [attempt, delay] of [250, 500, 1_000].entries()) {
      updates.resync("generation_changed");
      await vi.advanceTimersByTimeAsync(delay - 1);
      expect(refreshSnapshot).toHaveBeenCalledTimes(attempt);
      await vi.advanceTimersByTimeAsync(1);
      expect(refreshSnapshot).toHaveBeenCalledTimes(attempt + 1);
      expect(resume).toHaveBeenCalledTimes(attempt + 1);
    }
    updates.dispose();
  });

  it("keeps a resync pending across failed baseline reads and backs off", async () => {
    vi.useFakeTimers();
    const refreshSnapshot = vi
      .fn<() => Promise<OperationsSnapshotCursor>>()
      .mockRejectedValueOnce(new Error("snapshot unavailable"))
      .mockRejectedValueOnce(new Error("snapshot unavailable"))
      .mockResolvedValueOnce(cursor(12));
    const resume = vi.fn();
    const updates = new OperationsLiveRefresh({
      initial: cursor(7),
      refreshSnapshot,
      refreshPrincipals: vi.fn(async () => undefined),
      resume,
      random: () => 0,
    });
    updates.resync("cursor_unavailable");
    await vi.advanceTimersByTimeAsync(0);
    expect(refreshSnapshot).toHaveBeenCalledTimes(1);
    for (const [attempt, delay] of [500, 1_000].entries()) {
      await vi.advanceTimersByTimeAsync(delay - 1);
      expect(refreshSnapshot).toHaveBeenCalledTimes(attempt + 1);
      expect(resume).not.toHaveBeenCalled();
      await vi.advanceTimersByTimeAsync(1);
      expect(refreshSnapshot).toHaveBeenCalledTimes(attempt + 2);
    }
    expect(resume).toHaveBeenCalledTimes(1);
    expect(resume).toHaveBeenCalledWith(cursor(12));
    await vi.advanceTimersByTimeAsync(60_000);
    expect(refreshSnapshot).toHaveBeenCalledTimes(3);
    updates.dispose();
  });

  it("backs off repeated lost-cursor resyncs until the stream delivers an event", async () => {
    vi.useFakeTimers();
    const refreshSnapshot = vi.fn(async () => cursor(7));
    const resume = vi.fn();
    const updates = new OperationsLiveRefresh({
      initial: cursor(7),
      refreshSnapshot,
      refreshPrincipals: vi.fn(async () => undefined),
      resume,
      random: () => 0,
    });
    updates.resync("sequence_gap");
    await vi.advanceTimersByTimeAsync(0);
    expect(refreshSnapshot).toHaveBeenCalledTimes(1);
    updates.resync("cursor_unavailable");
    await vi.advanceTimersByTimeAsync(499);
    expect(refreshSnapshot).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(refreshSnapshot).toHaveBeenCalledTimes(2);
    // A delivered event shows that the resumed stream is healthy again.
    updates.event(cursor(8), "allocation");
    updates.resync("sequence_gap");
    await vi.advanceTimersByTimeAsync(0);
    expect(refreshSnapshot).toHaveBeenCalledTimes(3);
    expect(resume).toHaveBeenCalledTimes(3);
    updates.dispose();
  });
});
