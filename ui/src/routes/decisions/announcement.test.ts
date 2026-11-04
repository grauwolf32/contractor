import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useAnnouncement } from "./announcement";

/** How long useAnnouncement keeps a message (eight seconds). */
const ANNOUNCEMENT_MS = 8_000;

describe("useAnnouncement", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("keeps a message for a while and drops it when cleared", () => {
    const { result } = renderHook(() => useAnnouncement());
    expect(result.current.text).toBe("");
    act(() => result.current.announce("Assessment saved."));
    expect(result.current.text).toBe("Assessment saved.");
    act(() => vi.advanceTimersByTime(ANNOUNCEMENT_MS - 1));
    expect(result.current.text).toBe("Assessment saved.");
    act(() => vi.advanceTimersByTime(1));
    expect(result.current.text).toBe("");

    act(() => result.current.announce("Decision recorded: Rejected."));
    act(() => result.current.clear());
    expect(result.current.text).toBe("");
  });

  it("restarts the time when the same message is announced again", () => {
    const { result } = renderHook(() => useAnnouncement());
    act(() => result.current.announce("Assessment saved."));
    act(() => vi.advanceTimersByTime(ANNOUNCEMENT_MS - 1_000));
    act(() => result.current.announce("Assessment saved."));
    act(() => vi.advanceTimersByTime(ANNOUNCEMENT_MS - 1_000));
    expect(result.current.text).toBe("Assessment saved.");
    act(() => vi.advanceTimersByTime(1_000));
    expect(result.current.text).toBe("");
  });
});
