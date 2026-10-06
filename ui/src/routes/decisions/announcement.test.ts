import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ANNOUNCEMENT_MS, focusIfLost, useAnnouncement } from "./announcement";

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
    act(() => result.current.announce("Decision recorded: Approved."));
    expect(result.current.text).toBe("Decision recorded: Approved.");
    act(() => vi.advanceTimersByTime(ANNOUNCEMENT_MS - 1));
    expect(result.current.text).toBe("Decision recorded: Approved.");
    act(() => vi.advanceTimersByTime(1));
    expect(result.current.text).toBe("");

    act(() => result.current.announce("Decision recorded: Rejected."));
    act(() => result.current.clear());
    expect(result.current.text).toBe("");
  });

  it("restarts the time when the same message is announced again", () => {
    const { result } = renderHook(() => useAnnouncement());
    act(() => result.current.announce("Decision recorded: Approved."));
    act(() => vi.advanceTimersByTime(ANNOUNCEMENT_MS - 1_000));
    act(() => result.current.announce("Decision recorded: Approved."));
    act(() => vi.advanceTimersByTime(ANNOUNCEMENT_MS - 1_000));
    expect(result.current.text).toBe("Decision recorded: Approved.");
    act(() => vi.advanceTimersByTime(1_000));
    expect(result.current.text).toBe("");
  });
});

describe("focusIfLost", () => {
  function decision() {
    const container = document.createElement("div");
    container.tabIndex = -1;
    const inside = document.createElement("button");
    container.append(inside);
    const outside = document.createElement("input");
    document.body.append(container, outside);
    return { container, inside, outside };
  }

  afterEach(() => {
    document.body.replaceChildren();
  });

  it("takes focus from inside the container or from the page", () => {
    const { container, inside } = decision();
    inside.focus();
    focusIfLost(container);
    expect(container).toHaveFocus();

    container.blur();
    expect(document.body).toHaveFocus();
    focusIfLost(container);
    expect(container).toHaveFocus();
  });

  it("leaves focus the user moved elsewhere", () => {
    const { container, outside } = decision();
    outside.focus();
    focusIfLost(container);
    expect(outside).toHaveFocus();
    focusIfLost(null);
    expect(outside).toHaveFocus();
  });
});
