import { renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { useRecordingCallback } from "./recording";

describe("useRecordingCallback", () => {
  it("reports each change of recording, and stopping on unmount", () => {
    const first = vi.fn();
    const { rerender, unmount } = renderHook(
      ({ recording, onRecording }) =>
        useRecordingCallback(recording, onRecording),
      { initialProps: { recording: false, onRecording: first } },
    );
    expect(first).not.toHaveBeenCalled();
    rerender({ recording: true, onRecording: first });
    rerender({ recording: true, onRecording: first });
    expect(first.mock.calls).toEqual([[true]]);
    rerender({ recording: false, onRecording: first });
    expect(first.mock.calls).toEqual([[true], [false]]);
    // The latest callback is called; unmounting mid-way stops recording.
    const second = vi.fn();
    rerender({ recording: true, onRecording: second });
    unmount();
    expect(first.mock.calls).toEqual([[true], [false]]);
    expect(second.mock.calls).toEqual([[true], [false]]);
  });

  it("says nothing on unmount when it was not recording", () => {
    const onRecording = vi.fn();
    const { unmount } = renderHook(() =>
      useRecordingCallback(false, onRecording),
    );
    unmount();
    expect(onRecording).not.toHaveBeenCalled();
  });
});
