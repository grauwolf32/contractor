import { describe, expect, it } from "vitest";

import {
  describeStopReason,
  STOP_REASON_CODES,
  stopReasonTone,
} from "./stop-reason";

describe("stop reasons", () => {
  it("explains every code the Server sends in one plain sentence", () => {
    expect(STOP_REASON_CODES).toHaveLength(21);
    for (const code of STOP_REASON_CODES) {
      const stop = describeStopReason({
        state: "completed",
        stopReason: { code, message: `server message for ${code}` },
      });
      expect(stop?.sentence).toMatch(/^[A-Z].*\.$/u);
      expect(stop?.sentence).not.toContain(code);
      expect(stop?.sentence).not.toContain("server message");
      expect(stop?.label).toBeDefined();
      expect(stop?.message).toBe(`server message for ${code}`);
    }
  });

  it("never shows an unknown code, only the Server's message", () => {
    const stop = describeStopReason({
      state: "cancelled",
      stopReason: {
        code: "brand_new_reason",
        message: "Stopped for a new reason.",
      },
    });
    expect(stop).toMatchObject({
      label: undefined,
      sentence: "Stopped for a new reason.",
      tone: "neutral",
      deadline: false,
    });
  });

  it("words the time limit by state and keeps failures as errors", () => {
    const deadline = { code: "deadline_exhausted", message: "deadline" };
    const paused = describeStopReason({
      state: "paused",
      stopReason: deadline,
    });
    expect(paused?.sentence).toMatch(/^The time limit was reached\. Continue/u);
    expect(stopReasonTone(paused!)).toBe("warning");
    const ended = describeStopReason({
      state: "completed",
      stopReason: deadline,
    });
    expect(ended?.sentence).toBe(
      "Stopped by the time limit: no new work was started after it.",
    );
    const failed = describeStopReason({
      state: "failed",
      stopReason: deadline,
    });
    expect(failed?.tone).toBe("error");
    expect(stopReasonTone(failed!)).toBe("blocked");
    const budget = describeStopReason({
      state: "completed",
      stopReason: { code: "round_budget_exhausted", message: "rounds" },
    });
    expect(budget?.tone).toBe("error");
    expect(
      stopReasonTone(
        describeStopReason({
          state: "completed",
          stopReason: { code: "round_complete", message: "done" },
        })!,
      ),
    ).toBe("info");
    expect(describeStopReason({ state: "active" })).toBeNull();
  });
});
