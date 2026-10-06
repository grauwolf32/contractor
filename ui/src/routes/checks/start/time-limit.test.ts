import { describe, expect, it } from "vitest";

import {
  DEFAULT_TIME_LIMIT,
  describeTimeLimit,
  TIME_LIMIT_OPTIONS,
  timeLimitSeconds,
} from "./time-limit";

describe("time limit", () => {
  it("offers 24 hours (default), 7 days, no time limit and custom", () => {
    expect(DEFAULT_TIME_LIMIT).toBe("86400");
    expect(TIME_LIMIT_OPTIONS.map((option) => option.label)).toEqual([
      "24 hours",
      "7 days",
      "No time limit",
      "Custom",
    ]);
    expect(timeLimitSeconds({ choice: "86400", hours: "" })).toBe(86400);
    expect(timeLimitSeconds({ choice: "604800", hours: "" })).toBe(604800);
    expect(timeLimitSeconds({ choice: "0", hours: "" })).toBe(0);
  });

  it("accepts custom hours from 0.01 to 8760", () => {
    expect(timeLimitSeconds({ choice: "custom", hours: "0.01" })).toBe(36);
    expect(timeLimitSeconds({ choice: "custom", hours: "1.5" })).toBe(5400);
    expect(timeLimitSeconds({ choice: "custom", hours: "8760" })).toBe(
      31_536_000,
    );
    for (const hours of ["", "0", "0.001", "8760.01", "-1", "abc", "Infinity"])
      expect(timeLimitSeconds({ choice: "custom", hours })).toBeUndefined();
  });

  it("describes a limit in words", () => {
    expect(describeTimeLimit(86400)).toBe("24 hours");
    expect(describeTimeLimit(604800)).toBe("7 days");
    expect(describeTimeLimit(5400)).toBe("1.5 hours");
    expect(describeTimeLimit(36)).toBe("36 seconds");
    expect(describeTimeLimit(0)).toBe("no time limit");
  });
});
