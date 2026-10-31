import { describe, expect, it } from "vitest";
import {
  parseTimelineFilter,
  serializeTimelineFilter,
} from "./timeline-filter";

describe("Timeline checkbox URLs", () => {
  it("retains single, combined, all and no selections", () => {
    expect(parseTimelineFilter("issues")).toEqual(["issues"]);
    expect(parseTimelineFilter("issues,checks")).toEqual(["checks", "issues"]);
    expect(parseTimelineFilter("none")).toEqual([]);
    expect(parseTimelineFilter(null)).toBe("all");
    expect(parseTimelineFilter("checks,issues,runs,materials")).toBe("all");
    expect(serializeTimelineFilter(["runs", "checks"])).toBe("checks,runs");
    expect(serializeTimelineFilter([])).toBe("none");
  });
  it("rejects an unknown or oversized selection as a whole", () => {
    expect(parseTimelineFilter("issues,unknown")).toBe("all");
    expect(parseTimelineFilter("checks,checks,checks,checks,checks")).toBe(
      "all",
    );
  });
});
