import { describe, expect, it } from "vitest";

import buckets from "../../../../../api/testdata/public/audit-coverage-buckets.json";

import {
  isConcludedStatus,
  STATUS_GROUPS,
  statusGroup,
  type Status,
} from "./assessments";

describe("coverage result groups", () => {
  const all = [
    ...buckets.completed,
    ...buckets.gap,
    ...buckets.unchecked,
  ] as Status[];

  it("classify every coverage status of the shared bucket cases", () => {
    expect([...all].sort()).toEqual(Object.keys(STATUS_GROUPS).sort());
    expect(new Set(all).size).toBe(all.length);
  });

  it("nest in the Server's completed, gap and unchecked buckets", () => {
    for (const status of all) {
      const completed = buckets.completed.includes(status);
      expect(isConcludedStatus(status), status).toBe(completed);
      expect(statusGroup(status) === "uncertain", status).toBe(
        buckets.gap.includes(status),
      );
      expect(statusGroup(status) === "not-tested", status).toBe(
        buckets.unchecked.includes(status),
      );
      expect(statusGroup(status) === "issues", status).toBe(
        buckets.issue.includes(status),
      );
    }
  });

  it("count an item the owner rejected as follow-up, not as done", () => {
    expect(statusGroup("excluded")).toBe("uncertain");
    expect(isConcludedStatus("excluded")).toBe(false);
  });
});
