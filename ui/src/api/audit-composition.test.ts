import { describe, expect, it } from "vitest";
import { audit } from "../test/builders";
import { requireAuditComposition } from "./audit-composition";

describe("common Audit counters", () => {
  it("accepts a started Audit before its first task or Run", () => {
    const value = audit("project_1", "audit_1", "active");
    expect(requireAuditComposition(value, 200)).toEqual(value);
  });
  it.each([
    { taskCount: 9 },
    { submittedRunCount: 2, reservedRunCount: 1 },
    { outstandingRunCount: 2, reservedRunCount: 1 },
  ])("rejects inconsistent counters %j", (overrides) => {
    expect(() =>
      requireAuditComposition(
        audit("project_1", "audit_1", "active", overrides),
        200,
      ),
    ).toThrow();
  });
});
