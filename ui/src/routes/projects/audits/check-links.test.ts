import { expect, it } from "vitest";
import { checkPath, sectionPath } from "./check-links";
it("builds project-owned check links", () => {
  expect(checkPath("project_1", "audit_1")).toBe(
    "/projects/project_1/audits/audit_1",
  );
  expect(sectionPath("project_1", "audit_1", "findings")).toContain("audit_1");
});
