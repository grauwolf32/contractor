import workspacePathCases from "../../../../api/testdata/v1alpha1/workspace-path-cases.json";
import { describe, expect, it } from "vitest";

import { isWorkspaceTarget } from "./advanced-validation";

// Studio refuses exactly the workspace targets the Server and the Runtime
// refuse: the shared cases hold all three to one rule (WSP-R13).
type PathCase = (typeof workspacePathCases.cases)[number];

function caseValue(item: PathCase): string {
  if (typeof item.value === "string") return item.value;
  return item.repeat.unit.repeat(item.repeat.count) + item.repeat.suffix;
}

const cases = workspacePathCases.cases.map(
  (item) => [item.name, caseValue(item), item.workspaceTarget] as const,
);

describe("Studio workspace targets", () => {
  it("reads the shared cases", () => {
    expect(cases.length).toBeGreaterThan(0);
    expect(workspacePathCases.rules.workspaceTarget).toEqual({
      emptyAllowed: true,
      maxBytes: 1024,
      maxComponents: 32,
    });
  });

  it.each(cases)("%s", (_name, value, accepted) => {
    expect(isWorkspaceTarget(value)).toBe(accepted);
  });
});
