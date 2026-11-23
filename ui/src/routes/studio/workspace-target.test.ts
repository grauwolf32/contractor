import workspacePathCases from "../../../../api/testdata/v1alpha1/workspace-path-cases.json";
import { describe, expect, it } from "vitest";

import { patchDraft, starter, type Path } from "./document";
import { validateDraft } from "./validation";

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

const stage: Path = ["spec", "stages", "start"];
const target: Path = [...stage, "context", "workspace", "sources", 0, "target"];

/** Local check errors at a workspace whose only source mounts at `value`. */
function targetErrors(value: string): string[] {
  const draft = patchDraft(starter("Workflow"), [...stage, "context"], {
    artifacts: {
      source: { namespace: "inputs", name: "source", required: true },
    },
    workspace: {
      mode: "direct",
      sources: [{ artifact: "source", target: value }],
    },
  });
  return validateDraft(draft)
    .filter(
      (problem) =>
        problem.severity === "error" &&
        JSON.stringify(problem.path) === JSON.stringify(target),
    )
    .map((problem) => problem.message);
}

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
    expect(targetErrors(value)).toEqual(
      accepted
        ? []
        : [expect.stringMatching(/^Use a normalized relative POSIX directory/)],
    );
  });
});
