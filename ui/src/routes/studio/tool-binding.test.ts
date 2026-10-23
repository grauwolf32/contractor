import cases from "../../../../api/testdata/tool-parameter-binding-cases.json";
import { describe, expect, it } from "vitest";
import { patchDraft, starter } from "./document";
import { validateDraft } from "./validation";

describe("shared tool parameter binding grammar", () => {
  it.each(cases.bindings)("validates $name", ({ binding, valid }) => {
    let draft = starter("AgentTemplate");
    draft = patchDraft(draft, ["spec", "runtime"], "tool@1");
    draft = patchDraft(draft, ["spec", "sandboxProfile"], "local-workdir@1");
    draft = patchDraft(
      draft,
      ["spec", "toolsets"],
      [{ ref: "scan@1", tools: ["scan_nuclei"] }],
    );
    draft = patchDraft(draft, ["spec", "execution"], {
      tool: "scan_nuclei",
      arguments: { rate_limit: binding },
      resultArtifact: "report",
      timeoutSeconds: 60,
    });
    const path = ["spec", "execution", "arguments", "rate_limit"];
    const errors = validateDraft(draft).filter(
      (problem) =>
        problem.severity === "error" &&
        path.every((part, index) => problem.path[index] === part),
    );
    expect(errors.length === 0).toBe(valid);
  });
});
