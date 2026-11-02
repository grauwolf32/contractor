import { describe, expect, it } from "vitest";
import { patchDraft, starter } from "./document";
import { validateDraft } from "./validation";

describe("terminal summarizer reasoning option", () => {
  it.each([
    { value: true, valid: true },
    { value: false, valid: true },
    { value: null, valid: false },
    { value: "true", valid: false },
    { value: 1, valid: false },
  ])("validates includeReasoning $value", ({ value, valid }) => {
    let draft = starter("AgentTemplate");
    draft = patchDraft(draft, ["spec", "summarizer"], {
      includeReasoning: value,
    });
    const path = ["spec", "summarizer", "includeReasoning"];
    const errors = validateDraft(draft).filter(
      (problem) =>
        problem.severity === "error" &&
        path.every((part, index) => problem.path[index] === part),
    );
    expect(errors.length === 0).toBe(valid);
  });
});

describe("Workflow-owned model selection", () => {
  it.each([null, "worker@1"])(
    "rejects removed template model fields: %s",
    (value) => {
      for (const path of [
        ["spec", "modelPolicy"],
        ["spec", "summarizer", "modelPolicy"],
      ]) {
        const draft = patchDraft(starter("AgentTemplate"), path, value);
        expect(
          validateDraft(draft).some(
            (problem) =>
              problem.severity === "error" &&
              problem.path.join(".") === path.join("."),
          ),
        ).toBe(true);
      }
    },
  );
});
