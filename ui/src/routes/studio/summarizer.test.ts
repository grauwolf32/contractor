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
      modelPolicy: "terminal_summarizer@1",
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
