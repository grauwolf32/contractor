import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import cases from "../../../../api/testdata/workflow-context-management-cases.json";
import { patchDraft, readDraft } from "./document";
import { validateDraft } from "./validation";

const source = readFileSync(
  "../internal/config/testdata/context-compaction/workflows/compact_copy.yaml",
  "utf8",
);

describe("shared Workflow contextManagement contract", () => {
  const editorCases = cases.cases.filter(
    (entry) => !("catalogOnly" in entry && entry.catalogOnly),
  );

  it("covers editor cases", () => {
    expect(editorCases.length).toBeGreaterThan(10);
  });

  it.each(editorCases)("validates $name", ({ path, value, valid }) => {
    const draft = patchDraft(readDraft(source), ["spec", ...path], value);
    expect(
      validateDraft(draft).some((problem) => problem.severity === "error"),
    ).toBe(!valid);
  });
});
