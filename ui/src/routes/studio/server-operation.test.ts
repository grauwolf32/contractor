import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import cases from "../../../../api/testdata/server-operation-stage-cases.json";
import { patchDraft, readDraft } from "./document";
import { buildGraph } from "./graph";
import { validateDraft } from "./validation";

const source = readFileSync(
  "../internal/config/testdata/server-operations/workflows/join.yaml",
  "utf8",
);

describe("shared server operation stage contract", () => {
  it.each(cases.cases)("validates $name", ({ path, value, valid }) => {
    const draft = patchDraft(
      readDraft(source),
      ["spec", "stages", "join-api", ...path],
      value,
    );
    expect(
      validateDraft(draft).some((problem) => problem.severity === "error"),
    ).toBe(!valid);
  });

  it("labels the operation in the graph", () => {
    const stage = buildGraph(readDraft(source)).nodes.find(
      (node) => node.type === "stage",
    );
    expect(stage?.subtitle).toContain("sitemap-openapi-join@1");
  });
});
