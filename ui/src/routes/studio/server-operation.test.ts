import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import cases from "../../../../api/testdata/server-operation-stage-cases.json";
import snapshot from "../../../../api/testdata/server-operation-memory-snapshot.json";
import prepared from "../../../../api/testdata/server-operation-openapi-work.json";
import { patchDraft, readDraft } from "./document";
import { buildGraph } from "./graph";
import { validateDraft } from "./validation";

const source = readFileSync(
  "../internal/config/testdata/server-operations/workflows/join.yaml",
  "utf8",
);

describe("shared server operation stage contract", () => {
  it("accepts and displays passive OpenAPI preparation without Workers", () => {
    const draft = readDraft(prepared.workflow);
    expect(
      validateDraft(draft).filter((problem) => problem.severity === "error"),
    ).toEqual([]);
    expect(
      buildGraph(draft).nodes.find((node) => node.type === "stage")?.subtitle,
    ).toContain("openapi-prepare-work@1");
  });
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

describe("shared explicit Memory snapshot contract", () => {
  it.each(snapshot.cases)("validates $name", ({ path, value, valid }) => {
    const draft = patchDraft(
      readDraft(snapshot.workflow),
      ["spec", "stages", "snapshot", ...path],
      value,
    );
    expect(
      validateDraft(draft).some((problem) => problem.severity === "error"),
    ).toBe(!valid);
  });
  it("displays the model-free snapshot Stage", () => {
    expect(
      buildGraph(readDraft(snapshot.workflow)).nodes.find(
        (node) => node.type === "stage",
      )?.subtitle,
    ).toContain("memory-snapshot@1");
  });
});
