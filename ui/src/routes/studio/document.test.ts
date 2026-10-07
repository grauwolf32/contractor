import { readFileSync, readdirSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  addBlock,
  at,
  patchDraft,
  readDraft,
  removeBlock,
  renameBlock,
  sourceDiff,
  starter,
} from "./document";
import { buildGraph } from "./graph";
import { validateDraft } from "./validation";

const configRoot = resolve(process.cwd(), "../configs");
describe("authored YAML studio", () => {
  it("preserves comments, multiline instructions and opaque fields through graph edits", () => {
    const source = `# Bundle note\napiVersion: contractor/v1alpha1\nkind: Workflow\nmetadata:\n  name: sample # retain me\n  version: "1"\nspec:\n  x-future: {opaque: [one, two]}\n  entryStage: start\n  stages:\n    start:\n      objective: |\n        Keep this multiline objective.\n        Keep the second line.\n      planner: passthrough@1 # planner note\n      agents: {worker: {template: worker@1}}\n      on:\n        succeeded: {succeed: {}}\n        failed: {retry: {maxAttempts: 2, then: {fail: {}}}}\n        interrupted: {fail: {}}\n`;
    let draft = readDraft(source);
    draft = patchDraft(
      draft,
      ["spec", "stages", "start", "planner"],
      "router@1",
    );
    const added = addBlock(draft, "stage");
    draft = patchDraft(
      added.draft,
      ["spec", "stages", "start", "on", "succeeded"],
      { next: added.path[2] },
    );
    draft = removeBlock(draft, added.path);
    expect(draft.source).toContain("# Bundle note");
    expect(draft.source).toContain("# retain me");
    expect(draft.source).toContain("# planner note");
    expect(draft.source).toContain("objective: |");
    expect(at(draft.value, ["spec", "x-future"])).toEqual({
      opaque: ["one", "two"],
    });
    expect(
      at(draft.value, ["spec", "stages", "start", "on", "failed"]),
    ).toEqual({ retry: { maxAttempts: 2, then: { fail: {} } } });
    expect(
      validateDraft(draft).some((problem) =>
        problem.message.includes("does not exist"),
      ),
    ).toBe(true);
  });
  it("rejects malformed, duplicate, multiple, aliased and oversized documents", () => {
    for (const source of [
      "[",
      "kind: Workflow\nkind: AgentTemplate",
      "kind: Workflow\n---\nkind: Workflow",
      "kind: Workflow\nx: &shared {a: 1}\ny: *shared",
      `kind: Workflow\nx: ${"a".repeat(1024 * 1024)}`,
    ])
      expect(() => readDraft(source)).toThrow();
  });
  it("keeps layout outside authored definitions and shows a bounded diff", () => {
    const draft = starter("Workflow"),
      graph = buildGraph(draft);
    expect(graph.nodes.some((node) => node.type === "stage")).toBe(true);
    expect(draft.source).not.toContain("position");
    expect(sourceDiff("one\nsame\nlast", "one\nchanged\nlast")).toEqual([
      { kind: "same", line: "one" },
      { kind: "remove", line: "same" },
      { kind: "add", line: "changed" },
      { kind: "same", line: "last" },
    ]);
  });
  for (const [directory, kind] of [
    ["workflows", "Workflow"],
    ["audit-profiles", "AuditProfile"],
    ["agent-templates", "AgentTemplate"],
  ] as const) {
    for (const file of readdirSync(resolve(configRoot, directory)).filter(
      (file) => file.endsWith(".yaml"),
    )) {
      it(`loads real ${directory}/${file} without inventing local errors`, () => {
        const source = readFileSync(
          resolve(configRoot, directory, file),
          "utf8",
        );
        const draft = readDraft(source);
        expect(draft.kind).toBe(kind);
        expect(draft.source).toBe(source);
        expect(
          validateDraft(draft).filter(
            (problem) => problem.severity === "error",
          ),
        ).toEqual([]);
        expect(buildGraph(draft).nodes.length).toBeGreaterThan(0);
      });
    }
  }
});

describe("workflow graph rules", () => {
  it("renames stage keys and explicit continuation references without dropping node comments", () => {
    let draft = starter("Workflow");
    const added = addBlock(draft, "stage");
    draft = patchDraft(
      added.draft,
      ["spec", "stages", "start", "on", "failed"],
      { retry: { maxAttempts: 2, then: { next: added.path[2] } } },
    );
    draft = renameBlock(draft, added.path, "finish");
    expect(
      at(draft.value, [
        "spec",
        "stages",
        "start",
        "on",
        "failed",
        "retry",
        "then",
        "next",
      ]),
    ).toBe("finish");
    expect(at(draft.value, added.path)).toBeUndefined();
    expect(validateDraft(draft)).toEqual([]);
    expect(() =>
      renameBlock(draft, ["spec", "stages", "finish"], "start"),
    ).toThrow("already exists");
    draft = renameBlock(draft, ["spec", "stages", "start"], "entry");
    expect(at(draft.value, ["spec", "entryStage"])).toBe("entry");
  });
  it("renames check role and inventory/input references together", () => {
    let draft = starter("AuditProfile");
    draft = renameBlock(draft, ["spec", "workflows", "check"], "trace");
    expect(at(draft.value, ["spec", "inventory", "itemWorkflowRole"])).toBe(
      "trace",
    );
    draft = renameBlock(draft, ["spec", "inputs", "openapi"], "definition");
    expect(at(draft.value, ["spec", "inventory", "source", "name"])).toBe(
      "definition",
    );
    expect(validateDraft(draft)).toEqual([]);
  });
  it("distinguishes attempt retry from next cycles and unreachable drafts", () => {
    let draft = starter("Workflow");
    draft = patchDraft(draft, ["spec", "stages", "start", "on", "failed"], {
      retry: { maxAttempts: 2, then: { fail: {} } },
    });
    expect(validateDraft(draft)).toEqual([]);
    const added = addBlock(draft, "stage");
    expect(
      validateDraft(added.draft).some((problem) =>
        problem.message.includes("unreachable"),
      ),
    ).toBe(true);
    draft = patchDraft(
      added.draft,
      ["spec", "stages", "start", "on", "failed", "retry", "then"],
      { next: added.path[2] },
    );
    expect(validateDraft(draft)).toEqual([]);
    draft = patchDraft(draft, [...added.path, "on", "succeeded"], {
      next: "start",
    });
    expect(
      validateDraft(draft).some((problem) => problem.message.includes("cycle")),
    ).toBe(true);
  });
  it("requires result files on every success path, including failed branches at joins", () => {
    let draft = starter("Workflow");
    draft = patchDraft(draft, ["spec", "outputs", "report"], {
      required: true,
      mediaTypes: ["text/markdown"],
    });
    expect(
      validateDraft(draft).some((problem) =>
        problem.message.includes("without required output report"),
      ),
    ).toBe(true);
    draft = patchDraft(
      draft,
      ["spec", "stages", "start", "result", "artifacts", "report"],
      {
        required: true,
        mediaTypes: ["text/markdown"],
        from: { namespace: "work", name: "report" },
      },
    );
    draft = patchDraft(draft, ["spec", "stages", "start", "workflowOutputs"], {
      report: "report",
    });
    expect(validateDraft(draft)).toEqual([]);
    const added = addBlock(draft, "stage");
    draft = patchDraft(
      added.draft,
      ["spec", "stages", "start", "on", "succeeded"],
      { next: added.path[2] },
    );
    expect(validateDraft(draft)).toEqual([]);
    draft = patchDraft(draft, ["spec", "stages", "start", "on", "failed"], {
      next: added.path[2],
    });
    expect(
      validateDraft(draft).some((problem) =>
        problem.message.includes("without required output report"),
      ),
    ).toBe(true);
  });
  it("warns on a declared artifact read before its producer while allowing external project context", () => {
    let draft = starter("Workflow");
    const added = addBlock(draft, "stage");
    draft = patchDraft(
      added.draft,
      ["spec", "stages", "start", "on", "succeeded"],
      { next: added.path[2] },
    );
    draft = patchDraft(
      draft,
      [...added.path, "result", "artifacts", "report"],
      {
        required: true,
        mediaTypes: ["text/markdown"],
        from: { namespace: "work", name: "report" },
      },
    );
    draft = patchDraft(
      draft,
      ["spec", "stages", "start", "context", "artifacts"],
      {
        future: { namespace: "work", name: "report", required: true },
        external: { namespace: "project", name: "external", required: true },
      },
    );
    expect(
      validateDraft(draft).filter((problem) => problem.severity === "warning"),
    ).toHaveLength(1);
  });
  it("reports malformed media types instead of crashing validation", () => {
    let draft = starter("Workflow");
    draft = patchDraft(draft, ["spec", "outputs"], {
      report: { required: true, mediaTypes: {} },
    });
    draft = patchDraft(
      draft,
      ["spec", "stages", "start", "result", "artifacts"],
      {
        report: {
          required: true,
          mediaTypes: ["text/markdown"],
          from: { namespace: "work", name: "report" },
        },
      },
    );
    draft = patchDraft(draft, ["spec", "stages", "start", "workflowOutputs"], {
      report: "report",
    });
    expect(() => validateDraft(draft)).not.toThrow();
    expect(validateDraft(draft).length).toBeGreaterThan(0);
  });
});
