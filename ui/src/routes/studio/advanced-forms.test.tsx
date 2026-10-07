import { readFileSync } from "node:fs";
import { useState } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { parse } from "yaml";
import { describe, expect, it } from "vitest";
import {
  at,
  appendDraft,
  editDraft,
  patchDraft,
  readDraft,
  removeBlock,
  renameBlock,
  starter,
  type Draft,
  type Path,
} from "./document";
import { Inspector } from "./inspector";
import { buildGraph } from "./graph";
import { validateDraft } from "./validation";
import { setEscalationMode, setSelectionField } from "./advanced-edits";
import { renameArgument, setArgumentSource } from "./advanced-edits";
import { OVERLAY_MEDIA_TYPE } from "./workspace-form";

function workflow() {
  let draft = starter("Workflow");
  draft = patchDraft(draft, ["spec", "stages", "start", "planner"], "router@1");
  draft = patchDraft(draft, ["spec", "inputs"], {
    source: { required: true, mediaTypes: ["application/zip"] },
    state: { required: false, mediaTypes: [OVERLAY_MEDIA_TYPE] },
  });
  draft = patchDraft(
    draft,
    ["spec", "stages", "start", "context", "artifacts"],
    {
      source: { namespace: "inputs", name: "source", required: true },
      state: { namespace: "inputs", name: "state", required: false },
    },
  );
  return readDraft(`# Preserve bundle note\n${draft.source}`);
}
const workspacePath: Path = ["spec", "stages", "start", "context", "workspace"];
const tool = () =>
  readDraft(
    readFileSync("../configs/agent-templates/audit_sqlmap_scan.yaml", "utf8"),
  );
function mount(initial: Draft, type: string) {
  function Harness() {
    const [draft, setDraft] = useState(initial);
    const node = buildGraph(draft).nodes.find((node) => node.type === type)!;
    return (
      <>
        <Inspector
          draft={draft}
          node={node}
          onPatch={(path, value) => setDraft(patchDraft(draft, path, value))}
          onReplace={setDraft}
          onRename={() => {}}
          onConnect={() => {}}
          onRemove={(node) => setDraft(removeBlock(draft, node.path))}
        />
        <output data-testid="yaml">{draft.source}</output>
      </>
    );
  }
  render(<Harness />);
  return () => readDraft(screen.getByTestId("yaml").textContent!);
}
function change(label: string, value: string) {
  const field = screen.getByLabelText(label);
  fireEvent.change(field, { target: { value } });
  fireEvent.blur(field);
}

describe("Advanced Studio forms", () => {
  it("creates overlay exports atomically, preserves input fields and restores an optional state alias", async () => {
    const user = userEvent.setup(),
      initial = workflow(),
      current = mount(initial, "stage");
    await user.click(
      screen.getByRole("button", { name: "Configure workspace" }),
    );
    await user.selectOptions(
      screen.getByLabelText("Workspace mode"),
      "overlay",
    );
    await user.click(screen.getByRole("button", { name: "Add state input" }));
    await user.click(
      screen.getByRole("button", { name: "Add workspace export" }),
    );
    const draft = current(),
      exports = at(draft.value, [...workspacePath, "export"]) as Record<
        string,
        string
      >;
    expect(at(draft.value, [...workspacePath, "state"])).toEqual({
      artifact: "state",
    });
    expect(
      at(draft.value, [
        "spec",
        "stages",
        "start",
        "result",
        "artifacts",
        exports.state!,
      ]),
    ).toEqual({ required: true, mediaTypes: [OVERLAY_MEDIA_TYPE] });
    expect(
      at(draft.value, [
        "spec",
        "stages",
        "start",
        "result",
        "artifacts",
        exports.diff!,
      ]),
    ).toEqual({ required: true, mediaTypes: ["text/x-diff"] });
    expect(at(draft.value, ["spec", "inputs"])).toEqual(
      at(initial.value, ["spec", "inputs"]),
    );
    expect(draft.source).toContain("# Preserve bundle note");
    expect(validateDraft(draft).filter((p) => p.severity === "error")).toEqual(
      [],
    );
    await user.click(
      screen.getByRole("button", { name: "Add workspace source" }),
    );
    expect(
      at(current().value, [...workspacePath, "sources"]) as unknown[],
    ).toHaveLength(2);
    expect(
      validateDraft(current()).some((p) =>
        p.message.includes("non-overlapping"),
      ),
    ).toBe(true);
    await user.click(screen.getByRole("button", { name: "Remove source 2" }));
    expect(
      validateDraft(current()).filter((p) => p.severity === "error"),
    ).toEqual([]);
  });
  it("edits defaults and stage selections at their native paths and restores inheritance on clearing", () => {
    const initial = workflow(),
      current = mount(initial, "stage");
    fireEvent.click(
      screen.getByText("Stage execution overrides", { selector: "summary" }),
    );
    change("Stage agent worker model policy", "worker@2");
    change("Stage planner gateway", "gateway@2");
    expect(
      at(current().value, ["spec", "executionConfig", "stages", "start"]),
    ).toEqual({
      agents: { worker: { modelPolicy: "worker@2" } },
      planner: { llmGateway: "gateway@2" },
    });
    expect(
      at(current().value, ["spec", "stages", "start", "executionConfig"]),
    ).toBeUndefined();
    change("Stage agent worker model policy", "");
    expect(
      at(current().value, [
        "spec",
        "executionConfig",
        "stages",
        "start",
        "agents",
      ]),
    ).toBeUndefined();
    expect(
      at(current().value, [
        "spec",
        "executionConfig",
        "stages",
        "start",
        "planner",
      ]),
    ).toEqual({ llmGateway: "gateway@2" });
    expect(
      validateDraft(current()).filter((p) => p.severity === "error"),
    ).toEqual([]);
  });
  it("keeps false and numeric literals typed and switches an argument to an artifact binding", async () => {
    const user = userEvent.setup(),
      current = mount(tool(), "toolExecution");
    await user.selectOptions(
      screen.getByLabelText("level literal type"),
      "boolean",
    );
    expect(
      at(current().value, ["spec", "execution", "arguments", "level", "value"]),
    ).toBe(false);
    await user.click(screen.getByLabelText("level value"));
    expect(
      at(current().value, ["spec", "execution", "arguments", "level", "value"]),
    ).toBe(true);
    change("risk value", "2.5");
    expect(
      at(current().value, ["spec", "execution", "arguments", "risk", "value"]),
    ).toBe(2.5);
    await user.selectOptions(screen.getByLabelText("level source"), "artifact");
    change("level binding name", "request");
    expect(
      at(current().value, ["spec", "execution", "arguments", "level"]),
    ).toEqual({ source: "artifact", name: "request" });
    expect(
      validateDraft(current()).filter((p) => p.severity === "error"),
    ).toEqual([]);
    expect(current().source).toContain(
      "# SQLMap's lowest supported test level and risk",
    );
  });
  it("renames a stage with its routing overrides and preserves scalar comments", () => {
    let draft = patchDraft(workflow(), ["spec", "executionConfig"], {
      stages: { start: { planner: { modelPolicy: "planner@1" } } },
    });
    draft = readDraft(
      draft.source.replace(
        "modelPolicy: planner@1",
        "modelPolicy: planner@1 # Keep routing note",
      ),
    );
    const renamed = renameBlock(draft, ["spec", "stages", "start"], "prepare");
    expect(
      at(renamed.value, ["spec", "executionConfig", "stages", "start"]),
    ).toBeUndefined();
    expect(
      at(renamed.value, [
        "spec",
        "executionConfig",
        "stages",
        "prepare",
        "planner",
        "modelPolicy",
      ]),
    ).toBe("planner@1");
    expect(renamed.source).toContain("# Keep routing note");
    expect(
      validateDraft(renamed).filter((p) => p.severity === "error"),
    ).toEqual([]);
  });
  it("switches escalation forms exclusively, retains opaque siblings and permits explicit credential clearing only inline", () => {
    const path: Path = [
      "spec",
      "stages",
      "start",
      "on",
      "failed",
      "escalate",
      "executionConfig",
    ];
    let draft = patchDraft(workflow(), path.slice(0, -2), {
      escalate: {
        maxAttempts: 1,
        then: { fail: {} },
        executionConfig: { ref: "stronger@2", "x-future": { note: "keep" } },
      },
    });
    draft = editDraft(draft, (doc) => {
      setEscalationMode(doc, path, "inline");
      setSelectionField(doc, [...path, "planner"], "modelPolicy", "planner@2");
      setSelectionField(doc, [...path, "planner"], "credential", null);
    });
    expect(at(draft.value, path)).toEqual({
      planner: { modelPolicy: "planner@2", credential: null },
      "x-future": { note: "keep" },
    });
    expect(validateDraft(draft).filter((p) => p.severity === "error")).toEqual(
      [],
    );
    draft = editDraft(draft, (doc) => {
      setEscalationMode(doc, path, "reference");
      doc.setIn([...path, "ref"], "stronger@3");
    });
    expect(at(draft.value, path)).toEqual({
      ref: "stronger@3",
      "x-future": { note: "keep" },
    });
  });
  it("renames argument syntax keys, rejects duplicates and retains untouched binding comments", () => {
    const initial = tool(),
      path = ["spec", "execution", "arguments", "level"];
    const draft = editDraft(initial, (doc) =>
      renameArgument(doc, path, "test_level"),
    );
    expect(at(draft.value, [...path.slice(0, -1), "test_level"])).toEqual({
      source: "literal",
      value: 1,
    });
    expect(draft.source).toContain(
      "# SQLMap's lowest supported test level and risk",
    );
    expect(() =>
      editDraft(initial, (doc) => renameArgument(doc, path, "risk")),
    ).toThrow("already exists");
    expect(() =>
      editDraft(initial, (doc) => renameArgument(doc, path, "bad-name")),
    ).toThrow("identifier");
    expect(initial.source).toBe(tool().source);
    const switched = editDraft(draft, (doc) =>
      setArgumentSource(doc, [...path.slice(0, -1), "test_level"], "parameter"),
    );
    expect(at(switched.value, [...path.slice(0, -1), "test_level"])).toEqual({
      source: "parameter",
      name: "",
    });
  });
  it("appends a source without replacing comments or unknown fields on previous sources", () => {
    const initial = readDraft(
      patchDraft(workflow(), workspacePath, {
        mode: "direct",
        sources: [{ artifact: "source", target: "src", "x-note": 1 }],
      }).source.replace("target: src", "target: src # Keep target note"),
    );
    const draft = appendDraft(initial, [...workspacePath, "sources"], {
      artifact: "source",
      target: "lib",
    });
    expect(draft.source).toContain("# Keep target note");
    expect(at(draft.value, [...workspacePath, "sources", 0, "x-note"])).toBe(1);
    expect(parse(draft.source)).toEqual(draft.value);
  });
});

describe("Authored advanced contracts", () => {
  it.each([
    [
      "absolute target",
      { mode: "direct", sources: [{ artifact: "source", target: "/root" }] },
      "POSIX",
    ],
    [
      "parent target",
      { mode: "direct", sources: [{ artifact: "source", target: "../src" }] },
      "POSIX",
    ],
    [
      "overlap",
      {
        mode: "direct",
        sources: [
          { artifact: "source", target: "src" },
          { artifact: "source", target: "src/sub" },
        ],
      },
      "non-overlapping",
    ],
    [
      "optional source",
      { mode: "direct", sources: [{ artifact: "state", target: "" }] },
      "required stage input",
    ],
    ["no sources", { mode: "overlay", sources: [] }, "1 and 32"],
    [
      "unknown state",
      {
        mode: "direct",
        sources: [{ artifact: "source", target: "" }],
        state: { artifact: "missing" },
      },
      "declared stage input",
    ],
    [
      "direct export",
      {
        mode: "direct",
        sources: [{ artifact: "source", target: "" }],
        export: { state: "state", diff: "diff" },
      },
      "overlay mode",
    ],
    [
      "missing slots",
      {
        mode: "overlay",
        sources: [{ artifact: "source", target: "" }],
        export: { state: "state", diff: "diff" },
      },
      "runtime-owned",
    ],
  ])("rejects workspace %s", (_name, value, message) => {
    expect(
      validateDraft(patchDraft(workflow(), workspacePath, value)).some((p) =>
        p.message.includes(message as string),
      ),
    ).toBe(true);
  });
  it("permits restoring state in direct mode", () => {
    const draft = patchDraft(workflow(), workspacePath, {
      mode: "direct",
      sources: [{ artifact: "source", target: "" }],
      state: { artifact: "state" },
    });
    expect(validateDraft(draft).filter((p) => p.severity === "error")).toEqual(
      [],
    );
  });
  it.each([
    [{ workers: { credential: null } }, "null is escalation-only"],
    [{ planner: {} }, "at least one execution"],
    [{ workers: { modelPolicy: null } }, "non-null versioned"],
    [
      { stages: { missing: { planner: { modelPolicy: "model@1" } } } },
      "stage does not exist",
    ],
    [
      {
        stages: { start: { agents: { missing: { modelPolicy: "model@1" } } } },
      },
      "logical agent does not exist",
    ],
  ])("rejects invalid routing %#", (value, message) => {
    expect(
      validateDraft(
        patchDraft(workflow(), ["spec", "executionConfig"], value),
      ).some((p) => p.message.includes(message)),
    ).toBe(true);
  });
  it.each([
    [["timeoutSeconds"], 0, "1 to 3600"],
    [["timeoutSeconds"], 1.5, "integer"],
    [["arguments", "level", "value"], null, "non-null"],
    [["arguments", "level", "value"], {}, "non-null"],
    [["arguments", "level", "value"], Number.MAX_SAFE_INTEGER + 1, "bounded"],
    [["arguments", "level", "name"], "extra", "exactly a source"],
    [["arguments"], null, "mapping"],
    [["tool"], "another_tool", "matching execution.tool"],
    [["resultArtifact"], "tool-invocation.secret", "reserved"],
  ])("rejects invalid tool execution %#", (path, value, message) => {
    expect(
      validateDraft(
        patchDraft(tool(), ["spec", "execution", ...(path as string[])], value),
      ).some((p) => p.message.includes(message as string)),
    ).toBe(true);
  });
});
