import {
  isAlias,
  isMap,
  isScalar,
  parseDocument,
  visit,
  type Document,
} from "yaml";

export const KINDS = ["Workflow", "AuditProfile", "AgentTemplate"] as const;
export type Kind = (typeof KINDS)[number];
export type Path = (string | number)[];
export const MAX_SOURCE_BYTES = 1024 * 1024;
export const MAX_BLOCKS = 256;
export const kindLabel = (kind: Kind) =>
  ({
    Workflow: "Workflow",
    AuditProfile: "Check type",
    AgentTemplate: "Agent",
  })[kind];

export function record(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}
export const textValue = (value: unknown): string =>
  typeof value === "string"
    ? value
    : typeof value === "number"
      ? String(value)
      : "";
export function at(value: unknown, path: Path): unknown {
  return path.reduce<unknown>(
    (current, key) =>
      Array.isArray(current) && typeof key === "number"
        ? current[key]
        : Object.hasOwn(record(current), key)
          ? record(current)[key]
          : undefined,
    value,
  );
}

export interface Draft {
  document: Document;
  value: Record<string, unknown>;
  kind: Kind;
  source: string;
}

/** Keep the syntax tree: visual edits must retain comments and opaque fields. */
export function readDraft(source: string): Draft {
  if (new TextEncoder().encode(source).length > MAX_SOURCE_BYTES)
    throw new Error("YAML must be at most 1 MiB.");
  const document = parseDocument(source, {
    uniqueKeys: true,
    keepSourceTokens: true,
  });
  if (document.errors.length) throw new Error(document.errors[0]!.message);
  let nodes = 0;
  visit(document, (_, node) => {
    if (++nodes > 20000) throw new Error("YAML has too many entries.");
    // Editing an alias could also change its anchor. Require explicit values
    // so the inspector never silently changes another block.
    if (isAlias(node))
      throw new Error(
        "Expand YAML aliases before importing into the visual editor.",
      );
  });
  if (document.warnings.length) throw new Error(document.warnings[0]!.message);
  const value = record(document.toJS({ maxAliasCount: 0 }));
  const kind = KINDS.find((candidate) => candidate === value.kind);
  if (!kind)
    throw new Error(
      "Expected one Workflow, AuditProfile or AgentTemplate YAML document.",
    );
  const spec = record(value.spec);
  const blockCount =
    ["stages", "inputs", "outputs", "parameters", "workflows"].reduce(
      (count, key) => count + Object.keys(record(spec[key])).length,
      0,
    ) +
    ["toolsets", "skills"].reduce(
      (count, key) => count + (Array.isArray(spec[key]) ? spec[key].length : 0),
      0,
    );
  if (blockCount > MAX_BLOCKS)
    throw new Error(`A studio document supports at most ${MAX_BLOCKS} blocks.`);
  return { document, value, kind, source };
}

export function patchDraft(draft: Draft, path: Path, value: unknown): Draft {
  const document = draft.document.clone();
  document.setIn(path, value);
  return readDraft(document.toString());
}
export function removeBlock(draft: Draft, path: Path): Draft {
  const document = draft.document.clone();
  document.deleteIn(path);
  return readDraft(document.toString());
}
/** Rename a mapped block and the explicit references understood by its kind. */
export function renameBlock(draft: Draft, path: Path, name: string): Draft {
  const old = path.at(-1),
    section = textValue(path[1]);
  if (
    typeof old !== "string" ||
    path.length !== 3 ||
    !/^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(name)
  )
    throw new Error(
      "Use a block identifier containing letters, numbers, dots, underscores or hyphens.",
    );
  if (name === old) return draft;
  const document = draft.document.clone();
  const renameKey = (parent: Path, old: string, next: string) => {
    const map = document.getIn(parent, true);
    if (!isMap(map)) throw new Error("This block is not in a mapping.");
    if (map.has(next)) throw new Error(`A block named ${next} already exists.`);
    const pair = map.items.find(
      (pair) => isScalar(pair.key) && pair.key.value === old,
    );
    if (pair && isScalar(pair.key)) pair.key.value = next;
  };
  renameKey(path.slice(0, -1), old, name);
  const spec = record(draft.value.spec);
  if (draft.kind === "Workflow") {
    if (section === "stages" && spec.entryStage === old)
      document.setIn(["spec", "entryStage"], name);
    for (const [stageName, value] of Object.entries(record(spec.stages))) {
      const stage = record(value),
        p: Path = [
          "spec",
          "stages",
          section === "stages" && stageName === old ? name : stageName,
        ];
      for (const outcome of ["succeeded", "failed", "interrupted"]) {
        const action = record(record(stage.on)[outcome]);
        for (const nested of [[], ["retry", "then"], ["escalate", "then"]]) {
          if (section === "stages" && at(action, [...nested, "next"]) === old)
            document.setIn([...p, "on", outcome, ...nested, "next"], name);
        }
      }
      if (section === "inputs")
        for (const [binding, value] of Object.entries(
          record(record(stage.context).artifacts),
        )) {
          if (
            record(value).namespace === "inputs" &&
            record(value).name === old
          )
            document.setIn(
              [...p, "context", "artifacts", binding, "name"],
              name,
            );
        }
      if (
        section === "outputs" &&
        Object.hasOwn(record(stage.workflowOutputs), old)
      )
        renameKey([...p, "workflowOutputs"], old, name);
    }
  } else if (draft.kind === "AuditProfile") {
    const inventory = record(spec.inventory);
    if (section === "workflows" && inventory.itemWorkflowRole === old)
      document.setIn(["spec", "inventory", "itemWorkflowRole"], name);
    for (const key of ["source", "settings"]) {
      const row = record(inventory[key]);
      if (section === "workflows" && row.role === old)
        document.setIn(["spec", "inventory", key, "role"], name);
      if (
        section === "inputs" &&
        row.source === "audit-input" &&
        row.name === old
      )
        document.setIn(["spec", "inventory", key, "name"], name);
    }
    for (const [role, value] of Object.entries(record(spec.workflows))) {
      const p: Path = [
        "spec",
        "workflows",
        section === "workflows" && role === old ? name : role,
      ];
      for (const [binding, bindingValue] of Object.entries(
        record(record(value).inputs),
      )) {
        const row = record(bindingValue);
        if (section === "workflows" && row.role === old)
          document.setIn([...p, "inputs", binding, "role"], name);
        if (
          section === "inputs" &&
          row.source === "audit-input" &&
          row.name === old
        )
          document.setIn([...p, "inputs", binding, "name"], name);
      }
    }
  }
  return readDraft(document.toString());
}
export function uniqueName(value: unknown, prefix: string): string {
  let suffix = 1;
  while (Object.hasOwn(record(value), `${prefix}_${suffix}`)) suffix++;
  return `${prefix}_${suffix}`;
}

const stage = () => ({
  objective: "Describe the stage objective",
  planner: "passthrough@1",
  agents: { worker: { template: "worker@1", namespace: "work" } },
  context: { artifacts: {} },
  result: { artifacts: {} },
  on: {
    succeeded: { succeed: {} },
    failed: { fail: {} },
    interrupted: { fail: {} },
  },
});

export function starter(kind: Kind): Draft {
  const document = parseDocument("");
  document.setIn(["apiVersion"], "contractor/v1alpha1");
  document.setIn(["kind"], kind);
  document.setIn(["metadata"], { name: "untitled", version: "1" });
  document.setIn(
    ["spec"],
    kind === "Workflow"
      ? {
          parameters: {},
          inputs: {},
          outputs: {},
          entryStage: "start",
          stages: { start: stage() },
        }
      : kind === "AgentTemplate"
        ? {
            description: "Describe this agent",
            runtime: "adk@1",
            instructions: { ref: "instructions/worker.md" },
            modelPolicy: "worker@1",
            sandboxProfile: "local-workdir@1",
            toolsets: [],
          }
        : {
            mode: "operation-tracing",
            standards: [],
            inputs: {
              source: { required: true, mediaTypes: ["application/zip"] },
              openapi: { required: true, mediaTypes: ["application/yaml"] },
            },
            inventory: {
              implementation: "openapi-operations@1",
              source: { source: "audit-input", name: "openapi" },
              itemWorkflowRole: "check",
            },
            workflows: {
              check: {
                kind: "check",
                ref: "check@1",
                inputs: {
                  task: { source: "item-package" },
                  source: { source: "audit-input", name: "source" },
                },
                parameters: {},
                outputs: { result: "result" },
              },
            },
            execution: {
              roundMode: "fixed-barrier",
              maxRounds: 1,
              batchSize: 1,
              maxItemsPerRound: 256,
              maxItemsTotal: 256,
              maxSubmittedRuns: 768,
              maxItemRunAttempts: 3,
              deadlineSeconds: 86400,
              maxEvidenceBytes: 67108864,
              incompleteRound: "assess-with-gaps",
            },
            interaction: {
              activeChecks: "prohibited",
              findingConfirmation: "human-required",
              notApplicable: "profile-rule",
              reportAcceptance: "automatic",
            },
          },
  );
  document.commentBefore =
    " Local draft. Resolve selectors and instruction files in your configuration bundle before use.";
  return readDraft(document.toString());
}

export type BlockType =
  "stage" | "input" | "output" | "parameter" | "role" | "toolset" | "skill";
export function addBlock(
  draft: Draft,
  type: BlockType,
): { draft: Draft; path: Path } {
  const section = {
    stage: "stages",
    input: "inputs",
    output: "outputs",
    parameter: "parameters",
    role: "workflows",
    toolset: "toolsets",
    skill: "skills",
  }[type];
  const collection = at(draft.value, ["spec", section]);
  const isList = type === "toolset" || type === "skill";
  const key = isList
    ? Array.isArray(collection)
      ? collection.length
      : 0
    : uniqueName(collection, type);
  const path: Path = ["spec", section, key];
  const value =
    type === "stage"
      ? stage()
      : type === "parameter"
        ? { required: false }
        : type === "role"
          ? {
              kind: "check",
              ref: "check@1",
              inputs: {},
              parameters: {},
              outputs: {},
            }
          : type === "toolset"
            ? { ref: "toolset@1", tools: [] }
            : type === "skill"
              ? { namespace: "skills", name: `skill_${key}` }
              : { required: true, mediaTypes: ["application/yaml"] };
  return { draft: patchDraft(draft, path, value), path };
}

export interface Difference {
  kind: "same" | "add" | "remove";
  line: string;
}
/** A bounded single-hunk diff; avoids quadratic work on imported documents. */
export function sourceDiff(original: string, current: string): Difference[] {
  const before = original.split("\n"),
    after = current.split("\n");
  let prefix = 0,
    suffix = 0;
  while (
    prefix < before.length &&
    prefix < after.length &&
    before[prefix] === after[prefix]
  )
    prefix++;
  while (
    suffix < before.length - prefix &&
    suffix < after.length - prefix &&
    before[before.length - 1 - suffix] === after[after.length - 1 - suffix]
  )
    suffix++;
  if (prefix === before.length && prefix === after.length) return [];
  return [
    ...before
      .slice(Math.max(0, prefix - 3), prefix)
      .map((line) => ({ kind: "same" as const, line })),
    ...before
      .slice(prefix, before.length - suffix)
      .map((line) => ({ kind: "remove" as const, line })),
    ...after
      .slice(prefix, after.length - suffix)
      .map((line) => ({ kind: "add" as const, line })),
    ...after
      .slice(after.length - suffix, after.length - suffix + 3)
      .map((line) => ({ kind: "same" as const, line })),
  ];
}
