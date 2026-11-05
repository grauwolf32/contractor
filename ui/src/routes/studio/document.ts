import {
  isAlias,
  isMap,
  isScalar,
  parseDocument,
  visit,
  type Document,
} from "yaml";

export const KINDS = ["Workflow", "AuditProfile", "AgentTemplate"] as const;
type Kind = (typeof KINDS)[number];
export type Path = (string | number)[];
export const MAX_SOURCE_BYTES = 1024 * 1024;
const MAX_BLOCKS = 256;
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
/** Several related edits form one undo entry and keep untouched syntax nodes. */
export function editDraft(
  draft: Draft,
  edit: (document: Document) => void,
): Draft {
  const document = draft.document.clone();
  edit(document);
  return readDraft(document.toString());
}
export function appendDraft(draft: Draft, path: Path, value: unknown): Draft {
  return editDraft(draft, (document) => {
    if (!document.hasIn(path)) document.setIn(path, []);
    document.addIn(path, value);
  });
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
    if (
      section === "stages" &&
      Object.hasOwn(record(record(spec.executionConfig).stages), old)
    )
      renameKey(["spec", "executionConfig", "stages"], old, name);
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
    for (const key of ["seed_source", "continuation"]) {
      if (section === "workflows" && record(spec[key]).workflow === old)
        document.setIn(["spec", key, "workflow"], name);
    }
    if (section === "inputs" && record(spec.seed_source).input === old)
      document.setIn(["spec", "seed_source", "input"], name);
    if (Array.isArray(spec.initial_work))
      spec.initial_work.forEach((work, index) => {
        if (section === "workflows" && record(work).workflow === old)
          document.setIn(["spec", "initial_work", index, "workflow"], name);
        if (section === "inputs")
          for (const [binding, input] of Object.entries(
            record(record(work).inputs),
          ))
            if (input === old)
              document.setIn(
                ["spec", "initial_work", index, "inputs", binding],
                name,
              );
      });
    if (section === "inputs")
      for (const [workflow, value] of Object.entries(record(spec.workflows)))
        for (const [binding, input] of Object.entries(
          record(record(value).inputs),
        ))
          if (
            record(input).source === "audit-input" &&
            record(input).name === old
          )
            document.setIn(
              ["spec", "workflows", workflow, "inputs", binding, "name"],
              name,
            );
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
            sandboxProfile: "local-workdir@1",
            toolsets: [],
          }
        : {
            schema: "contractor.audit.program.v1",
            inputs: { source: { required: true, media_types: ["text/plain"] } },
            workflows: {
              check: {
                ref: "check@1",
                inputs: { source: { source: "audit-input", name: "source" } },
                parameters: {},
                outputs: { result: "result" },
              },
            },
            hypotheses: {},
            initial_work: [
              {
                key: "check-source",
                workflow: "check",
                objective: "Inspect selected source",
              },
            ],
            limits: {
              max_tasks: 256,
              max_iterations: 1,
              max_tasks_per_iteration: 256,
              batch_size: 1,
              max_submitted_runs: 768,
              max_attempts: 3,
              deadline_seconds: 86400,
              max_evidence_bytes: 67108864,
              max_hypotheses: 256,
              max_hypothesis_revisions: 768,
              max_hypothesis_bytes: 16777216,
            },
            policy: { active_operations: "prohibited" },
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
              ref: "check@1",
              inputs: {},
              parameters: {},
              outputs: {},
            }
          : type === "toolset"
            ? { ref: "toolset@1", tools: [] }
            : type === "skill"
              ? { namespace: "skills", name: `skill_${key}` }
              : {
                  required: true,
                  [draft.kind === "AuditProfile"
                    ? "media_types"
                    : "mediaTypes"]: ["application/yaml"],
                };
  return { draft: patchDraft(draft, path, value), path };
}

interface Difference {
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
