import type { RunStatus } from "../../api/runs";
import { isTerminalRunState } from "../../api/runs";
import type { ArtifactMetadata } from "../../api/artifacts";
import {
  CONFIG_ID_PATTERN,
  CONFIG_VERSION_PATTERN,
  type WorkflowResource,
} from "../../api/workflows";

export type WorkflowOutputSlot = WorkflowResource["outputs"][string];

export interface WorkflowIdentity {
  name: string;
  version: string;
}

export interface OutputEntry {
  slot: string;
  declaration?: WorkflowOutputSlot;
  artifact?: ArtifactMetadata["artifact"];
  kind: "primary" | "declared" | "unclassified";
}

export function parseWorkflowIdentity(
  selector: string,
): WorkflowIdentity | undefined {
  const separator = selector.lastIndexOf("@");
  if (separator <= 0 || separator === selector.length - 1) {
    return undefined;
  }
  const name = selector.slice(0, separator);
  const version = selector.slice(separator + 1);
  return CONFIG_ID_PATTERN.test(name) && CONFIG_VERSION_PATTERN.test(version)
    ? { name, version }
    : undefined;
}

export function requireWorkflowOutputs(
  workflow: WorkflowResource,
  expected: WorkflowIdentity,
): Record<string, WorkflowOutputSlot> {
  if (
    workflow.ref.name !== expected.name ||
    workflow.ref.version !== expected.version ||
    typeof workflow.outputs !== "object" ||
    workflow.outputs === null ||
    Array.isArray(workflow.outputs)
  ) {
    throw new Error("Server returned an invalid exact Workflow contract");
  }
  for (const [slot, declaration] of Object.entries(workflow.outputs)) {
    if (
      !CONFIG_ID_PATTERN.test(slot) ||
      typeof declaration !== "object" ||
      declaration === null ||
      typeof declaration.required !== "boolean" ||
      !Array.isArray(declaration.mediaTypes) ||
      declaration.mediaTypes.some(
        (mediaType) => typeof mediaType !== "string",
      ) ||
      (declaration.primary !== undefined &&
        typeof declaration.primary !== "boolean")
    ) {
      throw new Error("Server returned an invalid exact Workflow contract");
    }
  }
  return workflow.outputs;
}

export function organizeRunOutputs(
  outputs: RunStatus["outputs"],
  declarations?: Record<string, WorkflowOutputSlot>,
): OutputEntry[] {
  if (declarations === undefined) {
    return Object.entries(outputs)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([slot, artifact]) => ({
        slot,
        artifact,
        kind: "unclassified" as const,
      }));
  }

  const declared = Object.entries(declarations)
    .map(([slot, declaration]): OutputEntry => ({
      slot,
      declaration,
      ...(outputs[slot] === undefined ? {} : { artifact: outputs[slot] }),
      kind: declaration.primary === true ? "primary" : "declared",
    }))
    .sort((left, right) => {
      if (left.kind !== right.kind) {
        return left.kind === "primary" ? -1 : 1;
      }
      if ((left.artifact === undefined) !== (right.artifact === undefined)) {
        return left.artifact === undefined ? 1 : -1;
      }
      return left.slot.localeCompare(right.slot);
    });
  const undeclared = Object.entries(outputs)
    .filter(([slot]) => declarations[slot] === undefined)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([slot, artifact]): OutputEntry => ({
      slot,
      artifact,
      kind: "unclassified",
    }));
  return [...declared, ...undeclared];
}

export function outputRole(entry: OutputEntry): string {
  switch (entry.kind) {
    case "primary":
      return "Declared primary result";
    case "declared":
      return "Declared supporting result";
    case "unclassified":
      return "Unclassified Run output";
  }
}

export function missingOutputCopy(
  declaration: WorkflowOutputSlot,
  state: RunStatus["state"],
): string {
  const requirement = declaration.required ? "Required" : "Optional";
  if (!isTerminalRunState(state)) {
    return `${requirement} output is not available yet.`;
  }
  if (state === "succeeded") {
    return declaration.required
      ? "Required output is missing from this completed Run."
      : "Optional output was not produced by this completed Run.";
  }
  return `${requirement} output is unavailable because the Run ${state}.`;
}
