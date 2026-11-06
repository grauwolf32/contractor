import { dereference, validate, type Schema } from "@cfworker/json-schema";
import auditSchema from "../../../../../api/audit/v1/contract.schema.json";
import materialSchema from "../../../../../api/v1alpha1/material-set.schema.json";
import commonSchema from "../../../../../api/v1alpha1/common.schema.json";
import type { ResearchTask } from "./research-data";

const lookup = dereference(commonSchema as Schema);
dereference(materialSchema as Schema, lookup);
dereference(auditSchema as Schema, lookup);

/** A display projection validated against the owning shared document schema. */
interface ScopedTaskResult {
  verdict?: "supported" | "refuted" | "inconclusive";
  rationale: string;
  tested_conditions: string[];
  limitations: string[];
  evidence: unknown[];
  origin: {
    audit_id: string;
    task_id: string;
    run_id: string;
    hypothesis?: { id: string; revision: number };
  };
}

export function scopedTaskResult(
  text: string,
  task: ResearchTask,
): ScopedTaskResult | undefined {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return undefined;
  }
  if (
    typeof value !== "object" ||
    value === null ||
    !("schema" in value) ||
    value.schema !== "contractor.audit.result.v1"
  )
    return undefined;
  if (
    !validate(
      value,
      { $ref: `${auditSchema.$id}#/$defs/result` },
      "2020-12",
      lookup,
      true,
    ).valid
  )
    throw new Error(
      "The saved check result violates its shared document contract",
    );
  const items = (
    value as unknown as { items: ScopedTaskResult[] }
  ).items.filter((item) => item.origin.task_id === task.document.task_id);
  const item = items[0];
  const expected = task.document.hypothesis;
  if (
    items.length !== 1 ||
    !item ||
    item.origin.audit_id !== task.document.audit_id ||
    item.origin.hypothesis?.id !== expected?.id ||
    item.origin.hypothesis?.revision !== expected?.revision
  )
    throw new Error(
      "Saved result does not match this task's exact hypothesis context",
    );
  return item;
}

/** A batch remains one immutable artifact. Select its trusted Task member only
 * for display; downloading still returns the original exact bytes. */
export function taskResultPreview(text: string, taskId: string): string {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return text.slice(0, 32768);
  }
  if (
    typeof value !== "object" ||
    value === null ||
    !("schema" in value) ||
    value.schema !== "contractor.audit.result.v1" ||
    !("items" in value) ||
    !Array.isArray(value.items)
  ) {
    return text.slice(0, 32768);
  }
  const items = value.items.filter(
    (item: unknown) =>
      typeof item === "object" &&
      item !== null &&
      "origin" in item &&
      typeof item.origin === "object" &&
      item.origin !== null &&
      "task_id" in item.origin &&
      item.origin.task_id === taskId,
  );
  if (items.length !== 1)
    throw new Error("Saved result has no unique member for this task");
  return JSON.stringify(items[0], null, 2).slice(0, 32768);
}
