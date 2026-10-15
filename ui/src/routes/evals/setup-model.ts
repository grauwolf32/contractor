import { stringRule } from "../../api/contract";
import type { EvalCapabilities, EvalDraft, EvalVariant } from "../../api/evals";
import { compareVersions } from "../../app/format";
import { estimatedPlanBytes, MAX_EVAL_DOCUMENT_BYTES } from "./plan-size";

export const MAX_EVAL_CASES = 1000;
export const MAX_EVAL_MEMBERS = 1000;
export const MAX_EVAL_REPETITIONS = 100;
const DEFAULT_WALL_MS = 90 * 60 * 1000;
// Eval case and criterion IDs follow the public EvalId rule.
const EVAL_ID_PATTERN = stringRule("EvalId");
export const PIN_DIMENSIONS = [
  "source",
  "tasks",
  "scorers",
  "expected",
  "instructions",
  "models",
  "sampling",
  "tools",
  "skills",
  "runtime-config",
  "execution",
  "standards",
  "inventory",
  "audit-execution",
  "audit-interaction",
] as const;
export const COMPARISON_PURPOSES = {
  workflow: {
    label: "Workflow or check configuration",
    equal: ["source", "tasks", "scorers"],
    different: [
      "instructions",
      "models",
      "sampling",
      "tools",
      "skills",
      "runtime-config",
      "execution",
      "standards",
      "inventory",
      "audit-execution",
      "audit-interaction",
    ],
  },
  instructions: {
    label: "Instructions only",
    equal: [
      "source",
      "tasks",
      "scorers",
      "models",
      "sampling",
      "tools",
      "skills",
      "runtime-config",
      "execution",
    ],
    different: ["instructions"],
  },
  models: {
    label: "Model configuration",
    equal: [
      "source",
      "tasks",
      "scorers",
      "instructions",
      "tools",
      "skills",
      "runtime-config",
      "execution",
    ],
    different: ["models", "sampling"],
  },
} as const;

export function initialEvalDraft(): EvalDraft {
  return {
    dataset: { id: "", revision: "" },
    caseIds: [],
    variants: ["a", "b"].map((id) => ({
      id,
      kind: "workflow",
      selector: "",
      executionConfig: {},
    })),
    repetitions: 1,
    order: { kind: "alternating" },
    checks: [],
    comparison: {
      baseline: "a",
      candidate: "b",
      gates: { minCandidateEndToEndPass: 1, maxQualityDrop: 0 },
      requiredEqual: [...COMPARISON_PURPOSES.workflow.equal],
      allowedDifferences: [...COMPARISON_PURPOSES.workflow.different],
    },
    budgets: {
      maxMembers: MAX_EVAL_MEMBERS,
      maxInFlight: 1,
      wallMs: DEFAULT_WALL_MS,
      maxObservedTotalTokens: null,
    },
  };
}

export function sortedBindings(
  capabilities: EvalCapabilities | undefined,
  kind: EvalVariant["kind"],
) {
  return [...(capabilities?.bindings ?? [])]
    .filter((x) => x.kind === kind)
    .sort((a, b) => {
      const [aName, aVersion = ""] = a.selector.split("@");
      const [bName, bVersion = ""] = b.selector.split("@");
      return (
        aName!.localeCompare(bName!) || compareVersions(bVersion, aVersion)
      );
    });
}

export function draftProblem(name: string, draft: EvalDraft): string | null {
  if (!name.trim()) return "Name the experiment.";
  if (draft.variants.some((v) => !v.selector))
    return "Choose both A/B versions.";
  if (!draft.dataset.id || !draft.dataset.revision || !draft.caseIds.length)
    return "Select a dataset revision and at least one case.";
  if (
    !Number.isInteger(draft.repetitions) ||
    draft.repetitions < 1 ||
    draft.repetitions > MAX_EVAL_REPETITIONS
  )
    return `Choose 1–${MAX_EVAL_REPETITIONS} repetitions.`;
  const expected = draft.caseIds.length * 2 * draft.repetitions;
  if (expected > MAX_EVAL_MEMBERS || expected > draft.budgets.maxMembers)
    return "The matrix exceeds the selected member allowance.";
  if (estimatedPlanBytes(draft) > MAX_EVAL_DOCUMENT_BYTES)
    return "The matrix is too large for a frozen 1 MiB plan. Reduce cases or repetitions.";
  if (
    !Number.isInteger(draft.budgets.maxInFlight) ||
    draft.budgets.maxInFlight < 1 ||
    draft.budgets.maxInFlight > draft.budgets.maxMembers
  )
    return "Choose valid concurrency within the member allowance.";
  if (!Number.isSafeInteger(draft.budgets.wallMs) || draft.budgets.wallMs <= 0)
    return "Set a positive time allowance.";
  if (!draft.checks.length || !draft.checks.some((c) => c.required))
    return "Select at least one required criterion.";
  if (
    draft.checks.some(
      (c) =>
        !EVAL_ID_PATTERN.test(c.id) ||
        (c.evaluator === "human-review@1" && !c.rubricRevision),
    )
  )
    return "Give each criterion an ID and pin the human review rubric revision.";
  if (new Set(draft.checks.map((c) => c.id)).size !== draft.checks.length)
    return "Criterion IDs must be distinct.";
  return null;
}
