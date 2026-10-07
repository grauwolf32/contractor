/**
 * User-facing words for managed Evals (docs/design/ui/v3b-build-contract.md
 * §3): experiment states and conclusions with their status tones, the
 * execution kind (an Audit is a "check"), control modes and the list
 * filters. Pages take labels from here instead of printing API values.
 */
import type {
  EvalExperiment,
  EvalMemberQuery,
  EvalPairQuery,
  EvalSummary,
} from "../../api/evals";
import type { StatusTone } from "../../app/status-tone";

export type EvalState = EvalExperiment["state"];
export type EvalConclusion = EvalSummary["conclusion"];
export type EvalExecutionKind = EvalExperiment["executionKind"];
export type EvalControlMode = EvalExperiment["controlMode"];
export type EvalFreshness = NonNullable<EvalExperiment["freshness"]>;
export type EvalSection = "overview" | "comparison" | "attempts" | "setup";

/** A status word and the tone of its glyph or chip. */
export interface EvalLabel {
  readonly label: string;
  readonly tone: StatusTone;
}

/** Readable fallback for a value this client does not know yet. */
function unknownLabel(value: string): string {
  const words = value.replaceAll(/[_-]+/g, " ").trim();
  return words === ""
    ? "Unknown"
    : words.charAt(0).toUpperCase() + words.slice(1);
}

function lookup(
  table: Readonly<Record<string, EvalLabel>>,
  value: string,
): EvalLabel {
  return Object.hasOwn(table, value)
    ? table[value]!
    : { label: unknownLabel(value), tone: "neutral" };
}

// Key order is the lifecycle order of the list filter.
export const EVAL_STATE_LABELS: Readonly<Record<EvalState, EvalLabel>> = {
  draft: { label: "Draft", tone: "idle" },
  preparing: { label: "Preparing", tone: "progress" },
  ready: { label: "Ready to start", tone: "info" },
  running: { label: "Running", tone: "progress" },
  settling: { label: "Finishing", tone: "progress" },
  finished: { label: "Finished", tone: "done" },
  pausing: { label: "Pausing", tone: "warning" },
  paused: { label: "Paused", tone: "warning" },
  cancelling: { label: "Cancelling", tone: "warning" },
  cancelled: { label: "Cancelled", tone: "neutral" },
};

/** Lifecycle state of an experiment: "Draft", "Running", "Finished", … */
export function evalStateLabel(state: EvalState): EvalLabel {
  return lookup(EVAL_STATE_LABELS, state);
}

// "Meets declared gates" is the only passing wording (S30:529-530): it is not
// a universal quality approval.
export const EVAL_CONCLUSION_LABELS: Readonly<
  Record<EvalConclusion, EvalLabel>
> = {
  pass: { label: "Meets declared gates", tone: "success" },
  regressions: { label: "Regressions found", tone: "blocked" },
  inconclusive: { label: "Inconclusive", tone: "warning" },
};

/** Comparison conclusion; without a summary nothing is concluded yet. */
export function evalConclusionLabel(
  conclusion: EvalConclusion | null | undefined,
): EvalLabel {
  return conclusion === null || conclusion === undefined
    ? { label: "Not concluded yet", tone: "idle" }
    : lookup(EVAL_CONCLUSION_LABELS, conclusion);
}

const FRESHNESS: Readonly<Record<EvalFreshness, string>> = {
  current: "View is current",
  pending: "View is pending",
  stale: "Evidence changed · refresh to review the current result",
};

/** Whether the comparison view matches the selected evidence. */
export function evalFreshnessText(
  freshness: EvalFreshness | null | undefined,
): string | undefined {
  if (freshness === null || freshness === undefined) return undefined;
  return Object.hasOwn(FRESHNESS, freshness)
    ? FRESHNESS[freshness]
    : unknownLabel(freshness);
}

/** What one experiment member executes: a Workflow Run or a whole check. */
export function executionKindLabel(kind: EvalExecutionKind | string): string {
  return kind === "audit"
    ? "Check"
    : kind === "workflow"
      ? "Workflow"
      : unknownLabel(kind);
}

/** The executable a variant selects: a Workflow or a check type. */
export function executableLabel(kind: EvalExecutionKind | string): string {
  return kind === "audit" ? "Check type" : "Workflow";
}

/** An execution reference: "Run" or "Check" (an Audit). */
export function executionRefLabel(kind: string): string {
  return kind === "run"
    ? "Run"
    : kind === "audit"
      ? "Check"
      : unknownLabel(kind);
}

/** Who dispatches the experiment's members. */
export function controlModeLabel(mode: EvalControlMode | string): string {
  return mode === "server"
    ? "Server controlled"
    : mode === "external"
      ? "External producer"
      : unknownLabel(mode);
}

/** "8 expected members". */
export function expectedMembersText(count: number): string {
  return `${count.toLocaleString("en-US")} expected ${count === 1 ? "member" : "members"}`;
}

export const PAIR_FILTERS: readonly {
  value: NonNullable<EvalPairQuery["filter"]>;
  label: string;
}[] = [
  { value: "regressions", label: "Quality regressions" },
  { value: "unresolved", label: "Unresolved pairs" },
  { value: "all", label: "All pairs" },
];

export const ATTEMPT_FILTERS: readonly {
  value: NonNullable<EvalMemberQuery["filter"]>;
  label: string;
}[] = [
  { value: "all", label: "All" },
  { value: "unresolved", label: "Unresolved" },
  { value: "failed", label: "Failed" },
  { value: "unscored", label: "Unscored" },
  { value: "unsupported", label: "Unsupported" },
  { value: "blocked", label: "Blocked" },
  { value: "conflicting", label: "Conflicting" },
];

export const EVAL_SECTIONS: readonly { value: EvalSection; label: string }[] = [
  { value: "overview", label: "Overview" },
  { value: "comparison", label: "Comparison" },
  { value: "attempts", label: "Attempts" },
  { value: "setup", label: "Setup" },
];

/** The page of one experiment section. */
export function experimentPath(
  experimentId: string,
  section: EvalSection | `pairs/${string}`,
): string {
  return `/evals/experiments/${encodeURIComponent(experimentId)}/${section}`;
}

/** Where opening an experiment lands: drafts continue their setup. */
export function experimentLanding(experiment: {
  experimentId: string;
  state: EvalState;
}): string {
  return experimentPath(
    experiment.experimentId,
    experiment.state === "draft" ? "setup" : "overview",
  );
}

/** The experiments list with this experiment selected. */
export function experimentListPath(experimentId: string): string {
  return `/evals?experiment=${encodeURIComponent(experimentId)}`;
}

/** Which arm a variant is: A is the baseline, B the candidate. */
export function variantArm(
  experiment: Pick<EvalExperiment, "setup" | "draft">,
  variantId: string,
): "a" | "b" | undefined {
  const comparison =
    experiment.setup?.comparison ?? experiment.draft?.comparison;
  if (comparison?.baseline === variantId) return "a";
  if (comparison?.candidate === variantId) return "b";
  return undefined;
}

const PIN_LABELS: Readonly<Record<string, string>> = {
  source: "Source",
  tasks: "Tasks",
  scorers: "Scorers",
  expected: "Expected results",
  instructions: "Instructions",
  models: "Models",
  sampling: "Sampling",
  tools: "Tools",
  skills: "Skills",
  "runtime-config": "Runtime configuration",
  execution: "Execution",
  standards: "Standards",
  inventory: "Inventory",
  "audit-execution": "Check execution",
  "audit-interaction": "Check interaction",
};

/** A comparison dimension that A and B must share or may differ in. */
export function pinLabel(dimension: string): string {
  return Object.hasOwn(PIN_LABELS, dimension)
    ? PIN_LABELS[dimension]!
    : unknownLabel(dimension);
}

// Each evaluator assesses one declared property; the UI calls an assessment
// check a criterion (docs/design/ui/v3b-build-contract.md §3).
const EVALUATOR_LABELS: Readonly<Record<string, string>> = {
  "required-artifact@1": "Required output artifacts",
  "media-type@1": "Output media type",
  "json-schema@1": "Registered JSON schema",
  "human-review@1": "Human review",
};

/** A registered evaluator, or its selector when the UI has no name for it. */
export function evaluatorLabel(evaluator: string): string {
  return Object.hasOwn(EVALUATOR_LABELS, evaluator)
    ? EVALUATOR_LABELS[evaluator]!
    : evaluator;
}
