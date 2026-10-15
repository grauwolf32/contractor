import type { WorkflowResource } from "../../api/workflows";
import { plural } from "../../app/format";
import type { StatusTone } from "../../app/status-tone";
import type { RunDraftState } from "../../run-drafts/memory";

/**
 * How far a Run draft is from starting, for the readiness chip and the
 * submit row: "Ready", "Review 2", "Defaults" or "3/5" required fields,
 * with a sentence and a tone.
 */
export function runDraftReadiness({
  workflow,
  draft,
  valid,
  artifactsPending,
  inputNoun,
}: {
  workflow: WorkflowResource;
  draft: RunDraftState;
  /** Whether the draft validates into a request. */
  valid: boolean;
  /** Whether the input choices are still loading. */
  artifactsPending: boolean;
  inputNoun: "file" | "material";
}): {
  ready: boolean;
  repeatReviewPending: boolean;
  value: string;
  copy: string;
  tone: StatusTone;
} {
  const { parameters, artifactSelections, artifactReviews } = draft;
  const requiredParameterNames = Object.entries(workflow.parameters)
    .filter(([, slot]) => slot.required)
    .map(([name]) => name);
  const requiredArtifactNames = Object.entries(workflow.inputs)
    .filter(([, slot]) => slot.required)
    .map(([name]) => name);
  const requiredFieldCount =
    requiredParameterNames.length + requiredArtifactNames.length;
  const completedRequiredFieldCount =
    requiredParameterNames.filter((name) => parameters[name] !== undefined)
      .length +
    requiredArtifactNames.filter(
      (name) => (artifactSelections[name] ?? "") !== "",
    ).length;
  const remaining = Math.max(
    0,
    requiredFieldCount - completedRequiredFieldCount,
  );
  const reviewRequired = Object.keys(workflow.inputs).filter((name) => {
    const selected = artifactSelections[name] ?? "";
    return selected !== "" && artifactReviews[name] !== selected;
  }).length;
  const repeatReviewPending =
    draft.repeat !== undefined && !draft.repeat.reviewed;
  const ready = valid && !artifactsPending && !repeatReviewPending;
  const value = ready
    ? "Ready"
    : remaining === 0 && reviewRequired > 0
      ? `Review ${reviewRequired}`
      : requiredFieldCount === 0
        ? "Defaults"
        : `${completedRequiredFieldCount}/${requiredFieldCount}`;
  const copy = ready
    ? "Fields complete · inputs reviewed"
    : repeatReviewPending
      ? "Review retained values for this new Run"
      : artifactsPending && requiredArtifactNames.length > 0
        ? `Loading ${inputNoun} choices…`
        : remaining > 0
          ? `${plural(remaining, "required field")} remaining`
          : reviewRequired > 0
            ? `Fields complete · ${plural(reviewRequired, "input review")} needed`
            : "Review highlighted settings";
  const tone: StatusTone = ready
    ? "success"
    : remaining === 0 && reviewRequired > 0
      ? "review"
      : requiredFieldCount === 0
        ? "neutral"
        : "idle";
  return { ready, repeatReviewPending, value, copy, tone };
}
