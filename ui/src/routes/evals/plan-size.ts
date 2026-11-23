import type { EvalDraft } from "../../api/evals";

/** The Server's bound on one evaluation document (dataset, frozen plan). */
export const MAX_EVAL_DOCUMENT_BYTES = 1024 * 1024;

// Match the server's conservative authoring bound. The portable plan stores
// each member twice (member row and execution order) and repeats comparison
// dimensions in both arms' pin maps. JSON escaping follows Go's encoder;
// api/testdata/evals/plan-size-cases.json holds both estimates to the same
// cases.
export function estimatedPlanBytes(draft: EvalDraft): number {
  const draftJSON = JSON.stringify(draft).replace(
    /[<>&\u2028\u2029]/g,
    (char) => `\\u${char.charCodeAt(0).toString(16).padStart(4, "0")}`,
  );
  const draftBytes = new TextEncoder().encode(draftJSON).length;
  const members = draft.caseIds.length * 2 * draft.repetitions;
  const dimensions =
    draft.comparison.requiredEqual.length +
    draft.comparison.allowedDifferences.length;
  return 3 * draftBytes + 900 * members + 128 * dimensions + 32 * 1024;
}
