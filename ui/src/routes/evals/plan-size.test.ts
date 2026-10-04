import planSizeCases from "../../../../api/testdata/evals/plan-size-cases.json";
import { describe, expect, it } from "vitest";

import type { EvalDraft } from "../../api/evals";
import { estimatedPlanBytes, MAX_EVAL_DOCUMENT_BYTES } from "./setup-model";

describe("native plan-size estimate", () => {
  it("uses the Server document bound", () => {
    expect(MAX_EVAL_DOCUMENT_BYTES).toBe(planSizeCases.maxDocumentBytes);
  });

  it.each(planSizeCases.cases)("matches the shared $name case", (testCase) => {
    const draft = structuredClone(testCase.draft) as unknown as EvalDraft;
    if ("padding" in testCase) {
      const variant = draft.variants[0]!;
      variant.parameters = {
        ...variant.parameters,
        padding: testCase.padding.unit.repeat(testCase.padding.repeat),
      };
    }
    const estimated = estimatedPlanBytes(draft);
    expect(estimated).toBe(testCase.estimatedBytes);
    expect(estimated <= MAX_EVAL_DOCUMENT_BYTES).toBe(testCase.withinLimit);
  });
});
