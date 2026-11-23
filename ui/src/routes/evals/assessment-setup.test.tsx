import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import emptyHuman from "../../../../api/testdata/evals/valid/human-review-empty-parameters.json" with { type: "json" };
import unsupportedHuman from "../../../../api/testdata/evals/invalid/human-review-empty-parameters-human-review-parameters.json" with { type: "json" };
import { conformsTo } from "../../api/contract";
import type { EvalCapabilities, EvalDraft } from "../../api/evals";
import {
  createEvalFixture,
  EVAL_FIXTURE_ORIGIN,
} from "../../test/evals-fixture";
import { AssessmentSetup } from "./assessment-setup";
import { initialEvalDraft } from "./setup-model";

describe("human-review parameter contract", () => {
  it("uses the shared closed rule without narrowing other evaluators", () => {
    expect(conformsTo("EvalCheck", emptyHuman)).toBe(true);
    expect(conformsTo("EvalCheck", unsupportedHuman)).toBe(false);
    expect(conformsTo("EvalCheck", { ...emptyHuman, parameters: null })).toBe(
      false,
    );
    expect(
      conformsTo("EvalCheck", {
        ...unsupportedHuman,
        evaluator: "producer-review@2",
      }),
    ).toBe(true);
  });

  it.each([
    ["required-artifact@1", { output: "report" }],
    ["media-type@1", { output: "report", mediaType: "text/plain" }],
    ["json-schema@1", { output: "report", schema: "json@1" }],
  ] as const)(
    "clears %s parameters only when switching to human review",
    async (evaluator, parameters) => {
      const user = userEvent.setup();
      const fixture = createEvalFixture();
      const reply = await fixture.handle(
        new Request(`${EVAL_FIXTURE_ORIGIN}/v1/eval-capabilities`),
      );
      const capabilities = reply.body as EvalCapabilities;
      const initial = initialEvalDraft();
      initial.checks = [
        { id: "review", evaluator, required: true, parameters },
      ];
      const changed = vi.fn();
      function Harness() {
        const [draft, setDraft] = useState<EvalDraft>(initial);
        return (
          <AssessmentSetup
            draft={draft}
            capabilities={capabilities}
            onChange={(next) => {
              changed(next);
              setDraft(next);
            }}
          />
        );
      }
      render(<Harness />);
      await user.clear(screen.getByLabelText("Checked output role"));
      await user.type(screen.getByLabelText("Checked output role"), "changed");
      expect(changed.mock.lastCall?.[0].checks[0].parameters).toEqual({
        ...parameters,
        output: "changed",
      });
      await user.selectOptions(
        screen.getByLabelText("Evaluator"),
        "human-review@1",
      );
      const check = changed.mock.lastCall?.[0].checks[0];
      expect(check).toMatchObject({
        id: "review",
        evaluator: "human-review@1",
        required: true,
      });
      expect(check.parameters).toBeUndefined();
    },
  );
});
