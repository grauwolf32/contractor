import { expect, it } from "vitest";
import {
  draftProblem,
  initialEvalDraft,
  MAX_EVAL_MEMBERS,
} from "./setup-model";

it("flags native plan overflow before the setup is saved", () => {
  const draft = initialEvalDraft();
  draft.dataset = { id: "trace-small", revision: "r1" };
  draft.variants[0]!.selector = "trace-a@1";
  draft.variants[1]!.selector = "trace-b@1";
  draft.checks = [
    {
      id: "review",
      evaluator: "human-review@1",
      required: true,
      rubricRevision: "r1",
    },
  ];
  expect(draft.budgets.maxMembers).toBe(MAX_EVAL_MEMBERS);
  draft.caseIds = Array.from({ length: 500 }, (_, i) => `case-${i}`);
  expect(draftProblem("Boundary", draft)).toBeNull();

  draft.caseIds = Array.from({ length: 25 }, (_, i) => `case-${i}`);
  draft.repetitions = 50;
  expect(draftProblem("Too many", draft)).toMatch(/matrix exceeds/);

  draft.repetitions = 1;
  draft.caseIds = Array.from({ length: 500 }, (_, i) =>
    `case-${i}`.padEnd(128, "x"),
  );
  expect(draftProblem("Too large", draft)).toMatch(/frozen 1 MiB plan/);
});
