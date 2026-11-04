import { describe, expect, it } from "vitest";

import {
  findingDecisionBody,
  findingDraftProblem,
  SEVERITY_OPTIONS,
} from "./model";
import { FINDING_ID } from "./test-support";

describe("finding decision drafts", () => {
  it("names what DecisionBar cannot check", () => {
    const base = { rationale: "Reason", verdict: "duplicate" as const };
    expect(findingDraftProblem(base, FINDING_ID)).toBe(
      "Choose the possible issue this one duplicates.",
    );
    expect(
      findingDraftProblem(
        { ...base, duplicateTargetId: FINDING_ID },
        FINDING_ID,
      ),
    ).toBe("A possible issue cannot duplicate itself. Choose another one.");
    expect(
      findingDraftProblem(
        { ...base, duplicateTargetId: "not an id" },
        FINDING_ID,
      ),
    ).toBe("That is not a valid possible issue ID.");
    expect(
      findingDraftProblem(
        { ...base, duplicateTargetId: "finding_2" },
        FINDING_ID,
      ),
    ).toBeUndefined();
    expect(
      findingDraftProblem(
        { verdict: "true_positive", rationale: "Reason" },
        FINDING_ID,
      ),
    ).toBe("Choose severity.");
    expect(
      findingDraftProblem(
        { verdict: "false_positive", rationale: "  " },
        FINDING_ID,
      ),
    ).toBe("Write a short reason.");
  });

  it("limits the reason to 64 KiB of UTF-8", () => {
    const draft = { verdict: "false_positive" as const };
    expect(
      findingDraftProblem({ ...draft, rationale: "a".repeat(65_536) }, "f"),
    ).toBeUndefined();
    // 21,846 three-byte characters are fewer than 65,536 characters but
    // more than 64 KiB.
    expect(
      findingDraftProblem({ ...draft, rationale: "€".repeat(21_846) }, "f"),
    ).toBe("The reason is too long. Keep it under 64 KiB.");
  });

  it("builds the request body of each verdict with a trimmed reason", () => {
    expect(
      findingDecisionBody({
        verdict: "true_positive",
        severity: "critical",
        duplicateTargetId: "ignored",
        rationale: " Confirmed. ",
      }),
    ).toEqual({
      verdict: "true_positive",
      severity: "critical",
      rationale: "Confirmed.",
    });
    expect(
      findingDecisionBody({
        verdict: "duplicate",
        severity: "low",
        duplicateTargetId: " finding_2 ",
        rationale: "Same.",
      }),
    ).toEqual({
      verdict: "duplicate",
      duplicateTargetId: "finding_2",
      rationale: "Same.",
    });
    for (const verdict of [
      "false_positive",
      "needs_evidence",
      "reopen",
    ] as const)
      expect(
        findingDecisionBody({ verdict, severity: "high", rationale: "Why" }),
      ).toEqual({ verdict, rationale: "Why" });
    expect(() =>
      findingDecisionBody({ verdict: "true_positive", rationale: "Why" }),
    ).toThrow(TypeError);
  });

  it("lists every severity in the vocabulary order", () => {
    expect(SEVERITY_OPTIONS.map((option) => option.label)).toEqual([
      "Informational",
      "Low",
      "Medium",
      "High",
      "Critical",
    ]);
  });
});
