import { describe, expect, it } from "vitest";

import { PublicAPIError } from "../../api/error";
import {
  decisionErrorMessage,
  decisionOutcome,
  findingDecisionBody,
  findingDraftProblem,
  SEVERITY_OPTIONS,
} from "./model";
import { FINDING_ID, makeDecision } from "./test-support";

function apiError(status: number, code: string) {
  return new PublicAPIError({ status, code, message: `${code} message` });
}

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

describe("recorded decisions", () => {
  it("reads outcomes in the shared vocabulary", () => {
    expect(decisionOutcome(makeDecision())).toEqual({
      label: "Approved",
      tone: "success",
    });
    expect(decisionOutcome({ action: "reject" })).toEqual({
      label: "Rejected",
      tone: "neutral",
    });
    expect(decisionOutcome({ action: "not_applicable" })).toEqual({
      label: "Not applicable",
      tone: "neutral",
    });
  });
});

describe("refused decisions", () => {
  it.each([
    [
      412,
      "precondition_failed",
      "finding",
      "Not saved: this possible issue or its check changed first. The latest version is now shown. Check it, then record your decision again.",
    ],
    [
      412,
      "precondition_failed",
      "request",
      "Not saved: this request changed or expired first. The latest version is now shown. Check it, then decide again.",
    ],
    [
      409,
      "conflict",
      "finding",
      "Not saved: the decision conflicts with the current state, so nothing was retried. The latest version is now shown.",
    ],
    [
      404,
      "not_found",
      "finding",
      "Not saved: the review or the chosen possible issue no longer exists. The latest version is now shown.",
    ],
    [
      0,
      "network_error",
      "request",
      "The Server's answer did not arrive, so the decision may have been saved. Check the latest state before you record it again.",
    ],
    [400, "invalid_argument", "finding", "Not saved: invalid_argument message"],
  ] as const)("explains %i %s for a %s", (status, code, subject, message) => {
    expect(decisionErrorMessage(apiError(status, code), subject)).toBe(message);
  });

  it("explains errors raised before a request", () => {
    expect(
      decisionErrorMessage(
        new Error("this review does not offer it"),
        "finding",
      ),
    ).toBe("Not saved: this review does not offer it");
    expect(decisionErrorMessage("unknown", "request")).toBe(
      "Not saved: the request failed.",
    );
  });
});
