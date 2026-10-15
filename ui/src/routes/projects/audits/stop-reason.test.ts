import { describe, expect, it } from "vitest";

import { describeStopReason, stopReasonTone } from "./stop-reason";

/** The Audit stop reason codes of internal/auditcontroller. */
const STOP_REASON_CODES = [
  "deadline_exhausted",
  "round_complete",
  "cancel_requested",
  "delete_requested",
  "project_deleting",
  "report_rejected",
  "report_acceptance_expired",
  "submission_budget_exhausted",
  "item_budget_exhausted",
  "item_attempt_budget_exhausted",
  "round_budget_exhausted",
  "role_attempt_budget_exhausted",
  "proposal_scan_budget_exhausted",
  "proposal_inventory_limit_exceeded",
  "next_round_contract_invalid",
  "controller_contract_invalid",
  "dispatch_contract_invalid",
  "role_contract_invalid",
  "role_dispatch_contract_invalid",
  "role_dependency_unresolved",
  "role_execution_not_retryable",
];

describe("stop reasons", () => {
  it("explains every code the Server sends in one plain sentence", () => {
    for (const code of STOP_REASON_CODES) {
      const stop = describeStopReason({
        state: "completed",
        stopReason: { code, message: `server message for ${code}` },
      });
      expect(stop?.sentence).toMatch(/^[A-Z].*\.$/u);
      expect(stop?.sentence).not.toContain(code);
      expect(stop?.sentence).not.toContain("server message");
      expect(stop?.label).toBeDefined();
      expect(stop?.message).toBe(`server message for ${code}`);
    }
  });

  it("never shows an unknown code, only the Server's message", () => {
    const stop = describeStopReason({
      state: "cancelled",
      stopReason: {
        code: "brand_new_reason",
        message: "Stopped for a new reason.",
      },
    });
    expect(stop).toMatchObject({
      label: undefined,
      sentence: "Stopped for a new reason.",
      tone: "neutral",
      deadline: false,
    });
  });

  it("words the time limit by state and keeps failures as errors", () => {
    const deadline = { code: "deadline_exhausted", message: "deadline" };
    const paused = describeStopReason({
      state: "paused",
      stopReason: deadline,
    });
    expect(paused?.sentence).toMatch(/^The time limit was reached\. Continue/u);
    expect(stopReasonTone(paused!)).toBe("warning");
    const ended = describeStopReason({
      state: "completed",
      stopReason: deadline,
    });
    expect(ended?.sentence).toBe(
      "Stopped by the time limit: no new work was started after it.",
    );
    const failed = describeStopReason({
      state: "failed",
      stopReason: deadline,
    });
    expect(failed?.tone).toBe("error");
    expect(stopReasonTone(failed!)).toBe("blocked");
    const budget = describeStopReason({
      state: "completed",
      stopReason: { code: "round_budget_exhausted", message: "rounds" },
    });
    expect(budget?.tone).toBe("error");
    expect(
      stopReasonTone(
        describeStopReason({
          state: "completed",
          stopReason: { code: "round_complete", message: "done" },
        })!,
      ),
    ).toBe("info");
    expect(describeStopReason({ state: "active" })).toBeNull();
  });
});
