import { describe, expect, it } from "vitest";

import { EVAL_STATES, type EvalExperiment } from "../../api/evals";
import {
  controlModeLabel,
  EVAL_STATE_LABELS,
  evalConclusionLabel,
  evalFreshnessText,
  evalStateLabel,
  evaluatorLabel,
  executableLabel,
  executionKindLabel,
  executionRefLabel,
  expectedMembersText,
  experimentLanding,
  experimentListPath,
  pinLabel,
  variantArm,
} from "./labels";

describe("Evals vocabulary", () => {
  it("labels every lifecycle state with a word and a tone", () => {
    expect(Object.keys(EVAL_STATE_LABELS)).toEqual([...EVAL_STATES]);
    expect(evalStateLabel("ready")).toEqual({
      label: "Ready to start",
      tone: "info",
    });
    expect(evalStateLabel("settling")).toEqual({
      label: "Finishing",
      tone: "progress",
    });
    expect(evalStateLabel("finished").tone).toBe("done");
    expect(evalStateLabel("cancelled").label).toBe("Cancelled");
    expect(evalStateLabel("archived" as EvalExperiment["state"])).toEqual({
      label: "Archived",
      tone: "neutral",
    });
  });

  it("says Meets declared gates for a pass and never concludes without a summary", () => {
    expect(evalConclusionLabel("pass")).toEqual({
      label: "Meets declared gates",
      tone: "success",
    });
    expect(evalConclusionLabel("regressions").tone).toBe("blocked");
    expect(evalConclusionLabel("inconclusive").label).toBe("Inconclusive");
    expect(evalConclusionLabel(null)).toEqual({
      label: "Not concluded yet",
      tone: "idle",
    });
    expect(evalFreshnessText("stale")).toMatch(/refresh/);
    expect(evalFreshnessText(undefined)).toBeUndefined();
  });

  it("calls an Audit a check and keeps Workflow and Run", () => {
    expect(executionKindLabel("audit")).toBe("Check");
    expect(executionKindLabel("workflow")).toBe("Workflow");
    expect(executableLabel("audit")).toBe("Check type");
    expect(executableLabel("workflow")).toBe("Workflow");
    expect(executionRefLabel("audit")).toBe("Check");
    expect(executionRefLabel("run")).toBe("Run");
    expect(controlModeLabel("server")).toBe("Server controlled");
    expect(controlModeLabel("external")).toBe("External producer");
    expect(pinLabel("audit-execution")).toBe("Check execution");
    expect(pinLabel("runtime-config")).toBe("Runtime configuration");
    expect(pinLabel("new-dimension")).toBe("New dimension");
    expect(evaluatorLabel("human-review@1")).toBe("Human review");
    expect(evaluatorLabel("required-artifact@1")).toBe("Required output files");
    expect(evaluatorLabel("custom@2")).toBe("custom@2");
  });

  it("counts members and builds experiment links", () => {
    expect(expectedMembersText(1)).toBe("1 expected member");
    expect(expectedMembersText(1200)).toBe("1,200 expected members");
    expect(experimentLanding({ experimentId: "e 1", state: "draft" })).toBe(
      "/evals/experiments/e%201/setup",
    );
    expect(experimentLanding({ experimentId: "e1", state: "running" })).toBe(
      "/evals/experiments/e1/overview",
    );
    expect(experimentListPath("e/1")).toBe("/evals?experiment=e%2F1");
  });

  it("maps variants to arms by the declared baseline, not by array order", () => {
    const comparison = {
      baseline: "candidate-x",
      candidate: "baseline-y",
    } as NonNullable<EvalExperiment["setup"]>["comparison"];
    const experiment = {
      setup: { comparison } as NonNullable<EvalExperiment["setup"]>,
    };
    expect(variantArm(experiment, "candidate-x")).toBe("a");
    expect(variantArm(experiment, "baseline-y")).toBe("b");
    expect(variantArm(experiment, "other")).toBeUndefined();
  });
});
