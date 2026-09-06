import { describe, expect, it } from "vitest";

import type { WorkflowResource } from "../../api/workflows";
import {
  missingOutputCopy,
  organizeRunOutputs,
  parseWorkflowIdentity,
  requireWorkflowOutputs,
} from "./output-model";

const declarations: WorkflowResource["outputs"] = {
  report: {
    required: true,
    mediaTypes: ["text/markdown"],
    primary: true,
  },
  diagram: {
    required: false,
    mediaTypes: ["text/vnd.likec4"],
    primary: true,
  },
  trace: { required: false, mediaTypes: ["application/json"] },
};

describe("Run output presentation model", () => {
  it("orders present primary results before missing and supporting outputs", () => {
    const entries = organizeRunOutputs(
      {
        report: {
          namespace: "outputs",
          name: "report",
          revision: "report-r1",
        },
        legacy: {
          namespace: "outputs",
          name: "legacy",
          revision: "legacy-r1",
        },
        trace: {
          namespace: "outputs",
          name: "trace",
          revision: "trace-r1",
        },
      },
      declarations,
    );

    expect(entries.map(({ slot, kind }) => [slot, kind])).toEqual([
      ["report", "primary"],
      ["diagram", "primary"],
      ["trace", "declared"],
      ["legacy", "unclassified"],
    ]);
    expect(entries[1]?.artifact).toBeUndefined();
  });

  it("never promotes present outputs when the exact contract is unavailable", () => {
    expect(
      organizeRunOutputs({
        report: {
          namespace: "outputs",
          name: "report",
          revision: "report-r1",
        },
      }),
    ).toMatchObject([{ slot: "report", kind: "unclassified" }]);
  });

  it("validates exact Workflow identity and distinguishes pending from missing", () => {
    const identity = parseWorkflowIdentity("review-source@2026.09");
    expect(identity).toEqual({ name: "review-source", version: "2026.09" });
    expect(parseWorkflowIdentity("review-source@latest@1")).toBeUndefined();

    const workflow = {
      ref: { name: "review-source", version: "2026.09" },
      entryStage: "review",
      parameters: {},
      inputs: {},
      outputs: declarations,
      stages: {},
    } satisfies WorkflowResource;
    expect(requireWorkflowOutputs(workflow, identity!)).toBe(declarations);
    expect(() =>
      requireWorkflowOutputs(workflow, {
        name: "review-source",
        version: "older",
      }),
    ).toThrow("invalid exact Workflow contract");
    expect(missingOutputCopy(declarations.report!, "running")).toContain(
      "not available yet",
    );
    expect(missingOutputCopy(declarations.report!, "succeeded")).toContain(
      "missing",
    );
  });
});
