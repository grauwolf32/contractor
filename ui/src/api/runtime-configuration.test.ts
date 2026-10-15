import { describe, expect, it } from "vitest";

import {
  canonicalRunRuntimeConfiguration,
  canonicalStageRuntimeConfiguration,
} from "./runtime-configuration";

const digest = `sha256:${"1".repeat(64)}`;

function pin(label: string) {
  return {
    label,
    bindingRevision: "1",
    config: { name: label, version: "1", digest },
  };
}

function allocation(
  logicalAgent: string,
  overrides: Partial<{
    agentLabels: ReturnType<typeof pin>[];
    runtimeAdapters: string[];
    origins: Record<string, { layer: string; configs?: unknown[] }>;
  }> = {},
) {
  return {
    logicalAgent,
    agentLabels: [pin("debug")],
    runtimeAdapters: ["caido-graphql@1", "otlp-http@1"],
    origins: {
      caido: { layer: "run_execution_config" },
      workerTelemetry: {
        layer: "agent_labels",
        configs: [{ name: "debug", version: "1", digest }],
      },
    },
    status: "pinned" as const,
    ...overrides,
  } as Parameters<
    typeof canonicalStageRuntimeConfiguration
  >[0]["allocations"][number];
}

describe("Runtime configuration invariants", () => {
  it("accepts canonical Run pins and rejects disordered or misplaced labels", () => {
    expect(
      canonicalRunRuntimeConfiguration({
        default: pin("default"),
        labels: [pin("debug"), pin("proxy")],
      }),
    ).toBe(true);
    for (const value of [
      { default: pin("debug"), labels: [] },
      { default: pin("default"), labels: [pin("proxy"), pin("debug")] },
      { default: pin("default"), labels: [pin("debug"), pin("debug")] },
      { default: pin("default"), labels: [pin("default")] },
    ]) {
      expect(canonicalRunRuntimeConfiguration(value)).toBe(false);
    }
  });

  it("accepts a canonical Stage projection and rejects broken invariants", () => {
    expect(
      canonicalStageRuntimeConfiguration({
        allocations: [allocation("planner"), allocation("reviewer")],
      }),
    ).toBe(true);
    for (const allocations of [
      [allocation("reviewer"), allocation("planner")],
      [allocation("reviewer", { agentLabels: [pin("x"), pin("a")] })],
      [allocation("reviewer", { runtimeAdapters: ["z@1", "a@1"] })],
      [
        allocation("reviewer", {
          origins: { caido: { layer: "agent_labels" } },
        }),
      ],
      [
        allocation("reviewer", {
          origins: {
            caido: {
              layer: "run_execution_config",
              configs: [{ name: "debug", version: "1", digest }],
            },
          },
        }),
      ],
    ]) {
      expect(canonicalStageRuntimeConfiguration({ allocations })).toBe(false);
    }
  });
});
