import { describe, expect, it } from "vitest";

import {
  safeRunRuntimeConfiguration,
  safeStageRuntimeConfiguration,
} from "./runtime-configuration";

const digest = `sha256:${"1".repeat(64)}`;

describe("safe Runtime configuration projections", () => {
  it("retains only the closed pinned Run and Stage provenance shapes", () => {
    expect(
      safeRunRuntimeConfiguration({
        default: {
          label: "default",
          bindingRevision: "1",
          config: { name: "empty", version: "1", digest },
        },
        labels: [
          {
            label: "debug",
            bindingRevision: "2",
            config: { name: "debug", version: "1", digest },
          },
        ],
      }),
    ).toMatchObject({ labels: [{ label: "debug" }] });
    expect(
      safeStageRuntimeConfiguration({
        allocations: [
          {
            logicalAgent: "reviewer",
            agentLabels: [
              {
                label: "debug",
                bindingRevision: "2",
                config: { name: "debug", version: "1", digest },
              },
            ],
            runtimeAdapters: ["caido-graphql@1", "otlp-http@1"],
            origins: {
              caido: { layer: "run_execution_config" },
              workerTelemetry: {
                layer: "agent_labels",
                configs: [{ name: "debug", version: "1", digest }],
              },
            },
            status: "pinned",
          },
        ],
      }),
    ).toMatchObject({
      allocations: [
        {
          logicalAgent: "reviewer",
          origins: {
            caido: { layer: "run_execution_config" },
            workerTelemetry: { layer: "agent_labels" },
          },
        },
      ],
    });
  });

  it("accepts Run-owned toolset origins and rejects Agent overrides or secret fields", () => {
    const value = {
      allocations: [
        {
          logicalAgent: "reviewer",
          agentLabels: [],
          runtimeAdapters: [],
          status: "pinned",
          origins: {
            toolsets: {
              "mcp@1": {
                layer: "run_labels",
                configs: [{ name: "knowledge", version: "1", digest }],
              },
            },
          },
        },
      ],
    };
    expect(
      safeStageRuntimeConfiguration(value).allocations[0]?.origins.toolsets?.[
        "mcp@1"
      ]?.layer,
    ).toBe("run_labels");
    value.allocations[0]!.origins.toolsets["mcp@1"].layer = "agent_labels";
    expect(() => safeStageRuntimeConfiguration(value)).toThrow(
      "Toolset origin layer",
    );
    value.allocations[0]!.origins.toolsets["mcp@1"].layer = "run_labels";
    Object.assign(value.allocations[0]!.origins.toolsets["mcp@1"], {
      headers: { Authorization: "secret" },
    });
    expect(() => safeStageRuntimeConfiguration(value)).toThrow(
      "response shape",
    );
  });

  it("fails closed on unexpected secret-like fields", () => {
    expect(() =>
      safeRunRuntimeConfiguration({
        default: {
          label: "default",
          bindingRevision: "1",
          config: { name: "empty", version: "1", digest },
          token: "must-not-enter-query-state",
        },
        labels: [],
      }),
    ).toThrow("shape");
    expect(() =>
      safeStageRuntimeConfiguration({
        allocations: [
          {
            logicalAgent: "reviewer",
            agentLabels: [],
            runtimeAdapters: [],
            origins: {},
            status: "pinned",
            runtimeSettings: { token: "must-not-render" },
          },
        ],
      }),
    ).toThrow("shape");
  });
});
