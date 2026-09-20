import { describe, expect, it } from "vitest";

import {
  safeRunRuntimeConfiguration,
  safeStageRuntimeConfiguration,
} from "./runtime-configuration";
import {
  safeRuntimeChangedSections,
  safeToolsetConfigurationCoverage,
  safeToolsetConfigurationRequirements,
} from "./toolset-configuration";

const digest = `sha256:${"1".repeat(64)}`;

describe("safe Runtime configuration projections", () => {
  it("retains configured digests, explicit clears and all mixed sections", () => {
    expect(
      safeToolsetConfigurationCoverage([
        { ref: "mcp@1", state: "configured", valueDigest: digest },
        { ref: "test-native@2", state: "cleared" },
      ]),
    ).toEqual([
      { ref: "mcp@1", state: "configured", valueDigest: digest },
      { ref: "test-native@2", state: "cleared" },
    ]);
    expect(
      safeRuntimeChangedSections([
        "planner.telemetry",
        "worker.llmGateway",
        "worker.toolsets",
      ]),
    ).toHaveLength(3);
  });

  it.each(
    [
      [{ ref: "mcp@1", state: "configured" }],
      [{ ref: "mcp@1", state: "cleared", valueDigest: digest }],
      [{ ref: "mcp@1", state: "inherited" }],
      [
        {
          ref: "mcp@1",
          state: "configured",
          valueDigest: digest,
          credential: "private",
        },
      ],
      [{ ref: "mcp@1", state: "configured", valueDigest: "not-a-digest" }],
      [
        { ref: "mcp@1", state: "cleared" },
        { ref: "mcp@1", state: "cleared" },
      ],
      null,
      Array.from({ length: 33 }, () => ({ ref: "mcp@1", state: "cleared" })),
    ].map((value) => ({ value })),
  )("rejects malformed/unknown/unbounded coverage %#", ({ value }) => {
    expect(() => safeToolsetConfigurationCoverage(value)).toThrow(
      "projection is invalid",
    );
  });

  it("rejects unknown/duplicate effects and oversized requirements", () => {
    expect(() => safeRuntimeChangedSections(["worker.secret"])).toThrow();
    expect(() =>
      safeRuntimeChangedSections(["worker.toolsets", "worker.toolsets"]),
    ).toThrow();
    expect(() =>
      safeToolsetConfigurationRequirements(
        Array.from({ length: 33 }, () => ({})),
      ),
    ).toThrow();
  });
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
