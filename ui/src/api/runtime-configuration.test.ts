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
            runtimeAdapters: ["otlp-http@1"],
            origins: {
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
          origins: { workerTelemetry: { layer: "agent_labels" } },
        },
      ],
    });
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
