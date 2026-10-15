import { screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { RuntimeAgentObservation } from "../../api/operations";
import { renderApplication } from "../../test/application";
import { operationsSnapshot } from "../../test/builders-execution";
import { json, page } from "../../test/responses";

function agent(
  instanceId: string,
  slotState: RuntimeAgentObservation["slotState"],
  extra: Partial<RuntimeAgentObservation> = {},
): RuntimeAgentObservation {
  return {
    instanceId,
    softwareVersion: "0.1.0",
    supportedRuntimes: ["adk@1"],
    supportedToolsets: [],
    supportedSandboxProfiles: ["local-workdir@1"],
    supportedRuntimeAdapters: [],
    observedState: slotState === "fenced" ? "fenced" : "idle",
    slotState,
    ...extra,
  };
}

function renderOverview(runtimeAgents: RuntimeAgentObservation[]) {
  renderApplication("/operations", {
    server: {
      "GET /v1/operations/snapshot": () =>
        json(operationsSnapshot("4", { runtimeAgents })),
      "GET /v1/projects": () => json(page([])),
    },
  });
}

describe("Operations overview", () => {
  it("counts every slot state on its own and keeps idle apart from capacity", async () => {
    renderOverview([
      agent("runtime-idle", "idle"),
      agent("runtime-busy", "busy"),
      agent("runtime-reserved-a", "reserved"),
      agent("runtime-reserved-b", "reserved"),
      agent("runtime-draining", "draining"),
      agent("runtime-fenced", "fenced", {
        currentAllocationId: "allocation-observed",
        authoritativeAllocationId: "allocation-authoritative",
      }),
    ]);
    expect(
      await screen.findByRole("heading", { name: "Execution readiness" }),
    ).toBeInTheDocument();
    const legend = screen.getByRole("list", { name: "Slots by state" });
    expect(
      within(legend)
        .getAllByRole("listitem")
        .map((item) => item.textContent),
    ).toEqual(["Idle 1", "Busy 1", "Reserved 2", "Draining 1", "Fenced 1"]);
    expect(
      screen.getByRole("img", {
        name: "6 Runtime Agent slots: 1 idle, 1 busy, 2 reserved, 1 draining, 1 fenced",
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "Idle slots do not guarantee capacity for a particular Run.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", {
        name: /^Runtime Agents 6 deployed processes/,
      }),
    ).toHaveAttribute("href", "/operations/runtime-agents");
    expect(
      screen.getByRole("link", {
        name: /^Reconciliation 1 visible mismatch /i,
      }),
    ).toHaveAttribute("href", "/operations/runtime-agents");
    expect(
      screen.queryByRole("button", { name: /force|release|reassign/i }),
    ).not.toBeInTheDocument();
  });

  it("says when the snapshot holds no Runtime process", async () => {
    renderOverview([]);
    expect(
      await screen.findByText(/No Runtime processes are present/),
    ).toBeInTheDocument();
    expect(screen.getByText("0 slots")).toBeInTheDocument();
    expect(screen.queryByRole("img", { name: /slots:/ })).toBeNull();
    const diagnostics = screen
      .getByText("Diagnostics: snapshot and live connection")
      .closest("details");
    expect(diagnostics).not.toHaveAttribute("open");
    expect(
      within(diagnostics as HTMLElement).getByText("operations-generation-1"),
    ).toBeInTheDocument();
  });
});
