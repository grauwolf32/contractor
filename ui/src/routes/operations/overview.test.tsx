import { screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { RuntimeAgentObservation } from "../../api/operations";
import { renderApplication } from "../../test/application";
import { operationsSnapshot } from "../../test/builders-execution";
import { json, page } from "../../test/responses";

function agent(
  instanceId: string,
  slotState: RuntimeAgentObservation["state"],
  extra: Partial<RuntimeAgentObservation> = {},
): RuntimeAgentObservation {
  return {
    allocationSlots: 1,
    availableSlots: Number(slotState === "idle"),
    busySlots: Number(slotState === "busy"),
    reservedSlots: Number(slotState === "reserved"),
    drainingSlots: Number(slotState === "draining"),
    fencedSlots: Number(slotState === "fenced"),
    observedAllocations: [],
    authoritativeAllocationIds: [],
    pendingReleaseAllocationIds: [],
    instanceId,
    softwareVersion: "0.1.0",
    supportedRuntimes: ["adk@1"],
    supportedToolsets: [],
    supportedSandboxProfiles: ["local-workdir@1"],
    supportedRuntimeAdapters: [],
    observedState: slotState === "fenced" ? "fenced" : "idle",
    state: slotState,
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
        observedAllocations: [
          { allocationId: "allocation-observed", observedState: "allocated" },
        ],
        authoritativeAllocationIds: ["allocation-authoritative"],
      }),
    ]);
    expect(
      await screen.findByRole("heading", { name: "Execution slots" }),
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
        name: /^Runtime processes\s*6/,
      }),
    ).toHaveAttribute("href", "/operations/runtime-agents");
    expect(
      screen.getByRole("link", {
        name: /^Reconciliation\s*1/i,
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

  it("bounds the Overview preview while counting every process", async () => {
    renderOverview(
      Array.from({ length: 120 }, (_, index) =>
        agent(`runtime-${index}`, "reserved", {
          authoritativeAllocationIds: [`allocation-${index}`],
        }),
      ),
    );
    const processes = await screen.findByRole("region", {
      name: "Runtime processes",
    });
    expect(processes.querySelectorAll("details")).toHaveLength(6);
    expect(
      screen.getByRole("link", { name: /^Runtime processes\s*120/ }),
    ).toHaveAttribute("href", "/operations/runtime-agents");
    expect(
      screen.getByRole("link", { name: /^Reconciliation\s*120/ }),
    ).toBeInTheDocument();
    const inspection = screen.getByRole("region", {
      name: "Needs inspection",
    });
    expect(
      within(inspection).getAllByRole("link", { name: /^runtime-/ }),
    ).toHaveLength(4);
    expect(
      within(inspection).getByRole("link", { name: "runtime-0" }),
    ).toHaveAttribute("href", "/operations#process-runtime-0");
    expect(
      within(inspection).getByRole("link", {
        name: "Inspect Runtime Agents →",
      }),
    ).toHaveAttribute("href", "/operations/runtime-agents");
  });

  it("links an inspection target beyond the preview to the full inventory", async () => {
    renderOverview([
      ...Array.from({ length: 6 }, (_, index) =>
        agent(`idle-${index}`, "idle"),
      ),
      agent("runtime-outside-preview", "reserved", {
        authoritativeAllocationIds: ["allocation-pending"],
      }),
    ]);
    const inspection = await screen.findByRole("region", {
      name: "Needs inspection",
    });
    expect(
      within(inspection).getByRole("link", { name: "runtime-outside-preview" }),
    ).toHaveAttribute(
      "href",
      "/operations/runtime-agents#process-runtime-outside-preview",
    );
    expect(
      screen
        .getByRole("region", { name: "Runtime processes" })
        .querySelector("#process-runtime-outside-preview"),
    ).toBeNull();
  });
});
