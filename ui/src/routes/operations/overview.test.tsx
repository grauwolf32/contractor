import { render, screen, within } from "@testing-library/react";
import { createMemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";

import { PublicAPI } from "../../api/client";
import type { RuntimeAgentObservation } from "../../api/operations";
import { Application } from "../../app/application";
import { applicationRoutes } from "../../app/router";
import type { RuntimeConfig } from "../../config/runtime-config";
import { RunEventsManager } from "../../events/run-events";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

const session = {
  principal: {
    userId: "user_local",
    username: "owner",
    capabilities: ["user", "operations"],
  },
  csrfToken: "a".repeat(43),
  idleExpiresAt: "2099-08-31T20:00:00Z",
  absoluteExpiresAt: "2099-09-01T12:00:00Z",
};

function apiResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "content-type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
    },
  });
}

class IdleWebSocket {
  readyState = 0;
  onopen = null;
  onmessage = null;
  onerror = null;
  onclose = null;
  send(): void {}
  close(): void {}
}

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
  const api = new PublicAPI(
    runtimeConfig,
    vi.fn(async (input) => {
      const request = input instanceof Request ? input : new Request(input);
      const path = new URL(request.url).pathname;
      if (path === "/v1/auth/session") return apiResponse(session);
      if (path === "/v1/operations/snapshot")
        return apiResponse({
          cursor: { generation: "overview-generation", revision: "4" },
          runtimeAgents,
          allocations: [],
        });
      if (path === "/v1/projects")
        return apiResponse({ items: [], page: { hasMore: false } });
      throw new Error(`unexpected ${request.method} ${request.url}`);
    }),
  );
  const router = createMemoryRouter(applicationRoutes(), {
    initialEntries: ["/operations"],
  });
  const events = new RunEventsManager(runtimeConfig.apiBaseUrl, {
    WebSocketImplementation: IdleWebSocket as unknown as typeof WebSocket,
  });
  render(
    <Application
      api={api}
      publicAPI={api}
      runEvents={events}
      router={router}
    />,
  );
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
      within(diagnostics as HTMLElement).getByText("overview-generation"),
    ).toBeInTheDocument();
  });
});
