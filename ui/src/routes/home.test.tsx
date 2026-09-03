import { render, screen, within } from "@testing-library/react";
import { createMemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";

import { PublicAPI, type AuthSession } from "../api/client";
import { Application } from "../app/application";
import { applicationRoutes } from "../app/router";
import type { RuntimeConfig } from "../config/runtime-config";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

const session: AuthSession = {
  principal: {
    userId: "user_local",
    username: "owner",
    capabilities: ["user", "operations"],
  },
  csrfToken: "a".repeat(43),
  idleExpiresAt: "2026-09-02T20:00:00Z",
  absoluteExpiresAt: "2026-09-03T12:00:00Z",
};

function apiResponse(value: unknown): Response {
  return new Response(JSON.stringify(value), {
    headers: {
      "content-type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
    },
  });
}

function workflow(name: string, version: string) {
  return {
    ref: { name, version },
    entryStage: "execute",
    parameters: { objective: { required: true } },
    inputs: {
      source: { required: true, mediaTypes: ["application/zip"] },
    },
    outputs: {
      report: { required: true, mediaTypes: ["text/markdown"] },
    },
  };
}

describe("Action center", () => {
  it("combines recent work, active Runs, Runtime health, and quick starts", async () => {
    const requests: URL[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        requests.push(url);
        if (url.pathname === "/v1/auth/session") {
          return apiResponse(session);
        }
        if (url.pathname === "/v1/runs") {
          const state = url.searchParams.get("state");
          if (state === "running") {
            return apiResponse({
              items: [
                {
                  runId: "run-active",
                  workflow: "review@2",
                  state: "running",
                  labels: {},
                  createdAt: "2026-09-02T10:00:00Z",
                  updatedAt: "2026-09-02T10:02:00Z",
                },
              ],
              page: { hasMore: false },
            });
          }
          if (state === "initializing" || state === "cancelling") {
            return apiResponse({ items: [], page: { hasMore: false } });
          }
          return apiResponse({
            items: [
              {
                runId: "run-failed",
                workflow: "review@2",
                state: "failed",
                labels: {},
                createdAt: "2026-09-02T09:00:00Z",
                updatedAt: "2026-09-02T09:01:00Z",
                finishedAt: "2026-09-02T09:01:00Z",
              },
              {
                runId: "run-success",
                workflow: "generate@1",
                state: "succeeded",
                labels: {},
                createdAt: "2026-09-02T08:00:00Z",
                updatedAt: "2026-09-02T08:05:00Z",
                finishedAt: "2026-09-02T08:05:00Z",
              },
            ],
            page: { hasMore: false },
          });
        }
        if (url.pathname === "/v1/workflows") {
          return apiResponse({
            items: [
              workflow("review", "1"),
              workflow("generate", "1"),
              workflow("review", "2"),
            ],
            page: { hasMore: false },
          });
        }
        if (url.pathname === "/v1/operations/snapshot") {
          return apiResponse({
            cursor: { generation: "operations-generation", revision: "7" },
            runtimeAgents: [
              {
                instanceId: "agent-idle",
                softwareVersion: "0.1.0",
                supportedRuntimes: ["adk@1"],
                supportedToolsets: [],
                supportedSandboxProfiles: ["default@1"],
                supportedRuntimeAdapters: [],
                observedState: "idle",
                slotState: "idle",
              },
              {
                instanceId: "agent-busy",
                softwareVersion: "0.1.0",
                supportedRuntimes: ["adk@1"],
                supportedToolsets: [],
                supportedSandboxProfiles: ["default@1"],
                supportedRuntimeAdapters: [],
                observedState: "allocated",
                slotState: "busy",
                currentAllocationId: "allocation-observed",
                authoritativeAllocationId: "allocation-authoritative",
                reconciliationReason: {
                  code: "allocation_mismatch",
                  retryable: true,
                },
              },
            ],
            allocations: [],
          });
        }
        throw new Error(`unexpected ${request.method} ${url}`);
      }),
    );
    const router = createMemoryRouter(applicationRoutes(), {
      initialEntries: ["/"],
    });
    const view = render(
      <Application api={api} publicAPI={api} router={router} />,
    );

    expect(
      await screen.findByRole("heading", { name: "Action center" }),
    ).toBeInTheDocument();
    expect(
      await screen.findByRole("link", { name: /review@2.*run-failed/i }),
    ).toHaveAttribute("href", "/runs/run-failed");
    expect(
      screen.getByRole("link", { name: /review@2.*run-active/i }),
    ).toHaveAttribute("href", "/runs/run-active");
    expect(
      screen.getByRole("link", { name: /generate@1.*run-success/i }),
    ).toHaveAttribute("href", "/runs/run-success");
    expect(
      await screen.findByRole("heading", { name: "Capacity needs attention" }),
    ).toBeInTheDocument();

    const health = view.container.querySelector(".runtime-health-panel");
    expect(health).not.toBeNull();
    expect(
      within(health as HTMLElement).getByText("Attention"),
    ).toBeInTheDocument();
    const quickStart = view.container.querySelector(".quick-start-panel");
    expect(quickStart).not.toBeNull();
    expect(
      within(quickStart as HTMLElement).getByRole("link", {
        name: /review.*@2/i,
      }),
    ).toHaveAttribute("href", "/workflows/review/2");
    expect(
      within(quickStart as HTMLElement).queryByRole("link", {
        name: /review.*@1/i,
      }),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Home" })).toHaveAttribute(
      "aria-current",
      "page",
    );
    expect(
      screen.getByRole("link", { name: "View failed Runs →" }),
    ).toHaveAttribute("href", "/runs?state=failed");
    expect(requests.filter((url) => url.pathname === "/v1/runs")).toHaveLength(
      4,
    );
  });
});
