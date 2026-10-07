import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";

import { PublicAPI } from "../../../api/client";
import { Application } from "../../../app/application";
import { applicationRoutes } from "../../../app/router";
import type { RuntimeConfig } from "../../../config/runtime-config";
import { RunEventsManager } from "../../../events/run-events";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};
const digest = `sha256:${"5".repeat(64)}`;

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

const gateway = {
  ref: { kind: "llm-gateways", name: "local-litellm", version: "1", digest },
  body: { protocol: "openai-compatible@1", url: "http://127.0.0.1:4000/v1" },
  source: "operator",
};
const policy = {
  ref: { kind: "model-policies", name: "worker", version: "1", digest },
  body: { model: "qwen-worker" },
  source: "managed",
};

function setup(path: string) {
  const reads: string[] = [];
  const api = new PublicAPI(
    runtimeConfig,
    vi.fn(async (input) => {
      const request = input instanceof Request ? input : new Request(input);
      const url = new URL(request.url);
      if (url.pathname === "/v1/auth/session")
        return apiResponse({
          principal: {
            userId: "user_local",
            username: "owner",
            capabilities: ["user", "operations"],
          },
          csrfToken: "a".repeat(43),
          idleExpiresAt: "2099-08-31T20:00:00Z",
          absoluteExpiresAt: "2099-09-01T12:00:00Z",
        });
      if (url.pathname === "/v1/operations/snapshot")
        return apiResponse({
          cursor: { generation: "configurations", revision: "1" },
          runtimeAgents: [],
          allocations: [],
        });
      if (url.pathname.startsWith("/v1/configurations/")) {
        reads.push(url.pathname);
        if (url.pathname === "/v1/configurations/llm-gateways")
          return apiResponse({ items: [gateway], page: { hasMore: false } });
        if (url.pathname === "/v1/configurations/model-policies")
          return apiResponse({ items: [policy], page: { hasMore: false } });
        if (
          url.pathname ===
          "/v1/configurations/llm-gateways/local-litellm/versions/1"
        )
          return apiResponse(gateway);
        return apiResponse({ items: [], page: { hasMore: false } });
      }
      if (url.pathname === "/v1/projects")
        return apiResponse({ items: [], page: { hasMore: false } });
      throw new Error(`unexpected ${request.method} ${request.url}`);
    }),
  );
  const router = createMemoryRouter(applicationRoutes(), {
    initialEntries: [path],
  });
  render(
    <Application
      api={api}
      publicAPI={api}
      runEvents={
        new RunEventsManager(runtimeConfig.apiBaseUrl, {
          WebSocketImplementation: IdleWebSocket as unknown as typeof WebSocket,
        })
      }
      router={router}
    />,
  );
  return { router, reads };
}

describe("LLM configurations", () => {
  it("keeps the chosen kind in the address and lists that kind", async () => {
    const { router, reads } = setup(
      "/operations/configurations?kind=llm-gateways",
    );
    expect(
      await screen.findByText("local-litellm@1", { selector: "strong" }),
    ).toBeInTheDocument();
    const kinds = screen.getByRole("group", { name: "Kind" });
    expect(
      within(kinds)
        .getAllByRole("button")
        .map((button) => button.textContent),
    ).toEqual([
      "Model policies",
      "LLM gateways",
      "Agent templates",
      "Execution configs",
    ]);
    expect(
      within(kinds).getByRole("button", { name: "LLM gateways" }),
    ).toHaveAttribute("aria-pressed", "true");
    expect(reads).toEqual(["/v1/configurations/llm-gateways"]);
    expect(
      screen.getByRole("link", { name: "Inspect / clone" }),
    ).toHaveAttribute(
      "href",
      "/operations/configurations/llm-gateways/local-litellm/1",
    );

    await userEvent
      .setup()
      .click(within(kinds).getByRole("button", { name: "Model policies" }));
    expect(
      await screen.findByText("worker@1", { selector: "strong" }),
    ).toBeInTheDocument();
    expect(router.state.location.search).toBe("");
    expect(reads).toContain("/v1/configurations/model-policies");
  });

  it("leads back to the list of the same kind from a version", async () => {
    setup("/operations/configurations/llm-gateways/local-litellm/1");
    expect(
      await screen.findByRole("heading", { name: "Published values" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "local-litellm@1" }),
    ).toBeInTheDocument();
    const breadcrumb = screen.getByRole("navigation", { name: "Breadcrumb" });
    expect(
      within(breadcrumb).getByRole("link", { name: "LLM gateways" }),
    ).toHaveAttribute("href", "/operations/configurations?kind=llm-gateways");
    expect(
      within(breadcrumb).getByRole("link", { name: "LLM configurations" }),
    ).toHaveAttribute("href", "/operations/configurations");
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Clone to new version" }),
      ).toBeEnabled(),
    );
  });
});
