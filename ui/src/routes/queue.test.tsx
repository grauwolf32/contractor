import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { PublicAPI } from "../api/client";
import { Application } from "../app/application";
import { applicationRoutes } from "../app/router";
import type { RuntimeConfig } from "../config/runtime-config";
import { RunEventsManager } from "../events/run-events";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

const session = {
  principal: {
    userId: "user_local",
    username: "owner",
    capabilities: ["user", "operations"] as const,
  },
  csrfToken: "a".repeat(43),
  idleExpiresAt: "2026-09-05T20:00:00Z",
  absoluteExpiresAt: "2026-09-06T12:00:00Z",
};

function apiResponse(value: unknown): Response {
  return new Response(JSON.stringify(value), {
    headers: {
      "content-type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
    },
  });
}

function queueItem(
  runId: string,
  state: "initializing" | "running" | "cancelling" = "running",
) {
  return {
    runId,
    workflow: "openapi-from-workspace@4",
    state,
    labels: {},
    eventCursor: { generation: `events-${runId}`, sequence: "1" },
    createdAt: "2026-09-05T08:00:00Z",
    updatedAt: "2026-09-05T08:01:00Z",
  };
}

class QueueWebSocket {
  static instances: QueueWebSocket[] = [];

  protocol = "contractor.events.v1";
  readyState = 0;
  sent: string[] = [];
  onopen: ((event: Event) => unknown) | null = null;
  onmessage: ((event: MessageEvent) => unknown) | null = null;
  onerror: ((event: Event) => unknown) | null = null;
  onclose: ((event: CloseEvent) => unknown) | null = null;

  constructor(
    readonly url: string | URL,
    readonly protocols?: string | string[],
  ) {
    QueueWebSocket.instances.push(this);
  }

  send(value: string): void {
    this.sent.push(value);
  }

  close(code?: number): void {
    this.readyState = 3;
    this.onclose?.({ code: code ?? 1000 } as CloseEvent);
  }

  open(): void {
    this.readyState = 1;
    this.onopen?.(new Event("open"));
  }

  message(value: unknown): void {
    this.onmessage?.({ data: JSON.stringify(value) } as MessageEvent);
  }
}

function renderQueueApplication(api: PublicAPI, path = "/runs") {
  const router = createMemoryRouter(applicationRoutes(), {
    initialEntries: [path],
  });
  const events = new RunEventsManager(runtimeConfig.apiBaseUrl, {
    WebSocketImplementation: QueueWebSocket as unknown as typeof WebSocket,
  });
  return {
    ...render(
      <Application
        api={api}
        publicAPI={api}
        runEvents={events}
        router={router}
      />,
    ),
    router,
  };
}

beforeEach(() => {
  QueueWebSocket.instances = [];
});

describe("Runs Queue view", () => {
  it("groups all active Run contexts and keeps filters across stable pages", async () => {
    const requests: URL[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") {
          return apiResponse(session);
        }
        if (url.pathname !== "/v1/queue") {
          throw new Error(`unexpected ${request.method} ${url.pathname}`);
        }
        requests.push(url);
        if (url.searchParams.get("cursor") === "next-queue") {
          return apiResponse({
            items: [queueItem("run-next", "cancelling")],
            page: { hasMore: false },
          });
        }
        if (url.searchParams.get("membership") === "project") {
          return apiResponse({
            items: [
              {
                ...queueItem("run-project"),
                project: {
                  projectId: "project-payment",
                  name: "Payment service",
                  kind: "project",
                },
              },
            ],
            page: { hasMore: false },
          });
        }
        return apiResponse({
          items: [
            queueItem("run-standalone", "initializing"),
            {
              ...queueItem("run-project"),
              project: {
                projectId: "project-payment",
                name: "Payment service",
                kind: "project",
              },
            },
            {
              ...queueItem("run-eval"),
              labels: { "eval.leg": "a", purpose: "eval" },
              project: {
                projectId: "evaluation-openapi",
                name: "OpenAPI regression",
                kind: "evaluation",
              },
            },
          ],
          page: { hasMore: true, nextCursor: "next-queue" },
        });
      }),
    );
    renderQueueApplication(api);

    expect(
      await screen.findByRole("link", { name: "run-standalone" }),
    ).toBeInTheDocument();
    expect(screen.getByText("No Project")).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: "Payment service" }),
    ).toHaveAttribute("href", "/projects/project-payment");
    expect(screen.getByText("OpenAPI regression")).toBeInTheDocument();
    expect(screen.getByText("eval.leg").closest("code")).toHaveTextContent(
      "eval.leg=a",
    );
    expect(
      screen.queryByRole("columnheader", { name: /^(position|rank|eta)$/i }),
    ).not.toBeInTheDocument();
    const views = screen.getByRole("navigation", { name: "Run views" });
    expect(within(views).getByRole("link", { name: /Queue/ })).toHaveAttribute(
      "aria-current",
      "page",
    );
    expect(
      within(views).getByRole("link", { name: /Completed/ }),
    ).toHaveAttribute("href", "/runs?view=completed");
    const primary = screen.getByRole("navigation", {
      name: "Primary navigation",
    });
    expect(within(primary).queryByRole("link", { name: "Queue" })).toBeNull();
    expect(within(primary).getByRole("link", { name: "Runs" })).toHaveAttribute(
      "aria-current",
      "page",
    );

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(
      await screen.findByRole("link", { name: "run-next" }),
    ).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText("Context"), "project");
    expect(
      await screen.findByRole("link", { name: "run-project" }),
    ).toBeInTheDocument();
    expect(requests.at(-1)?.searchParams.get("membership")).toBe("project");
    expect(requests.at(-1)?.searchParams.has("cursor")).toBe(false);

    await user.selectOptions(screen.getByLabelText("State"), "running");
    await waitFor(() =>
      expect(requests.at(-1)?.searchParams.get("state")).toBe("running"),
    );
  });

  it("removes a terminal Run after its lifecycle event", async () => {
    let terminal = false;
    let queueReads = 0;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") {
          return apiResponse(session);
        }
        if (url.pathname === "/v1/queue") {
          queueReads += 1;
          return apiResponse({
            items: terminal ? [] : [queueItem("run-live")],
            page: { hasMore: false },
          });
        }
        throw new Error(`unexpected ${request.method} ${url.pathname}`);
      }),
    );
    renderQueueApplication(api);

    expect(
      await screen.findByRole("link", { name: "run-live" }),
    ).toBeInTheDocument();
    await waitFor(() => expect(QueueWebSocket.instances).toHaveLength(1));
    const socket = QueueWebSocket.instances[0];
    act(() => socket?.open());
    const subscription = JSON.parse(socket?.sent[0] ?? "{}");
    expect(subscription).toMatchObject({
      stream: { kind: "run", id: "run-live" },
      after: { generation: "events-run-live", sequence: "1" },
    });
    act(() =>
      socket?.message({
        version: "contractor.events.v1",
        type: "subscribed",
        subscriptionId: subscription.subscriptionId,
        stream: subscription.stream,
        cursor: subscription.after,
      }),
    );

    terminal = true;
    act(() =>
      socket?.message({
        version: "contractor.events.v1",
        type: "event",
        subscriptionId: subscription.subscriptionId,
        stream: subscription.stream,
        cursor: { generation: "events-run-live", sequence: "2" },
        kind: "lifecycle.changed",
        occurredAt: "2026-09-05T08:02:00Z",
        data: { runId: "run-live", resource: "run", state: "succeeded" },
      }),
    );

    expect(
      await screen.findByText("No active Runs match this view."),
    ).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "run-live" })).toBeNull();
    expect(queueReads).toBeGreaterThan(1);
  });

  it("redirects legacy Queue links and preserves active filters", async () => {
    const requests: URL[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") {
          return apiResponse(session);
        }
        if (url.pathname === "/v1/queue") {
          requests.push(url);
          return apiResponse({ items: [], page: { hasMore: false } });
        }
        throw new Error(`unexpected ${request.method} ${url.pathname}`);
      }),
    );
    const { router } = renderQueueApplication(
      api,
      "/queue?membership=project&state=running",
    );

    expect(
      await screen.findByText("No active Runs match this view."),
    ).toBeInTheDocument();
    expect(router.state.location.pathname).toBe("/runs");
    expect(router.state.location.search).toBe(
      "?membership=project&state=running",
    );
    expect(requests).toHaveLength(1);
    expect(requests[0]?.searchParams.get("membership")).toBe("project");
    expect(requests[0]?.searchParams.get("state")).toBe("running");
  });
});
