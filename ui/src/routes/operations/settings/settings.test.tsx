import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";

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
const session = {
  principal: {
    userId: "user_local",
    username: "owner",
    capabilities: ["user", "operations"] as const,
  },
  csrfToken: "a".repeat(43),
  idleExpiresAt: "2026-09-06T20:00:00Z",
  absoluteExpiresAt: "2026-09-07T12:00:00Z",
};

function response(
  value: unknown,
  status = 200,
  headers: Record<string, string> = {},
): Response {
  return new Response(value === undefined ? undefined : JSON.stringify(value), {
    status,
    headers: {
      "content-type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
      ...headers,
    },
  });
}

function schedulerSettings(maxConcurrentRuns: number, revision: string) {
  return response(
    {
      maxConcurrentRuns,
      revision,
      updatedAt: `2026-09-06T01:00:0${revision}Z`,
    },
    200,
    { ETag: `"${revision}"`, "Cache-Control": "no-store" },
  );
}

function operationsSnapshot(revision = "7") {
  return {
    cursor: { generation: "operations-settings-generation", revision },
    runtimeAgents: [],
    allocations: [],
  };
}

class SettingsWebSocket {
  static instances: SettingsWebSocket[] = [];
  protocol = "contractor.events.v1";
  readyState = 0;
  sent: string[] = [];
  onopen: ((event: Event) => unknown) | null = null;
  onmessage: ((event: MessageEvent) => unknown) | null = null;
  onerror: ((event: Event) => unknown) | null = null;
  onclose: ((event: CloseEvent) => unknown) | null = null;

  constructor() {
    SettingsWebSocket.instances.push(this);
  }

  send(value: string): void {
    this.sent.push(value);
  }

  close(): void {
    this.readyState = 3;
  }

  open(): void {
    this.readyState = 1;
    this.onopen?.(new Event("open"));
  }

  message(value: unknown): void {
    this.onmessage?.({ data: JSON.stringify(value) } as MessageEvent);
  }
}

function renderSettings(api: PublicAPI) {
  const router = createMemoryRouter(applicationRoutes(), {
    initialEntries: ["/operations/settings"],
  });
  const events = new RunEventsManager(runtimeConfig.apiBaseUrl, {
    WebSocketImplementation: SettingsWebSocket as unknown as typeof WebSocket,
  });
  return render(
    <Application
      api={api}
      publicAPI={api}
      runEvents={events}
      router={router}
    />,
  );
}

function commonResponse(request: Request): Response | undefined {
  const path = new URL(request.url).pathname;
  if (path === "/v1/auth/session") {
    return response(session);
  }
  if (path === "/v1/operations/snapshot") {
    return response(operationsSnapshot());
  }
  if (path === "/v1/settings/git-key") {
    return response({ configured: false });
  }
  return undefined;
}

beforeEach(() => {
  SettingsWebSocket.instances = [];
});

describe("Operations Scheduler settings", () => {
  it("keeps personal repository settings available without Operations capability", async () => {
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const path = new URL(request.url).pathname;
        if (path === "/v1/auth/session") {
          return response({
            ...session,
            principal: {
              ...session.principal,
              capabilities: ["user"],
            },
          });
        }
        if (path === "/v1/settings/git-key") {
          return response({ configured: false });
        }
        throw new Error(`unexpected ${request.method} ${request.url}`);
      }),
    );

    renderSettings(api);

    expect(
      await screen.findByRole("heading", { name: "Git SSH key" }),
    ).toBeInTheDocument();
    expect(screen.getByText("1 configuration area")).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: "Workflow scheduling" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText("Operations capability required"),
    ).not.toBeInTheDocument();
  });

  it("validates, CAS-saves once and settles on the returned value", async () => {
    const requests: Request[] = [];
    let finishPut: ((value: Response) => void) | undefined;
    const pendingPut = new Promise<Response>((resolve) => {
      finishPut = resolve;
    });
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const common = commonResponse(request);
        if (common !== undefined) return common;
        if (
          new URL(request.url).pathname === "/v1/operations/settings/scheduler"
        ) {
          requests.push(request);
          return request.method === "GET"
            ? schedulerSettings(1, "1")
            : pendingPut;
        }
        throw new Error(`unexpected ${request.method} ${request.url}`);
      }),
    );
    renderSettings(api);
    const input = await screen.findByLabelText(
      "Maximum concurrent Workflow Runs",
    );
    expect(input).toHaveValue(1);
    expect(screen.getByText("Admission, not capacity.")).toBeInTheDocument();
    expect(
      screen.getByText(/one Workflow Run may require several agents/),
    ).toBeInTheDocument();

    fireEvent.change(input, { target: { value: "" } });
    expect(
      screen.getByText("Enter a maximum from 1 through 32."),
    ).toBeInTheDocument();
    expect(requests.filter((request) => request.method === "PUT")).toHaveLength(
      0,
    );
    fireEvent.change(input, { target: { value: "1.5" } });
    expect(screen.getByText(/must be a whole number/)).toBeInTheDocument();
    fireEvent.change(input, { target: { value: "33" } });
    expect(screen.getByText(/must be from 1 through 32/)).toBeInTheDocument();

    fireEvent.change(input, { target: { value: "2" } });
    const user = userEvent.setup();
    await user.click(
      screen.getByRole("button", { name: "Save scheduling limit" }),
    );
    expect(screen.getByRole("button", { name: "Saving…" })).toBeDisabled();
    await waitFor(() =>
      expect(
        requests.filter((request) => request.method === "PUT"),
      ).toHaveLength(1),
    );
    const put = requests.find((request) => request.method === "PUT");
    expect(put?.headers.get("If-Match")).toBe('"1"');
    await expect(put?.json()).resolves.toEqual({ maxConcurrentRuns: 2 });

    finishPut?.(schedulerSettings(2, "2"));
    expect(
      await screen.findByText("Saved maximum 2 at revision 2."),
    ).toBeInTheDocument();
    expect(input).toHaveValue(2);
    expect(
      screen.getByRole("button", { name: "Save scheduling limit" }),
    ).toBeDisabled();
  });

  it("reloads after a 412 and preserves the unsaved draft without overwriting", async () => {
    let saved = 1;
    let revision = "1";
    let puts = 0;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const common = commonResponse(request);
        if (common !== undefined) return common;
        if (
          new URL(request.url).pathname !== "/v1/operations/settings/scheduler"
        ) {
          throw new Error(`unexpected ${request.method} ${request.url}`);
        }
        if (request.method === "GET") return schedulerSettings(saved, revision);
        puts += 1;
        saved = 3;
        revision = "2";
        return response(
          {
            code: "precondition_failed",
            message: "resource revision precondition failed",
            retryable: false,
            requestId: "request-conflict",
          },
          412,
        );
      }),
    );
    renderSettings(api);
    const input = await screen.findByLabelText(
      "Maximum concurrent Workflow Runs",
    );
    fireEvent.change(input, { target: { value: "2" } });
    await userEvent.click(
      screen.getByRole("button", { name: "Save scheduling limit" }),
    );

    expect(
      await screen.findByText(/changed on the Server|Another operator/),
    ).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("3")).toBeInTheDocument());
    expect(input).toHaveValue(2);
    expect(puts).toBe(1);
    expect(
      screen.getByRole("button", { name: "Save scheduling limit" }),
    ).toBeDisabled();

    await userEvent.click(
      screen.getByRole("button", { name: "Reset to saved value" }),
    );
    expect(input).toHaveValue(3);
    expect(
      screen.queryByText(/changed on the Server|Another operator/),
    ).not.toBeInTheDocument();
  });

  it("refetches the singleton on typed Operations invalidation and live reconnect", async () => {
    let saved = 1;
    let revision = "1";
    let settingsReads = 0;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const common = commonResponse(request);
        if (common !== undefined) return common;
        if (
          new URL(request.url).pathname === "/v1/operations/settings/scheduler"
        ) {
          settingsReads += 1;
          return schedulerSettings(saved, revision);
        }
        throw new Error(`unexpected ${request.method} ${request.url}`);
      }),
    );
    renderSettings(api);
    expect(
      await screen.findByRole("heading", { name: "Workflow scheduling" }),
    ).toBeInTheDocument();
    await waitFor(() => expect(SettingsWebSocket.instances).toHaveLength(1));
    const socket = SettingsWebSocket.instances[0];
    act(() => socket?.open());
    const subscription = JSON.parse(socket?.sent[0] ?? "{}");
    act(() =>
      socket?.message({
        version: "contractor.events.v1",
        type: "subscribed",
        subscriptionId: subscription.subscriptionId,
        stream: { kind: "operations" },
        cursor: subscription.after,
      }),
    );
    await waitFor(() => expect(settingsReads).toBeGreaterThanOrEqual(2));

    saved = 4;
    revision = "2";
    act(() =>
      socket?.message({
        version: "contractor.events.v1",
        type: "event",
        subscriptionId: subscription.subscriptionId,
        stream: { kind: "operations" },
        cursor: {
          generation: "operations-settings-generation",
          sequence: "8",
        },
        kind: "operations.changed",
        occurredAt: "2026-09-06T01:00:02Z",
        data: { resource: "schedulerSettings", revision: "8" },
      }),
    );
    await waitFor(() =>
      expect(
        screen.getByLabelText("Maximum concurrent Workflow Runs"),
      ).toHaveValue(4),
    );
    expect(screen.getByText("4")).toBeInTheDocument();
  });
});
