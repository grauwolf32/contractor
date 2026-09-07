import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { PublicAPI } from "../../api/client";
import type { RunStatus } from "../../api/runs";
import { Application } from "../../app/application";
import { applicationRoutes } from "../../app/router";
import type { RuntimeConfig } from "../../config/runtime-config";
import { RunEventsManager } from "../../events/run-events";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

const digest = `sha256:${"1".repeat(64)}`;
const session = {
  principal: {
    userId: "user_local",
    username: "owner",
    capabilities: ["user", "operations"] as const,
  },
  csrfToken: "a".repeat(43),
  idleExpiresAt: "2026-08-31T20:00:00Z",
  absoluteExpiresAt: "2026-09-01T12:00:00Z",
};

const executionConfig = {
  variant: "failed_escalation" as const,
  escalationOrdinal: 1,
  ref: { configId: "strong-review", version: "2", digest },
  planner: {
    modelPolicy: { policyId: "planner-strong", version: "2", digest },
    llmGateway: { gatewayId: "local", version: "1", digest },
    credential: { credentialId: "planner-budget" },
    origins: {
      modelPolicy: "executionConfig:strong-review@2",
      llmGateway: "workflow:router-analysis@1",
      credential: "run override",
    },
  },
  agents: {
    reviewer: {
      modelPolicy: { policyId: "worker-strong", version: "2", digest },
      llmGateway: { gatewayId: "local", version: "1", digest },
      credential: { credentialId: "worker-budget" },
      origins: {
        modelPolicy: "executionConfig:strong-review@2",
        llmGateway: "workflow:router-analysis@1",
        credential: "run override",
      },
    },
  },
};

const basePlan = {
  revision: 1,
  subtasks: [
    {
      id: "0",
      objective: "Inspect the source tree",
      instructions: "Use bounded source search.",
      status: "pending" as const,
    },
  ],
  currentSubtaskId: "0",
};

type RunOverrides = Omit<
  Partial<RunStatus>,
  "eventCursor" | "activeStageExecutionId"
> & {
  eventCursor?: RunStatus["eventCursor"] | undefined;
  activeStageExecutionId?: RunStatus["activeStageExecutionId"] | undefined;
};

function runFixture(overrides: RunOverrides = {}): RunStatus {
  const result: RunStatus = {
    runId: "run-router",
    workflow: "router-analysis@1",
    state: "running",
    deletable: false,
    runtimeLabels: [],
    labels: {
      "eval.id": "eval-router-01",
      "eval.leg": "a",
      purpose: "eval",
    },
    runtimeConfiguration: {
      default: {
        label: "default",
        bindingRevision: "1",
        config: {
          name: "contractor-empty",
          version: "1",
          digest:
            "sha256:80a1754c01f8443c29fdc8f650a2254b2461694819918b204a55a7ad3425dc5f",
        },
      },
      labels: [],
    },
    parameters: { objective: "Review the project architecture" },
    inputs: {
      source: {
        namespace: "inputs",
        name: "source",
        revision: "input-r1",
      },
    },
    attempts: [
      {
        stageExecutionId: "stage-router-1",
        stage: "analysis",
        objective: "Produce a global architecture review.",
        attempt: 1,
        executionConfig,
        state: "running",
        plan: basePlan,
        createdAt: "2026-08-31T12:00:00Z",
        updatedAt: "2026-08-31T12:01:00Z",
        plannerStartedAt: "2026-08-31T12:00:01Z",
      },
    ],
    transitions: [],
    outputs: {},
    outputPublications: [],
    eventCursor: { generation: "run-generation-1", sequence: "10" },
    activeStageExecutionId: "stage-router-1",
    createdAt: "2026-08-31T12:00:00Z",
    updatedAt: "2026-08-31T12:01:00Z",
    startedAt: "2026-08-31T12:00:00Z",
  };
  Object.assign(result, overrides);
  if (
    Object.prototype.hasOwnProperty.call(overrides, "eventCursor") &&
    overrides.eventCursor === undefined
  ) {
    delete result.eventCursor;
  }
  if (
    Object.prototype.hasOwnProperty.call(overrides, "activeStageExecutionId") &&
    overrides.activeStageExecutionId === undefined
  ) {
    delete result.activeStageExecutionId;
  }
  return result;
}

function apiResponse(value: unknown, options: ResponseInit = {}): Response {
  const headers = new Headers(options.headers);
  headers.set("content-type", "application/json");
  headers.set("X-Contractor-API-Version", "contractor.public.v1");
  return new Response(JSON.stringify(value), { ...options, headers });
}

function byteResponse(value: string, mediaType: string): Response {
  return new Response(value, {
    headers: {
      "content-type": mediaType,
      "content-length": String(new TextEncoder().encode(value).byteLength),
      "X-Contractor-API-Version": "contractor.public.v1",
    },
  });
}

class RouteWebSocket {
  static instances: RouteWebSocket[] = [];

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
    RouteWebSocket.instances.push(this);
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

function renderRunApplication(api: PublicAPI, path: string) {
  const router = createMemoryRouter(applicationRoutes(), {
    initialEntries: [path],
  });
  const events = new RunEventsManager(runtimeConfig.apiBaseUrl, {
    WebSocketImplementation: RouteWebSocket as unknown as typeof WebSocket,
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
    events,
  };
}

function sessionOrArtifacts(request: Request): Response | undefined {
  const url = new URL(request.url);
  if (url.pathname === "/v1/auth/session") {
    return apiResponse(session);
  }
  if (url.pathname === "/v1/runs/run-router/artifacts") {
    return apiResponse({ items: [], page: { hasMore: false } });
  }
  return undefined;
}

beforeEach(() => {
  RouteWebSocket.instances = [];
});

describe("Run routes", () => {
  it("continues a failed stage only after confirmation and refreshes the authoritative Run", async () => {
    let posts = 0;
    let requestBody: unknown;
    const failed = { ...runFixture().attempts[0]!, state: "failed" as const };
    let current = runFixture({
      state: "failed",
      resumeStageExecutionId: failed.stageExecutionId,
      attempts: [failed],
      activeStageExecutionId: undefined,
    });
    const api = new PublicAPI(runtimeConfig, async (input, init) => {
      const request = new Request(input, init);
      const common = sessionOrArtifacts(request);
      if (common !== undefined) return common;
      const path = new URL(request.url).pathname;
      if (path === "/v1/runs/run-router/resume" && request.method === "POST") {
        posts++;
        requestBody = await request.json();
        current = runFixture({
          attempts: [
            failed,
            {
              ...failed,
              stageExecutionId: "stage-router-2",
              previousExecutionId: failed.stageExecutionId,
              attempt: 2,
              state: "preparing",
            },
          ],
          activeStageExecutionId: "stage-router-2",
        });
        return apiResponse(
          {
            runId: "run-router",
            sourceStageExecutionId: failed.stageExecutionId,
            stageExecutionId: "stage-router-2",
          },
          { status: 202 },
        );
      }
      if (path === "/v1/runs/run-router") return apiResponse(current);
      throw new Error(`Unexpected request: ${request.method} ${path}`);
    });
    renderRunApplication(api, "/runs/run-router");
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", { name: "Continue from failed stage" }),
    );
    expect(posts).toBe(0);
    expect(
      screen.getByText(/may repeat external side effects/),
    ).toBeInTheDocument();
    await user.click(
      screen.getByRole("button", { name: "Confirm continuation" }),
    );
    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: "Continue from failed stage" }),
      ).not.toBeInTheDocument(),
    );
    expect(posts).toBe(1);
    expect(requestBody).toEqual({ stageExecutionId: "stage-router-1" });
    expect(await screen.findByText("stage-router-2")).toBeInTheDocument();
  });

  it("does not offer continuation without the Server capability", async () => {
    const current = runFixture({
      state: "failed",
      activeStageExecutionId: undefined,
    });
    const api = new PublicAPI(runtimeConfig, async (input, init) => {
      const request = new Request(input, init);
      return sessionOrArtifacts(request) ?? apiResponse(current);
    });
    renderRunApplication(api, "/runs/run-router");
    expect(
      await screen.findByText(/Continuation is unavailable/),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Continue from failed stage" }),
    ).not.toBeInTheDocument();
  });

  it("keeps a failed Run unchanged after a continuation conflict", async () => {
    let gets = 0;
    const current = runFixture({
      state: "failed",
      resumeStageExecutionId: "stage-router-1",
      activeStageExecutionId: undefined,
    });
    const api = new PublicAPI(runtimeConfig, async (input, init) => {
      const request = new Request(input, init);
      const common = sessionOrArtifacts(request);
      if (common !== undefined) return common;
      if (request.method === "POST")
        return apiResponse(
          {
            code: "conflict",
            message: "Run cannot be continued",
            retryable: false,
          },
          { status: 409 },
        );
      gets++;
      return apiResponse(current);
    });
    renderRunApplication(api, "/runs/run-router");
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", { name: "Continue from failed stage" }),
    );
    await user.click(
      screen.getByRole("button", { name: "Confirm continuation" }),
    );
    expect(
      await screen.findByText("Run cannot be continued"),
    ).toBeInTheDocument();
    await waitFor(() => expect(gets).toBeGreaterThan(1));
    expect(
      screen.getByRole("heading", { name: /Run failed/ }),
    ).toBeInTheDocument();
  });

  it("offers confirmed deletion only for server-deletable completed Runs", async () => {
    const deleteRequests: Request[] = [];
    let deleted = false;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") {
          return apiResponse(session);
        }
        if (url.pathname === "/v1/runs" && request.method === "GET") {
          return apiResponse({
            items: [
              ...(!deleted
                ? [
                    {
                      runId: "run-delete",
                      workflow: "router-analysis@1",
                      state: "succeeded",
                      deletable: true,
                      labels: {},
                      createdAt: "2026-08-31T12:00:00Z",
                      updatedAt: "2026-08-31T12:01:00Z",
                      finishedAt: "2026-08-31T12:01:00Z",
                    },
                  ]
                : []),
              {
                runId: "run-release-pending",
                workflow: "router-analysis@1",
                state: "failed",
                deletable: false,
                labels: {},
                createdAt: "2026-08-31T11:00:00Z",
                updatedAt: "2026-08-31T11:01:00Z",
                finishedAt: "2026-08-31T11:01:00Z",
              },
            ],
            page: { hasMore: false },
          });
        }
        if (
          url.pathname === "/v1/runs/run-delete" &&
          request.method === "DELETE"
        ) {
          deleteRequests.push(request);
          deleted = true;
          return new Response(null, {
            status: 204,
            headers: {
              "X-Contractor-API-Version": "contractor.public.v1",
            },
          });
        }
        throw new Error(`unexpected ${request.method} ${url}`);
      }),
    );
    renderRunApplication(api, "/runs?view=completed");
    const user = userEvent.setup();

    const trigger = await screen.findByRole("button", {
      name: "Delete Run run-delete",
    });
    expect(
      screen.queryByRole("button", {
        name: "Delete Run run-release-pending",
      }),
    ).not.toBeInTheDocument();

    await user.click(trigger);
    expect(
      screen.getByRole("alertdialog", { name: "Delete completed Run?" }),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(deleteRequests).toHaveLength(0);

    await user.click(trigger);
    await user.click(screen.getByRole("button", { name: "Delete Run" }));
    await waitFor(() =>
      expect(
        screen.queryByRole("link", { name: "run-delete" }),
      ).not.toBeInTheDocument(),
    );
    expect(deleteRequests).toHaveLength(1);
    expect(deleteRequests[0]?.headers.get("X-CSRF-Token")).toBe(
      session.csrfToken,
    );
    expect(await deleteRequests[0]?.clone().text()).toBe("");
  });

  it("filters and paginates authoritative Run history", async () => {
    const requests: URL[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") {
          return apiResponse(session);
        }
        if (url.pathname === "/v1/runs") {
          requests.push(url);
          if (url.searchParams.get("cursor") === "next-run") {
            return apiResponse({
              items: [
                {
                  runId: "run-second",
                  workflow: "likec4-from-workspace@5",
                  state: "succeeded",
                  labels: {},
                  createdAt: "2026-08-31T10:00:00Z",
                  updatedAt: "2026-08-31T10:10:00Z",
                  finishedAt: "2026-08-31T10:10:00Z",
                },
              ],
              page: { hasMore: false },
            });
          }
          if (url.searchParams.get("state") === "failed") {
            return apiResponse({ items: [], page: { hasMore: false } });
          }
          return apiResponse({
            items: [
              {
                runId: "run-first",
                workflow: "openapi-from-workspace@5",
                state: "succeeded",
                labels: {},
                createdAt: "2026-08-31T12:00:00Z",
                updatedAt: "2026-08-31T12:01:00Z",
              },
            ],
            page: { hasMore: true, nextCursor: "next-run" },
          });
        }
        throw new Error(`unexpected ${request.method} ${url}`);
      }),
    );
    renderRunApplication(api, "/runs?view=completed");
    expect(
      await screen.findByRole("link", { name: "run-first" }),
    ).toBeInTheDocument();
    const user = userEvent.setup();
    const labelFilters =
      document.querySelector<HTMLDetailsElement>(".run-label-filters");
    expect(labelFilters?.open).toBe(false);
    await user.click(screen.getByText("Metadata & eval filters"));
    expect(labelFilters?.open).toBe(true);
    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(
      await screen.findByRole("link", { name: "run-second" }),
    ).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("State"), "failed");
    expect(
      await screen.findByText("No completed Runs match this view."),
    ).toBeInTheDocument();
    expect(requests.at(-1)?.searchParams.get("state")).toBe("failed");
    expect(requests.at(-1)?.searchParams.get("lifecycle")).toBe("terminal");
    expect(requests.at(-1)?.searchParams.has("cursor")).toBe(false);
  });

  it("honors a deep-linked Run state filter", async () => {
    const requests: URL[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") {
          return apiResponse(session);
        }
        if (url.pathname === "/v1/runs") {
          requests.push(url);
          return apiResponse({ items: [], page: { hasMore: false } });
        }
        throw new Error(`unexpected ${request.method} ${url}`);
      }),
    );

    renderRunApplication(api, "/runs?state=failed");

    expect(
      await screen.findByText("No completed Runs match this view."),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("State")).toHaveValue("failed");
    expect(requests).toHaveLength(1);
    expect(requests[0]?.searchParams.get("state")).toBe("failed");
    expect(requests[0]?.searchParams.get("lifecycle")).toBe("terminal");
  });

  it("keeps exact metadata selectors across paging and resets the cursor when they change", async () => {
    const requests: URL[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") {
          return apiResponse(session);
        }
        if (url.pathname === "/v1/runs") {
          requests.push(url);
          const labels = url.searchParams.getAll("label");
          if (url.searchParams.get("cursor") === "next-eval") {
            return apiResponse({
              items: [
                {
                  runId: "run-leg-a-page-2",
                  workflow: "router-analysis@1",
                  state: "succeeded",
                  labels: {
                    "eval.id": "eval-group=01",
                    "eval.leg": "a",
                    purpose: "eval",
                  },
                  createdAt: "2026-08-31T10:00:00Z",
                  updatedAt: "2026-08-31T10:10:00Z",
                  finishedAt: "2026-08-31T10:10:00Z",
                },
              ],
              page: { hasMore: false },
            });
          }
          const leg = labels.includes("eval.leg=b") ? "b" : "a";
          return apiResponse({
            items: [
              {
                runId: `run-leg-${leg}`,
                workflow: "router-analysis@1",
                state: "succeeded",
                labels: {
                  "eval.id": "eval-group=01",
                  "eval.leg": leg,
                  purpose: "eval",
                },
                createdAt: "2026-08-31T12:00:00Z",
                updatedAt: "2026-08-31T12:01:00Z",
              },
            ],
            page: { hasMore: true, nextCursor: "next-eval" },
          });
        }
        throw new Error(`unexpected ${request.method} ${url}`);
      }),
    );
    renderRunApplication(
      api,
      "/runs?view=completed&label=eval.id%3Deval-group%3D01&label=eval.leg%3Da",
    );
    expect(
      await screen.findByRole("link", { name: "run-leg-a" }),
    ).toBeInTheDocument();
    expect(
      document.querySelector<HTMLDetailsElement>(".run-label-filters")?.open,
    ).toBe(true);
    expect(screen.getByLabelText("2 active filters")).toBeInTheDocument();
    const firstRow = screen
      .getByRole("link", { name: "run-leg-a" })
      .closest("tr");
    expect(firstRow).toHaveTextContent("eval.id:eval-group=01");
    expect(firstRow).toHaveTextContent("eval.leg:a");

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(
      await screen.findByRole("link", { name: "run-leg-a-page-2" }),
    ).toBeInTheDocument();
    expect(requests.at(-1)?.searchParams.get("cursor")).toBe("next-eval");
    expect(requests.at(-1)?.searchParams.getAll("label")).toEqual([
      "eval.id=eval-group=01",
      "eval.leg=a",
    ]);

    const leg = screen.getByLabelText("Eval leg");
    await user.clear(leg);
    await user.type(leg, "b");
    await user.click(
      screen.getByRole("button", { name: "Apply eval filters" }),
    );
    expect(
      await screen.findByRole("link", { name: "run-leg-b" }),
    ).toBeInTheDocument();
    expect(requests.at(-1)?.searchParams.has("cursor")).toBe(false);
    expect(requests.at(-1)?.searchParams.getAll("label")).toEqual([
      "eval.id=eval-group=01",
      "eval.leg=b",
      "purpose=eval",
    ]);
  });

  it("shows a global task and advances only its nested typed Planner projection", async () => {
    let currentRun = runFixture();
    let detailReads = 0;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const shared = sessionOrArtifacts(request);
        if (shared !== undefined) {
          return shared;
        }
        const url = new URL(request.url);
        if (url.pathname === "/v1/runs/run-router") {
          detailReads += 1;
          return apiResponse(currentRun);
        }
        throw new Error(`unexpected ${request.method} ${url}`);
      }),
    );
    const view = renderRunApplication(api, "/runs/run-router");
    expect(
      await screen.findByRole("heading", {
        name: "Produce a global architecture review.",
      }),
    ).toBeInTheDocument();
    expect(screen.getByText("Inspect the source tree")).toBeInTheDocument();
    await waitFor(() => expect(RouteWebSocket.instances).toHaveLength(1));
    const socket = RouteWebSocket.instances[0];
    act(() => socket?.open());
    const subscription = JSON.parse(socket?.sent[0] ?? "{}");
    expect(subscription.after).toEqual({
      generation: "run-generation-1",
      sequence: "10",
    });
    act(() =>
      socket?.message({
        version: "contractor.events.v1",
        type: "subscribed",
        subscriptionId: subscription.subscriptionId,
        stream: { kind: "run", id: "run-router" },
        cursor: subscription.after,
      }),
    );
    const livePlan: NonNullable<RunStatus["attempts"][number]["plan"]> = {
      revision: 2,
      subtasks: [
        {
          id: "0",
          objective: "Inspect the source tree",
          instructions: "Use bounded source search.",
          status: "pending",
        },
        {
          id: "1",
          objective: "Review trust boundaries",
          instructions: "Inspect only public architecture facts.",
          status: "pending",
        },
      ],
      currentSubtaskId: "0",
    };
    act(() =>
      socket?.message({
        version: "contractor.events.v1",
        type: "event",
        subscriptionId: subscription.subscriptionId,
        stream: { kind: "run", id: "run-router" },
        cursor: { generation: "run-generation-1", sequence: "11" },
        kind: "planner.event",
        occurredAt: "2026-08-31T12:02:00Z",
        data: {
          stageExecutionId: "stage-router-1",
          sessionId: "session-router-1",
          invocationId: "invocation-router-1",
          eventKind: "planner.plan_changed",
          plan: livePlan,
        },
      }),
    );
    expect(
      await screen.findByText("Review trust boundaries"),
    ).toBeInTheDocument();
    const metadataPanel = screen
      .getByRole("heading", { name: "Run metadata labels" })
      .closest("section");
    expect(metadataPanel).toHaveTextContent("eval.id:eval-router-01");
    expect(metadataPanel).toHaveTextContent("eval.leg:a");
    expect(metadataPanel).toHaveTextContent("Labels are fixed at creation");
    expect(detailReads).toBe(1);
    act(() =>
      socket?.message({
        version: "contractor.events.v1",
        type: "event",
        subscriptionId: subscription.subscriptionId,
        stream: { kind: "run", id: "run-router" },
        cursor: { generation: "run-generation-1", sequence: "12" },
        kind: "planner.event",
        occurredAt: "2026-08-31T12:02:01Z",
        data: {
          stageExecutionId: "stage-router-1",
          sessionId: "session-router-1",
          invocationId: "invocation-router-1",
          eventKind: "planner.dispatch_selected",
          planRevision: 2,
          subtaskId: "0",
          callId: "dispatch-0002",
          workerName: "reviewer",
        },
      }),
    );
    await waitFor(() =>
      expect(
        view.container.querySelector(".dispatch-banner")?.textContent,
      ).toContain("Logical Worker reviewer"),
    );
    expect(
      screen.queryByText(/Runtime Agent reviewer/),
    ).not.toBeInTheDocument();

    currentRun = runFixture({
      state: "cancelling",
      attempts: [{ ...runFixture().attempts[0]!, plan: livePlan }],
      eventCursor: { generation: "run-generation-1", sequence: "13" },
      updatedAt: "2026-08-31T12:02:02Z",
    });
    act(() =>
      socket?.message({
        version: "contractor.events.v1",
        type: "event",
        subscriptionId: subscription.subscriptionId,
        stream: { kind: "run", id: "run-router" },
        cursor: { generation: "run-generation-1", sequence: "13" },
        kind: "lifecycle.changed",
        occurredAt: "2026-08-31T12:02:02Z",
        data: { runId: "run-router", resource: "run", state: "succeeded" },
      }),
    );
    await waitFor(() => expect(detailReads).toBeGreaterThan(1));
    expect(await screen.findByText("cancelling")).toBeInTheDocument();
    expect(
      view.container.querySelector(".run-triage .state-succeeded"),
    ).toBeNull();
  });

  it("reconciles a cancellation race without an optimistic terminal state", async () => {
    let currentRun = runFixture({ eventCursor: undefined });
    let cancellationBody: unknown;
    let cancellationCalls = 0;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const shared = sessionOrArtifacts(request);
        if (shared !== undefined) {
          return shared;
        }
        const url = new URL(request.url);
        if (
          url.pathname === "/v1/runs/run-router" &&
          request.method === "GET"
        ) {
          return apiResponse(currentRun);
        }
        if (
          url.pathname === "/v1/runs/run-router/cancel" &&
          request.method === "POST"
        ) {
          cancellationCalls += 1;
          cancellationBody = await request.clone().json();
          currentRun = runFixture({
            state: "succeeded",
            eventCursor: undefined,
            activeStageExecutionId: undefined,
            finishedAt: "2026-08-31T12:05:00Z",
          });
          return apiResponse(
            {
              code: "conflict",
              message: "Run completed concurrently",
              retryable: false,
              requestId: "request-cancel-race",
            },
            { status: 409 },
          );
        }
        throw new Error(`unexpected ${request.method} ${url}`);
      }),
    );
    renderRunApplication(api, "/runs/run-router");
    const user = userEvent.setup();
    const button = await screen.findByRole("button", {
      name: "Request cancellation",
    });
    await user.click(button);
    expect(
      await screen.findByText(
        "Give a cancellation reason of 1–4096 characters.",
      ),
    ).toBeInTheDocument();
    expect(cancellationCalls).toBe(0);
    await user.type(
      screen.getByLabelText("Explicit reason"),
      "Stop after the current review",
    );
    await user.click(button);
    expect(
      await screen.findByRole("heading", { name: "Run is terminal" }),
    ).toBeInTheDocument();
    expect(screen.getByText("succeeded")).toBeInTheDocument();
    expect(cancellationCalls).toBe(1);
    expect(cancellationBody).toEqual({
      reason: "Stop after the current review",
    });
  });

  it("surfaces the primary terminal cause and focuses its failed attempt", async () => {
    const failedRun = runFixture({
      state: "failed",
      eventCursor: undefined,
      activeStageExecutionId: undefined,
      attempts: [
        {
          ...runFixture().attempts[0]!,
          state: "failed",
          result: {
            apiVersion: "contractor/v1alpha1",
            outcome: "failed",
            summary: "Planner could not start.",
            artifacts: {},
            error: {
              code: "planner_gateway_unavailable",
              message: "The configured Planner gateway is unavailable.",
              retryable: true,
            },
          },
          diagnostics: {
            items: [
              {
                participant: "planner",
                code: "planner_gateway_unavailable",
                message: "The configured Planner gateway is unavailable.",
                retryable: true,
              },
            ],
            truncated: false,
          },
          metrics: {
            reportsComplete: true,
            modelCalls: 2,
            inputTokens: 900,
            outputTokens: 300,
            totalTokens: 1200,
            toolCalls: 1,
            toolFailures: 0,
            errorCount: 1,
            truncated: false,
          },
          terminalAt: "2026-08-31T12:00:34Z",
          updatedAt: "2026-08-31T12:00:34Z",
        },
      ],
      updatedAt: "2026-08-31T12:00:34Z",
      finishedAt: "2026-08-31T12:00:34Z",
    });
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const shared = sessionOrArtifacts(request);
        if (shared !== undefined) {
          return shared;
        }
        if (new URL(request.url).pathname === "/v1/runs/run-router") {
          return apiResponse(failedRun);
        }
        throw new Error(`unexpected ${request.method} ${request.url}`);
      }),
    );

    const view = renderRunApplication(api, "/runs/run-router");
    const heading = await screen.findByRole("heading", {
      name: "Run failed in analysis",
    });
    const triage = heading.closest(".run-triage");
    expect(triage).not.toBeNull();
    expect(
      within(triage as HTMLElement).getByText("planner_gateway_unavailable"),
    ).toBeInTheDocument();
    expect(
      within(triage as HTMLElement).getByText(
        "The configured Planner gateway is unavailable.",
      ),
    ).toBeInTheDocument();
    expect(
      within(triage as HTMLElement).getByText("retryable"),
    ).toBeInTheDocument();
    expect(within(triage as HTMLElement).getByText("34s")).toBeInTheDocument();
    expect(within(triage as HTMLElement).getByText("1.2K")).toBeInTheDocument();
    expect(
      within(triage as HTMLElement).getByRole("link", {
        name: "Inspect focused attempt",
      }),
    ).toHaveAttribute("href", "#attempt-stage-router-1");
    expect(
      view.container.querySelector<HTMLDetailsElement>(
        "#attempt-stage-router-1",
      )?.open,
    ).toBe(true);
    expect(
      view.container.querySelector<HTMLDetailsElement>(
        ".run-runtime-configuration",
      )?.open,
    ).toBe(false);
  });

  it("opens a reviewed new-Run draft without mutating the terminal source", async () => {
    const terminal = runFixture({
      state: "failed",
      labels: {},
      eventCursor: undefined,
      activeStageExecutionId: undefined,
      attempts: [
        {
          ...runFixture().attempts[0]!,
          state: "failed",
          result: {
            apiVersion: "contractor/v1alpha1",
            outcome: "failed",
            summary: "Gateway unavailable.",
            artifacts: {},
            error: {
              code: "planner_gateway_unavailable",
              message: "Gateway unavailable.",
              retryable: true,
            },
          },
        },
      ],
      finishedAt: "2026-08-31T12:02:00Z",
    });
    const workflow = {
      ref: { name: "router-analysis", version: "1" },
      entryStage: "analysis",
      parameters: { objective: { required: true } },
      inputs: {},
      outputs: {},
      stages: {},
    };
    const createRequests: Request[] = [];
    const api = new PublicAPI(runtimeConfig, async (input, init) => {
      const request = new Request(input, init);
      const common = sessionOrArtifacts(request);
      if (common !== undefined) return common;
      const path = new URL(request.url).pathname;
      if (path === "/v1/runs/run-router/repeat-draft") {
        return apiResponse({
          sourceRunId: "run-router",
          authority: "ordinary",
          workflow: { name: "router-analysis", version: "1" },
          notices: [
            {
              code: "runtime_binding_changed",
              severity: "warning",
              field: "runtimeLabels.default",
              message: "The default Runtime binding changed.",
            },
          ],
          draft: {
            parameters: { objective: "Review the original project" },
            runtimeLabels: [],
            labels: {},
            executionConfig: { status: "available", value: {} },
            inputs: {},
          },
        });
      }
      if (path === "/v1/workflows/router-analysis/versions/1") {
        return apiResponse(workflow);
      }
      if (path === "/v1/artifacts") {
        return apiResponse({ items: [], page: { hasMore: false } });
      }
      if (path === "/v1/runs" && request.method === "POST") {
        createRequests.push(request.clone());
        return apiResponse(
          {
            runId: "run-repeat-new",
            state: "initializing",
            runtimeLabels: [],
            labels: {},
            runtimeConfiguration: terminal.runtimeConfiguration,
          },
          { status: 202 },
        );
      }
      if (path === "/v1/runs/run-repeat-new") {
        return apiResponse({
          ...terminal,
          runId: "run-repeat-new",
          state: "initializing",
          attempts: [],
          finishedAt: undefined,
        });
      }
      if (path === "/v1/runs/run-router") return apiResponse(terminal);
      throw new Error(`unexpected ${request.method} ${path}`);
    });
    const view = renderRunApplication(api, "/runs/run-router");
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", { name: "Configure another Run" }),
    );
    await waitFor(() =>
      expect(view.router.state.location.pathname).toBe(
        "/catalog/workflows/router-analysis/1",
      ),
    );
    const review = await screen.findByLabelText(
      /I reviewed the retained inputs/,
    );
    expect(review).not.toBeChecked();
    expect(
      await screen.findByRole("button", { name: "Start Workflow Run" }),
    ).toBeDisabled();
    expect(
      await screen.findByDisplayValue("Review the original project"),
    ).toBeInTheDocument();
    expect(screen.getByText("runtime_binding_changed")).toBeInTheDocument();
    expect(createRequests).toHaveLength(0);

    await user.click(review);
    await user.click(
      screen.getByRole("button", { name: "Start Workflow Run" }),
    );
    await waitFor(() => expect(createRequests).toHaveLength(1));
    expect(await createRequests[0]!.json()).toEqual({
      workflow: "router-analysis@1",
      runtimeLabels: [],
      parameters: { objective: "Review the original project" },
      artifacts: {},
    });
    expect(terminal.state).toBe("failed");
  });

  it("routes an Audit-managed terminal Run back to its owning Audit", async () => {
    const terminal = runFixture({
      state: "failed",
      projectId: "project-audit",
      eventCursor: undefined,
      activeStageExecutionId: undefined,
      finishedAt: "2026-08-31T12:02:00Z",
    });
    const api = new PublicAPI(runtimeConfig, async (input, init) => {
      const request = new Request(input, init);
      const common = sessionOrArtifacts(request);
      if (common !== undefined) return common;
      const path = new URL(request.url).pathname;
      if (path === "/v1/runs/run-router") return apiResponse(terminal);
      if (path === "/v1/runs/run-router/repeat-draft") {
        return apiResponse({
          sourceRunId: "run-router",
          authority: "audit-managed",
          workflow: { name: "router-analysis", version: "1" },
          projectId: "project-audit",
          auditId: "audit-one",
          notices: [
            {
              code: "audit_managed_run",
              severity: "blocking",
              field: "authority",
              message: "Continue from Audit.",
            },
          ],
        });
      }
      return apiResponse(
        {
          code: "not_found",
          message: "Fixture stops after navigation",
          retryable: false,
          requestId: "request-audit-navigation",
        },
        { status: 404 },
      );
    });
    const view = renderRunApplication(api, "/runs/run-router");
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", { name: "Configure another Run" }),
    );
    await waitFor(() =>
      expect(view.router.state.location.pathname).toBe(
        "/projects/project-audit/audits/audit-one",
      ),
    );
  });

  it("renders terminal attempts, escalation, safe metrics, and frozen outputs", async () => {
    const output = {
      namespace: "outputs",
      name: "report",
      revision: "output-r2",
    };
    const terminalRun = runFixture({
      state: "succeeded",
      eventCursor: undefined,
      activeStageExecutionId: undefined,
      outputs: { report: output },
      attempts: [
        {
          stageExecutionId: "stage-router-1",
          stage: "analysis",
          objective: "Produce a global architecture review.",
          attempt: 1,
          executionConfig: {
            variant: "base",
            planner: executionConfig.planner,
            agents: executionConfig.agents,
          },
          state: "interrupted",
          termination: {
            outcome: "interrupted",
            code: "worker_lease_lost",
            message: "The first Worker lease was lost.",
            retryable: true,
            phase: "running",
            occurredAt: "2026-08-31T12:02:00Z",
          },
          diagnostics: {
            items: [],
            truncated: true,
          },
          createdAt: "2026-08-31T12:00:00Z",
          updatedAt: "2026-08-31T12:02:00Z",
          terminalAt: "2026-08-31T12:02:00Z",
        },
        {
          stageExecutionId: "stage-router-2",
          stage: "analysis",
          objective: "Produce a global architecture review.",
          attempt: 2,
          previousExecutionId: "stage-router-1",
          executionConfig,
          state: "succeeded",
          result: {
            apiVersion: "contractor/v1alpha1",
            outcome: "succeeded",
            summary: "Architecture review completed.",
            artifacts: { report: output },
          },
          metrics: {
            reportsComplete: true,
            modelCalls: 4,
            inputTokens: 1200,
            outputTokens: 400,
            totalTokens: 1600,
            toolCalls: 6,
            toolFailures: 1,
            errorCount: 1,
            truncated: false,
          },
          diagnostics: {
            items: [
              {
                participant: "worker",
                logicalAgent: "reviewer",
                code: "worker_result_schema_json_invalid",
                message: "Worker result did not match StageContentResult.",
                retryable: true,
              },
            ],
            truncated: false,
          },
          runtimeConfiguration: {
            allocations: [
              {
                logicalAgent: "reviewer",
                agentLabels: [
                  {
                    label: "debug",
                    bindingRevision: "7",
                    config: { name: "debug", version: "1", digest },
                  },
                ],
                runtimeAdapters: ["otlp-http@1"],
                origins: {
                  workerTelemetry: {
                    layer: "agent_labels",
                    configs: [{ name: "debug", version: "1", digest }],
                  },
                  llmGateway: { layer: "run_execution_config" },
                },
                status: "released",
              },
            ],
          },
          createdAt: "2026-08-31T12:02:01Z",
          updatedAt: "2026-08-31T12:05:00Z",
          terminalAt: "2026-08-31T12:05:00Z",
        },
      ],
      transitions: [
        {
          sourceExecutionId: "stage-router-1",
          action: "escalate",
          targetStage: "analysis",
          targetExecutionId: "stage-router-2",
          escalationOrdinal: 1,
          escalationExhausted: false,
          decidedAt: "2026-08-31T12:02:01Z",
        },
        {
          sourceExecutionId: "stage-router-2",
          action: "succeed",
          escalationOrdinal: 1,
          escalationExhausted: false,
          decidedAt: "2026-08-31T12:05:00Z",
        },
      ],
      finishedAt: "2026-08-31T12:05:00Z",
    });
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") {
          return apiResponse(session);
        }
        if (url.pathname === "/v1/runs/run-router") {
          return apiResponse(terminalRun);
        }
        if (url.pathname === "/v1/runs/run-router/artifacts") {
          return apiResponse({
            items: [
              {
                artifact: output,
                mediaType: "text/plain",
                size: 12,
                current: true,
                frozen: true,
                createdAt: "2026-08-31T12:05:00Z",
              },
            ],
            page: { hasMore: false },
          });
        }
        if (
          url.pathname ===
          "/v1/runs/run-router/artifacts/outputs/report/metadata"
        ) {
          return apiResponse({
            artifact: output,
            mediaType: "text/plain",
            size: 12,
            current: true,
            frozen: true,
            createdAt: "2026-08-31T12:05:00Z",
          });
        }
        if (url.pathname === "/v1/runs/run-router/artifacts/outputs/report") {
          return byteResponse("safe report\n", "text/plain");
        }
        throw new Error(`unexpected ${request.method} ${url}`);
      }),
    );
    const view = renderRunApplication(api, "/runs/run-router");
    expect(await screen.findByText("worker_lease_lost")).toBeInTheDocument();
    expect(
      screen.getByText("Architecture review completed."),
    ).toBeInTheDocument();
    expect(screen.getByText("strong-review@2")).toBeInTheDocument();
    expect(screen.getByText("Model calls")).toBeInTheDocument();
    expect(screen.getByText("1600")).toBeInTheDocument();
    expect(
      screen.getAllByRole("heading", { name: "Attempt diagnostics" }),
    ).toHaveLength(2);
    expect(
      screen.getByText("No normalized Planner or Worker errors were reported."),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "Older diagnostics were omitted by a bounded report or public response.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText("worker_result_schema_json_invalid"),
    ).toBeInTheDocument();
    expect(screen.getAllByText("reviewer").length).toBeGreaterThan(0);
    expect(screen.getByText("Final Agent labels")).toBeInTheDocument();
    expect(screen.getByText("otlp-http@1")).toBeInTheDocument();
    expect(
      view.container.querySelector(".runtime-agent-override"),
    ).toHaveTextContent("workerTelemetry: agent_labels");
    expect(
      screen.getByText("Worker result did not match StageContentResult."),
    ).toBeInTheDocument();
    expect(screen.getAllByText("escalation 1")).toHaveLength(2);
    expect(
      await screen.findAllByRole("link", { name: /outputs\/report@output-r2/ }),
    ).toHaveLength(2);
    expect(
      screen.getByRole("heading", { name: "Run is terminal" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Request cancellation" }),
    ).not.toBeInTheDocument();
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Preview result" }));
    expect(
      await screen.findByText("safe report", { selector: "pre" }),
    ).toBeInTheDocument();
    expect(view.container.textContent).not.toMatch(/prompt|tool arguments/i);
  });

  it("shows exact Project output publication receipts without changing Run outcome", async () => {
    const projectRun = runFixture({
      projectId: "project-payment-service",
      state: "succeeded",
      eventCursor: undefined,
      activeStageExecutionId: undefined,
      outputPublications: [
        {
          output: "openapi",
          status: "published",
          source: {
            namespace: "outputs",
            name: "openapi",
            revision: "run-output-r1",
          },
          target: {
            namespace: "outputs",
            name: "openapi",
            revision: "project-output-r1",
          },
          createdAt: "2026-08-31T12:05:00Z",
        },
        {
          output: "docs",
          status: "failed",
          source: {
            namespace: "outputs",
            name: "docs",
            revision: "run-docs-r1",
          },
          errorCode: "artifact_conflict",
          errorMessage: "The Project binding already changed.",
          createdAt: "2026-08-31T12:05:01Z",
        },
      ],
      finishedAt: "2026-08-31T12:05:01Z",
    });
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const shared = sessionOrArtifacts(request);
        if (shared !== undefined) {
          return shared;
        }
        if (new URL(request.url).pathname === "/v1/runs/run-router") {
          return apiResponse(projectRun);
        }
        throw new Error(`unexpected ${request.method} ${request.url}`);
      }),
    );

    renderRunApplication(api, "/runs/run-router");

    const panel = (
      await screen.findByRole("heading", {
        name: "Reusable output status",
      })
    ).closest("section");
    expect(panel).not.toBeNull();
    expect(within(panel as HTMLElement).getByText("published")).toBeVisible();
    expect(within(panel as HTMLElement).getByText("failed")).toBeVisible();
    expect(panel).toHaveTextContent("source outputs/openapi@run-output-r1");
    expect(
      within(panel as HTMLElement).getByRole("link", {
        name: "target outputs/openapi@project-output-r1",
      }),
    ).toHaveAttribute(
      "href",
      "/projects/project-payment-service/artifacts/outputs/openapi?revision=project-output-r1",
    );
    expect(panel).toHaveTextContent("artifact_conflict");
    expect(panel).toHaveTextContent("The Project binding already changed.");
    expect(screen.getByText("succeeded")).toBeInTheDocument();
  });

  it("inspects exact RunScope metadata, versions, lineage, and safe preview", async () => {
    const metadata = {
      artifact: {
        namespace: "outputs",
        name: "report",
        revision: "output-r2",
      },
      mediaType: "text/plain",
      size: 12,
      current: true,
      frozen: true,
      createdAt: "2026-08-31T12:05:00Z",
    };
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") {
          return apiResponse(session);
        }
        if (url.pathname.endsWith("/metadata")) {
          return apiResponse(metadata);
        }
        if (url.pathname.endsWith("/versions")) {
          return apiResponse({
            items: [metadata],
            page: { hasMore: false },
          });
        }
        if (url.pathname.endsWith("/lineage")) {
          return apiResponse({
            items: [
              {
                kind: "output_bind",
                sourceScope: "run",
                source: {
                  namespace: "reviewer",
                  name: "report",
                  revision: "worker-r1",
                },
                targetScope: "run",
                target: metadata.artifact,
                runId: "run-router",
                stageExecutionId: "stage-router-1",
                createdAt: "2026-08-31T12:05:00Z",
              },
            ],
            page: { hasMore: false },
          });
        }
        if (url.pathname === "/v1/runs/run-router/artifacts/outputs/report") {
          return byteResponse("safe report\n", "text/plain");
        }
        throw new Error(`unexpected ${request.method} ${url}`);
      }),
    );
    renderRunApplication(
      api,
      "/runs/run-router/artifacts/outputs/report?revision=output-r2",
    );
    expect(
      await screen.findByRole("heading", { name: "outputs/report" }),
    ).toBeInTheDocument();
    expect((await screen.findAllByText("output-r2")).length).toBeGreaterThan(0);
    expect(screen.getByText("yes")).toBeInTheDocument();
    expect(await screen.findByText("output bind")).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: /Upload/ }),
    ).not.toBeInTheDocument();
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Load preview" }));
    expect(
      await screen.findByText("safe report", { selector: "pre" }),
    ).toBeInTheDocument();
    expect(screen.queryByText(/token|secret prompt/i)).not.toBeInTheDocument();
  });
});
