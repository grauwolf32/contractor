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
    runtimeLabels: [],
    labels: {},
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
                  workflow: "likec4-from-source@1",
                  state: "succeeded",
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
                workflow: "openapi-from-source@1",
                state: "running",
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
    renderRunApplication(api, "/runs");
    expect(
      await screen.findByRole("link", { name: "run-first" }),
    ).toBeInTheDocument();
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(
      await screen.findByRole("link", { name: "run-second" }),
    ).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("State"), "failed");
    expect(
      await screen.findByText("No Runs match this view."),
    ).toBeInTheDocument();
    expect(requests.at(-1)?.searchParams.get("state")).toBe("failed");
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
      await screen.findByText("No Runs match this view."),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("State")).toHaveValue("failed");
    expect(requests).toHaveLength(1);
    expect(requests[0]?.searchParams.get("state")).toBe("failed");
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
    await user.click(screen.getByText("Preview on demand"));
    await user.click(
      await screen.findByRole("button", { name: "Load preview" }),
    );
    expect(
      await screen.findByText("safe report", { selector: "pre" }),
    ).toBeInTheDocument();
    expect(view.container.textContent).not.toMatch(/prompt|tool arguments/i);
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
