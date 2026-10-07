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
  it("keeps attempt and Planner disclosures through lifecycle refetches and Refresh", async () => {
    const first = { ...runFixture().attempts[0]!, state: "failed" as const };
    const second = {
      ...runFixture().attempts[0]!,
      stageExecutionId: "stage-router-2",
      attempt: 2,
    };
    let currentRun = runFixture({
      attempts: [first, second],
      activeStageExecutionId: "stage-router-2",
    });
    let detailReads = 0;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const shared = sessionOrArtifacts(request);
        if (shared !== undefined) return shared;
        if (new URL(request.url).pathname === "/v1/runs/run-router") {
          detailReads += 1;
          return apiResponse(currentRun);
        }
        throw new Error(`unexpected ${request.method} ${request.url}`);
      }),
    );
    const view = renderRunApplication(api, "/runs/run-router");
    await waitFor(() => expect(RouteWebSocket.instances).toHaveLength(1));
    const socket = RouteWebSocket.instances[0]!;
    act(() => socket.open());
    const subscription = JSON.parse(socket.sent[0] ?? "{}");
    act(() =>
      socket.message({
        version: "contractor.events.v1",
        type: "subscribed",
        subscriptionId: subscription.subscriptionId,
        stream: { kind: "run", id: "run-router" },
        cursor: subscription.after,
      }),
    );
    const attempt = (id: string) =>
      view.container.querySelector<HTMLDetailsElement>(`#attempt-${id}`)!;
    const instructions = (id: string) =>
      attempt(id).querySelector<HTMLDetailsElement>(
        ".runs-subtask-list details",
      )!;
    await waitFor(() => expect(attempt("stage-router-2").open).toBe(true));
    expect(attempt("stage-router-1").open).toBe(false);
    const user = userEvent.setup();
    await user.click(attempt("stage-router-1").querySelector("summary")!);
    await user.click(instructions("stage-router-2").querySelector("summary")!);
    await user.click(attempt("stage-router-2").querySelector("summary")!);
    expect(attempt("stage-router-1").open).toBe(true);
    expect(attempt("stage-router-2").open).toBe(false);
    expect(instructions("stage-router-2").open).toBe(true);

    const originalAttempt = attempt("stage-router-1");
    const third = {
      ...second,
      stageExecutionId: "stage-router-3",
      attempt: 3,
    };
    currentRun = {
      ...currentRun,
      attempts: [first, { ...second, state: "failed" }, third],
      activeStageExecutionId: "stage-router-3",
      eventCursor: { generation: "run-generation-1", sequence: "11" },
    };
    act(() =>
      socket.message({
        version: "contractor.events.v1",
        type: "event",
        subscriptionId: subscription.subscriptionId,
        stream: { kind: "run", id: "run-router" },
        cursor: { generation: "run-generation-1", sequence: "11" },
        kind: "lifecycle.changed",
        occurredAt: "2026-08-31T12:02:00Z",
        data: { runId: "run-router", resource: "run", state: "running" },
      }),
    );
    await waitFor(() => expect(detailReads).toBeGreaterThan(1));
    await waitFor(() => expect(attempt("stage-router-3").open).toBe(true));
    expect(attempt("stage-router-1")).not.toBe(originalAttempt);
    expect(attempt("stage-router-1").open).toBe(true);
    expect(attempt("stage-router-2").open).toBe(false);
    expect(instructions("stage-router-2").open).toBe(true);

    const priorRefreshAttempt = attempt("stage-router-1");
    await new Promise((resolve) => setTimeout(resolve, 5));
    await user.click(screen.getByRole("button", { name: "Refresh" }));
    await waitFor(() => expect(detailReads).toBeGreaterThan(2));
    await waitFor(() =>
      expect(attempt("stage-router-1")).not.toBe(priorRefreshAttempt),
    );
    expect(attempt("stage-router-1").open).toBe(true);
    expect(attempt("stage-router-2").open).toBe(false);
    expect(instructions("stage-router-2").open).toBe(true);
    expect(attempt("stage-router-3").open).toBe(true);
  });

  it("shows a waiting invocation and retries the model without a new stage", async () => {
    let posts = 0;
    let current = runFixture({
      state: "waiting",
      recovery: {
        code: "model_unavailable",
        since: "2026-09-20T20:00:00Z",
        automaticUntil: "2026-09-20T20:05:00Z",
        requiresRetry: true,
      },
    });
    const api = new PublicAPI(runtimeConfig, async (input, init) => {
      const request = new Request(input, init);
      const common = sessionOrArtifacts(request);
      if (common !== undefined) return common;
      const path = new URL(request.url).pathname;
      if (
        path === "/v1/runs/run-router/retry-gateway" &&
        request.method === "POST"
      ) {
        posts++;
        expect(await request.json()).toEqual({});
        current = runFixture();
        return apiResponse({ runId: "run-router" }, { status: 202 });
      }
      if (path === "/v1/runs/run-router") return apiResponse(current);
      throw new Error(`Unexpected request: ${request.method} ${path}`);
    });
    renderRunApplication(api, "/runs/run-router");
    const retry = await screen.findByRole("button", {
      name: "Retry model connection",
    });
    expect(
      screen.getByText("The model was unloaded or is unavailable."),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Continue from failed stage" }),
    ).not.toBeInTheDocument();
    const user = userEvent.setup();
    await user.click(retry);
    // Retry has its own confirmation, distinct from Continue and Repeat.
    const dialog = screen.getByRole("dialog", {
      name: "Retry the model connection?",
    });
    expect(dialog).toHaveTextContent(
      "The model was unloaded or is unavailable.",
    );
    expect(dialog).toHaveTextContent("no new stage attempt starts");
    await user.click(within(dialog).getByRole("button", { name: "Not now" }));
    expect(posts).toBe(0);
    await waitFor(() => expect(retry).toHaveFocus());

    await user.click(retry);
    await user.click(
      within(
        screen.getByRole("dialog", { name: "Retry the model connection?" }),
      ).getByRole("button", { name: "Confirm retry" }),
    );
    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: "Retry model connection" }),
      ).not.toBeInTheDocument(),
    );
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(posts).toBe(1);
    expect(current.attempts).toHaveLength(1);
    await waitFor(() =>
      expect(screen.getByRole("heading", { level: 1 })).toHaveFocus(),
    );
  });

  it("keeps refreshing unchanged recovery until manual retry is required", async () => {
    const now = Date.now();
    const automaticUntil = new Date(now + 3300).toISOString();
    const waiting = runFixture({
      state: "waiting",
      recovery: {
        code: "model_unavailable",
        since: new Date(now).toISOString(),
        nextRetryAt: new Date(now + 100).toISOString(),
        automaticUntil,
        requiresRetry: false,
      },
    });
    let detailReads = 0;
    let withoutRecovery = false;
    const api = new PublicAPI(runtimeConfig, async (input, init) => {
      const request = new Request(input, init);
      const common = sessionOrArtifacts(request);
      if (common !== undefined) return common;
      const path = new URL(request.url).pathname;
      if (path === "/v1/runs/run-router") {
        detailReads++;
        if (withoutRecovery)
          return apiResponse({ ...waiting, recovery: undefined });
        if (Date.now() >= Date.parse(automaticUntil)) {
          return apiResponse({
            ...waiting,
            recovery: { ...waiting.recovery!, requiresRetry: true },
          });
        }
        return apiResponse(waiting);
      }
      throw new Error(`Unexpected request: ${request.method} ${path}`);
    });
    renderRunApplication(api, "/runs/run-router");
    expect(await screen.findByText(/Next recovery check:/)).toBeInTheDocument();
    await waitFor(() => expect(detailReads).toBeGreaterThanOrEqual(2), {
      timeout: 6000,
    });
    expect(
      screen.queryByRole("button", { name: "Retry model connection" }),
    ).not.toBeInTheDocument();
    expect(
      await screen.findByRole(
        "button",
        { name: "Retry model connection" },
        { timeout: 9000 },
      ),
    ).toBeInTheDocument();
    expect(detailReads).toBeGreaterThanOrEqual(3);
    const readsAtRetry = detailReads;
    await new Promise((resolve) => setTimeout(resolve, 2300));
    expect(detailReads).toBe(readsAtRetry);

    withoutRecovery = true;
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Refresh" }));
    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: "Retry model connection" }),
      ).not.toBeInTheDocument(),
    );
    const readsWithoutRecovery = detailReads;
    await new Promise((resolve) => setTimeout(resolve, 2300));
    expect(detailReads).toBe(readsWithoutRecovery);
  }, 15000);

  it("keeps a loaded Run visible when a background refetch fails", async () => {
    let fail = false;
    const api = new PublicAPI(runtimeConfig, async (input, init) => {
      const request = new Request(input, init);
      const common = sessionOrArtifacts(request);
      if (common !== undefined) return common;
      const path = new URL(request.url).pathname;
      if (path === "/v1/runs/run-router") {
        if (fail) {
          return apiResponse(
            { error: { code: "unavailable", message: "Server unavailable" } },
            { status: 503 },
          );
        }
        return apiResponse(runFixture());
      }
      throw new Error(`Unexpected request: ${request.method} ${path}`);
    });
    renderRunApplication(api, "/runs/run-router");
    const stageHeading = {
      name: "Produce a global architecture review.",
    };
    expect(
      await screen.findByRole("heading", stageHeading),
    ).toBeInTheDocument();
    fail = true;
    await userEvent.setup().click(screen.getByLabelText("Refresh"));
    expect(
      await screen.findByText(/showing the last loaded data/i),
    ).toBeInTheDocument();
    expect(screen.getByRole("heading", stageHeading)).toBeInTheDocument();
    expect(
      screen.queryByText("Could not load this Run"),
    ).not.toBeInTheDocument();
  });

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
    // The new attempt comes only from the refetched Run, linked to its source.
    const continued = await waitFor(() => {
      const element = document.querySelector<HTMLElement>(
        "#attempt-stage-router-2",
      );
      expect(element).not.toBeNull();
      return element!;
    });
    expect(within(continued).getByText(/^Continues/)).toHaveTextContent(
      "Continues stage-router-1",
    );
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
    const dialog = screen.getByRole("alertdialog", {
      name: "Continue from failed stage?",
    });
    expect(
      await within(dialog).findByText("Run cannot be continued"),
    ).toBeInTheDocument();
    await waitFor(() => expect(gets).toBeGreaterThan(1));
    // The refusal stays in the confirmation; Back returns to the unchanged Run.
    await user.click(within(dialog).getByRole("button", { name: "Back" }));
    expect(
      screen.getByRole("heading", { name: /Run failed/ }),
    ).toBeInTheDocument();
    expect(screen.getByText("Failed")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Continue from failed stage" }),
    ).toBeEnabled();
  });

  it("keeps a refused continuation on its own stage and never reopens it unasked", async () => {
    const posts: unknown[] = [];
    const analysis = {
      ...runFixture().attempts[0]!,
      state: "failed" as const,
    };
    const review = {
      ...analysis,
      stageExecutionId: "stage-review-1",
      stage: "review",
    };
    const failedAt = (attempt: typeof analysis) =>
      runFixture({
        state: "failed",
        eventCursor: undefined,
        activeStageExecutionId: undefined,
        attempts: attempt === analysis ? [analysis] : [analysis, attempt],
        resumeStageExecutionId: attempt.stageExecutionId,
      });
    let current = failedAt(analysis);
    const api = new PublicAPI(runtimeConfig, async (input, init) => {
      const request = new Request(input, init);
      const common = sessionOrArtifacts(request);
      if (common !== undefined) return common;
      const path = new URL(request.url).pathname;
      if (path === "/v1/runs/run-router/resume" && request.method === "POST") {
        posts.push(await request.json());
        // Another tab continued the Run first: it is queued again.
        current = runFixture({
          state: "pending",
          eventCursor: undefined,
          activeStageExecutionId: undefined,
          attempts: [analysis],
        });
        return apiResponse(
          {
            code: "conflict",
            message: "The Run is no longer failed",
            retryable: false,
          },
          { status: 409 },
        );
      }
      if (path === "/v1/runs/run-router") return apiResponse(current);
      throw new Error(`Unexpected request: ${request.method} ${path}`);
    });
    const view = renderRunApplication(api, "/runs/run-router");
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", { name: "Continue from failed stage" }),
    );
    const dialog = screen.getByRole("alertdialog", {
      name: "Continue from failed stage?",
    });
    await user.click(
      within(dialog).getByRole("button", { name: "Confirm continuation" }),
    );
    expect(
      await within(dialog).findByText("The Run is no longer failed"),
    ).toBeInTheDocument();
    // The refetched Run offers no continuation, yet the confirmation stays
    // with its stage and the refusal until the user closes it.
    await waitFor(() =>
      expect(view.container.querySelector(".run-triage")).toHaveClass(
        "run-triage-pending",
      ),
    );
    expect(
      screen.queryByRole("button", {
        name: "Continue from failed stage",
        hidden: true,
      }),
    ).toBeNull();
    expect(dialog).toBeInTheDocument();
    expect(dialog).toHaveTextContent("Retry stage analysis?");
    await user.click(within(dialog).getByRole("button", { name: "Back" }));
    expect(screen.queryByRole("alertdialog")).toBeNull();
    // The button that opened it is gone, so the page heading takes focus.
    await waitFor(() =>
      expect(screen.getByRole("heading", { level: 1 })).toHaveFocus(),
    );

    // The Run fails again at another stage: nothing opens until a click.
    current = failedAt(review);
    await user.click(screen.getByRole("button", { name: "Refresh" }));
    const trigger = await screen.findByRole("button", {
      name: "Continue from failed stage",
    });
    expect(screen.queryByRole("alertdialog")).toBeNull();
    await user.click(trigger);
    const next = screen.getByRole("alertdialog", {
      name: "Continue from failed stage?",
    });
    expect(next).toHaveTextContent("Retry stage review?");
    expect(within(next).queryByText("The Run is no longer failed")).toBeNull();
    expect(posts).toEqual([{ stageExecutionId: "stage-router-1" }]);
  });

  it("keeps a refused model retry on its own recovery and never reopens it unasked", async () => {
    let posts = 0;
    const waitingFor = (code: "model_unavailable" | "gateway_timeout") =>
      runFixture({
        state: "waiting",
        eventCursor: undefined,
        recovery: {
          code,
          since: "2026-09-20T20:00:00Z",
          automaticUntil: "2026-09-20T20:05:00Z",
          requiresRetry: true,
        },
      });
    let current = waitingFor("model_unavailable");
    const api = new PublicAPI(runtimeConfig, async (input, init) => {
      const request = new Request(input, init);
      const common = sessionOrArtifacts(request);
      if (common !== undefined) return common;
      const path = new URL(request.url).pathname;
      if (
        path === "/v1/runs/run-router/retry-gateway" &&
        request.method === "POST"
      ) {
        posts++;
        // The model came back and the Run resumed before this request.
        current = runFixture({ eventCursor: undefined });
        return apiResponse(
          {
            code: "conflict",
            message: "The Run is not waiting for a model retry",
            retryable: false,
          },
          { status: 409 },
        );
      }
      if (path === "/v1/runs/run-router") return apiResponse(current);
      throw new Error(`Unexpected request: ${request.method} ${path}`);
    });
    const view = renderRunApplication(api, "/runs/run-router");
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", { name: "Retry model connection" }),
    );
    const dialog = screen.getByRole("dialog", {
      name: "Retry the model connection?",
    });
    await user.click(
      within(dialog).getByRole("button", { name: "Confirm retry" }),
    );
    expect(
      await within(dialog).findByText(
        "The Run is not waiting for a model retry",
      ),
    ).toBeInTheDocument();
    await waitFor(() =>
      expect(view.container.querySelector(".run-triage")).toHaveClass(
        "run-triage-running",
      ),
    );
    expect(dialog).toHaveTextContent(
      "The model was unloaded or is unavailable.",
    );
    await user.click(within(dialog).getByRole("button", { name: "Not now" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    await waitFor(() =>
      expect(screen.getByRole("heading", { level: 1 })).toHaveFocus(),
    );

    // A later recovery waits for its own click and names its own cause.
    current = waitingFor("gateway_timeout");
    await user.click(screen.getByRole("button", { name: "Refresh" }));
    const trigger = await screen.findByRole("button", {
      name: "Retry model connection",
    });
    expect(screen.queryByRole("dialog")).toBeNull();
    await user.click(trigger);
    expect(
      screen.getByRole("dialog", { name: "Retry the model connection?" }),
    ).toHaveTextContent("The model request timed out.");
    expect(posts).toBe(1);
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
    expect(screen.getByRole("button", { name: "Cancel" })).toHaveFocus();
    await user.tab({ shift: true });
    expect(screen.getByRole("button", { name: "Delete Run" })).toHaveFocus();
    await user.keyboard("{Escape}");
    await waitFor(() => expect(trigger).toHaveFocus());
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
    await user.click(
      within(screen.getByRole("group", { name: "State" })).getByRole("button", {
        name: "Failed",
      }),
    );
    expect(
      await screen.findByText("No completed Runs match this view."),
    ).toBeInTheDocument();
    expect(requests.at(-1)?.searchParams.get("state")).toBe("failed");
    expect(requests.at(-1)?.searchParams.get("lifecycle")).toBe("terminal");
    expect(requests.at(-1)?.searchParams.has("cursor")).toBe(false);
  });

  it("drops the completed page cursor when the tab link clears the filters", async () => {
    const requests: URL[] = [];
    const completedRun = (runId: string) => ({
      runId,
      workflow: "openapi-from-workspace@5",
      state: "failed",
      labels: {},
      createdAt: "2026-08-31T12:00:00Z",
      updatedAt: "2026-08-31T12:01:00Z",
      finishedAt: "2026-08-31T12:01:00Z",
    });
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
          if (url.searchParams.get("cursor") === "failed-2") {
            return apiResponse({
              items: [completedRun("run-failed-2")],
              page: { hasMore: false },
            });
          }
          if (url.searchParams.get("state") === "failed") {
            return apiResponse({
              items: [completedRun("run-failed-1")],
              page: { hasMore: true, nextCursor: "failed-2" },
            });
          }
          return apiResponse({
            items: [{ ...completedRun("run-any"), state: "succeeded" }],
            page: { hasMore: false },
          });
        }
        throw new Error(`unexpected ${request.method} ${url}`);
      }),
    );
    const { router } = renderRunApplication(
      api,
      "/runs?view=completed&state=failed",
    );
    const user = userEvent.setup();

    expect(
      await screen.findByRole("link", { name: "run-failed-1" }),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(
      await screen.findByRole("link", { name: "run-failed-2" }),
    ).toBeInTheDocument();
    expect(router.state.location.search).toBe(
      "?view=completed&state=failed&cursor=failed-2",
    );
    expect(document.title).toBe("Completed · Runs · Contractor");

    const views = screen.getByRole("navigation", { name: "Run views" });
    await user.click(within(views).getByRole("link", { name: "Completed" }));
    expect(
      await screen.findByRole("link", { name: "run-any" }),
    ).toBeInTheDocument();
    expect(requests.at(-1)?.searchParams.has("state")).toBe(false);
    expect(requests.at(-1)?.searchParams.has("cursor")).toBe(false);
    expect(screen.queryByRole("navigation", { name: "Run pages" })).toBeNull();
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
    const state = screen.getByRole("group", { name: "State" });
    expect(
      within(state).getByRole("button", { name: "Failed" }),
    ).toHaveAttribute("aria-pressed", "true");
    expect(within(state).getByRole("button", { name: "All" })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
    expect(
      within(screen.getByRole("navigation", { name: "Run views" })).getByRole(
        "link",
        { name: "Completed" },
      ),
    ).toHaveAttribute("aria-current", "page");
    expect(requests).toHaveLength(1);
    expect(requests[0]?.searchParams.get("state")).toBe("failed");
    expect(requests[0]?.searchParams.get("lifecycle")).toBe("terminal");
  });

  it("stays in Completed when a state deep link is cleared or filtered further", async () => {
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
    const { router } = renderRunApplication(api, "/runs?state=failed");
    const user = userEvent.setup();
    expect(
      await screen.findByText("No completed Runs match this view."),
    ).toBeInTheDocument();
    const completedLink = () =>
      within(screen.getByRole("navigation", { name: "Run views" })).getByRole(
        "link",
        { name: "Completed" },
      );

    // "All" drops the state that selected Completed, so the view is named.
    await user.click(
      within(screen.getByRole("group", { name: "State" })).getByRole("button", {
        name: "All",
      }),
    );
    await waitFor(() =>
      expect(router.state.location.search).toBe("?view=completed"),
    );
    expect(completedLink()).toHaveAttribute("aria-current", "page");
    expect(document.title).toBe("Completed · Runs · Contractor");
    await waitFor(() =>
      expect(requests.at(-1)?.searchParams.has("state")).toBe(false),
    );
    expect(requests.at(-1)?.searchParams.get("lifecycle")).toBe("terminal");

    // A metadata filter added to a state deep link keeps the view as well.
    await act(() => router.navigate("/runs?state=cancelled"));
    await user.click(await screen.findByText("Metadata & eval filters"));
    await user.type(screen.getByLabelText("Label key"), "team");
    await user.type(screen.getByLabelText("Label value"), "red");
    await user.click(screen.getByRole("button", { name: "Add filter" }));
    await waitFor(() =>
      expect(router.state.location.search).toBe(
        "?view=completed&state=cancelled&label=team%3Dred",
      ),
    );
    expect(completedLink()).toHaveAttribute("aria-current", "page");
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
    expect(screen.getByText("2 active filters")).toBeInTheDocument();
    const firstRow = screen
      .getByRole("link", { name: "run-leg-a" })
      .closest("tr");
    const user = userEvent.setup();
    const contextToggle = within(firstRow!).getByRole("button", {
      name: "Context for run-leg-a",
    });
    expect(contextToggle).toHaveAttribute("aria-expanded", "false");
    await user.click(contextToggle);
    const context = screen.getByRole("region", {
      name: "Context for run-leg-a",
    });
    expect(context).toBeVisible();
    expect(context).toHaveTextContent("eval.id:eval-group=01");
    expect(context).toHaveTextContent("eval.leg:a");
    expect(contextToggle).toHaveAttribute("aria-expanded", "true");
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
    const metadataPanel = view.container.querySelector(
      ".run-metadata-label-panel",
    );
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
        view.container.querySelector(".runs-dispatch")?.textContent,
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
    // The lifecycle hint said succeeded; only the refetched Run state shows.
    expect(await screen.findByText("Cancelling")).toBeInTheDocument();
    expect(screen.queryByText("Succeeded")).toBeNull();
    expect(view.container.querySelector(".run-triage")).toHaveClass(
      "run-triage-cancelling",
    );
  });

  it("follows a new attempt's first Planner fact without a projection resync", async () => {
    let detailReads = 0;
    let releaseRefetch!: () => void;
    const refetchReleased = new Promise<void>((resolve) => {
      releaseRefetch = resolve;
    });
    const retry = {
      ...runFixture().attempts[0]!,
      stageExecutionId: "stage-router-2",
      objective: "Retry the architecture review.",
      attempt: 2,
      createdAt: "2026-08-31T12:02:00Z",
      updatedAt: "2026-08-31T12:02:00Z",
      plannerStartedAt: "2026-08-31T12:02:00Z",
    };
    delete retry.plan;
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
          if (detailReads === 1) {
            return apiResponse(runFixture());
          }
          await refetchReleased;
          return apiResponse(
            runFixture({
              attempts: [
                { ...runFixture().attempts[0]!, state: "failed" },
                retry,
              ],
              activeStageExecutionId: "stage-router-2",
              eventCursor: { generation: "run-generation-1", sequence: "12" },
            }),
          );
        }
        throw new Error(`unexpected ${request.method} ${url}`);
      }),
    );
    renderRunApplication(api, "/runs/run-router");
    await waitFor(() => expect(RouteWebSocket.instances).toHaveLength(1));
    const socket = RouteWebSocket.instances[0]!;
    act(() => socket.open());
    const subscription = JSON.parse(socket.sent[0] ?? "{}");
    const frame = (sequence: string) => ({
      version: "contractor.events.v1",
      type: "event",
      subscriptionId: subscription.subscriptionId,
      stream: { kind: "run", id: "run-router" },
      cursor: { generation: "run-generation-1", sequence },
      occurredAt: "2026-08-31T12:02:00Z",
    });
    act(() => {
      socket.message({
        version: "contractor.events.v1",
        type: "subscribed",
        subscriptionId: subscription.subscriptionId,
        stream: { kind: "run", id: "run-router" },
        cursor: subscription.after,
      });
      // The Stage start commits its lifecycle hint and planner.started
      // together; the Planner fact arrives before the Run refetch returns.
      socket.message({
        ...frame("11"),
        kind: "lifecycle.changed",
        data: {
          runId: "run-router",
          resource: "stageExecution",
          stageExecutionId: "stage-router-2",
          state: "running",
        },
      });
      socket.message({
        ...frame("12"),
        kind: "planner.event",
        data: {
          stageExecutionId: "stage-router-2",
          sessionId: "session-router-2",
          invocationId: "invocation-router-2",
          eventKind: "planner.started",
        },
      });
    });
    await waitFor(() => expect(detailReads).toBe(2));
    expect(socket.readyState).toBe(1);
    expect(screen.queryByText(/REST resync after/)).toBeNull();

    await act(async () => {
      releaseRefetch();
      await refetchReleased;
    });
    expect(
      await screen.findByRole("heading", {
        name: "Retry the architecture review.",
      }),
    ).toBeInTheDocument();
    expect(detailReads).toBe(2);
    expect(RouteWebSocket.instances).toHaveLength(1);
  });

  it("titles the Run detail, Run Artifact and Runs pages", async () => {
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
          return apiResponse(runFixture({ eventCursor: undefined }));
        }
        if (url.pathname === "/v1/runs/run-router/artifacts/inputs/source") {
          return apiResponse({
            artifact: {
              namespace: "inputs",
              name: "source",
              revision: "input-r1",
            },
            mediaType: "application/zip",
            sizeBytes: 3,
            digest,
            createdAt: "2026-08-31T12:00:00Z",
          });
        }
        return apiResponse({ items: [], page: { hasMore: false } });
      }),
    );
    const { router } = renderRunApplication(api, "/runs/run-router");
    await waitFor(() =>
      expect(document.title).toBe("router-analysis@1 · Run · Contractor"),
    );
    await act(async () => {
      await router.navigate("/runs/run-router/artifacts/inputs/source");
    });
    await waitFor(() =>
      expect(document.title).toBe("inputs/source · Run · Contractor"),
    );
    await act(async () => {
      await router.navigate("/runs");
    });
    await waitFor(() =>
      expect(document.title).toBe("Queue · Runs · Contractor"),
    );
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
            { runId: "run-router", state: "succeeded" },
            { status: 200 },
          );
        }
        throw new Error(`unexpected ${request.method} ${url}`);
      }),
    );
    renderRunApplication(api, "/runs/run-router");
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Cancel Run" }));
    const dialog = screen.getByRole("dialog", { name: "Cancel this Run?" });
    expect(within(dialog).getByLabelText("Reason")).toHaveFocus();
    const button = within(dialog).getByRole("button", {
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
      within(dialog).getByLabelText("Reason"),
      "Stop after the current review",
    );
    await user.click(button);
    expect(await screen.findByText("Succeeded")).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getByRole("heading", { level: 1 })).toHaveFocus(),
    );
    // A terminal Run shows no cancellation panel at all.
    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: "Request cancellation" }),
      ).not.toBeInTheDocument(),
    );
    expect(screen.queryByText(/Cancellation/)).not.toBeInTheDocument();
    expect(cancellationCalls).toBe(1);
    expect(cancellationBody).toEqual({
      reason: "Stop after the current review",
    });
  });

  it("keeps cancellation retryable after a failed request while the Run is active", async () => {
    let currentRun = runFixture({ eventCursor: undefined });
    let cancellationCalls = 0;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const shared = sessionOrArtifacts(request);
        if (shared !== undefined) return shared;
        const path = new URL(request.url).pathname;
        if (path === "/v1/runs/run-router" && request.method === "GET") {
          return apiResponse(currentRun);
        }
        if (
          path === "/v1/runs/run-router/cancel" &&
          request.method === "POST"
        ) {
          cancellationCalls += 1;
          if (cancellationCalls === 1) {
            return apiResponse(
              {
                code: "unavailable",
                message: "Cancellation service unavailable",
                retryable: true,
              },
              { status: 500 },
            );
          }
          currentRun = runFixture({
            state: "cancelling",
            eventCursor: undefined,
          });
          return apiResponse(
            { runId: "run-router", state: "cancelling" },
            { status: 202 },
          );
        }
        throw new Error(`unexpected ${request.method} ${path}`);
      }),
    );
    renderRunApplication(api, "/runs/run-router");
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Cancel Run" }));
    const dialog = screen.getByRole("dialog", { name: "Cancel this Run?" });
    const button = within(dialog).getByRole("button", {
      name: "Request cancellation",
    });
    await user.type(within(dialog).getByLabelText("Reason"), "Stop this Run");
    await user.click(button);
    expect(
      await within(dialog).findByText("Cancellation service unavailable"),
    ).toBeVisible();
    expect(within(dialog).getByLabelText("Reason")).toHaveValue(
      "Stop this Run",
    );
    expect(button).toBeEnabled();
    expect(screen.queryByText(/had finished before/)).toBeNull();
    expect(cancellationCalls).toBe(1);
    await user.click(button);
    expect(await screen.findByText("Cleanup in progress")).toBeVisible();
    expect(screen.getByText("Cancelling")).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.queryByRole("button", { name: "Cancel Run" })).toBeNull();
    expect(cancellationCalls).toBe(2);
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
      within(triage as HTMLElement).getByText("Retryable"),
    ).toBeInTheDocument();
    // Execution metrics live under Technical details and stay open for a
    // failed Run.
    const metrics =
      view.container.querySelector<HTMLDetailsElement>("#run-metrics")!;
    expect(metrics.open).toBe(true);
    expect(within(metrics).getAllByText("34s").length).toBeGreaterThan(0);
    expect(within(metrics).getByText("1.2K")).toBeInTheDocument();
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
    // A failed Run keeps its attempts and diagnostics expanded.
    expect(
      view.container.querySelector<HTMLDetailsElement>("#run-attempts")?.open,
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

  it.each([
    [
      "repeat_request_unavailable",
      "The original Run request is unavailable. Configure a new Run from the Workflow.",
    ],
    [
      "repeat_request_invalid",
      "The saved Run request could not be verified. Configure a new Run from the Workflow.",
    ],
  ])(
    "keeps a Run with %s on its detail page without preparing a draft",
    async (code, message) => {
      const terminal = runFixture({
        state: "failed",
        eventCursor: undefined,
        activeStageExecutionId: undefined,
        finishedAt: "2026-08-31T12:02:00Z",
      });
      const requests: Request[] = [];
      const api = new PublicAPI(runtimeConfig, async (input, init) => {
        const request = new Request(input, init);
        requests.push(request);
        const common = sessionOrArtifacts(request);
        if (common !== undefined) return common;
        const path = new URL(request.url).pathname;
        if (path === "/v1/runs/run-router") return apiResponse(terminal);
        if (path === "/v1/workflows/router-analysis/versions/1")
          return apiResponse({
            ref: { name: "router-analysis", version: "1" },
            entryStage: "analysis",
            parameters: {},
            inputs: {},
            outputs: {},
            stages: {},
          });
        if (path === "/v1/runs/run-router/repeat-draft")
          return apiResponse({
            sourceRunId: "run-router",
            authority: "ordinary",
            workflow: { name: "router-analysis", version: "1" },
            notices: [{ code, severity: "blocking", message }],
          });
        throw new Error(`Unexpected request: ${request.method} ${path}`);
      });
      const view = renderRunApplication(api, "/runs/run-router");
      const user = userEvent.setup();
      for (let attempt = 0; attempt < 2; attempt++) {
        await user.click(
          await screen.findByRole("button", { name: "Configure another Run" }),
        );
        expect(await screen.findByText(message)).toBeInTheDocument();
        expect(view.router.state.location.pathname).toBe("/runs/run-router");
      }
      expect(requests.filter((request) => request.method !== "GET")).toEqual(
        [],
      );
      expect(
        requests.filter((request) =>
          new URL(request.url).pathname.startsWith("/v1/workflows"),
        ),
      ).toHaveLength(1);
    },
  );

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
            summary: "## Summary\n\nArchitecture review completed.",
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
          resources: [
            {
              allocationId: "allocation-reviewer-2",
              runId: "run-router",
              stageExecutionId: "stage-router-2",
              stage: "analysis",
              logicalAgent: "reviewer",
              outcome: "succeeded",
              finishedAt: "2026-08-31T12:05:00Z",
              collectionPolicy: "requested",
              status: "available",
              resources: {
                version: 1,
                scope: "runtime_process",
                status: "complete",
                durationSeconds: 120,
                cpuUserSeconds: 18,
                cpuSystemSeconds: 6,
                rssStartBytes: 201_326_592,
                rssEndBytes: 234_881_024,
                rssPeakObservedBytes: 268_435_456,
                rssSampleCount: 9,
                maxSampleGapSeconds: 15,
              },
            },
          ],
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
                  caido: { layer: "run_execution_config" },
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
    // The Planner summary is Markdown, rendered inside a bounded preview.
    expect(
      await screen.findByText("Architecture review completed."),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Summary", level: 2 }),
    ).toBeInTheDocument();
    expect(view.container.querySelector(".result-summary")).not.toBeNull();
    expect(screen.getByText("strong-review@2")).toBeInTheDocument();
    expect(screen.getByText("Model calls")).toBeInTheDocument();
    expect(screen.getByText("1600")).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Allocation resource observations" }),
    ).toBeInTheDocument();
    expect(screen.getByText("256.0 MiB")).toBeInTheDocument();
    expect(
      screen.getByText("Runtime process scope · complete"),
    ).toBeInTheDocument();
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
    expect(screen.getByText("caido")).toBeInTheDocument();
    expect(screen.getByText("otlp-http@1")).toBeInTheDocument();
    expect(
      view.container.querySelector(".runtime-agent-override"),
    ).toHaveTextContent("workerTelemetry: agent_labels");
    expect(
      screen.getByText("Worker result did not match StageContentResult."),
    ).toBeInTheDocument();
    expect(screen.getAllByText("escalation 1")).toHaveLength(2);
    // Stage execution IDs stay readable in full and copyable, so attempts,
    // continuations and Scheduler decisions can be matched exactly.
    expect(
      screen.getAllByRole("button", { name: "Copy stage execution ID" }),
    ).toHaveLength(2);
    const second = view.container.querySelector<HTMLElement>(
      "#attempt-stage-router-2",
    )!;
    expect(
      within(second).getByText("stage-router-2", {
        selector: ".ui-id-chip-value",
      }),
    ).toHaveAttribute("title", "stage-router-2");
    expect(
      within(second).getByRole("button", {
        name: "Copy previous stage execution ID",
      }),
    ).toBeInTheDocument();
    expect(
      within(
        view.container.querySelector<HTMLElement>("#attempt-stage-router-1")!,
      ).getByRole("button", { name: "Copy target stage execution ID" }),
    ).toBeInTheDocument();
    expect(
      await screen.findAllByRole("link", { name: /outputs\/report@output-r2/ }),
    ).toHaveLength(2);
    // Terminal Runs hide the cancellation panel entirely and collapse the
    // heavy sections; the artifact table summary still reports its count.
    expect(screen.queryByText(/Cancellation/)).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Request cancellation" }),
    ).not.toBeInTheDocument();
    expect(
      view.container.querySelector<HTMLDetailsElement>("#run-attempts")?.open,
    ).toBe(false);
    const library =
      view.container.querySelector<HTMLDetailsElement>("#run-artifacts");
    expect(library?.open).toBe(false);
    await waitFor(() =>
      expect(library?.querySelector("summary")).toHaveTextContent("1 Artifact"),
    );
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
    expect(within(panel as HTMLElement).getByText("Published")).toBeVisible();
    expect(within(panel as HTMLElement).getByText("Failed")).toBeVisible();
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
    expect(screen.getByText("Succeeded")).toBeInTheDocument();
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
    expect(
      screen.getByText("Current revision", { selector: ".lede" }),
    ).toBeVisible();
    expect(screen.getByText("yes")).toBeInTheDocument();
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Versions" }));
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

  describe("Run page header", () => {
    function renderRun(run: RunStatus, path = "/runs/run-router") {
      const writes: Request[] = [];
      const api = new PublicAPI(runtimeConfig, async (input, init) => {
        const request = new Request(input, init);
        const common = sessionOrArtifacts(request);
        if (common !== undefined) return common;
        if (request.method !== "GET") {
          writes.push(request);
          throw new Error(`unexpected ${request.method} ${request.url}`);
        }
        const url = new URL(request.url);
        if (url.pathname === "/v1/runs/run-router") return apiResponse(run);
        if (url.pathname === "/v1/runs") {
          return apiResponse({
            items: [
              {
                runId: "run-router",
                workflow: run.workflow,
                state: run.state,
                labels: {},
                createdAt: "2026-08-31T12:00:00Z",
                updatedAt: "2026-08-31T12:01:00Z",
              },
            ],
            page: { hasMore: false },
          });
        }
        return apiResponse(
          {
            code: "not_found",
            message: "Not in this fixture",
            retryable: false,
          },
          { status: 404 },
        );
      });
      return { ...renderRunApplication(api, path), writes };
    }

    const failedAttempt = {
      ...runFixture().attempts[0]!,
      state: "failed" as const,
    };

    it.each([
      ["running", runFixture({ eventCursor: undefined }), ["Cancel Run"]],
      [
        "waiting for a manual model retry",
        runFixture({
          state: "waiting",
          eventCursor: undefined,
          recovery: {
            code: "gateway_timeout",
            since: "2026-09-20T20:00:00Z",
            automaticUntil: "2026-09-20T20:05:00Z",
            requiresRetry: true,
          },
        }),
        ["Retry model connection", "Cancel Run"],
      ],
      [
        "cancelling",
        runFixture({ state: "cancelling", eventCursor: undefined }),
        [],
      ],
      [
        "failed with a continuable stage",
        runFixture({
          state: "failed",
          eventCursor: undefined,
          activeStageExecutionId: undefined,
          attempts: [failedAttempt],
          resumeStageExecutionId: failedAttempt.stageExecutionId,
        }),
        ["Continue from failed stage", "Configure another Run"],
      ],
      [
        "failed without continuation",
        runFixture({
          state: "failed",
          eventCursor: undefined,
          activeStageExecutionId: undefined,
          attempts: [failedAttempt],
        }),
        ["Configure another Run"],
      ],
      [
        "succeeded",
        runFixture({
          state: "succeeded",
          eventCursor: undefined,
          activeStageExecutionId: undefined,
        }),
        ["Configure another Run"],
      ],
    ] as const)(
      "offers only the actions a %s Run allows, as separate buttons",
      async (_label, run, expected) => {
        renderRun(run);
        const heading = await screen.findByRole("heading", {
          level: 1,
          name: "router-analysis@1",
        });
        const header = heading.closest("header")!;
        await waitFor(() =>
          expect(
            within(header).getByRole("button", { name: "Refresh" }),
          ).toBeEnabled(),
        );
        const actions = within(header)
          .getAllByRole("button")
          .map((button) => button.textContent?.trim())
          .filter((name) =>
            [
              "Retry model connection",
              "Continue from failed stage",
              "Configure another Run",
              "Cancel Run",
            ].includes(name ?? ""),
          );
        expect(actions).toEqual(expected);
      },
    );

    it("names the failed stage before continuing and sends nothing on Back", async () => {
      const view = renderRun(
        runFixture({
          state: "failed",
          eventCursor: undefined,
          activeStageExecutionId: undefined,
          attempts: [failedAttempt],
          resumeStageExecutionId: failedAttempt.stageExecutionId,
        }),
      );
      const user = userEvent.setup();
      const trigger = await screen.findByRole("button", {
        name: "Continue from failed stage",
      });
      await user.click(trigger);
      const dialog = screen.getByRole("alertdialog", {
        name: "Continue from failed stage?",
      });
      expect(dialog).toHaveTextContent("Retry stage analysis?");
      expect(dialog).toHaveTextContent(/may repeat external side effects/);
      expect(
        within(dialog).getByRole("button", { name: "Back" }),
      ).toHaveFocus();
      // Repeat stays a separate action with its own review.
      expect(
        within(dialog).queryByRole("button", {
          name: "Configure another Run",
        }),
      ).toBeNull();
      await user.click(within(dialog).getByRole("button", { name: "Back" }));
      expect(screen.queryByRole("alertdialog")).toBeNull();
      await waitFor(() => expect(trigger).toHaveFocus());
      expect(view.writes).toHaveLength(0);
    });

    it("keeps a dismissed cancellation free of requests", async () => {
      const view = renderRun(runFixture({ eventCursor: undefined }));
      const user = userEvent.setup();
      const trigger = await screen.findByRole("button", {
        name: "Cancel Run",
      });
      await user.click(trigger);
      const dialog = screen.getByRole("dialog", { name: "Cancel this Run?" });
      await user.type(within(dialog).getByLabelText("Reason"), "Not needed");
      await user.click(
        within(dialog).getByRole("button", { name: "Keep running" }),
      );
      expect(screen.queryByRole("dialog")).toBeNull();
      await waitFor(() => expect(trigger).toHaveFocus());
      expect(view.writes).toHaveLength(0);
      expect(
        within(trigger.closest("header")!).getByText("Running"),
      ).toBeInTheDocument();
    });

    it("shows the identifiers, live status and every technical section", async () => {
      const cancelled = runFixture({
        state: "cancelled",
        projectId: "project-payment",
        activeStageExecutionId: undefined,
        attempts: [{ ...runFixture().attempts[0]!, state: "cancelled" }],
        cancellation: {
          code: "user_cancelled",
          requestedAt: "2026-08-31T12:03:00Z",
          requestedBy: "owner",
          reason: "Wrong project",
        },
        finishedAt: "2026-08-31T12:04:00Z",
      });
      const view = renderRun(cancelled);
      const user = userEvent.setup();
      expect(
        await screen.findByRole("heading", {
          level: 1,
          name: "router-analysis@1",
        }),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: "Copy Run ID" }),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: "Copy workflow version" }),
      ).toBeInTheDocument();
      const facts = view.container.querySelector<HTMLElement>(".run-metadata")!;
      expect(within(facts).getByText("router-analysis@1")).toBeInTheDocument();
      expect(facts).toHaveTextContent("project-payment");
      expect(
        within(facts.closest("header")!).getByText("Cancelled"),
      ).toBeInTheDocument();
      expect(screen.getByText("Live events: connecting")).toBeInTheDocument();
      const triage = view.container.querySelector(".run-triage")!;
      expect(triage).toHaveClass("run-triage-cancelled");
      expect(triage).toHaveTextContent("Lifecycle reason");
      expect(triage).toHaveTextContent("User requested");

      const technical = screen
        .getByRole("heading", { name: "Technical details" })
        .closest("section")!;
      const sections = [
        ...technical.querySelectorAll<HTMLDetailsElement>(
          "details.runs-section",
        ),
      ];
      expect(
        sections.map(
          (section) =>
            section.querySelector(".runs-section-title")?.textContent,
        ),
      ).toEqual([
        "Execution metrics",
        "Ordered Stage attempts",
        "Run metadata labels",
        "Parameters and input revisions",
        "Runtime infrastructure configuration",
        "Run artifacts",
        "Cancellation record",
      ]);
      // Only the execution metrics of an unsuccessful Run start open.
      expect(sections.map((section) => section.open)).toEqual([
        true,
        false,
        false,
        false,
        false,
        false,
        false,
      ]);
      for (const section of sections.slice(1)) {
        await user.click(section.querySelector(":scope > summary")!);
        expect(section.open).toBe(true);
      }
      expect(
        within(sections[3]!).getByText("Review the project architecture"),
      ).toBeInTheDocument();
      expect(
        within(sections[3]!).getByRole("link", {
          name: "source inputs/source@input-r1",
        }),
      ).toHaveAttribute(
        "href",
        "/runs/run-router/artifacts/inputs/source?revision=input-r1",
      );
      expect(sections[6]).toHaveTextContent("Wrong project");
      expect(sections[6]).toHaveTextContent("owner");
      expect(
        view.container.querySelector("#attempt-stage-router-1"),
      ).not.toBeNull();
    });

    it("opens the focused attempt before following the triage shortcut", async () => {
      const view = renderRun(
        runFixture({
          state: "succeeded",
          eventCursor: undefined,
          activeStageExecutionId: undefined,
          attempts: [{ ...runFixture().attempts[0]!, state: "succeeded" }],
        }),
      );
      const shortcut = await screen.findByRole("link", {
        name: "Inspect focused attempt",
      });
      const attempts =
        view.container.querySelector<HTMLDetailsElement>("#run-attempts")!;
      const attempt = view.container.querySelector<HTMLDetailsElement>(
        "#attempt-stage-router-1",
      )!;
      // The focused attempt is expanded inside its collapsed section.
      expect(attempts.open).toBe(false);
      expect(attempt.open).toBe(true);
      // A finished Run without an event stream shows no live status line.
      expect(screen.queryByText(/^Live events/)).toBeNull();
      await userEvent.setup().click(shortcut);
      expect(attempts.open).toBe(true);
      expect(attempt.open).toBe(true);
    });

    it("leads back to the list view the Run was opened from", async () => {
      renderRun(
        runFixture({
          state: "failed",
          eventCursor: undefined,
          activeStageExecutionId: undefined,
          attempts: [failedAttempt],
        }),
        "/runs?view=completed&state=failed",
      );
      const user = userEvent.setup();
      await user.click(await screen.findByRole("link", { name: "run-router" }));
      const breadcrumb = await screen.findByRole("navigation", {
        name: "Breadcrumb",
      });
      // The rail already names the destination; the trail starts at the view.
      expect(
        within(breadcrumb).queryByRole("link", { name: "Runs" }),
      ).toBeNull();
      expect(
        within(breadcrumb).getByRole("link", { name: "Completed" }),
      ).toHaveAttribute("href", "/runs?view=completed&state=failed");
      expect(within(breadcrumb).getByText("run-router")).toHaveAttribute(
        "aria-current",
        "page",
      );
    });
  });
});
