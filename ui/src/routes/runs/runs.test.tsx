import { act, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type {
  PlannerPlan,
  RunStatus,
  RunSummary,
  StageAttempt,
} from "../../api/runs";
import { createApplicationQueryClient } from "../../app/query-client";
import { conforming, DIGEST } from "../../test/builders";
import { deferred } from "../../test/deferred";
import {
  runStatus,
  stageAttempt,
  workflowResource,
} from "../../test/builders-execution";
import { fakeServer, type Routes } from "../../test/fake-server";
import {
  bytes,
  conflict,
  emptyPageOrNotFound,
  failure,
  json,
  noContent,
  page,
} from "../../test/responses";
import { CSRF_TOKEN } from "../../test/render";
import { FakeEventSocket, renderApplication } from "../../test/application";

const executionConfig: StageAttempt["executionConfig"] = {
  variant: "failed_escalation",
  escalationOrdinal: 1,
  ref: { configId: "strong-review", version: "2", digest: DIGEST },
  planner: {
    modelPolicy: { policyId: "planner-strong", version: "2", digest: DIGEST },
    llmGateway: { gatewayId: "local", version: "1", digest: DIGEST },
    credential: { credentialId: "planner-budget" },
    origins: {
      modelPolicy: "executionConfig:strong-review@2",
      llmGateway: "workflow:router-analysis@1",
      credential: "run override",
    },
  },
  agents: {
    reviewer: {
      modelPolicy: { policyId: "worker-strong", version: "2", digest: DIGEST },
      llmGateway: { gatewayId: "local", version: "1", digest: DIGEST },
      credential: { credentialId: "worker-budget" },
      origins: {
        modelPolicy: "executionConfig:strong-review@2",
        llmGateway: "workflow:router-analysis@1",
        credential: "run override",
      },
    },
  },
};

const basePlan: PlannerPlan = {
  revision: 1,
  subtasks: [
    {
      id: "0",
      objective: "Inspect the source tree",
      instructions: "Use bounded source search.",
      status: "pending",
    },
  ],
  currentSubtaskId: "0",
};

/** Overrides in which `undefined` removes the member. */
type RunOverrides = { [K in keyof RunStatus]?: RunStatus[K] | undefined };

/** The running analysis attempt of the default Run. */
const RUNNING_ATTEMPT = stageAttempt("stage-router-1", {
  stage: "analysis",
  objective: "Produce a global architecture review.",
  executionConfig,
  state: "running",
  plan: basePlan,
  createdAt: "2026-08-31T12:00:00Z",
  updatedAt: "2026-08-31T12:01:00Z",
  plannerStartedAt: "2026-08-31T12:00:01Z",
});

/**
 * Run "run-router" of router-analysis@1, running its first attempt with a
 * live event cursor, unless overridden; an override of `undefined` removes
 * the member.
 */
function runFixture(overrides: RunOverrides = {}): RunStatus {
  const run: Record<string, unknown> = {
    workflow: "router-analysis@1",
    state: "running",
    deletable: false,
    labels: {
      "eval.id": "eval-router-01",
      "eval.leg": "a",
      purpose: "eval",
    },
    parameters: { objective: "Review the project architecture" },
    inputs: {
      source: { namespace: "inputs", name: "source", revision: "input-r1" },
    },
    attempts: [RUNNING_ATTEMPT],
    eventCursor: { generation: "run-generation-1", sequence: "10" },
    activeStageExecutionId: "stage-router-1",
    createdAt: "2026-08-31T12:00:00Z",
    updatedAt: "2026-08-31T12:01:00Z",
    startedAt: "2026-08-31T12:00:00Z",
    ...overrides,
  };
  for (const [key, value] of Object.entries(run)) {
    if (value === undefined) delete run[key];
  }
  return runStatus("run-router", run as Partial<RunStatus>);
}

/** The list item of a terminal Run. */
function summary(
  runId: string,
  overrides: Partial<RunSummary> = {},
): RunSummary {
  return conforming("RunSummary", {
    runId,
    workflow: "router-analysis@1",
    state: "succeeded",
    deletable: true,
    labels: {},
    createdAt: "2026-08-31T12:00:00Z",
    updatedAt: "2026-08-31T12:01:00Z",
    ...overrides,
  });
}

/** Exact Artifact metadata of a Run. */
function artifactMetadata(
  artifact: { namespace: string; name: string; revision: string },
  mediaType = "text/plain",
  size = 12,
) {
  return conforming("ArtifactMetadata", {
    artifact,
    mediaType,
    size,
    current: true,
    frozen: true,
    createdAt: "2026-08-31T12:05:00Z",
  });
}

/** The Run's empty Artifact library. */
const RUN_PAGE: Routes = {
  "GET /v1/runs/{runId}/artifacts": () => json(page([])),
};

/** The Run page of `run()`, read again on every request. */
function runServer(run: () => RunStatus, routes: Routes = {}) {
  return fakeServer({
    ...RUN_PAGE,
    "GET /v1/runs/{runId}": () => json(run()),
    ...routes,
  });
}

/** How often the Run page read the Run. */
function runReads(view: ReturnType<typeof renderApplication>): number {
  return view.sent("GET", "/v1/runs/run-router").length;
}

/** Opens the Run's event stream and acknowledges its subscription. */
async function subscribe() {
  await waitFor(() => expect(FakeEventSocket.instances).toHaveLength(1));
  const socket = FakeEventSocket.instances[0]!;
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
  return {
    socket,
    subscription,
    /** An event frame of the Run's stream at `sequence`. */
    frame: (sequence: string, occurredAt = "2026-08-31T12:02:00Z") => ({
      version: "contractor.events.v1",
      type: "event",
      subscriptionId: subscription.subscriptionId,
      stream: { kind: "run", id: "run-router" },
      cursor: { generation: "run-generation-1", sequence },
      occurredAt,
    }),
  };
}

describe("Run routes", () => {
  it("keeps attempt and Planner disclosures through lifecycle refetches and Refresh", async () => {
    const first = { ...RUNNING_ATTEMPT, state: "failed" as const };
    const second = {
      ...RUNNING_ATTEMPT,
      stageExecutionId: "stage-router-2",
      attempt: 2,
    };
    let currentRun = runFixture({
      attempts: [first, second],
      activeStageExecutionId: "stage-router-2",
    });
    const view = renderApplication("/runs/run-router", {
      server: runServer(() => currentRun),
    });
    const { socket, frame } = await subscribe();
    const attempt = (id: string) =>
      view.container.querySelector<HTMLDetailsElement>(`#attempt-${id}`)!;
    const instructions = (id: string) =>
      attempt(id).querySelector<HTMLDetailsElement>(
        ".runs-subtask-list details",
      )!;
    await waitFor(() => expect(attempt("stage-router-2").open).toBe(true));
    expect(attempt("stage-router-1").open).toBe(false);
    const user = view.user;
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
    currentRun = runFixture({
      attempts: [first, { ...second, state: "failed" }, third],
      activeStageExecutionId: "stage-router-3",
      eventCursor: { generation: "run-generation-1", sequence: "11" },
    });
    act(() =>
      socket.message({
        ...frame("11"),
        kind: "lifecycle.changed",
        data: { runId: "run-router", resource: "run", state: "running" },
      }),
    );
    await waitFor(() => expect(runReads(view)).toBeGreaterThan(1));
    await waitFor(() => expect(attempt("stage-router-3").open).toBe(true));
    expect(attempt("stage-router-1")).not.toBe(originalAttempt);
    expect(attempt("stage-router-1").open).toBe(true);
    expect(attempt("stage-router-2").open).toBe(false);
    expect(instructions("stage-router-2").open).toBe(true);

    const priorRefreshAttempt = attempt("stage-router-1");
    const readsBeforeRefresh = runReads(view);
    await user.click(screen.getByRole("button", { name: "Refresh" }));
    await waitFor(() =>
      expect(runReads(view)).toBeGreaterThan(readsBeforeRefresh),
    );
    await waitFor(() =>
      expect(attempt("stage-router-1")).not.toBe(priorRefreshAttempt),
    );
    expect(attempt("stage-router-1").open).toBe(true);
    expect(attempt("stage-router-2").open).toBe(false);
    expect(instructions("stage-router-2").open).toBe(true);
    expect(attempt("stage-router-3").open).toBe(true);
  });

  it("shows a waiting invocation and retries the model without a new stage", async () => {
    let current = runFixture({
      state: "waiting",
      recovery: {
        code: "model_unavailable",
        since: "2026-09-20T20:00:00Z",
        automaticUntil: "2026-09-20T20:05:00Z",
        requiresRetry: true,
      },
    });
    const view = renderApplication("/runs/run-router", {
      server: runServer(() => current, {
        "POST /v1/runs/{runId}/retry-gateway": () => {
          current = runFixture();
          return json({ runId: "run-router" }, 202);
        },
      }),
    });
    const retries = () =>
      view.sent("POST", "/v1/runs/run-router/retry-gateway");
    const retry = await screen.findByRole("button", {
      name: "Retry model connection",
    });
    expect(
      screen.getByText("The model was unloaded or is unavailable."),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Continue from failed stage" }),
    ).not.toBeInTheDocument();
    const user = view.user;
    await user.click(retry);
    // Retry has its own confirmation, distinct from Continue and Repeat.
    const dialog = screen.getByRole("alertdialog", {
      name: "Retry the model connection?",
    });
    expect(dialog).toHaveTextContent(
      "The model was unloaded or is unavailable.",
    );
    expect(dialog).toHaveTextContent("no new stage attempt starts");
    await user.click(within(dialog).getByRole("button", { name: "Not now" }));
    expect(retries()).toHaveLength(0);
    await waitFor(() => expect(retry).toHaveFocus());

    await user.click(retry);
    await user.click(
      within(
        screen.getByRole("alertdialog", {
          name: "Retry the model connection?",
        }),
      ).getByRole("button", { name: "Confirm retry" }),
    );
    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: "Retry model connection" }),
      ).not.toBeInTheDocument(),
    );
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(retries()).toHaveLength(1);
    expect(await retries()[0]!.json()).toEqual({});
    expect(current.attempts).toHaveLength(1);
    await waitFor(() =>
      expect(screen.getByRole("heading", { level: 1 })).toHaveFocus(),
    );
  });

  it("reads a waiting Run when automatic recovery is due and slows down once a manual retry is required", async () => {
    // Fake timers that also follow real time: the jumps below replace the
    // seconds the polling waits, so no assertion depends on the wall clock.
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      const now = Date.now();
      const automaticUntil = new Date(now + 3_300).toISOString();
      const recovery: NonNullable<RunStatus["recovery"]> = {
        code: "model_unavailable",
        since: new Date(now).toISOString(),
        nextRetryAt: new Date(now + 100).toISOString(),
        automaticUntil,
        requiresRetry: false,
      };
      // No event stream: only the polling rule reads the Run again.
      const waiting = (value: RunStatus["recovery"]) =>
        runFixture({
          state: "waiting",
          eventCursor: undefined,
          recovery: value,
        });
      let withoutRecovery = false;
      const view = renderApplication("/runs/run-router", {
        server: runServer(() =>
          withoutRecovery
            ? waiting(undefined)
            : waiting({
                ...recovery,
                requiresRetry: Date.now() >= Date.parse(automaticUntil),
              }),
        ),
      });
      expect(
        await screen.findByText(/Next recovery check:/),
      ).toBeInTheDocument();
      // The recovery check is due: the Run is read at least every 2 s.
      await act(() => vi.advanceTimersByTimeAsync(2_000));
      await waitFor(() => expect(runReads(view)).toBeGreaterThanOrEqual(2));
      expect(
        screen.queryByRole("button", { name: "Retry model connection" }),
      ).not.toBeInTheDocument();
      // Automatic recovery ends without a Run event; the next read shows it.
      await act(() => vi.advanceTimersByTimeAsync(2_000));
      expect(
        await screen.findByRole("button", { name: "Retry model connection" }),
      ).toBeInTheDocument();
      expect(runReads(view)).toBeGreaterThanOrEqual(3);
      // A manual retry is required: the 2 s reads stop (the next one is
      // 20 s away).
      const readsAtRetry = runReads(view);
      await act(() => vi.advanceTimersByTimeAsync(10_000));
      expect(runReads(view)).toBe(readsAtRetry);

      withoutRecovery = true;
      await view.user.click(screen.getByRole("button", { name: "Refresh" }));
      await waitFor(() =>
        expect(
          screen.queryByRole("button", { name: "Retry model connection" }),
        ).not.toBeInTheDocument(),
      );
      const readsWithoutRecovery = runReads(view);
      await act(() => vi.advanceTimersByTimeAsync(10_000));
      expect(runReads(view)).toBe(readsWithoutRecovery);
    } finally {
      vi.useRealTimers();
    }
  });

  it("keeps a loaded Run visible when a background refetch fails", async () => {
    let fail = false;
    const view = renderApplication("/runs/run-router", {
      server: fakeServer({
        ...RUN_PAGE,
        "GET /v1/runs/{runId}": () =>
          fail
            ? failure(503, "unavailable", "Server unavailable", true)
            : json(runFixture()),
      }),
    });
    const stageHeading = {
      name: "Produce a global architecture review.",
    };
    expect(
      await screen.findByRole("heading", stageHeading),
    ).toBeInTheDocument();
    fail = true;
    await view.user.click(screen.getByLabelText("Refresh"));
    expect(
      await screen.findByText(/showing the last loaded data/i),
    ).toBeInTheDocument();
    expect(screen.getByRole("heading", stageHeading)).toBeInTheDocument();
    expect(
      screen.queryByText("Could not load this Run"),
    ).not.toBeInTheDocument();
  });

  it("continues a failed stage only after confirmation and refreshes the authoritative Run", async () => {
    const failed = { ...RUNNING_ATTEMPT, state: "failed" as const };
    let current = runFixture({
      state: "failed",
      resumeStageExecutionId: failed.stageExecutionId,
      attempts: [failed],
      activeStageExecutionId: undefined,
    });
    const view = renderApplication("/runs/run-router", {
      server: runServer(() => current, {
        "POST /v1/runs/{runId}/resume": () => {
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
          return json(
            {
              runId: "run-router",
              sourceStageExecutionId: failed.stageExecutionId,
              stageExecutionId: "stage-router-2",
            },
            202,
          );
        },
      }),
    });
    const posts = () => view.sent("POST", "/v1/runs/run-router/resume");
    const user = view.user;
    await user.click(
      await screen.findByRole("button", { name: "Continue from failed stage" }),
    );
    expect(posts()).toHaveLength(0);
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
    expect(posts()).toHaveLength(1);
    expect(await posts()[0]!.json()).toEqual({
      stageExecutionId: "stage-router-1",
    });
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
    renderApplication("/runs/run-router", { server: runServer(() => current) });
    expect(
      await screen.findByText(/Continuation is unavailable/),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Continue from failed stage" }),
    ).not.toBeInTheDocument();
  });

  it("keeps a failed Run unchanged after a continuation conflict", async () => {
    const current = runFixture({
      state: "failed",
      resumeStageExecutionId: "stage-router-1",
      activeStageExecutionId: undefined,
    });
    const view = renderApplication("/runs/run-router", {
      server: runServer(() => current, {
        "POST /v1/runs/{runId}/resume": () =>
          conflict("conflict", "Run cannot be continued"),
      }),
    });
    const user = view.user;
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
    await waitFor(() => expect(runReads(view)).toBeGreaterThan(1));
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
    const analysis = { ...RUNNING_ATTEMPT, state: "failed" as const };
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
    const view = renderApplication("/runs/run-router", {
      server: runServer(() => current, {
        "POST /v1/runs/{runId}/resume": () => {
          // Another tab continued the Run first: it is queued again.
          current = runFixture({
            state: "pending",
            eventCursor: undefined,
            activeStageExecutionId: undefined,
            attempts: [analysis],
          });
          return conflict("conflict", "The Run is no longer failed");
        },
      }),
    });
    const user = view.user;
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
    const posts = view.sent("POST", "/v1/runs/run-router/resume");
    expect(await Promise.all(posts.map((post) => post.json()))).toEqual([
      { stageExecutionId: "stage-router-1" },
    ]);
  });

  it("keeps a refused model retry on its own recovery and never reopens it unasked", async () => {
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
    const view = renderApplication("/runs/run-router", {
      server: runServer(() => current, {
        "POST /v1/runs/{runId}/retry-gateway": () => {
          // The model came back and the Run resumed before this request.
          current = runFixture({ eventCursor: undefined });
          return conflict(
            "conflict",
            "The Run is not waiting for a model retry",
          );
        },
      }),
    });
    const user = view.user;
    await user.click(
      await screen.findByRole("button", { name: "Retry model connection" }),
    );
    const dialog = screen.getByRole("alertdialog", {
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
      screen.getByRole("alertdialog", { name: "Retry the model connection?" }),
    ).toHaveTextContent("The model request timed out.");
    expect(view.sent("POST", "/v1/runs/run-router/retry-gateway")).toHaveLength(
      1,
    );
  });

  it("offers confirmed deletion only for server-deletable completed Runs", async () => {
    let deleted = false;
    const view = renderApplication("/runs?view=completed", {
      server: fakeServer({
        "GET /v1/runs": () =>
          json(
            page([
              ...(deleted
                ? []
                : [
                    summary("run-delete", {
                      finishedAt: "2026-08-31T12:01:00Z",
                    }),
                  ]),
              summary("run-release-pending", {
                state: "failed",
                deletable: false,
                createdAt: "2026-08-31T11:00:00Z",
                updatedAt: "2026-08-31T11:01:00Z",
                finishedAt: "2026-08-31T11:01:00Z",
              }),
            ]),
          ),
        "DELETE /v1/runs/{runId}": () => {
          deleted = true;
          return noContent();
        },
      }),
    });
    const deletions = () => view.sent("DELETE", "/v1/runs/run-delete");
    const user = view.user;

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
    expect(deletions()).toHaveLength(0);

    await user.click(trigger);
    await user.click(screen.getByRole("button", { name: "Delete Run" }));
    await waitFor(() =>
      expect(
        screen.queryByRole("link", { name: "run-delete" }),
      ).not.toBeInTheDocument(),
    );
    expect(deletions()).toHaveLength(1);
    expect(deletions()[0]?.headers.get("X-CSRF-Token")).toBe(CSRF_TOKEN);
    expect(await deletions()[0]?.text()).toBe("");
  });

  it("filters and paginates authoritative Run history", async () => {
    const view = renderApplication("/runs?view=completed", {
      server: fakeServer({
        "GET /v1/runs": ({ url }) => {
          if (url.searchParams.get("cursor") === "next-run") {
            return json(
              page([
                summary("run-second", {
                  workflow: "likec4-from-workspace@5",
                  createdAt: "2026-08-31T10:00:00Z",
                  updatedAt: "2026-08-31T10:10:00Z",
                  finishedAt: "2026-08-31T10:10:00Z",
                }),
              ]),
            );
          }
          if (url.searchParams.get("state") === "failed") {
            return json(page([]));
          }
          return json(
            page(
              [summary("run-first", { workflow: "openapi-from-workspace@5" })],
              "next-run",
            ),
          );
        },
      }),
    });
    const lastList = () =>
      new URL(view.sent("GET", "/v1/runs").at(-1)!.url).searchParams;
    expect(
      await screen.findByRole("link", { name: "run-first" }),
    ).toBeInTheDocument();
    const user = view.user;
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
    expect(lastList().get("state")).toBe("failed");
    expect(lastList().get("lifecycle")).toBe("terminal");
    expect(lastList().has("cursor")).toBe(false);
  });

  it("drops the completed page cursor when the tab link clears the filters", async () => {
    const failed = (runId: string) =>
      summary(runId, {
        workflow: "openapi-from-workspace@5",
        state: "failed",
        finishedAt: "2026-08-31T12:01:00Z",
      });
    const view = renderApplication("/runs?view=completed&state=failed", {
      server: fakeServer({
        "GET /v1/runs": ({ url }) => {
          if (url.searchParams.get("cursor") === "failed-2") {
            return json(page([failed("run-failed-2")]));
          }
          if (url.searchParams.get("state") === "failed") {
            return json(page([failed("run-failed-1")], "failed-2"));
          }
          return json(
            page([{ ...failed("run-any"), state: "succeeded" as const }]),
          );
        },
      }),
    });
    const lastList = () =>
      new URL(view.sent("GET", "/v1/runs").at(-1)!.url).searchParams;
    const user = view.user;

    expect(
      await screen.findByRole("link", { name: "run-failed-1" }),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(
      await screen.findByRole("link", { name: "run-failed-2" }),
    ).toBeInTheDocument();
    expect(view.router.state.location.search).toBe(
      "?view=completed&state=failed&cursor=failed-2",
    );
    expect(document.title).toBe("Completed · Runs · Contractor");

    const views = screen.getByRole("navigation", { name: "Run views" });
    await user.click(within(views).getByRole("link", { name: "Completed" }));
    expect(
      await screen.findByRole("link", { name: "run-any" }),
    ).toBeInTheDocument();
    expect(lastList().has("state")).toBe(false);
    expect(lastList().has("cursor")).toBe(false);
    expect(screen.queryByRole("navigation", { name: "Run pages" })).toBeNull();
  });

  it("honors a deep-linked Run state filter", async () => {
    const view = renderApplication("/runs?state=failed", {
      server: fakeServer({ "GET /v1/runs": () => json(page([])) }),
    });

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
    const lists = view.sent("GET", "/v1/runs");
    expect(lists).toHaveLength(1);
    const query = new URL(lists[0]!.url).searchParams;
    expect(query.get("state")).toBe("failed");
    expect(query.get("lifecycle")).toBe("terminal");
  });

  it("stays in Completed when a state deep link is cleared or filtered further", async () => {
    const view = renderApplication("/runs?state=failed", {
      server: fakeServer({ "GET /v1/runs": () => json(page([])) }),
    });
    const { router, user } = view;
    const lastList = () =>
      new URL(view.sent("GET", "/v1/runs").at(-1)!.url).searchParams;
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
    await waitFor(() => expect(lastList().has("state")).toBe(false));
    expect(lastList().get("lifecycle")).toBe("terminal");

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
    const evalLabels = (leg: string) => ({
      "eval.id": "eval-group=01",
      "eval.leg": leg,
      purpose: "eval",
    });
    const view = renderApplication(
      "/runs?view=completed&label=eval.id%3Deval-group%3D01&label=eval.leg%3Da",
      {
        server: fakeServer({
          "GET /v1/runs": ({ url }) => {
            if (url.searchParams.get("cursor") === "next-eval") {
              return json(
                page([
                  summary("run-leg-a-page-2", {
                    labels: evalLabels("a"),
                    createdAt: "2026-08-31T10:00:00Z",
                    updatedAt: "2026-08-31T10:10:00Z",
                    finishedAt: "2026-08-31T10:10:00Z",
                  }),
                ]),
              );
            }
            const leg = url.searchParams.getAll("label").includes("eval.leg=b")
              ? "b"
              : "a";
            return json(
              page(
                [summary(`run-leg-${leg}`, { labels: evalLabels(leg) })],
                "next-eval",
              ),
            );
          },
        }),
      },
    );
    const lastList = () =>
      new URL(view.sent("GET", "/v1/runs").at(-1)!.url).searchParams;
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
    const user = view.user;
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
    expect(lastList().get("cursor")).toBe("next-eval");
    expect(lastList().getAll("label")).toEqual([
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
    expect(lastList().has("cursor")).toBe(false);
    expect(lastList().getAll("label")).toEqual([
      "eval.id=eval-group=01",
      "eval.leg=b",
      "purpose=eval",
    ]);
  });

  it("shows a global task and advances only its nested typed Planner projection", async () => {
    let currentRun = runFixture();
    const view = renderApplication("/runs/run-router", {
      server: runServer(() => currentRun),
    });
    expect(
      await screen.findByRole("heading", {
        name: "Produce a global architecture review.",
      }),
    ).toBeInTheDocument();
    expect(screen.getByText("Inspect the source tree")).toBeInTheDocument();
    const { socket, subscription, frame } = await subscribe();
    expect(subscription.after).toEqual({
      generation: "run-generation-1",
      sequence: "10",
    });
    const livePlan: PlannerPlan = {
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
    const planner = {
      stageExecutionId: "stage-router-1",
      sessionId: "session-router-1",
      invocationId: "invocation-router-1",
    };
    act(() =>
      socket.message({
        ...frame("11"),
        kind: "planner.event",
        data: { ...planner, eventKind: "planner.plan_changed", plan: livePlan },
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
    expect(runReads(view)).toBe(1);
    act(() =>
      socket.message({
        ...frame("12", "2026-08-31T12:02:01Z"),
        kind: "planner.event",
        data: {
          ...planner,
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
      attempts: [{ ...RUNNING_ATTEMPT, plan: livePlan }],
      eventCursor: { generation: "run-generation-1", sequence: "13" },
      updatedAt: "2026-08-31T12:02:02Z",
    });
    act(() =>
      socket.message({
        ...frame("13", "2026-08-31T12:02:02Z"),
        kind: "lifecycle.changed",
        data: { runId: "run-router", resource: "run", state: "succeeded" },
      }),
    );
    await waitFor(() => expect(runReads(view)).toBeGreaterThan(1));
    // The lifecycle hint said succeeded; only the refetched Run state shows.
    expect(await screen.findByText("Cancelling")).toBeInTheDocument();
    expect(screen.queryByText("Succeeded")).toBeNull();
    expect(view.container.querySelector(".run-triage")).toHaveClass(
      "run-triage-cancelling",
    );
  });

  it("follows a new attempt's first Planner fact without a projection resync", async () => {
    const refetch = deferred();
    const retry = {
      ...RUNNING_ATTEMPT,
      stageExecutionId: "stage-router-2",
      objective: "Retry the architecture review.",
      attempt: 2,
      createdAt: "2026-08-31T12:02:00Z",
      updatedAt: "2026-08-31T12:02:00Z",
      plannerStartedAt: "2026-08-31T12:02:00Z",
    };
    delete retry.plan;
    let reads = 0;
    const view = renderApplication("/runs/run-router", {
      server: fakeServer({
        ...RUN_PAGE,
        "GET /v1/runs/{runId}": async () => {
          reads += 1;
          if (reads === 1) return json(runFixture());
          await refetch.promise;
          return json(
            runFixture({
              attempts: [{ ...RUNNING_ATTEMPT, state: "failed" }, retry],
              activeStageExecutionId: "stage-router-2",
              eventCursor: { generation: "run-generation-1", sequence: "12" },
            }),
          );
        },
      }),
    });
    const { socket, frame } = await subscribe();
    act(() => {
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
    await waitFor(() => expect(runReads(view)).toBe(2));
    expect(socket.readyState).toBe(1);
    expect(screen.queryByText(/REST resync after/)).toBeNull();

    await act(async () => {
      refetch.resolve();
      await refetch.promise;
    });
    expect(
      await screen.findByRole("heading", {
        name: "Retry the architecture review.",
      }),
    ).toBeInTheDocument();
    expect(runReads(view)).toBe(2);
    expect(FakeEventSocket.instances).toHaveLength(1);
  });

  it("titles the Run detail, Run Artifact and Runs pages", async () => {
    const { router } = renderApplication("/runs/run-router", {
      server: fakeServer(
        {
          ...RUN_PAGE,
          "GET /v1/runs/{runId}": () =>
            json(runFixture({ eventCursor: undefined })),
          "GET /v1/runs/{runId}/artifacts/{namespace}/{name}/metadata": () =>
            json(
              artifactMetadata(
                { namespace: "inputs", name: "source", revision: "input-r1" },
                "application/zip",
                3,
              ),
            ),
        },
        emptyPageOrNotFound,
      ),
    });
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
    const view = renderApplication("/runs/run-router", {
      server: runServer(() => currentRun, {
        "POST /v1/runs/{runId}/cancel": () => {
          currentRun = runFixture({
            state: "succeeded",
            eventCursor: undefined,
            activeStageExecutionId: undefined,
            finishedAt: "2026-08-31T12:05:00Z",
          });
          return json({ runId: "run-router", state: "succeeded" });
        },
      }),
    });
    const cancellations = () => view.sent("POST", "/v1/runs/run-router/cancel");
    const user = view.user;
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
    expect(cancellations()).toHaveLength(0);
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
    expect(cancellations()).toHaveLength(1);
    expect(await cancellations()[0]!.json()).toEqual({
      reason: "Stop after the current review",
    });
  });

  it("keeps cancellation retryable after a failed request while the Run is active", async () => {
    let currentRun = runFixture({ eventCursor: undefined });
    let calls = 0;
    const view = renderApplication("/runs/run-router", {
      server: runServer(() => currentRun, {
        "POST /v1/runs/{runId}/cancel": () => {
          calls += 1;
          if (calls === 1) {
            return failure(
              500,
              "unavailable",
              "Cancellation service unavailable",
              true,
            );
          }
          currentRun = runFixture({
            state: "cancelling",
            eventCursor: undefined,
          });
          return json({ runId: "run-router", state: "cancelling" }, 202);
        },
      }),
    });
    const cancellations = () => view.sent("POST", "/v1/runs/run-router/cancel");
    const user = view.user;
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
    expect(cancellations()).toHaveLength(1);
    await user.click(button);
    expect(await screen.findByText("Cleanup in progress")).toBeVisible();
    expect(screen.getByText("Cancelling")).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.queryByRole("button", { name: "Cancel Run" })).toBeNull();
    expect(cancellations()).toHaveLength(2);
  });

  it("surfaces the primary terminal cause and focuses its failed attempt", async () => {
    const failedRun = runFixture({
      state: "failed",
      eventCursor: undefined,
      activeStageExecutionId: undefined,
      attempts: [
        {
          ...RUNNING_ATTEMPT,
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
    const view = renderApplication("/runs/run-router", {
      server: runServer(() => failedRun),
    });
    const heading = await screen.findByRole("heading", {
      name: "Run failed in analysis",
    });
    const triage = heading.closest<HTMLElement>(".run-triage")!;
    expect(triage).not.toBeNull();
    expect(
      within(triage).getByText("planner_gateway_unavailable"),
    ).toBeInTheDocument();
    expect(
      within(triage).getByText(
        "The configured Planner gateway is unavailable.",
      ),
    ).toBeInTheDocument();
    expect(within(triage).getByText("Retryable")).toBeInTheDocument();
    // Execution metrics live under Technical details and stay open for a
    // failed Run.
    const metrics =
      view.container.querySelector<HTMLDetailsElement>("#run-metrics")!;
    expect(metrics.open).toBe(true);
    expect(within(metrics).getAllByText("34s").length).toBeGreaterThan(0);
    expect(within(metrics).getByText("1.2K")).toBeInTheDocument();
    expect(
      within(triage).getByRole("link", { name: "Inspect focused attempt" }),
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
      deletable: true,
      labels: {},
      eventCursor: undefined,
      activeStageExecutionId: undefined,
      attempts: [
        {
          ...RUNNING_ATTEMPT,
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
    const workflow = workflowResource("router-analysis", "1", {
      parameters: { objective: { required: true } },
    });
    const view = renderApplication("/runs/run-router", {
      server: fakeServer({
        ...RUN_PAGE,
        "GET /v1/runs/{runId}": ({ params }) =>
          json(
            params.runId === "run-repeat-new"
              ? {
                  ...terminal,
                  runId: "run-repeat-new",
                  state: "initializing",
                  attempts: [],
                  finishedAt: undefined,
                }
              : terminal,
          ),
        "GET /v1/runs/{runId}/repeat-draft": () =>
          json({
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
          }),
        "GET /v1/workflows/{name}/versions/{version}": () => json(workflow),
        "GET /v1/artifacts": () => json(page([])),
        "POST /v1/runs": () =>
          json(
            {
              runId: "run-repeat-new",
              state: "initializing",
              runtimeLabels: [],
              labels: {},
              runtimeConfiguration: terminal.runtimeConfiguration,
            },
            202,
          ),
      }),
    });
    const creations = () => view.sent("POST", "/v1/runs");
    const user = view.user;
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
    expect(creations()).toHaveLength(0);

    await user.click(review);
    await user.click(
      screen.getByRole("button", { name: "Start Workflow Run" }),
    );
    await waitFor(() => expect(creations()).toHaveLength(1));
    expect(await creations()[0]!.json()).toEqual({
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
      const view = renderApplication("/runs/run-router", {
        server: runServer(() => terminal, {
          "GET /v1/workflows/{name}/versions/{version}": () =>
            json(workflowResource("router-analysis", "1")),
          "GET /v1/runs/{runId}/repeat-draft": () =>
            json({
              sourceRunId: "run-router",
              authority: "ordinary",
              workflow: { name: "router-analysis", version: "1" },
              notices: [{ code, severity: "blocking", message }],
            }),
        }),
      });
      for (let attempt = 0; attempt < 2; attempt++) {
        await view.user.click(
          await screen.findByRole("button", { name: "Configure another Run" }),
        );
        expect(await screen.findByText(message)).toBeInTheDocument();
        expect(view.router.state.location.pathname).toBe("/runs/run-router");
      }
      expect(
        view.requests.filter((request) => request.method !== "GET"),
      ).toEqual([]);
      expect(
        view.sent("GET", "/v1/workflows/router-analysis/versions/1"),
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
    // Everything the Audit page reads answers 404: the fixture stops after
    // the navigation.
    const view = renderApplication("/runs/run-router", {
      server: runServer(() => terminal, {
        "GET /v1/runs/{runId}/repeat-draft": () =>
          json({
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
          }),
      }),
    });
    await view.user.click(
      await screen.findByRole("button", { name: "Configure another Run" }),
    );
    await waitFor(() =>
      expect(view.router.state.location.pathname).toBe(
        "/projects/project-audit/audits/audit-one",
      ),
    );
  });

  // AGENTS.md 3.8, V385-001 A7: a late answer never moves a user who left.
  it.each([
    [
      "an ordinary draft",
      {
        sourceRunId: "run-router",
        authority: "ordinary",
        workflow: { name: "router-analysis", version: "1" },
        notices: [],
        draft: {
          parameters: { objective: "Review the original project" },
          runtimeLabels: [],
          labels: {},
          executionConfig: { status: "available", value: {} },
          inputs: {},
        },
      },
    ],
    [
      "an Audit-managed Run",
      {
        sourceRunId: "run-router",
        authority: "audit-managed",
        workflow: { name: "router-analysis", version: "1" },
        projectId: "project-audit",
        auditId: "audit-one",
        notices: [],
      },
    ],
  ])(
    "stays where the user went when the repeat draft of %s answers late",
    async (_kind, repeatDraft) => {
      const terminal = runFixture({
        state: "failed",
        eventCursor: undefined,
        activeStageExecutionId: undefined,
        finishedAt: "2026-08-31T12:02:00Z",
      });
      const answer = deferred();
      const queryClient = createApplicationQueryClient();
      const view = renderApplication("/runs/run-router", {
        server: runServer(() => terminal, {
          "GET /v1/runs/{runId}/repeat-draft": async () => {
            await answer.promise;
            return json(repeatDraft);
          },
          "GET /v1/runs": () => json(page([])),
        }),
        queryClient,
      });
      await view.user.click(
        await screen.findByRole("button", { name: "Configure another Run" }),
      );
      await act(() => view.router.navigate("/settings"));
      expect(queryClient.isMutating()).toBe(1);
      answer.resolve();
      // Once the draft request settled, the code after it has run: the user
      // is still where they went, and no navigation is under way.
      await waitFor(() => expect(queryClient.isMutating()).toBe(0));
      await act(async () => undefined);
      expect(view.router.state.navigation.state).toBe("idle");
      expect(view.router.state.location.pathname).toBe("/settings");
    },
  );

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
        stageAttempt("stage-router-1", {
          stage: "analysis",
          objective: "Produce a global architecture review.",
          executionConfig: {
            variant: "base",
            planner: executionConfig.planner!,
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
        }),
        stageAttempt("stage-router-2", {
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
                    config: { name: "debug", version: "1", digest: DIGEST },
                  },
                ],
                runtimeAdapters: ["otlp-http@1"],
                origins: {
                  workerTelemetry: {
                    layer: "agent_labels",
                    configs: [{ name: "debug", version: "1", digest: DIGEST }],
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
        }),
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
    const view = renderApplication("/runs/run-router", {
      server: runServer(() => terminalRun, {
        "GET /v1/runs/{runId}/artifacts": () =>
          json(page([artifactMetadata(output)])),
        "GET /v1/runs/{runId}/artifacts/{namespace}/{name}/metadata": () =>
          json(artifactMetadata(output)),
        "GET /v1/runs/{runId}/artifacts/{namespace}/{name}": () =>
          bytes("safe report\n", "text/plain"),
      }),
    });
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
    await view.user.click(
      screen.getByRole("button", { name: "Preview result" }),
    );
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
    renderApplication("/runs/run-router", {
      server: runServer(() => projectRun),
    });

    const panel = (
      await screen.findByRole("heading", {
        name: "Reusable output status",
      })
    ).closest<HTMLElement>("section")!;
    expect(panel).not.toBeNull();
    expect(within(panel).getByText("Published")).toBeVisible();
    expect(within(panel).getByText("Failed")).toBeVisible();
    expect(panel).toHaveTextContent("source outputs/openapi@run-output-r1");
    expect(
      within(panel).getByRole("link", {
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
    const metadata = artifactMetadata({
      namespace: "outputs",
      name: "report",
      revision: "output-r2",
    });
    const view = renderApplication(
      "/runs/run-router/artifacts/outputs/report?revision=output-r2",
      {
        server: fakeServer({
          "GET /v1/runs/{runId}/artifacts/{namespace}/{name}/metadata": () =>
            json(metadata),
          "GET /v1/runs/{runId}/artifacts/{namespace}/{name}/versions": () =>
            json(page([metadata])),
          "GET /v1/runs/{runId}/artifacts/{namespace}/{name}/lineage": () =>
            json(
              page([
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
              ]),
            ),
          "GET /v1/runs/{runId}/artifacts/{namespace}/{name}": () =>
            bytes("safe report\n", "text/plain"),
        }),
      },
    );
    expect(
      await screen.findByRole("heading", { name: "outputs/report" }),
    ).toBeInTheDocument();
    expect((await screen.findAllByText("output-r2")).length).toBeGreaterThan(0);
    expect(
      screen.getByText("Current revision", { selector: ".lede" }),
    ).toBeVisible();
    expect(screen.getByText("yes")).toBeInTheDocument();
    await view.user.click(screen.getByRole("button", { name: "Versions" }));
    expect(await screen.findByText("output bind")).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: /Upload/ }),
    ).not.toBeInTheDocument();
    await view.user.click(screen.getByRole("button", { name: "Load preview" }));
    expect(
      await screen.findByText("safe report", { selector: "pre" }),
    ).toBeInTheDocument();
    expect(screen.queryByText(/token|secret prompt/i)).not.toBeInTheDocument();
  });

  describe("Run page header", () => {
    function renderRun(run: RunStatus, path = "/runs/run-router") {
      const view = renderApplication(path, {
        server: runServer(() => run, {
          "GET /v1/runs": () =>
            json(
              page([
                summary("run-router", {
                  workflow: run.workflow,
                  state: run.state,
                  deletable: run.deletable,
                }),
              ]),
            ),
        }),
      });
      return {
        ...view,
        /** Requests that would change something; none is expected. */
        writes: () =>
          view.requests.filter((request) => request.method !== "GET"),
      };
    }

    const failedAttempt = { ...RUNNING_ATTEMPT, state: "failed" as const };

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
      const user = view.user;
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
      expect(view.writes()).toHaveLength(0);
    });

    it("keeps a dismissed cancellation free of requests", async () => {
      const view = renderRun(runFixture({ eventCursor: undefined }));
      const user = view.user;
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
      expect(view.writes()).toHaveLength(0);
      expect(
        within(trigger.closest("header")!).getByText("Running"),
      ).toBeInTheDocument();
    });

    it("shows the identifiers, live status and every technical section", async () => {
      const cancelled = runFixture({
        state: "cancelled",
        projectId: "project-payment",
        activeStageExecutionId: undefined,
        attempts: [{ ...RUNNING_ATTEMPT, state: "cancelled" }],
        cancellation: {
          code: "user_cancelled",
          requestedAt: "2026-08-31T12:03:00Z",
          requestedBy: "owner",
          reason: "Wrong project",
        },
        finishedAt: "2026-08-31T12:04:00Z",
      });
      const view = renderRun(cancelled);
      const user = view.user;
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
          attempts: [{ ...RUNNING_ATTEMPT, state: "succeeded" }],
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
      await view.user.click(shortcut);
      expect(attempts.open).toBe(true);
      expect(attempt.open).toBe(true);
    });

    it("leads back to the list view the Run was opened from", async () => {
      const view = renderRun(
        runFixture({
          state: "failed",
          eventCursor: undefined,
          activeStageExecutionId: undefined,
          attempts: [failedAttempt],
        }),
        "/runs?view=completed&state=failed",
      );
      await view.user.click(
        await screen.findByRole("link", { name: "run-router" }),
      );
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
      expect(
        within(breadcrumb).getByText("run-router").closest("[aria-current]"),
      ).toHaveAttribute("aria-current", "page");
    });
  });
});
