import { act, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { components } from "../api/generated/public";
import { conforming } from "../test/builders";
import {
  emptyPageOrNotFound,
  failure,
  fakeServer,
  json,
  page,
  type FakeCall,
  type Routes,
} from "../test/fake-server";
import { CSRF_TOKEN } from "../test/render";
import { FakeEventSocket, renderApplication } from "../test/application";

type QueueItem = components["schemas"]["RunQueueItem"];

function queueItem(
  runId: string,
  state: QueueItem["state"] = "running",
  overrides: Partial<QueueItem> = {},
): QueueItem {
  return conforming("RunQueueItem", {
    runId,
    workflow: "openapi-from-workspace@5",
    state,
    labels: {},
    eventCursor: { generation: `events-${runId}`, sequence: "1" },
    createdAt: "2026-09-05T08:00:00Z",
    updatedAt: "2026-09-05T08:01:00Z",
    ...overrides,
  });
}

const PAYMENT = {
  projectId: "project-payment",
  name: "Payment service",
  kind: "project",
} as const;

/** The owner's admission gate at `revision`, with its ETag. */
function queueControl(paused = false, revision = "0"): Response {
  return json(
    {
      paused,
      revision,
      ...(revision === "0" ? {} : { updatedAt: "2026-09-05T08:02:00Z" }),
    },
    200,
    { ETag: `"${revision}"` },
  );
}

/**
 * A signed-in owner whose Queue `queue` answers, with admission running;
 * `routes` add or replace operations.
 */
function renderQueue(
  queue: (call: FakeCall) => Response,
  path = "/runs",
  routes: Routes = {},
) {
  const view = renderApplication(path, {
    server: fakeServer(
      {
        "GET /v1/queue/control": () => queueControl(),
        "GET /v1/queue": queue,
        ...routes,
      },
      emptyPageOrNotFound,
    ),
  });
  return {
    ...view,
    /** The Queue reads so far. */
    queueReads: () =>
      view.sent("GET", "/v1/queue").map((request) => new URL(request.url)),
  };
}

/** Opens the first event socket and confirms its first subscription. */
async function subscribed() {
  await waitFor(() => expect(FakeEventSocket.instances).toHaveLength(1));
  const socket = FakeEventSocket.instances[0]!;
  act(() => socket.open());
  const subscription = JSON.parse(socket.sent[0] ?? "{}");
  act(() =>
    socket.message({
      version: "contractor.events.v1",
      type: "subscribed",
      subscriptionId: subscription.subscriptionId,
      stream: subscription.stream,
      cursor: subscription.after,
    }),
  );
  return { socket, subscription };
}

describe("Runs Queue view", () => {
  it("groups all active Run contexts and keeps filters across stable pages", async () => {
    const { router, user, queueReads } = renderQueue(({ url }) => {
      if (url.searchParams.get("cursor") === "next-queue") {
        return json(page([queueItem("run-next", "cancelling")]));
      }
      if (url.searchParams.get("membership") === "project") {
        return json(
          page([queueItem("run-project", "running", { project: PAYMENT })]),
        );
      }
      return json(
        page(
          [
            queueItem("run-standalone", "initializing"),
            queueItem("run-project", "running", { project: PAYMENT }),
            queueItem("run-eval", "running", {
              labels: { "eval.leg": "a", purpose: "eval" },
              project: {
                projectId: "evaluation-openapi",
                name: "OpenAPI regression",
                kind: "evaluation",
              },
            }),
          ],
          "next-queue",
        ),
      );
    });

    expect(
      await screen.findByRole("link", { name: "run-standalone" }),
    ).toBeInTheDocument();
    expect(screen.getByText("No Project")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Oldest first. Position here is not Scheduler priority.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: "Payment service" }),
    ).toHaveAttribute("href", "/projects/project-payment");
    expect(screen.getByText("OpenAPI regression")).toBeInTheDocument();
    expect(screen.getByText("eval.leg").closest("code")).toHaveTextContent(
      "eval.leg:a",
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

    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(
      await screen.findByRole("link", { name: "run-next" }),
    ).toBeInTheDocument();

    const context = screen.getByRole("group", { name: "Context" });
    await user.click(within(context).getByRole("button", { name: "Projects" }));
    expect(
      await screen.findByRole("link", { name: "run-project" }),
    ).toBeInTheDocument();
    expect(queueReads().at(-1)?.searchParams.get("membership")).toBe("project");
    expect(queueReads().at(-1)?.searchParams.has("cursor")).toBe(false);
    expect(
      within(context).getByRole("button", { name: "Projects" }),
    ).toHaveAttribute("aria-pressed", "true");

    const state = screen.getByRole("group", { name: "State" });
    await user.click(within(state).getByRole("button", { name: "Running" }));
    await waitFor(() =>
      expect(queueReads().at(-1)?.searchParams.get("state")).toBe("running"),
    );
    expect(router.state.location.search).toBe(
      "?membership=project&state=running",
    );

    await user.click(within(context).getByRole("button", { name: "All" }));
    await waitFor(() =>
      expect(queueReads().at(-1)?.searchParams.has("membership")).toBe(false),
    );
    expect(router.state.location.search).toBe("?state=running");
  });

  it("drops the page cursor when a tab link changes the filters", async () => {
    const { router, user, queueReads } = renderQueue(({ url }) => {
      if (url.searchParams.get("cursor") === "running-2") {
        return json(page([queueItem("run-running-2")]));
      }
      if (url.searchParams.get("state") === "running") {
        return json(page([queueItem("run-running-1")], "running-2"));
      }
      return json(page([queueItem("run-any", "initializing")]));
    }, "/runs?state=running");

    expect(
      await screen.findByRole("link", { name: "run-running-1" }),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(
      await screen.findByRole("link", { name: "run-running-2" }),
    ).toBeInTheDocument();
    expect(router.state.location.search).toBe(
      "?state=running&cursor=running-2",
    );

    const views = screen.getByRole("navigation", { name: "Run views" });
    await user.click(within(views).getByRole("link", { name: "Queue" }));
    expect(
      await screen.findByRole("link", { name: "run-any" }),
    ).toBeInTheDocument();
    expect(router.state.location.search).toBe("");
    expect(queueReads().at(-1)?.searchParams.has("state")).toBe(false);
    expect(queueReads().at(-1)?.searchParams.has("cursor")).toBe(false);
    expect(
      screen.queryByRole("navigation", { name: "Queue pages" }),
    ).toBeNull();

    // History restores the filter together with the cursor issued for it.
    await act(async () => {
      await router.navigate(-1);
    });
    expect(
      await screen.findByRole("link", { name: "run-running-2" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Previous" })).toBeEnabled();
  });

  it("removes a terminal Run after its lifecycle event", async () => {
    let terminal = false;
    const { queueReads } = renderQueue(() =>
      json(page(terminal ? [] : [queueItem("run-live")])),
    );

    expect(
      await screen.findByRole("link", { name: "run-live" }),
    ).toBeInTheDocument();
    const { socket, subscription } = await subscribed();
    expect(subscription).toMatchObject({
      stream: { kind: "run", id: "run-live" },
      after: { generation: "events-run-live", sequence: "1" },
    });

    terminal = true;
    act(() =>
      socket.message({
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
    expect(queueReads().length).toBeGreaterThan(1);
  });

  it("resubscribes after a resync even when the refetched cursors are unchanged", async () => {
    const { queueReads } = renderQueue(() =>
      json(page([queueItem("run-live")])),
    );

    expect(
      await screen.findByRole("link", { name: "run-live" }),
    ).toBeInTheDocument();
    const { socket, subscription } = await subscribed();
    const readsBeforeResync = queueReads().length;
    act(() =>
      socket.message({
        version: "contractor.events.v1",
        type: "resync_required",
        subscriptionId: subscription.subscriptionId,
        stream: subscription.stream,
        reason: "sequence_gap",
      }),
    );

    await waitFor(() =>
      expect(queueReads().length).toBeGreaterThan(readsBeforeResync),
    );
    // Only that subscription is resynchronized; it resumes on the same socket.
    await waitFor(() =>
      expect(JSON.parse(socket.sent.at(-1) ?? "{}")).toMatchObject({
        type: "subscribe",
        stream: { kind: "run", id: "run-live" },
        after: { generation: "events-run-live", sequence: "1" },
      }),
    );
    expect(JSON.parse(socket.sent.at(-1) ?? "{}").subscriptionId).not.toBe(
      subscription.subscriptionId,
    );
    expect(FakeEventSocket.instances).toHaveLength(1);
  });

  it("retries live subscriptions later when the resync read fails", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      let failReads = false;
      const { queueReads } = renderQueue(() => {
        if (failReads) {
          failReads = false;
          return failure(503, "unavailable", "Queue unavailable", true);
        }
        return json(page([queueItem("run-live")]));
      });

      expect(
        await screen.findByRole("link", { name: "run-live" }),
      ).toBeInTheDocument();
      const { socket, subscription } = await subscribed();
      failReads = true;
      const readsBeforeResync = queueReads().length;
      act(() =>
        socket.message({
          version: "contractor.events.v1",
          type: "resync_required",
          subscriptionId: subscription.subscriptionId,
          stream: subscription.stream,
          reason: "sequence_gap",
        }),
      );
      await waitFor(() =>
        expect(queueReads()).toHaveLength(readsBeforeResync + 1),
      );
      const subscribes = () =>
        socket.sent
          .map((frame) => JSON.parse(frame))
          .filter((frame) => frame.type === "subscribe");
      expect(subscribes()).toHaveLength(1);

      await act(async () => {
        await vi.advanceTimersByTimeAsync(10_000);
      });
      await waitFor(() => expect(subscribes()).toHaveLength(2));
      expect(subscribes().at(-1)).toMatchObject({
        stream: { kind: "run", id: "run-live" },
        after: { generation: "events-run-live", sequence: "1" },
      });
    } finally {
      vi.useRealTimers();
    }
  });

  it("resubscribes after the Server refuses the live session", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      renderQueue(() => json(page([queueItem("run-live")])));

      expect(
        await screen.findByRole("link", { name: "run-live" }),
      ).toBeInTheDocument();
      const { socket } = await subscribed();
      // The Server's session check closes the socket with 1008; the manager
      // does not reconnect or request a resync on its own.
      act(() => socket.serverClose(1008));
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1_000);
      });
      expect(FakeEventSocket.instances).toHaveLength(1);

      await act(async () => {
        await vi.advanceTimersByTimeAsync(10_000);
      });
      await waitFor(() => expect(FakeEventSocket.instances).toHaveLength(2));
      const replacement = FakeEventSocket.instances[1]!;
      act(() => replacement.open());
      expect(JSON.parse(replacement.sent[0] ?? "{}")).toMatchObject({
        type: "subscribe",
        stream: { kind: "run", id: "run-live" },
        after: { generation: "events-run-live", sequence: "1" },
      });
    } finally {
      vi.useRealTimers();
    }
  });

  it("opens Queue deep links with active filters", async () => {
    const { router, queueReads } = renderQueue(
      () => json(page([])),
      "/runs?membership=project&state=running",
    );

    expect(
      await screen.findByText("No active Runs match this view."),
    ).toBeInTheDocument();
    expect(router.state.location.pathname).toBe("/runs");
    expect(router.state.location.search).toBe(
      "?membership=project&state=running",
    );
    expect(queueReads()).toHaveLength(1);
    expect(queueReads()[0]?.searchParams.get("membership")).toBe("project");
    expect(queueReads()[0]?.searchParams.get("state")).toBe("running");
  });

  it("pauses and resumes admission with one authoritative action", async () => {
    let paused = false;
    let revision = 0;
    const { user, sent } = renderQueue(() => json(page([])), "/runs", {
      "GET /v1/queue/control": () => queueControl(paused, String(revision)),
      "PUT /v1/queue/control": async ({ request }) => {
        const body = (await request.json()) as { paused: boolean };
        paused = body.paused;
        revision += 1;
        return queueControl(paused, String(revision));
      },
    });
    const controlWrites = () => sent("PUT", "/v1/queue/control");

    await user.click(
      await screen.findByRole("button", { name: "Pause queue" }),
    );
    expect(
      await screen.findByRole("button", { name: "Resume queue" }),
    ).toBeEnabled();
    expect(screen.getByText("Admission paused")).toBeInTheDocument();
    expect(
      screen.getByText(/No next Stage starts until you resume/),
    ).toBeInTheDocument();
    expect(controlWrites()[0]?.headers.get("If-Match")).toBe('"0"');
    expect(controlWrites()[0]?.headers.get("X-CSRF-Token")).toBe(CSRF_TOKEN);

    await user.click(screen.getByRole("button", { name: "Resume queue" }));
    expect(
      await screen.findByRole("button", { name: "Pause queue" }),
    ).toBeEnabled();
    expect(screen.getByText("Admission running")).toBeInTheDocument();
    expect(controlWrites()[1]?.headers.get("If-Match")).toBe('"1"');
  });

  it("shows an unknown admission state and keeps the control disabled when the gate cannot be read", async () => {
    renderQueue(() => json(page([])), "/runs", {
      "GET /v1/queue/control": () =>
        failure(503, "unavailable", "Queue control unavailable", true),
    });

    expect(
      await screen.findByText("Could not read the queue admission state"),
    ).toBeInTheDocument();
    expect(screen.getByText("Admission unknown")).toBeInTheDocument();
    expect(screen.queryByText("Admission running")).toBeNull();
    expect(screen.getByRole("button", { name: "Pause queue" })).toBeDisabled();
  });

  it("marks Queue rows for the compact narrow layout without hiding facts", async () => {
    renderQueue(() =>
      json(
        page([
          queueItem("run-plain", "initializing"),
          queueItem("run-labelled", "running", { labels: { purpose: "eval" } }),
        ]),
      ),
    );

    const plain = (
      await screen.findByRole("link", { name: "run-plain" })
    ).closest("tr")!;
    const labelled = screen
      .getByRole("link", { name: "run-labelled" })
      .closest("tr")!;
    expect(within(plain).getByText("Initializing")).toBeInTheDocument();
    expect(within(labelled).getByText("Running")).toBeInTheDocument();
    // Phone rows label each fact themselves, since the header row is hidden.
    expect(
      [...plain.querySelectorAll(".runs-cell-label")].map(
        (label) => label.textContent,
      ),
    ).toEqual(["Workflow", "Context", "Labels", "Updated"]);
    expect(plain.querySelector(".run-list-labels-cell")).toHaveAttribute(
      "data-empty",
    );
    expect(labelled.querySelector(".run-list-labels-cell")).not.toHaveAttribute(
      "data-empty",
    );
    expect(
      screen.queryByRole("columnheader", { name: /^(position|rank|eta)$/i }),
    ).toBeNull();
  });
});
