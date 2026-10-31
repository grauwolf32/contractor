import { act, screen, within, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { renderApplication } from "../../test/application";
import { runStatus, workflowResource } from "../../test/builders-execution";
import { conforming } from "../../test/builders";
import { json, page, failure } from "../../test/responses";
import { readRunListOrigin, type RunListOrigin } from "./source-context";

const first = runStatus("run-first", {
  workflow: "review@1",
  state: "succeeded",
  deletable: true,
});
const second = runStatus("run-second", {
  workflow: "review@1",
  state: "succeeded",
  deletable: true,
});
function summary(runId: string) {
  return conforming("RunSummary", {
    runId,
    workflow: "review@1",
    state: "succeeded",
    deletable: true,
    labels: {},
    createdAt: "2026-10-31T10:00:00Z",
    updatedAt: "2026-10-31T10:01:00Z",
  });
}
const items = [summary(first.runId), summary(second.runId)];
const completed: RunListOrigin["runSource"] = {
  kind: "completed",
  label: "Completed",
  options: {
    lifecycle: "terminal",
    state: "succeeded",
    cursor: "next-page",
    labelSelectors: [{ key: "purpose", value: "eval" }],
  },
};
const origin = {
  returnTo:
    "/runs?view=completed&state=succeeded&cursor=next-page&label=purpose%3Deval",
  returnLabel: "Completed",
  returnState: { cursorStack: ["previous-page"] },
  runSource: completed,
};
const routes = {
  "GET /v1/runs/{runId}": ({ params }: { params: Record<string, string> }) =>
    json(params.runId === first.runId ? first : second),
  "GET /v1/workflows/{name}/versions/{version}": () =>
    json(workflowResource("review", "1")),
} as const;

describe("originating Run list", () => {
  it("captures the real filtered page, keeps its context through sibling selection and returns exactly", async () => {
    const view = renderApplication(
      {
        pathname: "/runs",
        search: origin.returnTo.slice(5),
        state: origin.returnState,
      },
      { server: { ...routes, "GET /v1/runs": () => json(page(items)) } },
    );
    await view.user.click(
      await screen.findByRole("link", { name: "run-first" }),
    );
    const pane = await screen.findByRole("region", { name: "Source Runs" });
    expect(within(pane).queryByRole("tab")).toBeNull();
    expect(within(pane).queryByRole("combobox")).toBeNull();
    expect(within(pane).queryByRole("link", { name: "Queue" })).toBeNull();
    const selected = await within(pane).findByRole("link", {
      name: /run-first/,
    });
    expect(selected).toHaveAttribute("aria-current", "true");
    const captured = view.router.state.location.state;
    expect(captured).toMatchObject(origin);
    expect(view.sent("GET", "/run-second")).toHaveLength(0);
    await view.user.click(
      within(pane).getByRole("link", { name: /run-second/ }),
    );
    await waitFor(() => expect(view.location()).toBe("/runs/run-second"));
    expect(view.router.state.location.state).toEqual(captured);
    expect(
      within(pane).getByRole("link", { name: /run-second/ }),
    ).toHaveAttribute("aria-current", "true");
    for (const request of view.sent("GET", "/v1/runs")) {
      const url = new URL(request.url);
      expect(url.searchParams.get("cursor")).toBe("next-page");
      expect(url.searchParams.get("state")).toBe("succeeded");
      expect(url.searchParams.get("lifecycle")).toBe("terminal");
      expect(url.searchParams.getAll("label")).toEqual(["purpose=eval"]);
    }
    await view.user.click(
      within(pane).getByRole("link", { name: "Back to Completed" }),
    );
    await waitFor(() => expect(view.location()).toBe(origin.returnTo));
    expect(view.router.state.location.state).toEqual(origin.returnState);
  });

  it("captures Queue state, membership and cursor from the real source link", async () => {
    const queued = conforming("RunQueueItem", {
      runId: first.runId,
      workflow: "review@1",
      state: "running",
      labels: {},
      eventCursor: { generation: "events-1", sequence: "1" },
      createdAt: "2026-10-31T10:00:00Z",
      updatedAt: "2026-10-31T10:01:00Z",
    });
    const source =
      "/runs?state=running&membership=standalone&cursor=queue-page";
    const view = renderApplication(source, {
      server: {
        ...routes,
        "GET /v1/queue": () => json(page([queued])),
        "GET /v1/runs/{runId}": () =>
          json(
            runStatus(first.runId, {
              workflow: "review@1",
              state: "running",
              deletable: false,
              eventCursor: queued.eventCursor,
            }),
          ),
      },
    });
    await view.user.click(
      await screen.findByRole("link", { name: first.runId }),
    );
    await screen.findByRole("region", { name: "Source Runs" });
    expect(view.router.state.location.state).toMatchObject({
      returnTo: source,
      returnLabel: "Queue",
      runSource: {
        kind: "queue",
        label: "Queue",
        options: {
          state: "running",
          membership: "standalone",
          cursor: "queue-page",
        },
        refreshMs: 10000,
      },
    });
  });

  it.each([
    {
      kind: "queue",
      label: "Queue",
      options: {
        state: "running",
        membership: "standalone",
        cursor: "queue-page",
      },
      refreshMs: 10000,
    },
    {
      kind: "project",
      label: "Project Runs",
      projectId: "project_example",
      options: {
        limit: 25,
        lifecycle: "terminal",
        state: "succeeded",
        cursor: "project-page",
      },
      refreshMs: 5000,
      settleWhenTerminal: true,
    },
    {
      kind: "project",
      label: "Recent Runs",
      projectId: "project_example",
      options: { limit: 5 },
      refreshMs: 10000,
    },
    {
      kind: "project",
      label: "Successful Runs",
      projectId: "project_example",
      options: { limit: 3, state: "succeeded" },
      refreshMs: 15000,
    },
  ])("reads exactly one existing page for $label", async (source) => {
    const isQueue = source.kind === "queue";
    const entry = {
      ...origin,
      returnTo: isQueue
        ? "/runs?membership=standalone&cursor=queue-page"
        : "/projects/project_example/runs?view=completed",
      runSource: source,
    };
    const queueItem = conforming("RunQueueItem", {
      runId: "run-queued",
      workflow: "review@1",
      state: "running",
      labels: {},
      eventCursor: { generation: "events-1", sequence: "1" },
      createdAt: "2026-10-31T10:00:00Z",
      updatedAt: "2026-10-31T10:01:00Z",
    });
    const view = renderApplication(
      { pathname: "/runs/run-first", state: entry },
      {
        server: {
          ...routes,
          "GET /v1/queue": () => json(page([queueItem])),
          "GET /v1/projects/{projectId}/runs": () =>
            json(page(items, "another-page")),
        },
      },
    );
    const pane = await screen.findByRole("region", { name: "Source Runs" });
    await within(pane).findByRole("link", {
      name: isQueue ? /run-queued/ : /run-first/,
    });
    const reads = view.sent(
      "GET",
      isQueue ? "/v1/queue" : "/projects/project_example/runs",
    );
    expect(reads).toHaveLength(1);
    const params = new URL(reads[0]!.url).searchParams;
    for (const [key, value] of Object.entries(source.options))
      expect(params.get(key)).toBe(String(value));
    expect(view.sent("GET", "/run-second")).toHaveLength(0);
    if (isQueue)
      expect(
        within(pane).getByText("This Run is not in the current source list."),
      ).toBeVisible();
    else
      expect(
        within(pane).getByRole("link", {
          name: "More Runs in the source list",
        }),
      ).toHaveAttribute("href", entry.returnTo);
  });

  it("bounds a full 50-Run source page without per-row detail reads or following its cursor", async () => {
    const fullPage = [
      items[0]!,
      ...Array.from({ length: 49 }, (_, index) => summary(`run-page-${index}`)),
    ];
    const view = renderApplication(
      { pathname: "/runs/run-first", state: origin },
      {
        server: {
          ...routes,
          "GET /v1/runs": () => json(page(fullPage, "not-loaded")),
        },
      },
    );
    const pane = await screen.findByRole("region", { name: "Source Runs" });
    await within(pane).findByText("50 shown");
    expect(within(pane).getAllByRole("listitem")).toHaveLength(50);
    expect(view.sent("GET", "/v1/runs")).toHaveLength(1);
    expect(
      new URL(view.sent("GET", "/v1/runs")[0]!.url).searchParams.get("limit"),
    ).toBe("50");
    expect(
      view.requests.filter((request) =>
        /^\/v1\/runs\/run-page-/.test(new URL(request.url).pathname),
      ),
    ).toHaveLength(0);
    expect(view.location()).toBe("/runs/run-first");
  });

  it("keeps detail accessible when the source read fails, then refreshes explicitly", async () => {
    let unavailable = true;
    const view = renderApplication(
      { pathname: "/runs/run-first", state: origin },
      {
        server: {
          ...routes,
          "GET /v1/runs": () =>
            unavailable
              ? failure(
                  503,
                  "storage_transaction_conflict",
                  "Source unavailable",
                )
              : json(page([])),
        },
      },
    );
    const pane = await screen.findByRole("region", { name: "Source Runs" });
    await within(pane).findByText(/Could not load the source Run list/);
    expect(
      await screen.findByRole("heading", { name: "review" }),
    ).toBeVisible();
    unavailable = false;
    await view.user.click(
      within(pane).getByRole("button", { name: "Refresh" }),
    );
    expect(
      await within(pane).findByText("No Runs in this source list"),
    ).toBeVisible();
    expect(screen.getByRole("heading", { name: "review" })).toBeVisible();
  });

  it.each([
    undefined,
    { ...origin, returnTo: "/issues" },
    {
      ...origin,
      runSource: { ...completed, options: { lifecycle: "active" } },
    },
  ])(
    "does not invent a list for direct or unsupported context",
    async (state) => {
      const view = renderApplication(
        { pathname: "/runs/run-first", state },
        { server: routes },
      );
      await screen.findByRole("heading", { name: "review" });
      expect(screen.queryByRole("region", { name: "Source Runs" })).toBeNull();
      expect(view.sent("GET", "/v1/runs")).toHaveLength(0);
      await act(async () => view.router.navigate("/runs/run-second"));
      expect(screen.queryByRole("region", { name: "Source Runs" })).toBeNull();
    },
  );
});

describe("Run list origin validation", () => {
  it("accepts only supported bounded context and local return destinations", () => {
    expect(readRunListOrigin(origin)).toEqual(origin);
    for (const value of [
      null,
      {},
      { ...origin, returnTo: "https://example.test/runs" },
      { ...origin, returnLabel: "" },
      { ...origin, runSource: { ...completed, refreshMs: 1 } },
      {
        ...origin,
        runSource: {
          ...completed,
          options: { ...completed.options, limit: 100000 },
        },
      },
      {
        ...origin,
        runSource: {
          ...completed,
          options: {
            ...completed.options,
            labelSelectors: [{ key: 1, value: "eval" }],
          },
        },
      },
    ])
      expect(readRunListOrigin(value)).toBeUndefined();
  });
});
