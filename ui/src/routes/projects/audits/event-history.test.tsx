import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";

import type { Audit, AuditEvent } from "../../../api/audits";
import { PublicAPIProvider } from "../../../api/context";
import { queryKeys } from "../../../api/query-keys";
import { checkLinks } from "./check-links";
import { fakeAPI, jsonResponse, makeAudit } from "./check-test-support";
import { CheckEventHistory } from "./event-history";
import { eventEntry } from "./event-model";

const audit = makeAudit("audit_example", "project_example", "completed");
const links = checkLinks(
  "project_example",
  "audit_example",
  new URLSearchParams(),
  "overview",
);
const event = (
  sequence: number,
  overrides: Partial<AuditEvent> = {},
): AuditEvent => ({
  auditId: audit.auditId,
  sequence,
  kind: "review.decided",
  entityId: `decision_${sequence}`,
  summary: { action: "approve" },
  createdAt: "2026-10-07T10:00:00Z",
  ...overrides,
});

function mount(handler: Parameters<typeof fakeAPI>[0], current = audit) {
  const { api } = fakeAPI(handler);
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const tree = (value: Audit) => (
    <QueryClientProvider client={client}>
      <PublicAPIProvider api={api}>
        <MemoryRouter>
          <CheckEventHistory audit={value} now={undefined} links={links} />
        </MemoryRouter>
      </PublicAPIProvider>
    </QueryClientProvider>
  );
  const view = render(tree(current));
  return {
    client,
    ...view,
    setAudit: (value: Audit) => view.rerender(tree(value)),
  };
}

describe("durable check history", () => {
  it("polls while active, reads the final state and stops after completion or leaving the route", async () => {
    vi.useFakeTimers();
    try {
      let reads = 0;
      const view = mount(
        () => {
          reads += 1;
          return jsonResponse({
            items: [event(reads)],
            total: reads,
            throughSequence: reads,
            page: { hasMore: false },
          });
        },
        { ...audit, state: "active" },
      );
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });
      expect(reads).toBe(1);
      await act(async () => {
        await vi.advanceTimersByTimeAsync(5_000);
      });
      expect(reads).toBe(2);
      await act(async () => {
        view.setAudit({ ...audit, revision: audit.revision + 1 });
        await vi.advanceTimersByTimeAsync(0);
      });
      expect(reads).toBe(3);
      await act(async () => {
        await vi.advanceTimersByTimeAsync(15_000);
      });
      expect(reads).toBe(3);
      view.setAudit({ ...audit, state: "active" });
      view.unmount();
      await act(async () => {
        await vi.advanceTimersByTimeAsync(15_000);
      });
      expect(reads).toBe(3);
    } finally {
      vi.useRealTimers();
    }
  });
  it("uses event sequence for tied timestamps, loads older pages and retains them on failure", async () => {
    const reads: string[] = [];
    let fail = true;
    mount((_request, url) => {
      reads.push(url.searchParams.get("cursor") ?? "head");
      if (!url.searchParams.has("cursor"))
        return jsonResponse({
          items: [event(3), event(2, { kind: "audit.resumed", summary: {} })],
          total: 3,
          throughSequence: 3,
          page: { hasMore: true, nextCursor: "older" },
        });
      if (fail)
        return jsonResponse(
          {
            code: "internal_error",
            message: "Unavailable",
            requestId: "request-1",
          },
          { status: 500 },
        );
      return jsonResponse({
        items: [event(1, { kind: "audit.created", summary: {} })],
        total: 3,
        throughSequence: 3,
        page: { hasMore: false },
      });
    });
    const user = userEvent.setup();
    expect(await screen.findByText("Check continued.")).toBeVisible();
    const log = screen.getByRole("list", { name: "Activity on this check" });
    expect(
      within(log)
        .getAllByRole("listitem")
        .map((node) => node.textContent),
    ).toEqual([
      expect.stringContaining("Decision recorded: Approved."),
      expect.stringContaining("Check continued."),
    ]);
    await user.click(
      screen.getByRole("button", { name: "Load older activity" }),
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Older activity could not be loaded.",
    );
    expect(screen.getByText("Check continued.")).toBeVisible();
    fail = false;
    await user.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByText("Check created.")).toBeVisible();
    expect(screen.getByText("Showing 3 of 3 recorded events.")).toBeVisible();
    expect(
      screen.queryByRole("button", { name: "Load older activity" }),
    ).not.toBeInTheDocument();
    expect(reads).toEqual(["head", "older", "older"]);
  });

  it("refreshes the loaded cursor chain against a fresh prefix", async () => {
    let refreshed = false;
    const cursors: string[] = [];
    const { client } = mount((_request, url) => {
      const cursor = url.searchParams.get("cursor");
      cursors.push(cursor ?? "head");
      if (cursor === null)
        return jsonResponse({
          items: [event(refreshed ? 4 : 3)],
          total: 3,
          throughSequence: refreshed ? 4 : 3,
          page: { hasMore: true, nextCursor: refreshed ? "new" : "old" },
        });
      return jsonResponse({
        items: [event(2), event(1, { kind: "audit.created", summary: {} })],
        total: 3,
        throughSequence: refreshed ? 4 : 3,
        page: { hasMore: false },
      });
    });
    await userEvent
      .setup()
      .click(
        await screen.findByRole("button", { name: "Load older activity" }),
      );
    expect(await screen.findByText("Check created.")).toBeVisible();
    refreshed = true;
    await act(async () => {
      await client.invalidateQueries({
        queryKey: queryKeys.audits.events(audit.auditId),
      });
    });
    await waitFor(() =>
      expect(cursors).toEqual(["head", "old", "head", "new"]),
    );
    expect(screen.getByText("Showing 3 of 3 recorded events.")).toBeVisible();
    expect(screen.getAllByText("Decision recorded: Approved.")).toHaveLength(2);
  });

  it.each(["wrong-prefix", "repeated-cursor", "overlapping-sequence"])(
    "rejects %s without discarding settled events",
    async (failure) => {
      mount((_request, url) =>
        url.searchParams.has("cursor")
          ? jsonResponse({
              items: [event(failure === "overlapping-sequence" ? 3 : 1)],
              total: 3,
              throughSequence: failure === "wrong-prefix" ? 4 : 3,
              page:
                failure === "repeated-cursor"
                  ? { hasMore: true, nextCursor: "older" }
                  : { hasMore: false },
            })
          : jsonResponse({
              items: [event(3)],
              total: 3,
              throughSequence: 3,
              page: { hasMore: true, nextCursor: "older" },
            }),
      );
      await userEvent
        .setup()
        .click(
          await screen.findByRole("button", { name: "Load older activity" }),
        );
      expect(await screen.findByRole("alert")).toHaveTextContent(
        "Older activity could not be loaded.",
      );
      expect(screen.getByText("Showing 1 of 3 recorded events.")).toBeVisible();
    },
  );

  it("records loading errors without inventing history from the current check", async () => {
    mount(() =>
      jsonResponse(
        {
          code: "internal_error",
          message: "Unavailable",
          requestId: "request-1",
        },
        { status: 500 },
      ),
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Activity could not be refreshed.",
    );
    expect(screen.queryByText("Check started.")).not.toBeInTheDocument();
  });

  it("labels decisions, failures and unknown future events without treating a proposed report as ready", () => {
    expect(eventEntry(event(1), links).title).toBe(
      "Decision recorded: Approved.",
    );
    expect(
      eventEntry(event(2, { summary: { verdict: "needs_evidence" } }), links),
    ).toMatchObject({
      title: "Decision recorded: Needs evidence.",
      tone: "warning",
    });
    expect(
      eventEntry(
        event(3, {
          kind: "execution.terminal_observed",
          summary: { outcome: "failed" },
        }),
        links,
      ),
    ).toMatchObject({ title: "Run failed.", tone: "blocked" });
    expect(
      eventEntry(
        event(4, {
          kind: "review.requested",
          summary: { kind: "report-acceptance" },
        }),
        links,
      ).title,
    ).toBe("Waiting for your decision:");
    expect(
      eventEntry(event(5, { kind: "audit.future", summary: {} }), links),
    ).toMatchObject({
      title: "Check activity recorded.",
      text: "audit.future",
    });
  });
});
