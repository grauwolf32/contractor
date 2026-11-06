import { onlineManager } from "@tanstack/react-query";
import {
  act,
  fireEvent,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { queryKeys } from "../../../api/query-keys";
import { audit } from "../../../test/builders";
import { fakeServer } from "../../../test/fake-server";
import { renderWithAPI } from "../../../test/render";
import { failure, json } from "../../../test/responses";
import { AuditActivity } from "./activity";

const initial = audit("project_history", "audit_history", "active", {
  eventSequence: 51,
});
const events = Array.from({ length: 51 }, (_, index) => ({
  auditId: initial.auditId,
  sequence: 51 - index,
  kind: index === 50 ? "audit.created" : "task.settled",
  entityId: index === 50 ? initial.auditId : `task_${index}`,
  summary: {},
  createdAt: "2026-11-06T10:00:00Z",
}));

function fixture(changing = false) {
  const reads: string[] = [];
  let failOlder = false;
  const server = fakeServer({
    "GET /v1/audits/{auditId}/events": ({ url }) => {
      const cursor = url.searchParams.get("cursor");
      reads.push(cursor ?? "head");
      expect(url.searchParams.get("limit")).toBe("50");
      if (url.pathname.includes("audit_other"))
        return json({
          items: [{ ...events[50]!, auditId: "audit_other", sequence: 1 }],
          throughSequence: 1,
          total: 1,
          page: { hasMore: false },
        });
      if (cursor !== null && failOlder)
        return failure(503, "storage_transaction_conflict");
      return json({
        items: cursor === null ? events.slice(0, 50) : events.slice(50),
        throughSequence: 51,
        total: 51,
        page:
          cursor === null
            ? { hasMore: true, nextCursor: "older" }
            : { hasMore: false },
      });
    },
  });
  function Journey() {
    const [current, setCurrent] = useState(
      changing ? initial : { ...initial, state: "completed" as const },
    );
    return (
      <>
        <button
          onClick={() =>
            setCurrent(audit("project_history", "audit_other", "completed"))
          }
        >
          Switch check
        </button>
        <button
          onClick={() =>
            setCurrent({
              ...initial,
              state: "completed",
              revision: initial.revision + 1,
            })
          }
        >
          Finish check
        </button>
        <AuditActivity audit={current} />
      </>
    );
  }
  return {
    ...renderWithAPI(<Journey />, { server }),
    reads,
    rejectOlder: (reject: boolean) => {
      failOlder = reject;
    },
  };
}
const activity = () => screen.getByRole("region", { name: "Check activity" });

afterEach(() => {
  onlineManager.setOnline(true);
  vi.useRealTimers();
});

describe("Audit activity", () => {
  it("never presents another check's loaded history after navigation", async () => {
    const view = fixture();
    await view.user.click(
      screen.getByRole("button", { name: "Show activity" }),
    );
    await waitFor(() =>
      expect(within(activity()).getAllByRole("listitem")).toHaveLength(50),
    );
    await view.user.click(screen.getByRole("button", { name: "Switch check" }));
    await waitFor(() =>
      expect(within(activity()).getAllByRole("listitem")).toHaveLength(1),
    );
    expect(
      within(activity()).queryByText("task.settled"),
    ).not.toBeInTheDocument();
    expect(within(activity()).getByText("audit.created")).toBeInTheDocument();
  });
  it("retains successful pages across continuation and refresh errors and retries the failed cursor", async () => {
    const view = fixture();
    expect(view.reads).toEqual([]);
    await view.user.click(
      screen.getByRole("button", { name: "Show activity" }),
    );
    await waitFor(() =>
      expect(within(activity()).getAllByRole("listitem")).toHaveLength(50),
    );
    view.rejectOlder(true);
    await view.user.click(
      screen.getByRole("button", { name: "Load older activity" }),
    );
    await screen.findByRole("alert");
    expect(within(activity()).getAllByRole("listitem")).toHaveLength(50);
    view.rejectOlder(false);
    await view.user.click(screen.getByRole("button", { name: "Try again" }));
    await waitFor(() =>
      expect(within(activity()).getAllByRole("listitem")).toHaveLength(51),
    );
    expect(view.reads).toEqual(["head", "older", "older"]);
    expect(
      screen.getByRole("button", { name: "Load older activity" }),
    ).toBeDisabled();
    view.rejectOlder(true);
    await act(() =>
      view.queryClient.invalidateQueries({
        queryKey: queryKeys.audits.detail(initial.auditId),
      }),
    );
    await screen.findByText("Could not refresh; showing the last loaded data.");
    expect(within(activity()).getAllByRole("listitem")).toHaveLength(51);
    view.rejectOlder(false);
    await view.user.click(screen.getByRole("button", { name: "Try again" }));
    await waitFor(() =>
      expect(
        screen.queryByText("Could not refresh; showing the last loaded data."),
      ).not.toBeInTheDocument(),
    );
    expect(view.reads.slice(-2)).toEqual(["head", "older"]);
    await act(async () => {
      onlineManager.setOnline(false);
      onlineManager.setOnline(true);
    });
    await waitFor(() => expect(view.reads).toHaveLength(9));
    expect(view.reads.slice(-2)).toEqual(["head", "older"]);
    expect(within(activity()).getAllByRole("listitem")).toHaveLength(51);
  });

  it("refreshes the loaded chain while active and once at settlement, then stops polling", async () => {
    vi.useFakeTimers();
    const view = fixture(true);
    fireEvent.click(screen.getByRole("button", { name: "Show activity" }));
    await vi.waitFor(() =>
      expect(within(activity()).getAllByRole("listitem")).toHaveLength(50),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Load older activity" }),
    );
    await vi.waitFor(() =>
      expect(within(activity()).getAllByRole("listitem")).toHaveLength(51),
    );
    await act(() => vi.advanceTimersByTimeAsync(5_000));
    await vi.waitFor(() => expect(view.queryClient.isFetching()).toBe(0));
    expect(view.reads).toEqual(["head", "older", "head", "older"]);
    fireEvent.click(screen.getByRole("button", { name: "Finish check" }));
    await vi.waitFor(() => expect(view.reads).toHaveLength(6));
    await vi.waitFor(() => expect(view.queryClient.isFetching()).toBe(0));
    expect(view.reads.slice(-2)).toEqual(["head", "older"]);
    await act(() => vi.advanceTimersByTimeAsync(60_000));
    expect(view.reads).toHaveLength(6);
    expect(within(activity()).getAllByRole("listitem")).toHaveLength(51);
  });
});
