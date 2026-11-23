import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { PublicAPIProvider } from "../../../api/context";
import type { Audit } from "../../../api/audits";
import { researchFixture } from "../../../test/audit-research-fixtures";
import { failure, json } from "../../../test/responses";
import { testAPI } from "../../../test/render";
import { useResearchTask } from "./research-data";

function fixture() {
  const research = researchFixture();
  const collected = research.tasks[0]!;
  const ready = { ...collected, state: "ready" as const };
  delete ready.result;
  let current: typeof collected = ready;
  let failing = false;
  const reads = vi.fn(async () =>
    failing ? failure(503, "storage_transaction_conflict") : json(current),
  );
  const { api } = testAPI(reads);
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>
      <PublicAPIProvider api={api}>{children}</PublicAPIProvider>
    </QueryClientProvider>
  );
  const audit: Audit = { ...research.audit, state: "active" };
  return {
    audit,
    collected,
    reads,
    wrapper,
    collect: () => {
      current = collected;
    },
    fail: () => {
      failing = true;
    },
  };
}

describe("selected research Task freshness", () => {
  afterEach(() => vi.useRealTimers());
  it("polls an off-page Task and stops once its result is collected", async () => {
    const server = fixture();
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const { result, unmount } = renderHook(
      () =>
        useResearchTask(server.audit, server.collected.document.task_id, true),
      { wrapper: server.wrapper },
    );
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.result).toBeUndefined();
    try {
      server.collect();
      await act(() => vi.advanceTimersByTimeAsync(5000));
      await waitFor(() =>
        expect(result.current.data?.result).toEqual(server.collected.result),
      );
      expect(server.reads).toHaveBeenCalledTimes(2);
      await act(() => vi.advanceTimersByTimeAsync(60_000));
      expect(server.reads).toHaveBeenCalledTimes(2);
      unmount();
    } finally {
      vi.useRealTimers();
    }
  });

  it("takes a final selected Task snapshot when terminal collection finishes", async () => {
    const server = fixture();
    const { result, rerender } = renderHook(
      ({ audit }) =>
        useResearchTask(audit, server.collected.document.task_id, true),
      { wrapper: server.wrapper, initialProps: { audit: server.audit } },
    );
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    server.collect();
    rerender({
      audit: {
        ...server.audit,
        state: "completed",
        revision: server.audit.revision + 1,
        outstandingRunCount: 0,
        collectionPending: false,
      },
    });
    await waitFor(() =>
      expect(result.current.data?.result).toEqual(server.collected.result),
    );
    expect(server.reads).toHaveBeenCalledTimes(2);
  });

  it("does not read in-page selection and backs off a failed off-page read", async () => {
    const server = fixture();
    server.fail();
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const { result, rerender, unmount } = renderHook(
      ({ enabled }) =>
        useResearchTask(
          server.audit,
          server.collected.document.task_id,
          enabled,
        ),
      { wrapper: server.wrapper, initialProps: { enabled: false } },
    );
    expect(server.reads).not.toHaveBeenCalled();
    rerender({ enabled: true });
    await waitFor(() => expect(result.current.isError).toBe(true));
    try {
      await act(() => vi.advanceTimersByTimeAsync(5000));
      expect(server.reads).toHaveBeenCalledTimes(1);
      await act(() => vi.advanceTimersByTimeAsync(5000));
      expect(server.reads).toHaveBeenCalledTimes(2);
      unmount();
    } finally {
      vi.useRealTimers();
    }
  });
});
