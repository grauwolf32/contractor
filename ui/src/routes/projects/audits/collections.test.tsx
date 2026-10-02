import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { describe, expect, it } from "vitest";

import { useAuditCollections } from "./collections";

function renderCollection(size = 11) {
  const server = {
    revision: 1,
    size,
    failContinuation: false,
    calls: [] as string[],
  };
  const snapshots: number[][] = [];
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const hook = renderHook(
    () => {
      const collection = useAuditCollections(["audit_example"], {
        id: (id) => id,
        queryKey: (id) => [id, "findings"],
        load: () => async (cursor?: string) => {
          server.calls.push(cursor ?? "head");
          const offset =
            cursor === undefined ? 0 : Number(cursor.split(":")[1]);
          if (
            cursor !== undefined &&
            Number(cursor.split(":")[0]) !== server.revision
          )
            throw new Error("409 revision conflict");
          if (offset >= 5 && server.failContinuation)
            throw new Error("Continuation unavailable");
          return {
            items: [{ id: String(offset), revision: server.revision }],
            page:
              offset + 1 < server.size
                ? {
                    hasMore: true,
                    nextCursor: `${server.revision}:${offset + 1}`,
                  }
                : { hasMore: false },
          };
        },
        identity: (item) => item.id,
        refetchInterval: () => false,
      });
      snapshots.push(collection.results[0]!.items.map((item) => item.revision));
      return collection;
    },
    {
      wrapper: ({ children }: { children: ReactNode }) => (
        <QueryClientProvider client={queryClient}>
          {children}
        </QueryClientProvider>
      ),
    },
  );
  const expectSize = (expected: number) =>
    waitFor(() =>
      expect(hook.result.current.results[0]?.items).toHaveLength(expected),
    );
  const loadMore = async (expected: number) => {
    act(() => hook.result.current.loadMore());
    await expectSize(expected);
  };
  return { ...hook, queryClient, server, snapshots, expectSize, loadMore };
}

describe("useAuditCollections", () => {
  it.each(["refresh", "invalidation"] as const)(
    "rebuilds all loaded continuations after a revision change on %s",
    async (trigger) => {
      const { result, server, queryClient, snapshots, expectSize, loadMore } =
        renderCollection();
      await expectSize(5);
      await loadMore(10);
      await loadMore(11);
      server.revision = 2;
      server.calls = [];
      await act(async () => {
        if (trigger === "refresh") await result.current.results[0]!.refetch();
        else
          await queryClient.invalidateQueries({
            queryKey: ["audit_example", "findings"],
          });
      });
      await waitFor(() =>
        expect(
          result.current.results[0]?.items.map((item) => item.revision),
        ).toEqual(Array(11).fill(2)),
      );
      expect(result.current.results[0]?.moreError).toBeNull();
      expect(result.current.truncated).toBe(false);
      expect(server.calls).toContain("2:5");
      expect(server.calls).toContain("2:10");
      expect(snapshots.every((revisions) => new Set(revisions).size <= 1)).toBe(
        true,
      );
      if (trigger === "refresh")
        expect(server.calls.some((cursor) => cursor.startsWith("1:"))).toBe(
          false,
        );
    },
  );

  it("recovers from a stale Load more cursor by refreshing the head", async () => {
    const { result, server, expectSize } = renderCollection(6);
    await expectSize(5);
    server.revision = 2;
    act(() => result.current.loadMore());
    await waitFor(() =>
      expect(result.current.results[0]?.moreError?.message).toBe(
        "409 revision conflict",
      ),
    );
    await act(async () => {
      await result.current.results[0]!.refetch();
    });
    await waitFor(() =>
      expect(
        result.current.results[0]?.items.map((item) => item.revision),
      ).toEqual(Array(6).fill(2)),
    );
    expect(result.current.results[0]?.moreError).toBeNull();
  });

  it("exposes failed continuation refreshes even with cached rows and supports retry", async () => {
    const { result, server, expectSize, loadMore } = renderCollection(6);
    await expectSize(5);
    await loadMore(6);
    server.failContinuation = true;
    await act(async () => {
      await result.current.results[0]!.refetch();
    });
    await waitFor(() =>
      expect(result.current.results[0]?.moreError?.message).toBe(
        "Continuation unavailable",
      ),
    );
    expect(result.current.results[0]?.items).toHaveLength(6);
    server.failContinuation = false;
    act(() => result.current.loadMore());
    await waitFor(() =>
      expect(result.current.results[0]?.moreError).toBeNull(),
    );
    expect(result.current.results[0]?.items).toHaveLength(6);
  });

  it("drops old continuation rows when the refreshed collection fits in one batch", async () => {
    const { result, server, expectSize, loadMore } = renderCollection(6);
    await expectSize(5);
    await loadMore(6);
    server.revision = 2;
    server.size = 3;
    await act(async () => {
      await result.current.results[0]!.refetch();
    });
    await expectSize(3);
    expect(
      result.current.results[0]?.items.map((item) => item.revision),
    ).toEqual([2, 2, 2]);
    expect(result.current.truncated).toBe(false);
  });
});
