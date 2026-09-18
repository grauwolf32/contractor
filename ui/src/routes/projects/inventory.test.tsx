import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import { PublicAPI } from "../../api/client";
import { PublicAPIProvider } from "../../api/context";
import { useProjectArtifactInventory } from "./inventory";

function fixture(fetcher: (input: RequestInfo | URL) => Promise<Response>) {
  const api = new PublicAPI(
    {
      uiVersion: "0.1.0",
      supportedApiVersions: ["contractor.public.v1"],
      apiBaseUrl: "http://127.0.0.1:8080",
    },
    vi.fn(fetcher),
  );
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <PublicAPIProvider api={api}>{children}</PublicAPIProvider>
    </QueryClientProvider>
  );
}
function response(items: unknown[], page: object) {
  return new Response(JSON.stringify({ items, page }), {
    headers: {
      "content-type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
    },
  });
}
const artifact = (name: string) => ({
  artifact: { namespace: "sources", name, revision: "r1" },
  mediaType: "application/zip",
  size: 10,
  current: true,
  frozen: false,
  createdAt: "2026-09-01T10:00:00Z",
});

describe("Project matching inventory", () => {
  it("does not offer a partial sole match while later pages are loading", async () => {
    let finish!: () => void;
    const second = new Promise<void>((resolve) => {
      finish = resolve;
    });
    let laterPage = false;
    const wrapper = fixture(async (input) => {
      const url = new URL(input instanceof Request ? input.url : String(input));
      if (url.searchParams.has("cursor")) {
        laterPage = true;
        await second;
        return response([artifact("second")], { hasMore: false });
      }
      return response([artifact("first")], {
        hasMore: true,
        nextCursor: "next",
      });
    });
    const { result } = renderHook(
      () => useProjectArtifactInventory("project_example"),
      { wrapper },
    );
    await waitFor(() => expect(laterPage).toBe(true));
    expect(result.current.data).toBeUndefined();
    await act(async () => {
      finish();
      await second;
    });
    await waitFor(() => expect(result.current.data).toHaveLength(2));
  });
  it("rejects a cyclic inventory and supports an explicit fresh retry", async () => {
    let failed = true;
    let reads = 0;
    const wrapper = fixture(async () => {
      reads++;
      return response(
        [artifact("first")],
        failed ? { hasMore: true, nextCursor: "repeated" } : { hasMore: false },
      );
    });
    const { result } = renderHook(
      () => useProjectArtifactInventory("project_example"),
      { wrapper },
    );
    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.data).toBeUndefined();
    expect(reads).toBe(2);
    failed = false;
    await act(async () => {
      await result.current.refetch();
    });
    await waitFor(() => expect(result.current.data).toHaveLength(1));
  });
});
