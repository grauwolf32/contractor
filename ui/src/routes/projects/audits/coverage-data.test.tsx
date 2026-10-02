import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { MemoryRouter, useNavigate } from "react-router";
import { describe, expect, it, vi } from "vitest";

import type { Audit, AuditCoverageRow } from "../../../api/audits";
import { PublicAPI } from "../../../api/client";
import { PublicAPIProvider } from "../../../api/context";
import { queryKeys } from "../../../api/query-keys";
import { useAuditCoverage } from "./coverage-data";

function auditAt(state: Audit["state"], revision: number): Audit {
  return {
    auditId: "audit_example",
    projectId: "project_example",
    profile: {
      name: "source-risk",
      version: "1",
      digest: `sha256:${"1".repeat(64)}`,
    },
    inputs: {},
    scope: {},
    runtimeLabels: [],
    state,
    phase: "rounds",
    revision,
    currentRoundId: "round_example",
    dispatchState: state === "active" ? "open" : "closed",
    holdState: "held",
    limits: {
      maxRounds: 1,
      batchSize: 1,
      maxItemsPerRound: 8,
      maxItemsTotal: 8,
      maxSubmittedRuns: 8,
      maxItemRunAttempts: 2,
      maxEvidenceBytes: 1_048_576,
    },
    reservedRunCount: 1,
    submittedRunCount: 1,
    outstandingRunCount: state === "active" ? 1 : 0,
    retainedEvidenceBytes: 0,
    eventSequence: revision,
    createdAt: "2026-09-20T10:00:00Z",
    updatedAt: "2026-09-20T10:00:00Z",
  };
}

function coverage(audit: Audit): AuditCoverageRow {
  return {
    itemId: "item_example",
    roundId: "round_example",
    itemKey: "check",
    subjectKey: "check",
    ordinal: 0,
    coverage: {
      status: "satisfied",
      requested: [],
      completed: [],
      gaps: [],
      rationale: `revision ${audit.revision}`,
    },
    updatedAt: audit.updatedAt,
  };
}

function response(value: unknown, revision?: number): Response {
  return new Response(JSON.stringify(value), {
    headers: {
      "Content-Type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
      ...(revision === undefined ? {} : { ETag: `"${revision}"` }),
    },
  });
}

function fixture(initialPath = "/coverage") {
  let current = auditAt("active", 1);
  const loadCoverage = vi.fn<(url: URL) => Promise<Response>>(async () =>
    response({ items: [coverage(current)], page: { hasMore: false } }),
  );
  const api = new PublicAPI(
    {
      uiVersion: "0.1.0",
      supportedApiVersions: ["contractor.public.v1"],
      apiBaseUrl: "http://127.0.0.1:8080",
    },
    vi.fn(async (input) => {
      const url = new URL((input as Request).url);
      return url.pathname.endsWith("/coverage")
        ? loadCoverage(url)
        : response(current, current.revision);
    }),
  );
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <PublicAPIProvider api={api}>
        <MemoryRouter initialEntries={[initialPath]}>{children}</MemoryRouter>
      </PublicAPIProvider>
    </QueryClientProvider>
  );
  const hook = renderHook(
    ({ audit }) => ({
      query: useAuditCoverage(audit),
      navigate: useNavigate(),
    }),
    { initialProps: { audit: current }, wrapper },
  );
  return {
    ...hook,
    queryClient,
    loadCoverage,
    setServerRevision(state: Audit["state"], revision: number) {
      current = auditAt(state, revision);
    },
    update(state: Audit["state"], revision: number) {
      current = auditAt(state, revision);
      hook.rerender({ audit: current });
    },
  };
}

describe("Audit coverage completion", () => {
  it("refreshes final same-round revisions without retaining a cache entry per revision", async () => {
    const { result, update, queryClient, loadCoverage } = fixture();
    await waitFor(() =>
      expect(result.current.query.items[0]?.coverage.rationale).toBe(
        "revision 1",
      ),
    );
    for (const revision of [2, 3, 4]) {
      update("completed", revision);
      await waitFor(() =>
        expect(result.current.query.items[0]?.coverage.rationale).toBe(
          `revision ${revision}`,
        ),
      );
    }
    expect(loadCoverage).toHaveBeenCalledTimes(4);
    expect(
      queryClient
        .getQueryCache()
        .findAll({ queryKey: queryKeys.audits.allCoverage("audit_example") }),
    ).toHaveLength(1);
  });

  it("replaces an unfinished first coverage read when the Audit finishes", async () => {
    const { result, update, loadCoverage } = fixture();
    let release!: (value: Response) => void;
    const pending = new Promise<Response>((resolve) => {
      release = resolve;
    });
    loadCoverage.mockImplementationOnce(async () => pending);
    await waitFor(() => expect(loadCoverage).toHaveBeenCalledTimes(1));
    update("completed", 2);
    await waitFor(() =>
      expect(result.current.query.items[0]?.coverage.rationale).toBe(
        "revision 2",
      ),
    );
    await act(async () => {
      release(
        response({
          items: [coverage(auditAt("active", 1))],
          page: { hasMore: false },
        }),
      );
      await pending;
    });
    expect(result.current.query.items[0]?.coverage.rationale).toBe(
      "revision 2",
    );
  });

  it("keeps a loaded revision pin stable until the reader clears it", async () => {
    const { result, update, loadCoverage, queryClient } = fixture(
      "/coverage?auditRevision=1",
    );
    await waitFor(() => expect(result.current.query.isSuccess).toBe(true));
    update("completed", 2);
    expect(result.current.query.items[0]?.coverage.rationale).toBe(
      "revision 1",
    );
    expect(result.current.query.error).toBeNull();
    expect(loadCoverage).toHaveBeenCalledTimes(1);
    await act(async () =>
      queryClient.invalidateQueries({
        queryKey: queryKeys.audits.detail("audit_example"),
      }),
    );
    expect(loadCoverage).toHaveBeenCalledTimes(1);
    await act(async () => {
      await result.current.navigate("/coverage");
    });
    await waitFor(() =>
      expect(result.current.query.items[0]?.coverage.rationale).toBe(
        "revision 2",
      ),
    );
    expect(loadCoverage).toHaveBeenCalledTimes(2);
  });

  it("does not poll a loaded pinned view while the Audit stays active", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      const { result, loadCoverage } = fixture("/coverage?auditRevision=1");
      await waitFor(() => expect(result.current.query.isSuccess).toBe(true));
      await act(async () => vi.advanceTimersByTimeAsync(6_000));
      expect(loadCoverage).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it("retries one unpinned first batch after a revision straddle", async () => {
    const { result, loadCoverage, setServerRevision } = fixture();
    loadCoverage.mockImplementationOnce(async () => {
      setServerRevision("active", 2);
      return response({
        items: [coverage(auditAt("active", 1))],
        page: { hasMore: false },
      });
    });
    await waitFor(() =>
      expect(result.current.query.items[0]?.coverage.rationale).toBe(
        "revision 2",
      ),
    );
    expect(loadCoverage).toHaveBeenCalledTimes(2);
    expect(result.current.query.error).toBeNull();
  });

  it("does not replace retained coverage when both unpinned attempts straddle", async () => {
    const { result, update, loadCoverage, setServerRevision } = fixture();
    await waitFor(() => expect(result.current.query.isSuccess).toBe(true));
    loadCoverage.mockImplementationOnce(async (url) => {
      expect(url.searchParams.get("round")).toBe("round_example");
      setServerRevision("completed", 3);
      return response({
        items: [coverage(auditAt("completed", 2))],
        page: { hasMore: false },
      });
    });
    loadCoverage.mockImplementationOnce(async () => {
      setServerRevision("completed", 4);
      return response({
        items: [coverage(auditAt("completed", 3))],
        page: { hasMore: false },
      });
    });
    update("completed", 2);
    await waitFor(() =>
      expect(result.current.query.error).toMatchObject({
        status: 409,
        code: "revision_conflict",
      }),
    );
    expect(result.current.query.items[0]?.coverage.rationale).toBe(
      "revision 1",
    );
  });

  it("rejects an initially stale revision pin", async () => {
    const { result, loadCoverage } = fixture("/coverage?auditRevision=2");
    await waitFor(() =>
      expect(result.current.query.error).toMatchObject({
        status: 409,
        code: "revision_conflict",
      }),
    );
    expect(loadCoverage).not.toHaveBeenCalled();
  });

  it("rejects a continuation when its pinned first batch is no longer current", async () => {
    const { result, loadCoverage, setServerRevision } = fixture(
      "/coverage?auditRevision=1",
    );
    loadCoverage.mockImplementation(async (url) => {
      const page = Number(url.searchParams.get("cursor") ?? "0");
      return response({
        items: [coverage(auditAt("active", 1))],
        page: { hasMore: true, nextCursor: String(page + 1) },
      });
    });
    await waitFor(() => expect(result.current.query.truncated).toBe(true));
    expect(result.current.query.items).toHaveLength(5);
    setServerRevision("completed", 2);
    act(() => result.current.query.loadMore());
    await waitFor(() =>
      expect(result.current.query.moreError).toMatchObject({
        status: 409,
        code: "revision_conflict",
      }),
    );
    expect(result.current.query.items).toHaveLength(5);
    expect(loadCoverage).toHaveBeenCalledTimes(5);
  });
});
