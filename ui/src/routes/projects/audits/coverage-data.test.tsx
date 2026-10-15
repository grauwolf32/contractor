import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { MemoryRouter, useNavigate } from "react-router";
import { describe, expect, it, vi } from "vitest";

import type { Audit, AuditCoverageRow } from "../../../api/audits";
import { PublicAPIProvider } from "../../../api/context";
import { queryKeys } from "../../../api/query-keys";
import * as build from "../../../test/builders";
import { failure, fakeServer, json, page } from "../../../test/fake-server";
import { testAPI } from "../../../test/render";
import { useAuditCoverage } from "./coverage-data";

function auditAt(state: Audit["state"], revision: number): Audit {
  return build.audit("project_example", "audit_example", state, {
    profile: {
      name: "source-risk",
      version: "1",
      digest: `sha256:${"1".repeat(64)}`,
    },
    revision,
    currentRoundId: "round_example",
    holdState: "held",
    reservedRunCount: 1,
    submittedRunCount: 1,
    outstandingRunCount: state === "active" ? 1 : 0,
    eventSequence: revision,
    createdAt: "2026-09-20T10:00:00Z",
    updatedAt: "2026-09-20T10:00:00Z",
  });
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

/** The Server's answer to a coverage page read at another revision. */
function revisionConflict(): Response {
  return failure(
    409,
    "conflict",
    "resource state changed; retry with the current revision",
    true,
  );
}

function fixture(initialPath = "/coverage") {
  let current = auditAt("active", 1);
  // Like the Server, read every page at its pinned revision or answer 409.
  const loadCoverage = vi.fn<(url: URL) => Promise<Response>>(async (url) =>
    url.searchParams.get("auditRevision") === String(current.revision)
      ? json({ ...page([coverage(current)]), auditRevision: current.revision })
      : revisionConflict(),
  );
  const { api } = testAPI(
    fakeServer({
      "GET /v1/audits/{auditId}/coverage": ({ url }) => loadCoverage(url),
      "GET /v1/audits/{auditId}": () =>
        json(current, 200, { ETag: `"${current.revision}"` }),
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
        json({ ...page([coverage(auditAt("active", 1))]), auditRevision: 1 }),
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

  it("pins every coverage page to the revision its batch starts from", async () => {
    const { result, loadCoverage } = fixture();
    await waitFor(() => expect(result.current.query.isSuccess).toBe(true));
    expect(loadCoverage).toHaveBeenCalledTimes(1);
    const url = loadCoverage.mock.calls[0]?.[0];
    expect(url?.searchParams.get("auditRevision")).toBe("1");
    expect(url?.searchParams.get("round")).toBe("round_example");
  });

  it("retries one unpinned first batch after a revision straddle", async () => {
    const { result, loadCoverage, setServerRevision } = fixture();
    loadCoverage.mockImplementationOnce(async () => {
      setServerRevision("active", 2);
      return revisionConflict();
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
      expect(url.searchParams.get("auditRevision")).toBe("2");
      setServerRevision("completed", 3);
      return revisionConflict();
    });
    loadCoverage.mockImplementationOnce(async (url) => {
      expect(url.searchParams.get("auditRevision")).toBe("3");
      setServerRevision("completed", 4);
      return revisionConflict();
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
      expect(url.searchParams.get("auditRevision")).toBe("1");
      const next = Number(url.searchParams.get("cursor") ?? "0") + 1;
      return json({
        ...page([coverage(auditAt("active", 1))], String(next)),
        auditRevision: 1,
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
