import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";

import { PublicAPIProvider } from "../../api/context";
import { InvalidResponseError } from "../../api/error";
import { queryKeys } from "../../api/query-keys";
import { workspace } from "../../test/builders";
import { researchFixture } from "../../test/audit-research-fixtures";
import { deferred } from "../../test/deferred";
import type { Fallback } from "../../test/fake-server";
import { testAPI } from "../../test/render";
import { json, page } from "../../test/responses";
import { useResearchMaterials, useResearchTasks } from "./research";
import { checkKeys, useCheckWorkspaces } from "./workspace";

function context(read: Fallback) {
  const { api } = testAPI(read);
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: Infinity } },
  });
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>
      <PublicAPIProvider api={api}>{children}</PublicAPIProvider>
    </QueryClientProvider>
  );
  return { client, wrapper };
}

describe("shared check queries without a router", () => {
  it("uses the explicit Task cursor and hypothesis with the existing cache identity", async () => {
    const fixture = researchFixture();
    const requests: URL[] = [];
    const server = context(async (_request, url) => {
      requests.push(url);
      return json(page([fixture.tasks[0]!], "cursor_next"));
    });
    const { result, unmount } = renderHook(
      () =>
        useResearchTasks(
          fixture.audit,
          "cursor_current",
          "hypothesis_selected",
        ),
      { wrapper: server.wrapper },
    );
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(requests).toHaveLength(1);
    expect(requests[0]!.pathname).toBe(
      `/v1/audits/${fixture.audit.auditId}/tasks`,
    );
    expect(Object.fromEntries(requests[0]!.searchParams)).toEqual({
      limit: "50",
      cursor: "cursor_current",
      hypothesis_id: "hypothesis_selected",
    });
    expect(
      server.client.getQueryData([
        ...queryKeys.audits.detail(fixture.audit.auditId),
        "tasks",
        "hypothesis_selected",
        "cursor_current",
      ]),
    ).toEqual(result.current.data);
    unmount();
    server.client.clear();
  });

  it("rejects a continuation that repeats the current Task cursor", async () => {
    const fixture = researchFixture();
    const server = context(async () =>
      json(page([fixture.tasks[0]!], "cursor_current")),
    );
    const { result, unmount } = renderHook(
      () => useResearchTasks(fixture.audit, "cursor_current"),
      { wrapper: server.wrapper },
    );
    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.error).toBeInstanceOf(InvalidResponseError);
    expect(result.current.error).toMatchObject({
      code: "invalid_api_response",
      detail: "Invalid Task continuation",
    });
    unmount();
    server.client.clear();
  });

  it("does not fetch disabled materials and aborts their in-flight read on unmount", async () => {
    const fixture = researchFixture();
    const pending = deferred<Response>();
    let signal: AbortSignal | undefined;
    const read = vi.fn(async (request: Request) => {
      signal = request.signal;
      return pending.promise;
    });
    const server = context(read);
    const { rerender, unmount } = renderHook(
      ({ enabled }) => useResearchMaterials(fixture.audit, undefined, enabled),
      { wrapper: server.wrapper, initialProps: { enabled: false } },
    );
    expect(read).not.toHaveBeenCalled();
    rerender({ enabled: true });
    await waitFor(() => expect(read).toHaveBeenCalledOnce());
    expect(signal?.aborted).toBe(false);
    unmount();
    expect(signal?.aborted).toBe(true);
    await act(async () =>
      pending.resolve(
        json({ ...page([]), auditRevision: fixture.audit.revision }),
      ),
    );
    server.client.clear();
  });

  it("reuses exact workspace revisions and keeps previous counts while the next revision loads", async () => {
    const fixture = researchFixture();
    const first = workspace(fixture.audit);
    const nextAudit = {
      ...fixture.audit,
      revision: fixture.audit.revision + 1,
    };
    const second = workspace(nextAudit);
    const pending = deferred<Response>();
    let reads = 0;
    const read = vi.fn(async () =>
      ++reads === 1 ? json(first) : pending.promise,
    );
    const server = context(read);
    const { result, rerender, unmount } = renderHook(
      ({ audit }) => useCheckWorkspaces([audit]),
      { wrapper: server.wrapper, initialProps: { audit: fixture.audit } },
    );
    await waitFor(() =>
      expect(result.current.get(fixture.audit.auditId)).toEqual(first),
    );
    rerender({ audit: nextAudit });
    await waitFor(() => expect(read).toHaveBeenCalledTimes(2));
    expect(result.current.get(fixture.audit.auditId)).toEqual(first);
    await act(async () => pending.resolve(json(second)));
    await waitFor(() =>
      expect(result.current.get(fixture.audit.auditId)).toEqual(second),
    );
    expect(
      server.client.getQueryData(
        checkKeys.workspaceAt(fixture.audit.auditId, fixture.audit.revision),
      ),
    ).toEqual(first);
    rerender({ audit: fixture.audit });
    await waitFor(() =>
      expect(result.current.get(fixture.audit.auditId)).toEqual(first),
    );
    expect(read).toHaveBeenCalledTimes(2);
    unmount();
    server.client.clear();
  });
});
