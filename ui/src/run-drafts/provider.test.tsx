import { QueryClientProvider } from "@tanstack/react-query";
import { act, render } from "@testing-library/react";
import { useEffect } from "react";
import { describe, expect, it, vi } from "vitest";

import type { AuthSession } from "../api/client";
import { queryKeys } from "../api/query-keys";
import { createApplicationQueryClient } from "../app/query-client";
import { SessionProvider } from "../auth/session";
import { useRunDraftStore } from "./context";
import { initialRunDraftState, type RunDraftMemoryStore } from "./memory";
import { RunDraftProvider, SessionRunDraftProvider } from "./provider";
import { discardSessionRunDrafts } from "./session-stores";

function authSession(userId: string): AuthSession {
  return {
    principal: { userId, username: userId, capabilities: ["user"] },
    csrfToken: "a".repeat(43),
    idleExpiresAt: "2099-01-01T00:00:00Z",
    absoluteExpiresAt: "2099-01-02T00:00:00Z",
  };
}

function writeDraft(store: RunDraftMemoryStore, objective: string) {
  const acquired = store.acquire(
    { workflowName: "inspect", workflowVersion: "1" },
    initialRunDraftState(),
  );
  if (acquired.kind !== "acquired") throw new Error("draft not acquired");
  const changed = structuredClone(acquired.entry.state);
  changed.parameters.objective = objective;
  store.replaceState(acquired.entry, changed);
}

function renderSessionProvider(capture: (store: RunDraftMemoryStore) => void) {
  const queryClient = createApplicationQueryClient();
  queryClient.setQueryData(queryKeys.session, authSession("owner-a"));
  render(
    <QueryClientProvider client={queryClient}>
      <SessionProvider
        api={{
          getSession: vi.fn(async () => authSession("owner-a")),
          login: vi.fn(),
          logout: vi.fn(),
        }}
      >
        <SessionRunDraftProvider>
          <Capture onStore={capture} />
        </SessionRunDraftProvider>
      </SessionProvider>
    </QueryClientProvider>,
  );
  return (session: AuthSession | null) =>
    act(async () => {
      queryClient.setQueryData(queryKeys.session, session);
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
}

function Capture({
  onStore,
}: {
  onStore: (store: RunDraftMemoryStore) => void;
}) {
  const store = useRunDraftStore();
  useEffect(() => {
    onStore(store);
  }, [onStore, store]);
  return null;
}

describe("RunDraftProvider", () => {
  it("replaces all in-memory drafts when the authenticated owner changes", () => {
    const stores: RunDraftMemoryStore[] = [];
    const capture = (store: RunDraftMemoryStore) => stores.push(store);
    const view = render(
      <RunDraftProvider ownerId="owner-a">
        <Capture onStore={capture} />
      </RunDraftProvider>,
    );
    const first = stores.at(-1)!;
    const acquired = first.acquire(
      { workflowName: "inspect", workflowVersion: "1" },
      initialRunDraftState(),
    );
    if (acquired.kind !== "acquired") throw new Error("draft not acquired");
    const changed = structuredClone(acquired.entry.state);
    changed.parameters.objective = "private owner-a draft";
    first.replaceState(acquired.entry, changed);

    view.rerender(
      <RunDraftProvider ownerId="owner-b">
        <Capture onStore={capture} />
      </RunDraftProvider>,
    );
    const second = stores.at(-1)!;
    expect(second).not.toBe(first);
    expect(second.ownerId).toBe("owner-b");
    expect(second.summaries()).toEqual([]);
  });

  it("keeps session drafts when the session expires and the same user returns", async () => {
    const stores: RunDraftMemoryStore[] = [];
    const setSession = renderSessionProvider((store) => stores.push(store));
    const first = stores.at(-1)!;
    expect(first.ownerId).toBe("owner-a");
    writeDraft(first, "owner-a draft");

    await setSession(null);
    expect(stores.at(-1)!.ownerId).toBe("anonymous");

    await setSession(authSession("owner-a"));
    expect(stores.at(-1)).toBe(first);
    expect(first.summaries()).toHaveLength(1);
  });

  it("drops session drafts when a different user signs in or on logout", async () => {
    const stores: RunDraftMemoryStore[] = [];
    const setSession = renderSessionProvider((store) => stores.push(store));
    const first = stores.at(-1)!;
    writeDraft(first, "owner-a draft");

    await setSession(null);
    await setSession(authSession("owner-b"));
    await setSession(authSession("owner-a"));
    expect(stores.at(-1)).not.toBe(first);
    expect(stores.at(-1)!.summaries()).toEqual([]);

    writeDraft(stores.at(-1)!, "owner-a second draft");
    const second = stores.at(-1)!;
    discardSessionRunDrafts();
    await setSession(null);
    await setSession(authSession("owner-a"));
    expect(stores.at(-1)).not.toBe(second);
    expect(stores.at(-1)!.summaries()).toEqual([]);
  });
});
