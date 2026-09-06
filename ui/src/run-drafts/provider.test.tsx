import { render } from "@testing-library/react";
import { useEffect } from "react";
import { describe, expect, it } from "vitest";

import { useRunDraftStore } from "./context";
import { initialRunDraftState, type RunDraftMemoryStore } from "./memory";
import { RunDraftProvider } from "./provider";

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
});
