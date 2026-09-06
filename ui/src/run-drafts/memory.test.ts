import { describe, expect, it, vi } from "vitest";

import {
  initialRunDraftState,
  MAX_RETAINED_RUN_DRAFTS,
  RunDraftMemoryStore,
  type RunDraftEntry,
} from "./memory";

function acquired(
  store: RunDraftMemoryStore,
  name: string,
  projectId?: string,
): RunDraftEntry {
  const result = store.acquire(
    {
      workflowName: name,
      workflowVersion: "1",
      ...(projectId === undefined ? {} : { projectId }),
    },
    initialRunDraftState(),
  );
  if (result.kind !== "acquired") throw new Error("expected acquired draft");
  return result.entry;
}

describe("RunDraftMemoryStore", () => {
  it("retains exact owner/workflow/scope values and submission identity", () => {
    const store = new RunDraftMemoryStore("owner-a");
    const standalone = acquired(store, "inspect");
    const next = structuredClone(standalone.state);
    next.parameters.objective = "Inspect the API";
    expect(store.replaceState(standalone, next)).toBe(true);
    const request = {
      workflow: "inspect@1",
      parameters: { objective: "Inspect the API" },
    };
    const key = standalone.keyring.keyFor(request, "standalone");
    store.markSubmitted(standalone);
    store.setAmbiguousSubmission(standalone, true);

    const restored = acquired(store, "inspect");
    expect(restored).toBe(standalone);
    expect(restored.state.parameters.objective).toBe("Inspect the API");
    expect(restored.keyring.keyFor(request, "standalone")).toBe(key);
    expect(restored.ambiguousSubmission).toBe(true);

    const project = acquired(store, "inspect", "project-a");
    const otherVersion = store.acquire(
      { workflowName: "inspect", workflowVersion: "2" },
      initialRunDraftState(),
    );
    expect(project).not.toBe(standalone);
    expect(project.state.parameters.objective).toBeUndefined();
    expect(otherVersion.kind).toBe("acquired");
  });

  it("releases pristine forms without retaining an empty draft", async () => {
    const store = new RunDraftMemoryStore("owner-a");
    const first = acquired(store, "inspect");
    store.retain(first);
    store.release(first);
    await Promise.resolve();

    expect(store.isCurrent(first)).toBe(false);
    expect(store.summaries()).toEqual([]);
    const replacement = acquired(store, "inspect");
    expect(replacement.generation).toBeGreaterThan(first.generation);
  });

  it("requires an explicit discard at the retained draft limit", () => {
    const store = new RunDraftMemoryStore("owner-a");
    const first = acquired(store, "workflow-0");
    for (let index = 0; index < MAX_RETAINED_RUN_DRAFTS; index += 1) {
      const entry = index === 0 ? first : acquired(store, `workflow-${index}`);
      const next = structuredClone(entry.state);
      next.parameters.objective = `draft-${index}`;
      store.replaceState(entry, next);
    }

    const blocked = store.acquire(
      { workflowName: "next", workflowVersion: "1" },
      initialRunDraftState(),
    );
    expect(blocked.kind).toBe("capacity");
    if (blocked.kind !== "capacity") return;
    expect(blocked.drafts).toHaveLength(MAX_RETAINED_RUN_DRAFTS);
    expect(store.isCurrent(first)).toBe(true);

    store.discard(blocked.drafts[0]!.key);
    expect(
      store.acquire(
        { workflowName: "next", workflowVersion: "1" },
        initialRunDraftState(),
      ).kind,
    ).toBe("acquired");
  });

  it("rejects late updates from a discarded draft generation", () => {
    const store = new RunDraftMemoryStore("owner-a");
    const stale = acquired(store, "inspect");
    store.discard(stale);
    const replacement = acquired(store, "inspect");
    const late = structuredClone(stale.state);
    late.artifactSelections.source = "sources/late@r1";

    expect(store.replaceState(stale, late)).toBe(false);
    expect(store.setAmbiguousSubmission(stale, true)).toBe(false);
    expect(replacement.state.artifactSelections.source).toBeUndefined();
  });

  it("does not touch browser persistence APIs", () => {
    const local = vi.spyOn(Storage.prototype, "setItem");
    const session = vi.spyOn(Storage.prototype, "removeItem");
    const store = new RunDraftMemoryStore("owner-a");
    const entry = acquired(store, "inspect");
    const next = structuredClone(entry.state);
    next.runtimeLabels = ["debug"];
    store.replaceState(entry, next);
    store.discard(entry);
    expect(local).not.toHaveBeenCalled();
    expect(session).not.toHaveBeenCalled();
    local.mockRestore();
    session.mockRestore();
  });
});
