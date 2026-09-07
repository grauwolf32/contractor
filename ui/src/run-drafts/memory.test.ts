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

  it("invalidates only a changed automatic suggestion and retains old exact metadata", () => {
    const store = new RunDraftMemoryStore("owner-a");
    const metadata = (name: string, revision: string) => ({
      artifact: { namespace: "sources", name, revision },
      mediaType: "application/zip",
      size: 10,
      current: true,
      frozen: false,
      createdAt: "2026-09-07T10:00:00Z",
    });
    const first = store.acquire(
      { workflowName: "inspect", workflowVersion: "1", projectId: "p1" },
      initialRunDraftState({ source: "sources/one@r1" }, [
        metadata("one", "r1"),
      ]),
    );
    if (first.kind !== "acquired") throw new Error("expected acquired draft");

    const refreshed = store.acquire(
      first.entry.identity,
      initialRunDraftState({ source: "sources/two@r2" }, [
        metadata("two", "r2"),
      ]),
    );
    if (refreshed.kind !== "acquired") {
      throw new Error("expected refreshed draft");
    }
    expect(refreshed.entry.state.artifactSelections.source).toBe(
      "sources/two@r2",
    );
    expect(refreshed.entry.state.artifactReviews.source).toBeUndefined();
    expect(refreshed.entry.meaningful).toBe(false);

    const reviewed = structuredClone(refreshed.entry.state);
    reviewed.artifactReviews.source = "sources/two@r2";
    store.replaceState(refreshed.entry, reviewed);
    const unchanged = store.acquire(
      refreshed.entry.identity,
      initialRunDraftState({ source: "sources/two@r2" }, [
        metadata("two", "r2"),
      ]),
    );
    if (unchanged.kind !== "acquired") {
      throw new Error("expected unchanged draft");
    }
    expect(unchanged.entry.state.artifactReviews.source).toBe("sources/two@r2");

    const changedAgain = store.acquire(
      refreshed.entry.identity,
      initialRunDraftState({ source: "sources/three@r3" }, [
        metadata("three", "r3"),
      ]),
    );
    if (changedAgain.kind !== "acquired") {
      throw new Error("expected changed draft");
    }
    expect(changedAgain.entry.state.artifactSelections.source).toBe(
      "sources/two@r2",
    );
    expect(changedAgain.entry.state.artifactReviews.source).toBeUndefined();
    expect(
      changedAgain.entry.state.knownArtifacts.map(
        (item) =>
          `${item.artifact.namespace}/${item.artifact.name}@${item.artifact.revision}`,
      ),
    ).toEqual(["sources/one@r1", "sources/two@r2", "sources/three@r3"]);
    expect(changedAgain.entry.meaningful).toBe(true);
  });

  it("retains exact suggestion metadata as non-user draft state", async () => {
    const store = new RunDraftMemoryStore("owner-a");
    const metadata = {
      artifact: { namespace: "sources", name: "one", revision: "r1" },
      mediaType: "application/zip",
      size: 10,
      current: true,
      frozen: false,
      createdAt: "2026-09-07T10:00:00Z",
    };
    const result = store.acquire(
      { workflowName: "inspect", workflowVersion: "1", projectId: "p1" },
      initialRunDraftState({ source: "sources/one@r1" }, [metadata]),
    );
    if (result.kind !== "acquired") throw new Error("expected acquired draft");
    expect(result.entry.state.knownArtifacts).toEqual([metadata]);
    expect(result.entry.meaningful).toBe(false);

    store.retain(result.entry);
    store.release(result.entry);
    await Promise.resolve();
    expect(store.isCurrent(result.entry)).toBe(false);
  });

  it("invalidates review only for the slot whose suggestion changed", () => {
    const store = new RunDraftMemoryStore("owner-a");
    const result = store.acquire(
      { workflowName: "inspect", workflowVersion: "1", projectId: "p1" },
      initialRunDraftState({ left: "reports/a@r1", right: "reports/b@r1" }),
    );
    if (result.kind !== "acquired") throw new Error("expected acquired draft");
    const reviewed = structuredClone(result.entry.state);
    reviewed.artifactReviews = {
      left: "reports/a@r1",
      right: "reports/b@r1",
    };
    store.replaceState(result.entry, reviewed);

    const refreshed = store.acquire(
      result.entry.identity,
      initialRunDraftState({ left: "reports/a@r2", right: "reports/b@r1" }),
    );
    if (refreshed.kind !== "acquired") {
      throw new Error("expected refreshed draft");
    }
    expect(refreshed.entry.state.artifactSelections).toEqual({
      left: "reports/a@r1",
      right: "reports/b@r1",
    });
    expect(refreshed.entry.state.artifactReviews).toEqual({
      right: "reports/b@r1",
    });
  });

  it("seeds a repeat draft without replacing edited or mounted state", () => {
    const store = new RunDraftMemoryStore("owner-a");
    const identity = { workflowName: "inspect", workflowVersion: "1" };
    const repeat = initialRunDraftState({ source: "sources/old@r1" });
    repeat.repeat = {
      sourceRunId: "run-old",
      notices: [],
      reviewed: false,
    };
    const seeded = store.seed(identity, repeat);
    expect(seeded.kind).toBe("seeded");
    if (seeded.kind !== "seeded") return;

    const refreshed = store.acquire(
      identity,
      initialRunDraftState({ source: "sources/current@r2" }),
    );
    expect(refreshed.kind).toBe("acquired");
    if (refreshed.kind !== "acquired") return;
    expect(refreshed.entry.state.artifactSelections.source).toBe(
      "sources/old@r1",
    );

    const replacement = initialRunDraftState();
    replacement.repeat = {
      sourceRunId: "run-new",
      notices: [],
      reviewed: false,
    };
    expect(store.seed(identity, replacement)).toMatchObject({
      kind: "conflict",
    });

    store.discard(seeded.entry);
    const mounted = acquired(store, "inspect");
    store.retain(mounted);
    expect(store.seed(identity, replacement)).toMatchObject({
      kind: "conflict",
    });
  });
});
