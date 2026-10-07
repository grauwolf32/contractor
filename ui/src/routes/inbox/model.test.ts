import { describe, expect, it } from "vitest";

import type { Audit, AuditWorkspace } from "../../api/audits";
import type {
  CrossProjectCheck,
  CrossProjectReport,
} from "../../api/cross-project";
import type { RunStatus, RunSummary } from "../../api/runs";
import { auditFixture, projectFixture } from "../../test/shell-harness";
import {
  buildInbox,
  finishedRecently,
  inboxSearch,
  inboxSubtitle,
  moreRecentRunsMayFollow,
  nextToDecide,
  parseRefKey,
  RECENT_MS,
  recentRuns,
  refKey,
  rowIndex,
  type InboxInput,
} from "./model";
import { checkProgress, primaryOutput } from "./present";

const NOW = Date.parse("2026-10-07T12:00:00Z");

function ago(minutes: number): string {
  return new Date(NOW - minutes * 60_000).toISOString();
}

function check(
  auditId: string,
  state: Audit["state"],
  overrides: Partial<Audit> = {},
): CrossProjectCheck {
  return {
    project: projectFixture("project_a", { name: "Shop" }),
    audit: { ...auditFixture("project_a", auditId, state), ...overrides },
  };
}

function run(
  runId: string,
  state: RunSummary["state"],
  minutes: number,
): RunSummary {
  return {
    runId,
    workflow: "review@1",
    state,
    deletable: false,
    labels: {},
    createdAt: ago(minutes + 1),
    updatedAt: ago(minutes),
    finishedAt: ago(minutes),
  };
}

function workspace(auditId: string, gaps: number): AuditWorkspace {
  return {
    auditId,
    auditRevision: 1,
    asOf: ago(0),
    executionState: "active",
    outstandingRuns: 0,
    totalChecks: 4,
    completedChecks: 1,
    issues: 0,
    gaps,
    unchecked: 3 - gaps,
    findings: 0,
    unreviewedFindings: 0,
    pendingReviews: 0,
  };
}

function input(overrides: Partial<InboxInput> = {}): InboxInput {
  return {
    now: NOW,
    issues: [],
    decisions: [],
    checks: [],
    reports: [],
    workspaces: new Map(),
    failedRuns: { shown: [], hidden: 0 },
    succeededRuns: { shown: [], hidden: 0 },
    waitingRuns: [],
    runStatuses: new Map(),
    ...overrides,
  };
}

describe("Inbox selection keys", () => {
  it("round-trips every kind, also with colons in IDs", () => {
    const refs = [
      { kind: "issue", auditId: "audit:1", findingId: "finding_1" },
      { kind: "review", auditId: "audit_1", requestId: "req.2" },
      { kind: "run", runId: "run-1" },
      { kind: "check", auditId: "audit_1" },
      { kind: "report", auditId: "audit_1" },
    ] as const;
    for (const ref of refs) expect(parseRefKey(refKey(ref))).toEqual(ref);
    expect(refKey(refs[0])).toBe("issue:audit%3A1:finding_1");
    // URLSearchParams decodes "%25" back to "%", which the key then decodes.
    const search = inboxSearch(refs[0]);
    expect(search).toBe("?item=issue:audit%253A1:finding_1");
    expect(parseRefKey(new URLSearchParams(search).get("item"))).toEqual(
      refs[0],
    );
  });

  it("ignores malformed values", () => {
    for (const value of [
      null,
      "",
      "issue:only_one",
      "run:a:b",
      "check:",
      "unknown:a",
      "check:%E0%A4%A",
      "check:has space",
    ])
      expect(parseRefKey(value)).toBeUndefined();
  });
});

describe("Inbox sections", () => {
  it("lists a stuck running check in Unblock and Running and moves through both rows", () => {
    const model = buildInbox(
      input({
        checks: [check("audit_run", "active"), check("audit_late", "paused")],
        workspaces: new Map([["audit_run", workspace("audit_run", 2)]]),
      }),
    );
    expect(model.unblock.map((row) => row.key)).toEqual([
      "check:audit_late",
      "check:audit_run",
    ]);
    expect(model.running.map((row) => row.key)).toEqual(["check:audit_run"]);
    // The keys follow the rows as shown.
    expect(model.order.map((row) => `${row.section} ${row.key}`)).toEqual([
      "unblock check:audit_late",
      "unblock check:audit_run",
      "running check:audit_run",
    ]);
    // The URL names the item: its first row, or the row the user chose.
    expect(rowIndex(model.order, "check:audit_run")).toBe(1);
    expect(rowIndex(model.order, "check:audit_run", "running")).toBe(2);
    expect(rowIndex(model.order, "check:audit_late", "running")).toBe(0);
    expect(rowIndex(model.order, "check:gone", "running")).toBe(-1);
  });

  it("keeps failed and finished work of the last 7 days only", () => {
    const failed = recentRuns(
      [
        run("run_new", "failed", 10),
        run("run_old", "failed", RECENT_MS / 60_000 + 10),
      ],
      NOW,
    );
    expect(failed.shown.map((entry) => entry.runId)).toEqual(["run_new"]);
    const model = buildInbox(
      input({
        failedRuns: failed,
        checks: [
          check("audit_old_failure", "failed", {
            finishedAt: ago(60 * 24 * 9),
          }),
          check("audit_failure", "failed", { finishedAt: ago(30) }),
          check("audit_paused", "paused", { updatedAt: ago(60 * 24 * 30) }),
        ],
      }),
    );
    expect(model.unblock.map((row) => row.key)).toEqual([
      "check:audit_paused",
      "check:audit_failure",
      "run:run_new",
    ]);
  });

  it("shows a finished check with a ready report once, as the report", () => {
    const done = check("audit_done", "completed", { finishedAt: ago(5) });
    const report: CrossProjectReport = {
      ...done,
      report: { status: "ready", summary: "# Report" },
    };
    const model = buildInbox(
      input({
        checks: [
          done,
          check("audit_plain", "completed", { finishedAt: ago(9) }),
        ],
        reports: [report],
      }),
    );
    expect(model.ready.map((row) => row.key)).toEqual([
      "report:audit_done",
      "check:audit_plain",
    ]);
  });

  it("cuts long lists and says how many were left out", () => {
    const runs = recentRuns(
      Array.from({ length: 8 }, (_, index) =>
        run(`run_${index}`, "failed", index),
      ),
      NOW,
    );
    expect(runs.shown).toHaveLength(5);
    expect(runs.hidden).toBe(3);
    expect(runs.shown[0]?.runId).toBe("run_0");
  });

  it("knows when recent Runs may continue on the Server's next page", () => {
    const recent = [run("run_a", "failed", 10), run("run_b", "failed", 20)];
    const old = run("run_old", "failed", RECENT_MS / 60_000 + 10);
    // A full page of recent Runs: the next page may hold more of them.
    expect(moreRecentRunsMayFollow(recent, true, NOW)).toBe(true);
    // The page reaches past the 7 days, or it is the last page.
    expect(moreRecentRunsMayFollow([...recent, old], true, NOW)).toBe(false);
    expect(moreRecentRunsMayFollow(recent, false, NOW)).toBe(false);
    expect(moreRecentRunsMayFollow([], true, NOW)).toBe(false);
  });

  it("reads reports only of checks that finished in the last 7 days", () => {
    expect(
      finishedRecently(check("a", "completed", { finishedAt: ago(30) }), NOW),
    ).toBe(true);
    expect(
      finishedRecently(
        check("b", "completed", { finishedAt: ago(60 * 24 * 8) }),
        NOW,
      ),
    ).toBe(false);
    for (const state of ["failed", "cancelled", "waiting_review"] as const)
      expect(
        finishedRecently(check("c", state, { finishedAt: ago(30) }), NOW),
      ).toBe(false);
  });

  it("lists a waiting Run only once the Server says it needs a retry", () => {
    // A waiting Run has not ended.
    const waiting: RunSummary = {
      runId: "run_wait",
      workflow: "review@1",
      state: "waiting",
      deletable: false,
      labels: {},
      createdAt: ago(2),
      updatedAt: ago(1),
    };
    const status = (requiresRetry: boolean) =>
      ({
        runId: "run_wait",
        recovery: {
          code: "model_unavailable",
          since: ago(5),
          automaticUntil: ago(1),
          requiresRetry,
        },
      }) as RunStatus;
    const automatic = buildInbox(
      input({
        waitingRuns: [waiting],
        runStatuses: new Map([["run_wait", status(false)]]),
      }),
    );
    expect(automatic.unblock).toHaveLength(0);
    const manual = buildInbox(
      input({
        waitingRuns: [waiting],
        runStatuses: new Map([["run_wait", status(true)]]),
      }),
    );
    expect(manual.unblock.map((row) => row.key)).toEqual(["run:run_wait"]);
  });
});

describe("Inbox words", () => {
  it("summarises the counts in plain words", () => {
    expect(inboxSubtitle({ decide: 2, unblock: 1, ready: 2, running: 1 })).toBe(
      "2 need you, 1 is stuck, 2 are ready, 1 check is running.",
    );
    expect(inboxSubtitle({ decide: 1, unblock: 0, ready: 0, running: 3 })).toBe(
      "1 needs you, 3 checks are running.",
    );
    expect(inboxSubtitle({ decide: 0, unblock: 0, ready: 0, running: 0 })).toBe(
      "Nothing needs you right now.",
    );
  });

  it("names the next decision after one leaves the list", () => {
    const model = buildInbox(
      input({
        checks: [
          check("a", "paused"),
          check("b", "paused"),
          check("c", "paused"),
        ],
      }),
    );
    const rows = model.unblock;
    expect(nextToDecide(rows, "check:b")?.key).toBe("check:c");
    expect(nextToDecide(rows, "check:c")?.key).toBe("check:b");
    expect(nextToDecide(rows, "check:gone")?.key).toBe("check:a");
    expect(nextToDecide([], "check:a")).toBeUndefined();
    // A refresh removed the decided item first: its neighbour moved into
    // the place it held, or it was last and the new last item is next.
    expect(nextToDecide(rows, "check:gone", 1)?.key).toBe("check:b");
    expect(nextToDecide(rows, "check:gone", 3)?.key).toBe("check:c");
    expect(nextToDecide([], "check:gone", 1)).toBeUndefined();
  });

  it("builds a progress line from the workspace counts", () => {
    const progress = checkProgress(
      {
        ...workspace("a", 1),
        totalChecks: 6,
        completedChecks: 3,
        issues: 1,
        unchecked: 2,
      },
      "requirement",
    );
    expect(progress.summary).toBe("3 of 6 requirements done");
    expect(progress.label).toBe(
      "3 of 6 requirements done: 1 with an issue found, 1 needs follow-up, 2 not checked yet",
    );
    expect(progress.segments.map((segment) => segment.tone)).toEqual([
      "done",
      "done",
      "blocked",
      "partial",
      "idle",
      "idle",
    ]);
  });

  it("never picks a file when no primary output is declared", () => {
    const status = {
      outputs: {
        log: { namespace: "outputs", name: "log", revision: "r1" },
      },
    } as unknown as RunStatus;
    const declared = (primary: boolean) =>
      new Map([
        [
          "review@1",
          { log: { required: true, mediaTypes: ["text/plain"], primary } },
        ],
      ]);
    expect(
      primaryOutput({ workflow: "review@1" }, status, false, declared(false)),
    ).toEqual({ state: "none" });
    expect(
      primaryOutput({ workflow: "review@1" }, status, false, declared(true)),
    ).toMatchObject({ state: "present", slot: "log" });
    expect(
      primaryOutput(
        { workflow: "review@1" },
        { outputs: {} } as unknown as RunStatus,
        false,
        declared(true),
      ),
    ).toEqual({ state: "missing", slot: "log" });
    expect(
      primaryOutput(
        { workflow: "review@1" },
        status,
        false,
        new Map([["review@1", null]]),
      ),
    ).toEqual({ state: "unavailable" });
  });
});
