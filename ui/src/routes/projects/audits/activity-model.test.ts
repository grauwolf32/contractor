import { describe, expect, it } from "vitest";

import {
  CHECK_ACTIVITY_LIMIT,
  checkActivity,
  itemActivity,
  nowSentence,
} from "./activity-model";
import { buildEntries } from "./check-model";
import {
  endpointRow,
  makeAttempt,
  makeAudit,
  makeFinding,
  makeItem,
  makeReview,
} from "./check-test-support";

describe("activity", () => {
  it("tells one item's story newest first, from attempts to the result", () => {
    const [entry] = buildEntries(
      [
        endpointRow("item_1", 0, "GET", "/orders", "traced-partial", {
          resultSummary: "Read without an owner check.\n\nMore.",
        }),
      ],
      [
        makeItem("item_1", 0, "operation-trace", {
          createdAt: "2026-10-05T10:00:00Z",
          attempts: [
            makeAttempt("item_1", 1, {
              terminalOutcome: "failed",
              collectionDisposition: "execution-failed",
              createdAt: "2026-10-05T10:01:00Z",
              collectedAt: "2026-10-05T10:02:00Z",
            }),
            makeAttempt("item_1", 2, {
              createdAt: "2026-10-05T10:03:00Z",
              collectedAt: "2026-10-05T10:04:00Z",
            }),
          ],
        }),
      ],
      [
        makeFinding("audit_1", "f1", "IDOR on orders", "GET /orders", {
          createdAt: "2026-10-05T10:05:00Z",
        }),
      ],
      "endpoint",
    );
    const log = itemActivity(entry!);
    expect(log.map((step) => [step.title, step.text])).toEqual([
      ["Result: Partially traced.", "Read without an owner check."],
      ["Proposed a possible issue:", "IDOR on orders"],
      ["Attempt 2 finished.", "Its result was accepted."],
      ["Retried automatically: attempt 2 started.", undefined],
      ["Attempt 1 failed.", "The run failed."],
      ["Attempt 1 started.", undefined],
      ["Added to the check.", undefined],
    ]);
    expect(log[4]?.tone).toBe("blocked");
  });

  it("says what a check is doing now", () => {
    const entries = buildEntries(
      [endpointRow("item_1", 0, "GET", "/a", "not-tested")],
      [makeItem("item_1", 0, "operation-trace", { state: "collecting" })],
      [],
      "endpoint",
    );
    expect(
      nowSentence(makeAudit("a", "p", "active"), entries, "endpoint"),
    ).toBe("Running: 1 endpoint is being checked now.");
    expect(nowSentence(makeAudit("a", "p", "draft"), [], "requirement")).toBe(
      "Draft: start the check to create its requirements.",
    );
    expect(
      nowSentence(makeAudit("a", "p", "completed"), [], "item"),
    ).toBeUndefined();
  });

  it("keeps the whole-check log to the latest events", () => {
    const audit = makeAudit("audit_1", "project_1", "completed", {
      finishedAt: "2026-10-05T12:00:00Z",
    });
    const rows = Array.from({ length: CHECK_ACTIVITY_LIMIT + 5 }, (_, index) =>
      endpointRow(
        `item_${index}`,
        index,
        "GET",
        `/r${index}`,
        "traced-complete",
      ),
    );
    const log = checkActivity({
      audit,
      entries: buildEntries(rows, [], [], "endpoint"),
      findings: [],
      waiting: [makeReview("audit_1", "review_1")],
      now: undefined,
    });
    expect(log.entries).toHaveLength(CHECK_ACTIVITY_LIMIT);
    expect(log.total).toBe(CHECK_ACTIVITY_LIMIT + 5 + 4);
    expect(log.entries[0]).toMatchObject({
      title: "Check finished.",
      tone: "done",
    });
    const withNow = checkActivity({
      audit: makeAudit("audit_1", "project_1", "waiting_review"),
      entries: [],
      findings: [],
      waiting: [makeReview("audit_1", "review_1")],
      now: "Waiting for your decisions before it can go on.",
    });
    expect(withNow.entries[0]).toMatchObject({ time: "now", tone: "review" });
    expect(withNow.entries[1]).toMatchObject({
      title: "Waiting for your decision:",
      text: "Active test approval",
    });
  });
});
