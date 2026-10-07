import { describe, expect, it } from "vitest";

import {
  buildEntries,
  commonPathPrefix,
  countLegend,
  endpointArea,
  entryName,
  filterEntries,
  findingOnItem,
  groupByKind,
  issueSummary,
  itemStatus,
  legend,
  relativePath,
  splitIssues,
  workCounts,
} from "./check-model";
import {
  endpointRow,
  makeAttempt,
  makeAudit,
  makeFinding,
  makeItem,
  requirementRow,
  workspaceOf,
} from "./check-test-support";

describe("check model", () => {
  it("names items by their coverage, or by the work going on before it", () => {
    const row = endpointRow("item_1", 0, "GET", "/a", "not-tested");
    expect(itemStatus(row, undefined)).toEqual({
      label: "Not checked yet",
      tone: "idle",
      group: "not-tested",
    });
    expect(
      itemStatus(
        row,
        makeItem("item_1", 0, "operation-trace", { state: "submitted" }),
      ),
    ).toEqual({ label: "Checking now", tone: "progress", group: "not-tested" });
    expect(
      itemStatus(
        row,
        makeItem("item_1", 0, "operation-trace", { state: "awaiting_review" }),
      ),
    ).toEqual({
      label: "Waiting for you",
      tone: "review",
      group: "not-tested",
    });
    expect(
      itemStatus(
        endpointRow("item_1", 0, "GET", "/a", "traced-partial"),
        undefined,
      ),
    ).toEqual({
      label: "Partially traced",
      tone: "partial",
      group: "uncertain",
    });
    expect(
      itemStatus(
        requirementRow("r", 0, "A01:2025", "Access control", "violated"),
        makeItem("r", 0, "standard-mapping", { state: "submitted" }),
      ),
    ).toEqual({ label: "Issue found", tone: "blocked", group: "issues" });
  });

  it("joins rows, items and possible issues into endpoints and requirements", () => {
    const rows = [
      endpointRow("item_1", 0, "GET", "/shop/orders/{id}", "traced-complete"),
      requirementRow(
        "item_2",
        1,
        "A01:2025",
        "Trace access-control enforcement.\n\nMore text.",
        "violated",
      ),
    ];
    const items = [
      makeItem("item_1", 0, "operation-trace"),
      makeItem("item_2", 1, "standard-mapping", {
        origin: {
          ...makeItem("x", 0, "x").origin,
          standard: {
            scheme: "owasp-web-top10",
            version: "2025",
            mappingKey: "A01:2025",
            entryIds: ["A01:2025"],
            evidenceContract: { id: "bounded", version: "1" },
          },
        },
      }),
    ];
    const onEndpoint = makeFinding(
      "audit_1",
      "f1",
      "IDOR",
      "GET /shop/orders/{id}",
    );
    const onRequirement = makeFinding(
      "audit_1",
      "f2",
      "Missing check",
      "A01:2025",
    );
    const elsewhere = makeFinding("audit_1", "f3", "Other", "POST /other");
    const entries = buildEntries(
      rows,
      items,
      [onEndpoint, onRequirement, elsewhere],
      "item",
    );
    expect(entries.map((entry) => entry.kind)).toEqual([
      "endpoint",
      "requirement",
    ]);
    expect(entries[0]!.operation).toEqual({
      method: "GET",
      path: "/shop/orders/{id}",
    });
    expect(entries[0]!.summary).toBe("");
    expect(entryName(entries[0]!)).toBe("GET /shop/orders/{id}");
    expect(entries[1]!.summary).toBe("Trace access-control enforcement.");
    expect(entryName(entries[1]!)).toBe(
      "A01:2025 Trace access-control enforcement.",
    );
    expect(entries[0]!.findings.map((finding) => finding.findingId)).toEqual([
      "f1",
    ]);
    expect(entries[1]!.findings.map((finding) => finding.findingId)).toEqual([
      "f2",
    ]);
    expect(entries.map((entry) => entry.position)).toEqual([1, 2]);
  });

  it("ties a possible issue to its item by assessment, subject or operation", () => {
    const row = endpointRow("item_1", 0, "GET", "/orders", "traced-complete");
    const operation = { method: "GET", path: "/orders" };
    expect(
      findingOnItem(
        makeFinding("a", "f", "t", null, {
          currentAssessment: {
            assessmentId: "as",
            semanticAssessment: "supported",
            result: {
              ref: { namespace: "n", name: "r", revision: "1" },
              digest: `sha256:${"b".repeat(64)}`,
            },
            receiptId: "rc",
            itemId: "item_1",
            directVerification: false,
            acceptedAt: "2026-10-05T10:00:00Z",
          },
        }),
        row,
        operation,
      ),
    ).toBe(true);
    expect(
      findingOnItem(makeFinding("a", "f", "t", row.subjectKey), row, operation),
    ).toBe(true);
    expect(
      findingOnItem(
        makeFinding("a", "f", "t", "get https://host/orders"),
        row,
        operation,
      ),
    ).toBe(true);
    expect(
      findingOnItem(makeFinding("a", "f", "t", "POST /orders"), row, operation),
    ).toBe(false);
    expect(
      findingOnItem(makeFinding("a", "f", "t", null), row, operation),
    ).toBe(false);
  });

  it("filters by result group and searches task, result, evidence and attempts", () => {
    const rows = [
      endpointRow("item_1", 0, "GET", "/a", "violated", {
        resultSummary: "Anyone can read the record.",
        evidence: [
          { id: "e1", kind: "observation", summary: "Token accepted twice" },
        ],
      }),
      endpointRow("item_2", 1, "PUT", "/b", "blocked", { gaps: ["Too long"] }),
      endpointRow("item_3", 2, "POST", "/c", "not-tested"),
    ];
    const items = [
      makeItem("item_2", 1, "operation-trace", {
        attempts: [
          makeAttempt("item_2", 1, {
            terminalOutcome: "failed",
            collectionDisposition: "execution-failed",
          }),
        ],
      }),
    ];
    const entries = buildEntries(rows, items, [], "endpoint");
    const ids = (group: Parameters<typeof filterEntries>[1], search: string) =>
      filterEntries(entries, group, search).map((entry) => entry.row.itemId);
    expect(ids("all", "")).toEqual(["item_1", "item_2", "item_3"]);
    expect(ids("issues", "")).toEqual(["item_1"]);
    expect(ids("uncertain", "")).toEqual(["item_2"]);
    expect(ids("not-tested", "")).toEqual(["item_3"]);
    expect(ids("complete", "")).toEqual([]);
    expect(ids("all", "TOKEN ACCEPTED")).toEqual(["item_1"]);
    expect(ids("all", "execution-failed")).toEqual(["item_2"]);
    expect(ids("all", "too long")).toEqual(["item_2"]);
    expect(ids("issues", "too long")).toEqual([]);
  });

  it("groups entries by kind in order of appearance", () => {
    const entries = buildEntries(
      [
        endpointRow("e1", 0, "GET", "/a", "traced-complete"),
        requirementRow("r1", 1, "A01:2025", "Text", "satisfied"),
        endpointRow("e2", 2, "GET", "/b", "traced-complete"),
      ],
      [],
      [],
      "requirement",
    );
    expect(
      groupByKind(entries).map((group) => [
        group.kind,
        group.entries.map((entry) => entry.row.itemId),
      ]),
    ).toEqual([
      ["endpoint", ["e1", "e2"]],
      ["requirement", ["r1"]],
    ]);
  });

  it("shortens endpoint paths by their shared directory", () => {
    const paths = [
      "/workshop/api/mechanic/mechanic_report",
      "/workshop/api/shop/orders/{id}",
      "/workshop/api/shop/orders",
    ];
    const prefix = commonPathPrefix(paths);
    expect(prefix).toBe("/workshop/api/");
    expect(relativePath(paths[0]!, prefix)).toBe("mechanic/mechanic_report");
    expect(endpointArea(paths[0]!, prefix)).toBe("mechanic");
    expect(endpointArea("/workshop/api/orders", prefix)).toBeUndefined();
    expect(commonPathPrefix(["/only/one"])).toBe("");
    expect(commonPathPrefix(["/a", "/b"])).toBe("");
    // Every path keeps at least its last part.
    expect(commonPathPrefix(["/api/x", "/api/x/y"])).toBe("/api/");
  });

  it("separates issues from possible issues and ignores dismissed ones", () => {
    const findings = [
      makeFinding("a", "1", "t", null, { state: "confirmed" }),
      makeFinding("a", "2", "t", null),
      makeFinding("a", "3", "t", null, { state: "needs-evidence" }),
      makeFinding("a", "4", "t", null, { state: "rejected" }),
      makeFinding("a", "5", "t", null, { state: "duplicate" }),
    ];
    expect(issueSummary(findings)).toEqual(["1 issue", "2 possible issues"]);
    expect(issueSummary([])).toEqual([]);
    const split = splitIssues(findings);
    const ids = (list: readonly { findingId: string }[]) =>
      list.map((finding) => finding.findingId);
    expect(ids(split.issues)).toEqual(["1"]);
    expect(ids(split.possible)).toEqual(["2", "3"]);
    expect(ids(split.setAside)).toEqual(["4", "5"]);
  });

  it("orders the legend from finished work to work not started", () => {
    const entries = buildEntries(
      [
        endpointRow("a", 0, "GET", "/a", "not-tested"),
        endpointRow("b", 1, "GET", "/b", "blocked"),
        endpointRow("c", 2, "GET", "/c", "traced-complete"),
        endpointRow("d", 3, "GET", "/d", "traced-partial"),
        endpointRow("e", 4, "GET", "/e", "not-tested"),
      ],
      [makeItem("a", 0, "operation-trace", { state: "submitted" })],
      [],
      "endpoint",
    );
    expect(legend(entries).map((entry) => [entry.label, entry.count])).toEqual([
      ["Fully traced", 1],
      ["Partially traced", 1],
      ["Checking now", 1],
      ["Blocked", 1],
      ["Not checked yet", 1],
    ]);
  });

  it("reads concluded, issue, follow-up and unchecked counts from a snapshot", () => {
    const counts = workCounts(
      workspaceOf(makeAudit("a", "p", "active"), {
        totalChecks: 10,
        completedChecks: 6,
        issues: 2,
        gaps: 1,
        unchecked: 3,
      }),
    );
    expect(counts).toEqual({
      total: 10,
      done: 4,
      issues: 2,
      followUp: 1,
      unchecked: 3,
    });
    expect(
      countLegend(counts).map((entry) => [
        entry.label,
        entry.count,
        entry.group,
      ]),
    ).toEqual([
      ["Done", 4, "all"],
      ["Issues found", 2, "issues"],
      ["Need follow-up", 1, "uncertain"],
      ["Not checked yet", 3, "not-tested"],
    ]);
  });
});
