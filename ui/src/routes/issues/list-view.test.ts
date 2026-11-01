import { describe, expect, it } from "vitest";
import type { CrossProjectIssue } from "../../api/cross-project";
import { issueReviewFixtures } from "../../test/issue-review-fixtures";
import {
  ISSUE_SORTS,
  readIssueListView,
  visibleIssues,
  withIssueListView,
} from "./list-view";

function rows(): CrossProjectIssue[] {
  const data = issueReviewFixtures();
  return data.findings.map((finding) => {
    const audit = data.audits.find(
      (value) => value.auditId === finding.auditId,
    )!;
    return {
      finding,
      audit,
      project: data.projects.find(
        (value) => value.projectId === audit.projectId,
      )!,
    };
  });
}
const ids = (items: CrossProjectIssue[]) =>
  items.map((value) => value.finding.findingId);

describe("Issue list projection", () => {
  it("sorts severity using the analyst's rating before AI, keeping unset last", () => {
    const data = rows();
    const rated = data.find(
      (row) => row.finding.analystSeverity !== undefined,
    )!;
    rated.finding.firstProposal.document.severity_suggestion = "low";
    const sorted = visibleIssues(data, { sort: "severity", q: "" });
    expect(sorted.indexOf(rated)).toBeLessThan(
      sorted.findIndex(
        (row) =>
          row.finding.firstProposal.document.severity_suggestion === "medium",
      ),
    );
    expect(
      sorted.at(-1)?.finding.firstProposal.document.severity_suggestion,
    ).toBe("");
    expect(data).toEqual(
      rows().map((row) =>
        row.finding.findingId === rated.finding.findingId ? rated : row,
      ),
    );
  });
  it("orders time, state and title without mutating the input and breaks equal ties by identity", () => {
    const data = rows(),
      before = ids(data);
    const newest = visibleIssues(data, { sort: "newest", q: "" });
    expect(newest.map((row) => row.finding.createdAt)).toEqual(
      data
        .map((row) => row.finding.createdAt)
        .sort()
        .reverse(),
    );
    expect(ids(visibleIssues(data, { sort: "oldest", q: "" }))).toEqual(
      ids(newest).reverse(),
    );
    expect(
      visibleIssues(data, { sort: "status", q: "" }).at(-1)?.finding.state,
    ).toBe("confirmed");
    const titles = visibleIssues(data, { sort: "title", q: "" }).map((row) =>
      row.finding.firstProposal.document.title.toLowerCase(),
    );
    expect(titles).toEqual([...titles].sort());
    expect(ids(data)).toEqual(before);
    const left = structuredClone(data[0]!),
      right = structuredClone(left);
    left.finding.findingId = "a";
    right.finding.findingId = "z";
    for (const { value: sort } of ISSUE_SORTS)
      expect(ids(visibleIssues([right, left], { sort, q: "" }))).toEqual([
        "a",
        "z",
      ]);
  });
  it("searches title, subject, Project and check label, independently of sorting", () => {
    const data = rows();
    for (const q of ["SQL", "src/routes/products/search.ts"])
      expect(ids(visibleIssues(data, { sort: "title", q }))).toEqual([
        "issue-product-search-sql",
      ]);
    expect(
      visibleIssues(data, { sort: "newest", q: " Records API " }),
    ).toHaveLength(1);
    expect(
      visibleIssues(data, { sort: "status", q: "deployment review" }),
    ).toHaveLength(1);
    expect(
      visibleIssues(data, { sort: "severity", q: "no such issue" }),
    ).toEqual([]);
  });
  it("projects the maximum existing loaded collection with stable complete results", () => {
    const base = rows().find(
      (row) => row.finding.findingId === "issue-product-search-sql",
    )!;
    const data = Array.from({ length: 20_000 }, (_, i) => ({
      ...base,
      finding: {
        ...base.finding,
        findingId: `finding_${String(20_000 - i).padStart(5, "0")}`,
      },
    }));
    const sorted = visibleIssues(data, { sort: "severity", q: "search" });
    expect(sorted).toHaveLength(20_000);
    expect(sorted[0]?.finding.findingId).toBe("finding_00001");
    expect(sorted.at(-1)?.finding.findingId).toBe("finding_20000");
    expect(data[0]?.finding.findingId).toBe("finding_20000");
  });
  it("reads and writes canonical list context alongside existing state and project filters", () => {
    expect(readIssueListView(new URLSearchParams("sort=unknown"))).toEqual({
      sort: "severity",
      q: "",
    });
    const view = readIssueListView(
      new URLSearchParams("sort=oldest&q=SQL+parser"),
    );
    expect(withIssueListView("?state=all&project=project_shop", view)).toBe(
      "?state=all&project=project_shop&sort=oldest&q=SQL+parser",
    );
    expect(withIssueListView("", { sort: "severity", q: " " })).toBe("");
    expect(
      withIssueListView("?state=all&sort=oldest&q=SQL", {
        sort: "severity",
        q: "",
      }),
    ).toBe("?state=all");
  });
});
