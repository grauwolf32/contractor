import type { AuditFindingState } from "../../api/audits";
import type { CrossProjectIssue } from "../../api/cross-project";
import { auditProfileLabel } from "../projects/audits/labels";

export const ISSUE_SORTS = [
  { value: "severity", label: "Severity · high first" },
  { value: "newest", label: "Newest first" },
  { value: "oldest", label: "Oldest first" },
  { value: "status", label: "Status · needs review first" },
  { value: "title", label: "Title A–Z" },
] as const;
type IssueSort = (typeof ISSUE_SORTS)[number]["value"];
export interface IssueListView {
  sort: IssueSort;
  q: string;
}

export function readIssueListView(params: URLSearchParams): IssueListView {
  const value = params.get("sort");
  return {
    sort:
      ISSUE_SORTS.find((option) => option.value === value)?.value ?? "severity",
    q: params.get("q") ?? "",
  };
}

/** Canonical list context, independent of one finding's tab/review link. */
export function withIssueListView(search: string, view: IssueListView): string {
  const params = new URLSearchParams(search);
  params.delete("sort");
  params.delete("q");
  if (view.sort !== "severity") params.set("sort", view.sort);
  if (view.q.trim() !== "") params.set("q", view.q);
  const text = params.toString();
  return text === "" ? "" : `?${text}`;
}
const severityRank = {
  informational: 1,
  low: 2,
  medium: 3,
  high: 4,
  critical: 5,
};
const stateRank: Record<AuditFindingState, number> = {
  proposed: 0,
  "needs-evidence": 1,
  confirmed: 2,
  rejected: 3,
  duplicate: 4,
};
function severity(issue: CrossProjectIssue): number {
  const value =
    issue.finding.analystSeverity ??
    issue.finding.firstProposal.document.severity_suggestion;
  return value === "" ? 0 : severityRank[value];
}
function text(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}
function newer(left: CrossProjectIssue, right: CrossProjectIssue): number {
  return (
    Date.parse(right.finding.createdAt) - Date.parse(left.finding.createdAt)
  );
}
/** Only projects already-loaded rows; callers retain partial/error notices. */
export function visibleIssues(
  issues: readonly CrossProjectIssue[],
  view: IssueListView,
): CrossProjectIssue[] {
  const query = view.q.trim().toLowerCase();
  return issues
    .filter((issue) => {
      const document = issue.finding.firstProposal.document;
      return (
        query === "" ||
        [
          document.title,
          document.subject?.key,
          issue.project.name,
          auditProfileLabel(issue.audit),
        ]
          .join(" ")
          .toLowerCase()
          .includes(query)
      );
    })
    .sort((left, right) => {
      const order =
        view.sort === "severity"
          ? severity(right) - severity(left)
          : view.sort === "newest"
            ? newer(left, right)
            : view.sort === "oldest"
              ? newer(right, left)
              : view.sort === "status"
                ? stateRank[left.finding.state] - stateRank[right.finding.state]
                : text(
                    left.finding.firstProposal.document.title.toLowerCase(),
                    right.finding.firstProposal.document.title.toLowerCase(),
                  );
      return (
        order ||
        newer(left, right) ||
        text(left.finding.auditId, right.finding.auditId) ||
        text(left.finding.findingId, right.finding.findingId)
      );
    });
}
