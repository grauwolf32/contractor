/**
 * URLs of the Issues destination (docs/design/ui/v3b-build-contract.md §5):
 * `/issues?state=<finding state>&project=<projectId>&severity=<…>` filters
 * the list, and the selected possible issue is the path
 * `/issues/:projectId/:findingId`.
 */
import { findingNeedsReview, findingSeverity } from "../../api/audits";
import type {
  ProjectFinding,
  FindingSeverity,
  FindingState,
} from "../../api/audits";
import {
  isFindingSeverity,
  isFindingState,
} from "../projects/audits/finding-options";

/** A finding state, or every state. */
export type StateFilter = FindingState | "needs-review" | "unassessed" | "all";

export interface IssueFilters {
  /** Default "needs-review" (unassessed or reopened); "all" lists every state. */
  state: StateFilter;
  /** One project's possible issues; undefined for every project. */
  project?: string | undefined;
  /** Analyst severity when assessed, otherwise the producer suggestion. */
  severity?: FindingSeverity | undefined;
}

/** The state the list shows without a `state` parameter. */
const DEFAULT_STATE: StateFilter = "needs-review";

/** Filters from the list's query string; unknown values fall back. */
export function readIssueFilters(params: URLSearchParams): IssueFilters {
  const state = params.get("state");
  const project = params.get("project");
  const severity = params.get("severity");
  return {
    state:
      state === "all" || state === "unassessed" || state === "needs-review"
        ? state
        : state !== null && isFindingState(state)
          ? state
          : DEFAULT_STATE,
    project: project === null || project === "" ? undefined : project,
    severity:
      severity !== null && isFindingSeverity(severity) ? severity : undefined,
  };
}

/** The query string of the filters ("" or "?…"); the default state is left out. */
export function issueSearch(filters: IssueFilters): string {
  const params = new URLSearchParams();
  if (filters.state !== DEFAULT_STATE) params.set("state", filters.state);
  if (filters.project !== undefined) params.set("project", filters.project);
  if (filters.severity !== undefined) params.set("severity", filters.severity);
  const text = params.toString();
  return text === "" ? "" : `?${text}`;
}

/** Path of one possible issue on the Issues destination. */
export function issuePath(projectId: string, findingId: string): string {
  return `/issues/${encodeURIComponent(projectId)}/${encodeURIComponent(findingId)}`;
}

/** A possible issue on the Issues destination with the list's filters. */
export function issueHref(
  finding: Pick<ProjectFinding, "projectId" | "findingId">,
  filters: IssueFilters,
): string {
  return `${issuePath(finding.projectId, finding.findingId)}${issueSearch(filters)}`;
}

/** True when a possible issue still belongs in a list with these filters. */
export function matchesIssueFilters(
  finding: Pick<
    ProjectFinding,
    "state" | "analystSeverity" | "analystVerdict" | "claim"
  > & { severitySuggestion?: string },
  filters: IssueFilters,
): boolean {
  return (
    (filters.state === "all" ||
      (filters.state === "needs-review" && findingNeedsReview(finding)) ||
      (filters.state === "unassessed"
        ? finding.state === null
        : finding.state === filters.state)) &&
    (filters.severity === undefined ||
      findingSeverity(finding) === filters.severity)
  );
}
