/**
 * URLs of the Issues destination (docs/design/ui/v3b-build-contract.md §5):
 * `/issues?state=<finding state>&project=<projectId>&severity=<…>` filters
 * the list, and the selected possible issue is the path
 * `/issues/:auditId/:findingId`.
 */
import type {
  AuditFinding,
  AuditFindingSeverity,
  AuditFindingState,
} from "../../api/audits";
import {
  isFindingSeverity,
  isFindingState,
} from "../projects/audits/finding-options";

/** A finding state, or every state. */
export type StateFilter = AuditFindingState | "all";

export interface IssueFilters {
  /** Default "proposed" (Needs review); "all" lists every state. */
  state: StateFilter;
  /** One project's possible issues; undefined for every project. */
  project?: string | undefined;
  /** The analyst's rating; the AI suggestion is never filtered on. */
  severity?: AuditFindingSeverity | undefined;
}

/** The state the list shows without a `state` parameter. */
export const DEFAULT_STATE: StateFilter = "proposed";

/** Filters from the list's query string; unknown values fall back. */
export function readIssueFilters(params: URLSearchParams): IssueFilters {
  const state = params.get("state");
  const project = params.get("project");
  const severity = params.get("severity");
  return {
    state:
      state === "all"
        ? "all"
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
export function issuePath(auditId: string, findingId: string): string {
  return `/issues/${encodeURIComponent(auditId)}/${encodeURIComponent(findingId)}`;
}

/** A possible issue on the Issues destination with the list's filters. */
export function issueHref(
  finding: Pick<AuditFinding, "auditId" | "findingId">,
  filters: IssueFilters,
): string {
  return `${issuePath(finding.auditId, finding.findingId)}${issueSearch(filters)}`;
}

/** The check page's possible issues section, focused on one of them. */
export function checkIssuePath(
  projectId: string,
  auditId: string,
  findingId?: string,
): string {
  const section = `/projects/${encodeURIComponent(projectId)}/audits/${encodeURIComponent(auditId)}/findings`;
  return findingId === undefined
    ? section
    : `${section}?finding=${encodeURIComponent(findingId)}`;
}

/**
 * The check page's possible issues section filtered like an Issues list:
 * the section reads `state` (absent for every state) and `severity`.
 */
export function checkIssuesHref(
  projectId: string,
  auditId: string,
  filters: Pick<IssueFilters, "state" | "severity">,
): string {
  const params = new URLSearchParams();
  if (filters.state !== "all") params.set("state", filters.state);
  if (filters.severity !== undefined) params.set("severity", filters.severity);
  const text = params.toString();
  return `${checkIssuePath(projectId, auditId)}${text === "" ? "" : `?${text}`}`;
}

/** True when a possible issue still belongs in a list with these filters. */
export function matchesIssueFilters(
  finding: Pick<AuditFinding, "state" | "analystSeverity">,
  filters: IssueFilters,
): boolean {
  return (
    (filters.state === "all" || finding.state === filters.state) &&
    (filters.severity === undefined ||
      finding.analystSeverity === filters.severity)
  );
}
