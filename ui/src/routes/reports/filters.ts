import { PROJECT_ID_PATTERN } from "../../api/projects";

/**
 * The list filter of /reports: reports waiting for acceptance, ready
 * reports, or all reports (both). Pending and unavailable reports are not
 * reports yet; their checks show them, and /reports/:auditId opens them.
 */
export type ReportFilter = "proposed" | "ready" | "all";

export interface ReportFilters {
  status: ReportFilter;
  /** Only reports of this project; null for every project. */
  project: string | null;
}

/** URL parameters of the list; the selected report is the path. */
export const STATUS_PARAM = "status";
export const PROJECT_PARAM = "project";

const STATUS_FILTERS: readonly ReportFilter[] = ["proposed", "ready", "all"];

function isReportFilter(value: string | null): value is ReportFilter {
  return STATUS_FILTERS.some((filter) => filter === value);
}

/**
 * Reads `?status=proposed|ready|all&project=<projectId>`. A missing or
 * unknown status is "all"; an invalid project ID is no project filter.
 */
export function readFilters(params: URLSearchParams): ReportFilters {
  const status = params.get(STATUS_PARAM);
  const project = params.get(PROJECT_PARAM);
  return {
    status: isReportFilter(status) ? status : "all",
    project:
      project !== null && PROJECT_ID_PATTERN.test(project) ? project : null,
  };
}

/** The list's query string for these filters ("" for the defaults). */
export function filterSearch(filters: ReportFilters): string {
  const params = new URLSearchParams();
  if (filters.status !== "all") params.set(STATUS_PARAM, filters.status);
  if (filters.project !== null) params.set(PROJECT_PARAM, filters.project);
  const search = params.toString();
  return search === "" ? "" : `?${search}`;
}

/** /reports/:auditId with the list's filters, so the list stays as it was. */
export function reportPath(auditId: string, filters: ReportFilters): string {
  return `/reports/${encodeURIComponent(auditId)}${filterSearch(filters)}`;
}

/** The check page and its sections (coverage, report, …). */
export function checkPath(
  audit: { projectId: string; auditId: string },
  section?: string,
): string {
  const base = `/projects/${encodeURIComponent(audit.projectId)}/audits/${encodeURIComponent(audit.auditId)}`;
  return section === undefined ? base : `${base}/${section}`;
}
