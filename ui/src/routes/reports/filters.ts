import { PROJECT_ID_PATTERN } from "../../api/projects";

/**
 * The list filter of /reports: ready reports. Pending and unavailable report
 * statuses stay on their checks and are readable at /reports/:auditId.
 */
type ReportFilter = "ready" | "all";

export interface ReportFilters {
  status: ReportFilter;
  /** Only reports of this project; null for every project. */
  project: string | null;
  page: number;
}

/** URL parameters of the list; the selected report is the path. */
const STATUS_PARAM = "status";
export const PROJECT_PARAM = "project";
export const PAGE_PARAM = "page";

const STATUS_FILTERS: readonly ReportFilter[] = ["ready", "all"];

function isReportFilter(value: string | null): value is ReportFilter {
  return STATUS_FILTERS.some((filter) => filter === value);
}

/**
 * Reads `?status=ready|all&project=<projectId>`. A missing or
 * unknown status is "all"; an invalid project ID is no project filter.
 */
export function readFilters(params: URLSearchParams): ReportFilters {
  const status = params.get(STATUS_PARAM);
  const project = params.get(PROJECT_PARAM);
  const page = params.get(PAGE_PARAM) ?? "0";
  return {
    status: isReportFilter(status) ? status : "all",
    project:
      project !== null && PROJECT_ID_PATTERN.test(project) ? project : null,
    page: /^(?:0|[1-9][0-9]{0,5})$/.test(page) ? Number(page) : 0,
  };
}

/** The list's query string for these filters ("" for the defaults). */
export function filterSearch(filters: ReportFilters): string {
  const params = new URLSearchParams();
  if (filters.status !== "all") params.set(STATUS_PARAM, filters.status);
  if (filters.project !== null) params.set(PROJECT_PARAM, filters.project);
  if (filters.page > 0) params.set(PAGE_PARAM, String(filters.page));
  const search = params.toString();
  return search === "" ? "" : `?${search}`;
}

/** /reports/:auditId with the list's filters, so the list stays as it was. */
export function reportPath(auditId: string, filters: ReportFilters): string {
  return `/reports/${encodeURIComponent(auditId)}${filterSearch(filters)}`;
}
