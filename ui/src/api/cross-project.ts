/**
 * Cross-project reads for the V3B list destinations: Inbox, Checks, Issues
 * and Reports (docs/design/ui/v3b-build-contract.md §7). The public API has
 * no cross-project lists yet, so these hooks fan out over the per-project and
 * per-check endpoints within fixed bounds (CROSS_PROJECT_LIMITS):
 *
 * - the first page of `kind=project` projects, without projects that are
 *   being deleted;
 * - the first (newest) page of checks of each project;
 * - per check, the first page of its possible issues or pending review
 *   requests, or its report.
 *
 * Freshness. The project index and the check pages are the polled heads:
 * they refetch every CROSS_PROJECT_LIMITS.pollMs while the page is visible.
 * A per-check read is keyed by the check's revision, which the Server
 * advances on every finding, assessment, review and lifecycle change
 * (S19 §14.1), so a head that reports a new revision refetches exactly the
 * checks that changed and per-check reads never poll. While the new revision
 * loads, the previous one stays listed. A failed per-check read retries on
 * the poll interval.
 *
 * Failures never throw. A failed project or check is listed in `errors` and
 * sets `partial`; everything else stays listed. `error` is set only when the
 * project index failed and nothing can be listed. `truncated` says that a
 * bounded read had more records than it returned.
 *
 * Results keep their references while their data is unchanged (TanStack
 * `combine` with structural sharing), so they can feed memo and effect
 * dependencies without render loops.
 */
import {
  type Query,
  type QueryClient,
  type QueryKey,
  type UseQueryResult,
  useQueries,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { useCallback, useMemo } from "react";

import {
  type Audit,
  type AuditFinding,
  type AuditFindingPage,
  type AuditFindingState,
  type AuditPage,
  type AuditReport,
  type AuditReviewPage,
  type AuditReviewRequest,
  type AuditState,
  getAuditReport,
  listAuditFindings,
  listAuditReviews,
  listProjectAudits,
} from "./audits";
import type { PublicAPI } from "./client";
import { usePublicAPI } from "./context";
import { PublicAPIError } from "./error";
import type { components, operations } from "./generated/public";
import { listProjects, type Project, type ProjectPage } from "./projects";
import { queryKeys } from "./query-keys";

export const CROSS_PROJECT_LIMITS = {
  /** Projects read: the first page of `kind=project` projects. */
  projects: 50,
  /** Checks read per project: its first, newest page. */
  auditsPerProject: 50,
  /** Possible issues read per check and filter: its first page. */
  findingsPerAudit: 50,
  /** Poll interval of the project index and check pages while visible. */
  pollMs: 20_000,
} as const;

type AuditReviewKind = AuditReviewRequest["kind"];
type AuditReportStatus = components["schemas"]["AuditReportStatus"];

/** Analyst verdict filter values the findings endpoint accepts. */
export type FindingVerdictFilter = NonNullable<
  NonNullable<operations["listAuditFindings"]["parameters"]["query"]>["verdict"]
>;

/** A check and the project it belongs to. */
export interface CrossProjectCheck {
  project: Project;
  audit: Audit;
}

/** A possible issue (finding) with its check and project. */
export interface CrossProjectIssue extends CrossProjectCheck {
  finding: AuditFinding;
}

/** A pending review request with its check and project. */
export interface CrossProjectDecision extends CrossProjectCheck {
  review: AuditReviewRequest;
}

/** A check's report with the check and project. */
export interface CrossProjectReport extends CrossProjectCheck {
  report: AuditReport;
}

/** A project or check whose read failed. */
export interface CrossProjectError {
  projectId: string;
  /** Set when one check's read failed rather than its project's checks. */
  auditId?: string;
  error: Error;
}

/** Completeness and loading state shared by every cross-project list. */
export interface CrossProjectStatus {
  /** A bounded read had more records than it returned. */
  truncated: boolean;
  /** Some read failed; everything else is still listed. */
  partial: boolean;
  /** The failed projects and checks behind `partial`. */
  errors: CrossProjectError[];
  /** The project index failed, so nothing can be listed. */
  error: Error | null;
  /** Some read has not settled yet; what has settled is already listed. */
  isPending: boolean;
  /** Refetches every cross-project read now. */
  refetch: () => Promise<void>;
}

// The project index and the check pages: polled while the page is visible.
const POLLED = {
  staleTime: CROSS_PROJECT_LIMITS.pollMs,
  refetchInterval: CROSS_PROJECT_LIMITS.pollMs,
  refetchIntervalInBackground: false,
  refetchOnWindowFocus: false,
  retry: false,
} as const;

/** Which per-check reads a check in each state can have. */
const READS_BY_STATE: Readonly<
  Record<AuditState, { issues: boolean; decisions: boolean; report: boolean }>
> = {
  draft: { issues: false, decisions: false, report: false },
  active: { issues: true, decisions: true, report: false },
  waiting_review: { issues: true, decisions: true, report: true },
  paused: { issues: true, decisions: true, report: false },
  finalizing: { issues: true, decisions: true, report: true },
  cancelling: { issues: true, decisions: false, report: false },
  completed: { issues: true, decisions: false, report: true },
  cancelled: { issues: true, decisions: false, report: true },
  failed: { issues: true, decisions: false, report: true },
  deleting: { issues: false, decisions: false, report: false },
};

function statesWith(read: "issues" | "decisions" | "report"): AuditState[] {
  return (Object.keys(READS_BY_STATE) as AuditState[]).filter(
    (state) => READS_BY_STATE[state][read],
  );
}

// Every check that ran can hold possible issues.
const ISSUE_CHECK_STATES = statesWith("issues");
// Running and waiting checks can ask for decisions.
const DECISION_CHECK_STATES = statesWith("decisions");
// Finishing and ended checks can have a report.
const REPORT_CHECK_STATES = statesWith("report");

/** Review kinds that need the user beyond triaging a possible issue. */
const OTHER_DECISIONS: Readonly<Record<AuditReviewKind, boolean>> = {
  // Counted through its possible issue.
  "finding-triage": false,
  "active-check-approval": true,
  "requirement-applicability": true,
  "report-acceptance": true,
};
const OTHER_DECISION_KINDS = (
  Object.keys(OTHER_DECISIONS) as AuditReviewKind[]
).filter((kind) => OTHER_DECISIONS[kind]);

// Possible issues that need a review: never decided, reopened, or assessed
// again since the last decision. Every other state carries a decision or
// waits for evidence.
const NEEDS_REVIEW: readonly AuditFindingState[] = ["proposed"];

const DEFAULT_REPORT_STATUSES: readonly AuditReportStatus[] = [
  "proposed",
  "ready",
];

/** A stable cache key for a filter list: sorted, unique, comma-joined. */
function filterKey(values: readonly string[] | undefined): string {
  return values === undefined ? "" : [...new Set(values)].sort().join(",");
}

function filterValues<T extends string>(key: string): T[] {
  return key === "" ? [] : (key.split(",") as T[]);
}

/** No filter values means one unfiltered read. */
function orUnfiltered<T>(values: T[]): (T | undefined)[] {
  return values.length === 0 ? [undefined] : values;
}

function time(value: string): number {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : Number.NEGATIVE_INFINITY;
}

function newestFirst(left: string, right: string): number {
  const a = time(left);
  const b = time(right);
  return a === b ? 0 : a > b ? -1 : 1;
}

function byText(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function newerCheck(left: CrossProjectCheck, right: CrossProjectCheck) {
  return (
    newestFirst(left.audit.updatedAt, right.audit.updatedAt) ||
    newestFirst(left.audit.createdAt, right.audit.createdAt) ||
    byText(left.audit.auditId, right.audit.auditId)
  );
}

function newerIssue(left: CrossProjectIssue, right: CrossProjectIssue) {
  return (
    newestFirst(left.finding.createdAt, right.finding.createdAt) ||
    newestFirst(left.finding.updatedAt, right.finding.updatedAt) ||
    byText(left.audit.auditId, right.audit.auditId) ||
    byText(left.finding.findingId, right.finding.findingId)
  );
}

function newerDecision(
  left: CrossProjectDecision,
  right: CrossProjectDecision,
) {
  return (
    newestFirst(left.review.createdAt, right.review.createdAt) ||
    byText(left.audit.auditId, right.audit.auditId) ||
    byText(left.review.requestId, right.review.requestId)
  );
}

function useCrossProjectRefetch(): () => Promise<void> {
  const queryClient = useQueryClient();
  return useCallback(async () => {
    await queryClient.refetchQueries({
      queryKey: queryKeys.crossProject.all,
      type: "active",
    });
  }, [queryClient]);
}

/**
 * Refreshes the cross-project lists after a decision or any other mutation.
 * Every cross-project read is marked stale and the polled heads (project
 * index, check pages) refetch at once. Per-check reads follow from there: the
 * mutation advanced the changed check's revision, so its reads move to a new
 * key, while unchanged checks keep their data until their next observer
 * mounts.
 */
export async function invalidateCrossProject(
  queryClient: QueryClient,
): Promise<void> {
  await queryClient.invalidateQueries({
    queryKey: queryKeys.crossProject.all,
    refetchType: "none",
  });
  await Promise.all([
    queryClient.refetchQueries({
      queryKey: queryKeys.crossProject.projects,
      type: "active",
    }),
    queryClient.refetchQueries({
      queryKey: queryKeys.crossProject.allAudits,
      type: "active",
    }),
  ]);
}

export interface ProjectsIndex {
  /** Projects of the first page, newest first, without those in deletion. */
  projects: Project[];
  /** More projects exist beyond the first page. */
  truncated: boolean;
  /** The latest refresh failed; the last page read is still listed. */
  partial: boolean;
  /** The index failed and there is no earlier page to list. */
  error: Error | null;
  isPending: boolean;
  refetch: () => Promise<void>;
}

interface ProjectSelection {
  projects: Project[];
  truncated: boolean;
}

function selectProjects(page: ProjectPage): ProjectSelection {
  const limit = CROSS_PROJECT_LIMITS.projects;
  return {
    projects: page.items
      .slice(0, limit)
      .filter(
        (project) =>
          project.kind === "project" && project.lifecycle !== "deleting",
      ),
    truncated: page.page.hasMore || page.items.length > limit,
  };
}

const NO_PROJECTS: Project[] = [];

/** The first page of projects (kind `project`), polled while visible. */
export function useProjectsIndex(
  options: { enabled?: boolean } = {},
): ProjectsIndex {
  const api = usePublicAPI();
  const refetch = useCrossProjectRefetch();
  const { data, error, isPending } = useQuery({
    queryKey: queryKeys.crossProject.projects,
    queryFn: () => listProjects(api, { kind: "project" }),
    select: selectProjects,
    enabled: options.enabled ?? true,
    ...POLLED,
  });
  return useMemo(
    () => ({
      projects: data?.projects ?? NO_PROJECTS,
      truncated: data?.truncated ?? false,
      partial: data !== undefined && error !== null,
      error: data === undefined ? error : null,
      isPending,
      refetch,
    }),
    [data, error, isPending, refetch],
  );
}

export interface AllChecksOptions {
  /**
   * Only checks in these states; omit (or pass none) for every state. A
   * single state is filtered by the Server, so each project contributes up
   * to 50 checks in that state. Several states filter each project's first
   * page of checks, the read every other cross-project list shares.
   */
  states?: readonly AuditState[];
  /** False keeps every read idle (and `isPending` true). */
  enabled?: boolean;
}

export interface AllChecks extends CrossProjectStatus {
  /** Checks newest first by last update, then by creation. */
  checks: CrossProjectCheck[];
}

interface CombinedChecks {
  checks: CrossProjectCheck[];
  errors: CrossProjectError[];
  truncated: boolean;
  pending: boolean;
}

function combineChecks(
  projects: readonly Project[],
  results: readonly UseQueryResult<AuditPage>[],
  stateKey: string,
): CombinedChecks {
  const wanted = stateKey === "" ? undefined : new Set(stateKey.split(","));
  const limit = CROSS_PROJECT_LIMITS.auditsPerProject;
  const checks: CrossProjectCheck[] = [];
  const errors: CrossProjectError[] = [];
  const seen = new Set<string>();
  let truncated = false;
  let pending = false;
  results.forEach((result, position) => {
    const project = projects[position];
    if (project === undefined) return;
    if (result.isPending) pending = true;
    if (result.isError)
      errors.push({ projectId: project.projectId, error: result.error });
    const page = result.data;
    if (page === undefined) return;
    if (page.page.hasMore || page.items.length > limit) truncated = true;
    for (const audit of page.items.slice(0, limit)) {
      if (
        audit.projectId !== project.projectId ||
        seen.has(audit.auditId) ||
        (wanted !== undefined && !wanted.has(audit.state))
      )
        continue;
      seen.add(audit.auditId);
      checks.push({ project, audit });
    }
  });
  checks.sort(newerCheck);
  return { checks, errors, truncated, pending };
}

/** Checks of every listed project, newest first. */
export function useAllChecks(options: AllChecksOptions = {}): AllChecks {
  const enabled = options.enabled ?? true;
  const stateKey = filterKey(options.states);
  const api = usePublicAPI();
  const index = useProjectsIndex({ enabled });
  const projects = index.projects;
  const queries = useMemo(() => {
    const states = filterValues<AuditState>(stateKey);
    const state = states.length === 1 ? states[0] : undefined;
    return projects.map((project) => ({
      queryKey: queryKeys.crossProject.audits(project.projectId, state ?? null),
      queryFn: () =>
        listProjectAudits(api, {
          projectId: project.projectId,
          limit: CROSS_PROJECT_LIMITS.auditsPerProject,
          ...(state === undefined ? {} : { state }),
        }),
      enabled,
      ...POLLED,
    }));
  }, [api, enabled, projects, stateKey]);
  const combine = useCallback(
    (results: UseQueryResult<AuditPage>[]) =>
      combineChecks(projects, results, stateKey),
    [projects, stateKey],
  );
  const combined = useQueries({ queries, combine });
  return useMemo(
    () => ({
      checks: combined.checks,
      truncated: index.truncated || combined.truncated,
      partial: index.partial || combined.errors.length > 0,
      errors: combined.errors,
      error: index.error,
      isPending: index.isPending || combined.pending,
      refetch: index.refetch,
    }),
    [combined, index],
  );
}

/** One read about one check, keyed by the check's revision. */
interface CheckRead<T> {
  check: CrossProjectCheck;
  /** Key prefix; the check revision is appended as the last element. */
  key: QueryKey;
  load: () => Promise<T>;
}

interface SettledRead<T> {
  check: CrossProjectCheck;
  data: T;
}

interface CheckReads<R> {
  merged: R;
  errors: CrossProjectError[];
  pending: boolean;
}

/** Data of the newest cached revision of a per-check read. */
function latestRevision<T>(
  queryClient: QueryClient,
  prefix: QueryKey,
): T | undefined {
  let latest: T | undefined;
  let latestRevisionSeen = 0;
  for (const [key, data] of queryClient.getQueriesData<T>({
    queryKey: prefix,
  })) {
    const revision = key[prefix.length];
    if (
      data !== undefined &&
      key.length === prefix.length + 1 &&
      typeof revision === "number" &&
      revision > latestRevisionSeen
    ) {
      latest = data;
      latestRevisionSeen = revision;
    }
  }
  return latest;
}

function useCheckReads<T, R>(
  reads: readonly CheckRead<T>[],
  enabled: boolean,
  merge: (settled: SettledRead<T>[]) => R,
): CheckReads<R> {
  const queryClient = useQueryClient();
  const queries = useMemo(
    () =>
      reads.map((read) => ({
        queryKey: [...read.key, read.check.audit.revision],
        queryFn: read.load,
        enabled,
        // A revision of a check never changes; a change advances the key.
        staleTime: Number.POSITIVE_INFINITY,
        refetchInterval: (query: Query<T, Error, T, QueryKey>) =>
          query.state.status === "error" ? CROSS_PROJECT_LIMITS.pollMs : false,
        refetchIntervalInBackground: false,
        refetchOnWindowFocus: false,
        retry: false,
        // The previous revision stays listed while the new one loads.
        placeholderData: () => latestRevision<T>(queryClient, read.key),
      })),
    [enabled, queryClient, reads],
  );
  const combine = useCallback(
    (results: UseQueryResult<T>[]): CheckReads<R> => {
      const settled: SettledRead<T>[] = [];
      const errors: CrossProjectError[] = [];
      let pending = false;
      results.forEach((result, position) => {
        const read = reads[position];
        if (read === undefined) return;
        if (result.isPending) pending = true;
        if (result.isError)
          errors.push({
            projectId: read.check.project.projectId,
            auditId: read.check.audit.auditId,
            error: result.error,
          });
        if (result.data !== undefined)
          settled.push({ check: read.check, data: result.data });
      });
      return { merged: merge(settled), errors, pending };
    },
    [merge, reads],
  );
  return useQueries({ queries, combine });
}

function statusOf(
  checks: AllChecks,
  reads: CheckReads<unknown>,
  truncated: boolean,
): CrossProjectStatus {
  return {
    truncated: checks.truncated || truncated,
    partial: checks.partial || reads.errors.length > 0,
    errors:
      reads.errors.length === 0
        ? checks.errors
        : [...checks.errors, ...reads.errors],
    error: checks.error,
    isPending: checks.isPending || reads.pending,
    refetch: checks.refetch,
  };
}

export interface PossibleIssuesOptions {
  /**
   * Only possible issues in these states; omit (or pass none) for every
   * state. Filtered by the Server: one read per check and state (and
   * verdict), so each state gets its own first page.
   */
  states?: readonly AuditFindingState[];
  /**
   * Only possible issues with these analyst verdicts ("unreviewed" means no
   * effective decision); filtered by the Server like `states`.
   */
  verdicts?: readonly FindingVerdictFilter[];
  /** False keeps every read idle (and `isPending` true). */
  enabled?: boolean;
}

export interface AllPossibleIssues extends CrossProjectStatus {
  /** Possible issues newest first. */
  issues: CrossProjectIssue[];
  /**
   * The Server's count of matching possible issues in the listed checks,
   * including those beyond each check's first page; undefined until every
   * read has settled.
   */
  total: number | undefined;
}

interface MergedIssues {
  issues: CrossProjectIssue[];
  truncated: boolean;
  total: number;
}

function mergeIssues(settled: SettledRead<AuditFindingPage>[]): MergedIssues {
  const limit = CROSS_PROJECT_LIMITS.findingsPerAudit;
  const issues: CrossProjectIssue[] = [];
  const seen = new Set<string>();
  let truncated = false;
  let total = 0;
  for (const { check, data } of settled) {
    if (data.page.hasMore || data.items.length > limit) truncated = true;
    total +=
      Number.isSafeInteger(data.total) && data.total >= data.items.length
        ? data.total
        : data.items.length;
    for (const finding of data.items.slice(0, limit)) {
      const id = `${finding.auditId}/${finding.findingId}`;
      if (finding.auditId !== check.audit.auditId || seen.has(id)) continue;
      seen.add(id);
      issues.push({ project: check.project, audit: check.audit, finding });
    }
  }
  issues.sort(newerIssue);
  return { issues, truncated, total };
}

function usePossibleIssueReads(options: PossibleIssuesOptions) {
  const enabled = options.enabled ?? true;
  const stateKey = filterKey(options.states);
  const verdictKey = filterKey(options.verdicts);
  const api = usePublicAPI();
  const checks = useAllChecks({ states: ISSUE_CHECK_STATES, enabled });
  const reads = useMemo(() => {
    const states = orUnfiltered(filterValues<AuditFindingState>(stateKey));
    const verdicts = orUnfiltered(
      filterValues<FindingVerdictFilter>(verdictKey),
    );
    // The Server takes one state and one verdict per read.
    return checks.checks.flatMap((check) =>
      states.flatMap((state) =>
        verdicts.map((verdict): CheckRead<AuditFindingPage> => ({
          check,
          key: queryKeys.crossProject.findingsOf(
            check.audit.auditId,
            state ?? null,
            verdict ?? null,
          ),
          load: () =>
            listAuditFindings(api, check.audit.auditId, {
              ...(state === undefined ? {} : { state }),
              ...(verdict === undefined ? {} : { verdict }),
            }),
        })),
      ),
    );
  }, [api, checks.checks, stateKey, verdictKey]);
  const findings = useCheckReads(reads, enabled, mergeIssues);
  return { checks, findings };
}

/** Possible issues of every listed check that ran, newest first. */
export function useAllPossibleIssues(
  options: PossibleIssuesOptions = {},
): AllPossibleIssues {
  const { checks, findings } = usePossibleIssueReads(options);
  return useMemo(() => {
    const status = statusOf(checks, findings, findings.merged.truncated);
    return {
      ...status,
      issues: findings.merged.issues,
      total:
        status.isPending || status.error !== null
          ? undefined
          : findings.merged.total,
    };
  }, [checks, findings]);
}

export interface PendingDecisionsOptions {
  /**
   * Only these kinds of review request; omit (or pass none) for every kind.
   * Filtered on the client, so every kind shares one read per check.
   */
  kinds?: readonly AuditReviewKind[];
  /** False keeps every read idle (and `isPending` true). */
  enabled?: boolean;
}

export interface PendingDecisions extends CrossProjectStatus {
  /** Pending review requests of every kind, newest first. */
  decisions: CrossProjectDecision[];
}

interface MergedDecisions {
  decisions: CrossProjectDecision[];
  truncated: boolean;
}

function decisionMerger(kindKey: string) {
  const kinds = kindKey === "" ? undefined : new Set(kindKey.split(","));
  return (settled: SettledRead<AuditReviewPage>[]): MergedDecisions => {
    const decisions: CrossProjectDecision[] = [];
    const seen = new Set<string>();
    let truncated = false;
    for (const { check, data } of settled) {
      if (data.page.hasMore) truncated = true;
      for (const review of data.items) {
        const id = `${review.auditId}/${review.requestId}`;
        if (
          review.auditId !== check.audit.auditId ||
          review.state !== "pending" ||
          (kinds !== undefined && !kinds.has(review.kind)) ||
          seen.has(id)
        )
          continue;
        seen.add(id);
        decisions.push({ project: check.project, audit: check.audit, review });
      }
    }
    decisions.sort(newerDecision);
    return { decisions, truncated };
  };
}

/** Pending review requests of running and waiting checks, newest first. */
export function usePendingDecisions(
  options: PendingDecisionsOptions = {},
): PendingDecisions {
  const enabled = options.enabled ?? true;
  const kindKey = filterKey(options.kinds);
  const api = usePublicAPI();
  const checks = useAllChecks({ states: DECISION_CHECK_STATES, enabled });
  const reads = useMemo(
    () =>
      checks.checks.map((check): CheckRead<AuditReviewPage> => ({
        check,
        key: queryKeys.crossProject.pendingReviewsOf(check.audit.auditId),
        load: () =>
          listAuditReviews(api, check.audit.auditId, { state: "pending" }),
      })),
    [api, checks.checks],
  );
  const merge = useMemo(() => decisionMerger(kindKey), [kindKey]);
  const reviews = useCheckReads(reads, enabled, merge);
  return useMemo(
    () => ({
      ...statusOf(checks, reviews, reviews.merged.truncated),
      decisions: reviews.merged.decisions,
    }),
    [checks, reviews],
  );
}

export interface AllReportsOptions {
  /**
   * Report statuses to list. Omitted: the reports that exist (proposed and
   * ready). An empty list, like every empty filter here: every status.
   */
  statuses?: readonly AuditReportStatus[];
  /** False keeps every read idle (and `isPending` true). */
  enabled?: boolean;
}

export interface AllReports extends CrossProjectStatus {
  /** Reports in the order of their checks: newest check first. */
  reports: CrossProjectReport[];
}

async function loadReport(
  api: PublicAPI,
  auditId: string,
): Promise<AuditReport | null> {
  try {
    return await getAuditReport(api, auditId);
  } catch (error) {
    // A check deleted since its project's page was read has no report.
    if (error instanceof PublicAPIError && error.status === 404) return null;
    throw error;
  }
}

function reportMerger(statusKey: string) {
  const statuses = statusKey === "" ? undefined : new Set(statusKey.split(","));
  return (settled: SettledRead<AuditReport | null>[]) => ({
    reports: settled.flatMap(({ check, data }): CrossProjectReport[] =>
      data !== null && (statuses === undefined || statuses.has(data.status))
        ? [{ project: check.project, audit: check.audit, report: data }]
        : [],
    ),
  });
}

/**
 * Reports of checks that are finishing, waiting for a decision or ended.
 * A report that is still pending or unavailable (failed and stopped checks)
 * is not listed by default, and a missing report is not an error.
 */
export function useAllReports(options: AllReportsOptions = {}): AllReports {
  const enabled = options.enabled ?? true;
  const statusKey = filterKey(options.statuses ?? DEFAULT_REPORT_STATUSES);
  const api = usePublicAPI();
  const checks = useAllChecks({ states: REPORT_CHECK_STATES, enabled });
  const reads = useMemo(
    () =>
      checks.checks.map((check): CheckRead<AuditReport | null> => ({
        check,
        key: queryKeys.crossProject.reportOf(check.audit.auditId),
        load: () => loadReport(api, check.audit.auditId),
      })),
    [api, checks.checks],
  );
  const merge = useMemo(() => reportMerger(statusKey), [statusKey]);
  const reports = useCheckReads(reads, enabled, merge);
  return useMemo(
    () => ({
      ...statusOf(checks, reports, false),
      reports: reports.merged.reports,
    }),
    [checks, reports],
  );
}

export interface InboxSummary {
  /**
   * Everything that waits for the user: `possibleIssues` plus
   * `otherDecisions`. Undefined until both are known.
   */
  needsDecision: number | undefined;
  /**
   * Possible issues in state `proposed` (never decided, reopened, or assessed
   * again since the last decision), counted by the Server per check.
   */
  possibleIssues: number | undefined;
  /**
   * Pending active test approvals, requirement applicability and report
   * acceptance requests. Pending finding-triage requests are not counted
   * again: their possible issue is.
   */
  otherDecisions: number | undefined;
  /** Some read failed: the counts are lower bounds. */
  partial: boolean;
  /** Some bounded read was capped: the counts are lower bounds. */
  truncated: boolean;
}

/** Counts for the Inbox badge and header. */
export function useInboxSummary(): InboxSummary {
  const { checks, findings } = usePossibleIssueReads({ states: NEEDS_REVIEW });
  const decisions = usePendingDecisions({ kinds: OTHER_DECISION_KINDS });
  return useMemo(() => {
    const possibleIssues =
      checks.isPending || findings.pending || checks.error !== null
        ? undefined
        : findings.merged.total;
    const otherDecisions =
      decisions.isPending || decisions.error !== null
        ? undefined
        : decisions.decisions.length;
    return {
      needsDecision:
        possibleIssues === undefined || otherDecisions === undefined
          ? undefined
          : possibleIssues + otherDecisions,
      possibleIssues,
      otherDecisions,
      partial:
        checks.partial || findings.errors.length > 0 || decisions.partial,
      // A check's possible-issue count is exact beyond its first page.
      truncated: checks.truncated || decisions.truncated,
    };
  }, [checks, decisions, findings]);
}
