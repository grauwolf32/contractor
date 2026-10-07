/**
 * Owner-wide V3B lists. Checks, findings and pending decisions use Server
 * keysets, following every page sequentially. The shared project index also
 * follows all pages. Heads poll while visible; report payloads remain pinned
 * to each check revision. Failed continuations keep their settled pages and
 * mark the result partial; no collection is silently capped at 50 records.
 */
import {
  hashKey,
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
  type AuditFindingSeverity,
  type AuditFindingState,
  type AuditReport,
  type AuditReviewRequest,
  type AuditState,
  getAuditReport,
  listOwnerFindings,
  listOwnerReviews,
  listOwnerAudits,
} from "./audits";
import type { PublicAPI } from "./client";
import { usePublicAPI } from "./context";
import { PublicAPIError } from "./error";
import type { components, operations } from "./generated/public";
import { listProjects, type Project } from "./projects";
import { queryKeys } from "./query-keys";
import { collectOwnerPages, type OwnerCollection } from "./owner-pages";

/** Page size is enforced by the Server; collections have no fixed cap. */
export const CROSS_PROJECT_LIMITS = { pageSize: 200, pollMs: 20_000 } as const;

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

/**
 * A read that failed, by `scope`: a refresh of the project index (its last
 * page stays listed), one project's checks, or one read about one check.
 * `projectId` and `auditId` are set where the scope has them.
 */
export type CrossProjectError =
  | {
      scope: "index";
      projectId?: undefined;
      auditId?: undefined;
      error: Error;
    }
  | { scope: "list"; projectId?: undefined; auditId?: undefined; error: Error }
  | { scope: "project"; projectId: string; auditId?: undefined; error: Error }
  | { scope: "check"; projectId: string; auditId: string; error: Error };

/** Completeness and loading state shared by every cross-project list. */
export interface CrossProjectStatus {
  /** A bounded read had more records than it returned. */
  truncated: boolean;
  /** Some read failed; everything else is still listed. */
  partial: boolean;
  /** The failed reads behind `partial`. */
  errors: CrossProjectError[];
  /** A required list failed before any page could be listed. */
  error: Error | null;
  /** Some read has not settled yet; what has settled is already listed. */
  isPending: boolean;
  /**
   * Refetches the project index and check pages now, and every per-check
   * read that failed. Per-check reads with data are pinned to their check's
   * revision and are not refetched.
   */
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

/**
 * The checks the Inbox reads: running ones and those waiting for the user
 * (docs/design/ui/v3b-implementation.md, "API gaps"). useInboxSummary counts
 * only these. The Inbox page passes them as `checkStates` so its lists match
 * the badge and share its reads; possible issues of other checks stay on the
 * Issues page and the check's own page.
 */
export const INBOX_CHECK_STATES: readonly AuditState[] = [
  "active",
  "waiting_review",
];

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

/**
 * The check revision a per-check read's key ends with; undefined for the
 * polled heads (project index, check pages), whose keys end otherwise.
 */
function checkRevisionOf(key: QueryKey): number | undefined {
  const last = key.at(-1);
  return typeof last === "number" ? last : undefined;
}

function useCrossProjectRefetch(): () => Promise<void> {
  const queryClient = useQueryClient();
  return useCallback(async () => {
    await queryClient.refetchQueries({
      queryKey: queryKeys.crossProject.all,
      type: "active",
      // The heads, and per-check reads that failed: one with data is pinned
      // to its revision, and a loading one is already on its way.
      predicate: (query) =>
        checkRevisionOf(query.queryKey) === undefined ||
        query.state.status === "error",
    });
  }, [queryClient]);
}

/**
 * Refreshes the cross-project lists after a decision or any other mutation.
 * The polled heads (project index, check pages) are invalidated and the
 * mounted ones refetch at once. Per-check reads follow from there: the
 * mutation advanced the changed check's revision, so its reads move to a new
 * key, while the reads of unchanged checks keep their data.
 */
export async function invalidateCrossProject(
  queryClient: QueryClient,
): Promise<void> {
  await queryClient.invalidateQueries({
    queryKey: queryKeys.crossProject.all,
    predicate: (query) => checkRevisionOf(query.queryKey) === undefined,
  });
}

/**
 * The project index. `partial`: the latest refresh failed and the last page
 * read is still listed; `errors` then holds that failure (scope "index").
 * `error`: the index failed and there is no earlier page to list.
 */
export interface ProjectsIndex extends CrossProjectStatus {
  /** Every ordinary Project, newest first, without those in deletion. */
  projects: Project[];
}

const NO_PROJECTS: Project[] = [];
const NO_ERRORS: CrossProjectError[] = [];

/** Every ordinary Project, polled while visible. */
export function useProjectsIndex(
  options: { enabled?: boolean } = {},
): ProjectsIndex {
  const api = usePublicAPI();
  const refetch = useCrossProjectRefetch();
  const { data, error, isPending } = useQuery({
    queryKey: queryKeys.crossProject.projects,
    queryFn: ({ signal }) =>
      collectOwnerPages((cursor) =>
        listProjects(
          api,
          {
            kind: "project",
            ...(cursor === undefined ? {} : { cursor }),
          },
          signal,
        ),
      ),
    enabled: options.enabled ?? true,
    ...POLLED,
  });
  return useMemo(() => {
    const failure = error ?? data?.error ?? null;
    return {
      projects:
        data?.items.filter(
          (project) =>
            project.kind === "project" && project.lifecycle !== "deleting",
        ) ?? NO_PROJECTS,
      truncated: false,
      partial: failure !== null,
      errors:
        failure === null
          ? NO_ERRORS
          : [{ scope: "index" as const, error: failure }],
      error: data === undefined ? error : null,
      isPending,
      refetch,
    };
  }, [data, error, isPending, refetch]);
}

export interface AllChecksOptions {
  /** Checks in these states; omitted or empty means every state. Server-filtered. */
  states?: readonly AuditState[];
  /** False keeps every read idle (and `isPending` true). */
  enabled?: boolean;
}

export interface AllChecks extends CrossProjectStatus {
  /** Checks newest first by last update, then by creation. */
  checks: CrossProjectCheck[];
}

/** Checks across all active Projects; state filters run on the Server. */
export function useAllChecks(options: AllChecksOptions = {}): AllChecks {
  const enabled = options.enabled ?? true;
  const stateKey = filterKey(options.states);
  const api = usePublicAPI();
  const index = useProjectsIndex({ enabled });
  const query = useQuery({
    queryKey: queryKeys.crossProject.ownerAudits(stateKey),
    queryFn: ({ signal }) =>
      collectOwnerPages((cursor) =>
        listOwnerAudits(
          api,
          {
            ...(stateKey === "" ? {} : { state: stateKey }),
            ...(cursor === undefined ? {} : { cursor }),
          },
          signal,
        ),
      ),
    enabled,
    ...POLLED,
  });
  return useMemo(() => {
    const projects = new Map(
      index.projects.map((project) => [project.projectId, project]),
    );
    const seen = new Set<string>();
    const checks: CrossProjectCheck[] = [];
    for (const audit of query.data?.items ?? []) {
      const project = projects.get(audit.projectId);
      if (project === undefined || seen.has(audit.auditId)) continue;
      seen.add(audit.auditId);
      checks.push({ project, audit });
    }
    checks.sort(newerCheck);
    const failure = query.error ?? query.data?.error ?? null;
    const errors: CrossProjectError[] =
      failure === null
        ? index.errors
        : [...index.errors, { scope: "list", error: failure }];
    return {
      checks,
      truncated: false,
      partial: index.partial || failure !== null,
      errors,
      error: index.error ?? (query.data === undefined ? query.error : null),
      isPending: index.isPending || query.isPending,
      refetch: index.refetch,
    };
  }, [index, query.data, query.error, query.isPending]);
}

const NO_CHECKS: CrossProjectCheck[] = [];

/**
 * Listed checks in the `wanted` states that can have a per-check read
 * (`applicable`); every applicable state when `wanted` is omitted or empty.
 * When no wanted state is applicable, no check is listed; the shared check
 * pages are still read, so the status matches every other list.
 */
function useScopedChecks(
  wanted: readonly AuditState[] | undefined,
  applicable: readonly AuditState[],
  enabled: boolean,
): AllChecks {
  const wantedKey = filterKey(wanted);
  const scope = useMemo(() => {
    const states = filterValues<AuditState>(wantedKey);
    return states.length === 0
      ? applicable
      : applicable.filter((state) => states.includes(state));
  }, [applicable, wantedKey]);
  const checks = useAllChecks({
    states: scope.length === 0 ? applicable : scope,
    enabled,
  });
  return useMemo(
    () => (scope.length === 0 ? { ...checks, checks: NO_CHECKS } : checks),
    [checks, scope],
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

interface CachedRevision {
  revision: number;
  data: unknown;
}

/**
 * The newest cached revision that has data, of every per-check read, by the
 * hash of its key prefix (the key without the revision). One pass over the
 * cache.
 */
function cachedRevisions(queryClient: QueryClient) {
  const newest = new Map<string, CachedRevision>();
  for (const query of queryClient
    .getQueryCache()
    .findAll({ queryKey: queryKeys.crossProject.all })) {
    const revision = checkRevisionOf(query.queryKey);
    const data: unknown = query.state.data;
    if (revision === undefined || data === undefined) continue;
    const prefix = hashKey(query.queryKey.slice(0, -1));
    const known = newest.get(prefix);
    if (known === undefined || revision > known.revision)
      newest.set(prefix, { revision, data });
  }
  return newest;
}

function useCheckReads<T, R>(
  reads: readonly CheckRead<T>[],
  enabled: boolean,
  merge: (settled: SettledRead<T>[]) => R,
): CheckReads<R> {
  const queryClient = useQueryClient();
  const queries = useMemo(() => {
    // Looked up once per change of the reads, not on every render: the
    // previous revision of a check stays listed while its new one loads.
    const previous =
      reads.length === 0 ? undefined : cachedRevisions(queryClient);
    return reads.map((read) => {
      const placeholder = previous?.get(hashKey(read.key));
      return {
        queryKey: [...read.key, read.check.audit.revision],
        queryFn: read.load,
        enabled,
        // A revision of a check never changes, so a read that has data is
        // never refetched: not on mount, invalidation or refresh. A read
        // without data (failed, or not loaded yet) can still be fetched.
        staleTime: (query: Query<T, Error, T, QueryKey>) =>
          query.state.data === undefined ? 0 : ("static" as const),
        refetchInterval: (query: Query<T, Error, T, QueryKey>) =>
          query.state.status === "error" ? CROSS_PROJECT_LIMITS.pollMs : false,
        refetchIntervalInBackground: false,
        refetchOnWindowFocus: false,
        retry: false,
        ...(placeholder === undefined
          ? {}
          : { placeholderData: placeholder.data as T }),
      };
    });
  }, [enabled, queryClient, reads]);
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
            scope: "check",
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
   * Only checks in these states; omit (or pass none) for every check that
   * can hold possible issues (every state but draft and deleting). The Inbox
   * passes INBOX_CHECK_STATES. Filtered like AllChecksOptions.states.
   */
  checkStates?: readonly AuditState[];
  /**
   * Only possible issues in these states; omit (or pass none) for every
   * state. Filtered by the Server: one read per check and state (and
   * verdict and severity), so each filter combination follows every global page.
   */
  states?: readonly AuditFindingState[];
  /**
   * Only possible issues with these analyst verdicts ("unreviewed" means no
   * effective decision); filtered by the Server like `states`.
   */
  verdicts?: readonly FindingVerdictFilter[];
  /**
   * Only possible issues the analyst rated with these severities (the
   * Worker's suggestion is never a rating); filtered by the Server like
   * `states`. An unreviewed possible issue has no analyst severity, so with
   * this filter the verdict "unreviewed" matches nothing and is not read.
   */
  severities?: readonly AuditFindingSeverity[];
  /** False keeps every read idle (and `isPending` true). */
  enabled?: boolean;
}

export interface AllPossibleIssues extends CrossProjectStatus {
  /** Matching possible issues from every owner-wide page, newest first. */
  issues: CrossProjectIssue[];
  /** Compatibility field for consumers; owner-wide lists have no per-check cap. */
  truncatedAuditIds: string[];
  /** Number of listed matches; a partial collection is a lower bound. */
  total: number | undefined;
}

interface MergedIssues {
  issues: CrossProjectIssue[];
  truncatedAuditIds: string[];
  total: number;
}

interface FindingFilter {
  state: AuditFindingState | undefined;
  verdict: FindingVerdictFilter | undefined;
  severity: AuditFindingSeverity | undefined;
}

/**
 * One filter per read: the Server takes one state, verdict and severity per
 * request, and rejects a severity with the verdict "unreviewed" (an
 * unreviewed possible issue has no analyst severity, so the pair matches
 * nothing and is left out).
 */
function findingFilters(
  stateKey: string,
  verdictKey: string,
  severityKey: string,
): FindingFilter[] {
  const states = orUnfiltered(filterValues<AuditFindingState>(stateKey));
  const verdicts = orUnfiltered(filterValues<FindingVerdictFilter>(verdictKey));
  const severities = orUnfiltered(
    filterValues<AuditFindingSeverity>(severityKey),
  );
  return states.flatMap((state) =>
    verdicts.flatMap((verdict) =>
      severities
        .filter(
          (severity) => severity === undefined || verdict !== "unreviewed",
        )
        .map((severity) => ({ state, verdict, severity })),
    ),
  );
}

function usePossibleIssueReads(options: PossibleIssuesOptions) {
  const enabled = options.enabled ?? true;
  const stateKey = filterKey(options.states);
  const verdictKey = filterKey(options.verdicts);
  const severityKey = filterKey(options.severities);
  const api = usePublicAPI();
  const checks = useScopedChecks(
    options.checkStates,
    ISSUE_CHECK_STATES,
    enabled,
  );
  const auditStateKey = filterKey(
    options.checkStates?.length ? options.checkStates : ISSUE_CHECK_STATES,
  );
  const filters = useMemo(
    () => findingFilters(stateKey, verdictKey, severityKey),
    [stateKey, verdictKey, severityKey],
  );
  const queries = useMemo(
    () =>
      filters.map(({ state, verdict, severity }) => ({
        queryKey: queryKeys.crossProject.ownerFindings(
          auditStateKey,
          state ?? null,
          verdict ?? null,
          severity ?? null,
        ),
        queryFn: ({ signal }: { signal: AbortSignal }) =>
          collectOwnerPages((cursor) =>
            listOwnerFindings(
              api,
              {
                auditState: auditStateKey,
                ...(state === undefined ? {} : { state }),
                ...(verdict === undefined ? {} : { verdict }),
                ...(severity === undefined ? {} : { severity }),
                ...(cursor === undefined ? {} : { cursor }),
              },
              signal,
            ),
          ),
        enabled,
        ...POLLED,
      })),
    [api, auditStateKey, enabled, filters],
  );
  const combine = useCallback(
    (
      results: UseQueryResult<OwnerCollection<AuditFinding>>[],
    ): CheckReads<MergedIssues> => {
      const byAudit = new Map(
        checks.checks.map((check) => [check.audit.auditId, check]),
      );
      const issues: CrossProjectIssue[] = [];
      const seen = new Set<string>();
      const errors: CrossProjectError[] = [];
      for (const result of results) {
        const failure = result.error ?? result.data?.error ?? null;
        if (failure !== null) errors.push({ scope: "list", error: failure });
        for (const finding of result.data?.items ?? []) {
          const check = byAudit.get(finding.auditId);
          const key = `${finding.auditId}/${finding.findingId}`;
          if (check === undefined || seen.has(key)) continue;
          seen.add(key);
          issues.push({ ...check, finding });
        }
      }
      issues.sort(newerIssue);
      return {
        merged: { issues, truncatedAuditIds: [], total: issues.length },
        errors,
        pending: results.some((result) => result.isPending),
      };
    },
    [checks.checks],
  );
  const findings = useQueries({ queries, combine });
  return { checks, findings };
}

/** Possible issues of every listed check that ran, newest first. */
export function useAllPossibleIssues(
  options: PossibleIssuesOptions = {},
): AllPossibleIssues {
  const { checks, findings } = usePossibleIssueReads(options);
  return useMemo(() => {
    const { issues, truncatedAuditIds, total } = findings.merged;
    const status = statusOf(checks, findings, truncatedAuditIds.length > 0);
    return {
      ...status,
      issues,
      truncatedAuditIds,
      total: status.isPending || status.error !== null ? undefined : total,
    };
  }, [checks, findings]);
}

export interface PendingDecisionsOptions {
  /**
   * Only checks in these states; omit (or pass none) for every check that
   * can ask for a decision (running, waiting, paused or finishing). The
   * Inbox passes INBOX_CHECK_STATES. Filtered like AllChecksOptions.states.
   */
  checkStates?: readonly AuditState[];
  /**
   * Only these kinds of review request; omit (or pass none) for every kind.
   * Filtered on the client, so every kind shares one read per check.
   */
  kinds?: readonly AuditReviewKind[];
  /** False keeps every read idle (and `isPending` true). */
  enabled?: boolean;
}

export interface PendingDecisions extends CrossProjectStatus {
  /** All matching pending requests, newest first. */
  decisions: CrossProjectDecision[];
  /** Compatibility field for consumers; owner-wide lists have no per-check cap. */
  truncatedAuditIds: string[];
}

/** All pending requests across owned Projects; kinds share one read. */
export function usePendingDecisions(
  options: PendingDecisionsOptions = {},
): PendingDecisions {
  const enabled = options.enabled ?? true;
  const kindKey = filterKey(options.kinds);
  const auditStateKey = filterKey(
    options.checkStates?.length ? options.checkStates : DECISION_CHECK_STATES,
  );
  const api = usePublicAPI();
  const checks = useScopedChecks(
    options.checkStates,
    DECISION_CHECK_STATES,
    enabled,
  );
  const query = useQuery({
    queryKey: queryKeys.crossProject.ownerReviews(auditStateKey),
    queryFn: ({ signal }) =>
      collectOwnerPages((cursor) =>
        listOwnerReviews(
          api,
          {
            state: "pending",
            auditState: auditStateKey,
            ...(cursor === undefined ? {} : { cursor }),
          },
          signal,
        ),
      ),
    enabled,
    ...POLLED,
  });
  return useMemo(() => {
    const byAudit = new Map(
      checks.checks.map((check) => [check.audit.auditId, check]),
    );
    const kinds = kindKey === "" ? undefined : new Set(kindKey.split(","));
    const seen = new Set<string>();
    const decisions: CrossProjectDecision[] = [];
    for (const review of query.data?.items ?? []) {
      const check = byAudit.get(review.auditId);
      const key = `${review.auditId}/${review.requestId}`;
      if (
        check === undefined ||
        review.state !== "pending" ||
        seen.has(key) ||
        (kinds !== undefined && !kinds.has(review.kind))
      )
        continue;
      seen.add(key);
      decisions.push({ ...check, review });
    }
    decisions.sort(newerDecision);
    const failure = query.error ?? query.data?.error ?? null;
    const reads = {
      merged: decisions,
      errors:
        failure === null
          ? NO_ERRORS
          : [{ scope: "list" as const, error: failure }],
      pending: query.isPending,
    };
    return {
      ...statusOf(checks, reads, false),
      decisions,
      truncatedAuditIds: [],
    };
  }, [checks, kindKey, query.data, query.error, query.isPending]);
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
   * again since the last decision) of Inbox checks (INBOX_CHECK_STATES),
   * counted by the Server per check.
   */
  possibleIssues: number | undefined;
  /**
   * Pending active test approvals, requirement applicability and report
   * acceptance requests of Inbox checks. Pending finding-triage requests are
   * not counted again: their possible issue is.
   */
  otherDecisions: number | undefined;
  /** Some read failed: the counts are lower bounds. */
  partial: boolean;
  /** Some bounded read was capped: the counts are lower bounds. */
  truncated: boolean;
}

/**
 * Counts for the Inbox badge and header. Like the Inbox, they cover only
 * running checks and checks waiting for the user (INBOX_CHECK_STATES).
 */
export function useInboxSummary(): InboxSummary {
  const { checks, findings } = usePossibleIssueReads({
    checkStates: INBOX_CHECK_STATES,
    states: NEEDS_REVIEW,
  });
  const decisions = usePendingDecisions({
    checkStates: INBOX_CHECK_STATES,
    kinds: OTHER_DECISION_KINDS,
  });
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
