/**
 * The Inbox reads: the cross-project lists (src/api/cross-project.ts) with
 * the Inbox check states, so they match the rail badge and share its reads;
 * the owner's Runs (failed, succeeded and active, first page each); the
 * workspace counters of listed checks; the reports of checks that finished
 * in the last 7 days; the work items waiting for a decision; and queue
 * admission.
 *
 * Per-check reads (workspace counters, reports, work items) are pinned to the
 * check's revision like the cross-project reads: the polled check pages
 * decide when they change. Run lists and queue admission are polled on the
 * same 20 s interval. Failures never throw: they are counted for the page's
 * notice.
 */
import {
  useQueries,
  useQuery,
  useQueryClient,
  type Query,
  type QueryClient,
  type QueryKey,
  type UseQueryResult,
} from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useState } from "react";

import {
  collectAuditPages,
  type AuditCollection,
} from "../../api/audit-collections";
import {
  getAuditReport,
  getAuditWorkspace,
  type AuditFindingState,
  type AuditItem,
  type AuditReport,
  type AuditReviewRequest,
  type AuditWorkspace,
} from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import {
  CROSS_PROJECT_LIMITS,
  INBOX_CHECK_STATES,
  useAllChecks,
  useAllPossibleIssues,
  useInboxSummary,
  usePendingDecisions,
  useProjectsIndex,
  type AllChecks,
  type AllPossibleIssues,
  type CrossProjectCheck,
  type CrossProjectReport,
  type InboxSummary,
  type PendingDecisions,
} from "../../api/cross-project";
import { PublicAPIError } from "../../api/error";
import { listAwaitingReviewItems } from "../../api/inbox";
import { queryKeys } from "../../api/query-keys";
import { getOwnerQueueControl, type OwnerQueueControl } from "../../api/queue";
import {
  getRun,
  listRuns,
  type RunPage,
  type RunStatus,
  type RunSummary,
} from "../../api/runs";
import { getWorkflow, type WorkflowResource } from "../../api/workflows";
import {
  parseWorkflowIdentity,
  requireWorkflowOutputs,
  type WorkflowIdentity,
  type WorkflowOutputSlot,
} from "../runs/output-model";
import {
  buildInbox,
  COUNTED_CHECK_STATES,
  finishedRecently,
  LISTED_CHECK_STATES,
  recentRuns,
  waitingRuns,
  type InboxModel,
  type InboxSectionId,
  type Shown,
} from "./model";

const POLL_MS = CROSS_PROJECT_LIMITS.pollMs;

const PROPOSED: readonly AuditFindingState[] = ["proposed"];

/** Decisions beyond triaging a possible issue: the Inbox lists all three. */
const DECISION_KINDS: readonly AuditReviewRequest["kind"][] = [
  "active-check-approval",
  "requirement-applicability",
  "report-acceptance",
];

// Run lists and queue admission: polled while the page is visible.
const POLLED = {
  staleTime: POLL_MS,
  refetchInterval: POLL_MS,
  refetchIntervalInBackground: false,
  refetchOnWindowFocus: false,
  retry: false,
} as const;

// A terminal Run's status no longer changes the Inbox's view of it.
const SETTLED = {
  staleTime: 60_000,
  refetchOnWindowFocus: false,
  retry: false,
} as const;

/**
 * A read pinned to a check revision: once it has data it is never
 * refetched; a failed read retries on the poll interval.
 */
function pinned<T>() {
  return {
    staleTime: (query: Query<T, Error, T, QueryKey>) =>
      query.state.data === undefined ? 0 : ("static" as const),
    refetchInterval: (query: Query<T, Error, T, QueryKey>) =>
      query.state.status === "error" ? POLL_MS : false,
    refetchIntervalInBackground: false,
    refetchOnWindowFocus: false,
    retry: false,
  };
}

/**
 * The newest cached data under a per-check key prefix (the key without its
 * revision), so a check's previous counts stay shown while its new revision
 * loads.
 */
export function newestCached<T>(
  queryClient: QueryClient,
  prefix: QueryKey,
): T | undefined {
  let newest: { revision: number; data: T } | undefined;
  for (const [key, data] of queryClient.getQueriesData<T>({
    queryKey: prefix,
  })) {
    const revision = key.at(-1);
    if (data === undefined || typeof revision !== "number") continue;
    if (newest === undefined || revision > newest.revision)
      newest = { revision, data };
  }
  return newest?.data;
}

function uniqueBy<T>(items: readonly T[], key: (item: T) => string): T[] {
  const seen = new Set<string>();
  return items.filter((item) => {
    const value = key(item);
    if (seen.has(value)) return false;
    seen.add(value);
    return true;
  });
}

/** The current time, advanced every minute, for the 7-day windows. */
function useNow(intervalMs = 60_000): number {
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), intervalMs);
    return () => window.clearInterval(timer);
  }, [intervalMs]);
  return now;
}

export interface PerCheck<T> {
  /** Settled data (or the previous revision's) by check ID. */
  byCheck: Map<string, T>;
  failed: number;
  pending: boolean;
}

function usePerCheckReads<T>(
  checks: readonly CrossProjectCheck[],
  keyOf: (auditId: string, revision: number) => QueryKey,
  load: (auditId: string) => Promise<T>,
): PerCheck<T> {
  const queryClient = useQueryClient();
  const queries = useMemo(
    () =>
      checks.map((check) => {
        const { auditId, revision } = check.audit;
        const key = keyOf(auditId, revision);
        const placeholder = newestCached<T>(queryClient, key.slice(0, -1));
        return {
          queryKey: key,
          queryFn: () => load(auditId),
          ...pinned<T>(),
          ...(placeholder === undefined
            ? {}
            : { placeholderData: placeholder }),
        };
      }),
    [checks, keyOf, load, queryClient],
  );
  const combine = useCallback(
    (results: UseQueryResult<T>[]): PerCheck<T> => {
      const byCheck = new Map<string, T>();
      let failed = 0;
      let pending = false;
      results.forEach((result, position) => {
        const check = checks[position];
        if (check === undefined) return;
        if (result.data !== undefined)
          byCheck.set(check.audit.auditId, result.data);
        if (result.isError) failed += 1;
        if (result.isPending) pending = true;
      });
      return { byCheck, failed, pending };
    },
    [checks],
  );
  return useQueries({ queries, combine });
}

/** Declared outputs of a Workflow version; null when they cannot be read. */
export type WorkflowOutputs = Record<string, WorkflowOutputSlot> | null;

export function workflowKey(identity: WorkflowIdentity): string {
  return `${identity.name}@${identity.version}`;
}

export interface RunStatuses {
  byRun: Map<string, RunStatus>;
  /** Runs whose status could not be read. */
  failed: Set<string>;
  pending: boolean;
}

export interface RunList {
  query: UseQueryResult<RunPage>;
  recent: Shown<RunSummary>;
}

/** Ready reports of the checks that finished in the last 7 days. */
export interface ReadyReports {
  /** Newest check first. */
  reports: CrossProjectReport[];
  /** Some of those reports have not been read yet. */
  pending: boolean;
  /** Reports that could not be read. */
  failed: number;
}

export interface InboxData {
  now: number;
  model: InboxModel;
  summary: InboxSummary;
  issues: AllPossibleIssues;
  decisions: PendingDecisions;
  checks: AllChecks;
  reports: ReadyReports;
  failedRuns: RunList;
  succeededRuns: RunList;
  activeRuns: UseQueryResult<RunPage>;
  /** Active Runs on the first page, "50+" when it is full; undefined while loading. */
  activeRunCount: string | undefined;
  /** Project names by ID, for Runs that belong to a project. */
  projectNames: Map<string, string>;
  runStatuses: RunStatuses;
  /** Declared outputs by "name@version"; absent while loading. */
  workflowOutputs: Map<string, WorkflowOutputs>;
  workspaces: PerCheck<AuditWorkspace>;
  /**
   * Work items that wait for a decision, by check ID, for checks with
   * pending approval or applicability requests.
   */
  items: PerCheck<AuditCollection<AuditItem>>;
  queue: UseQueryResult<OwnerQueueControl>;
  /** A section's reads have not settled yet. */
  pending: Record<InboxSectionId, boolean>;
  /** A read behind a section failed, so an empty section may not be empty. */
  unavailable: Record<InboxSectionId, boolean>;
  /** Some read failed; the lists may be incomplete. */
  incomplete: boolean;
  /** Retries every failed read and refreshes the polled ones. */
  retry: () => Promise<void>;
}

const NO_RUNS: RunSummary[] = [];

export function useInboxData(): InboxData {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const now = useNow();

  // The same reads as the rail badge (useInboxSummary).
  const summary = useInboxSummary();
  const issues = useAllPossibleIssues({
    checkStates: INBOX_CHECK_STATES,
    states: PROPOSED,
  });
  const decisions = usePendingDecisions({
    checkStates: INBOX_CHECK_STATES,
    kinds: DECISION_KINDS,
  });
  // Several states: the per-project check pages every list above shares.
  const checks = useAllChecks({ states: LISTED_CHECK_STATES });
  const index = useProjectsIndex();
  const projectNames = useMemo(
    () =>
      new Map(
        index.projects.map((project) => [project.projectId, project.name]),
      ),
    [index.projects],
  );

  const counted = useMemo(
    () =>
      checks.checks.filter((check) =>
        COUNTED_CHECK_STATES.has(check.audit.state),
      ),
    [checks.checks],
  );
  const loadWorkspace = useCallback(
    (auditId: string) => getAuditWorkspace(api, auditId),
    [api],
  );
  const workspaces = usePerCheckReads(
    counted,
    queryKeys.inbox.workspace,
    loadWorkspace,
  );

  // Ready lists reports of checks that finished in the last 7 days only, so
  // only those are read: failed and stopped checks rarely have one.
  const finished = useMemo(
    () => checks.checks.filter((check) => finishedRecently(check, now)),
    [checks.checks, now],
  );
  const loadReport = useCallback(
    async (auditId: string): Promise<AuditReport | null> => {
      try {
        return await getAuditReport(api, auditId);
      } catch (error) {
        // A check deleted since its project's page was read has no report.
        if (error instanceof PublicAPIError && error.status === 404)
          return null;
        throw error;
      }
    },
    [api],
  );
  const reportReads = usePerCheckReads(
    finished,
    queryKeys.inbox.report,
    loadReport,
  );
  const readyReports = useMemo(
    () =>
      finished.flatMap((check): CrossProjectReport[] => {
        const report = reportReads.byCheck.get(check.audit.auditId);
        // A proposed report waits for acceptance: a decision, not a result.
        return report === undefined ||
          report === null ||
          report.status !== "ready"
          ? []
          : [{ project: check.project, audit: check.audit, report }];
      }),
    [finished, reportReads.byCheck],
  );

  const itemChecks = useMemo(
    () =>
      uniqueBy(
        decisions.decisions
          .filter(
            (decision) => decision.review.subjectKind === "audit-item-action",
          )
          .map(({ project, audit }) => ({ project, audit })),
        (check) => check.audit.auditId,
      ),
    [decisions.decisions],
  );
  // Only the items under a pending request: one small page, however many
  // items (an ASVS check has hundreds) the check has.
  const loadItems = useCallback(
    (auditId: string) =>
      collectAuditPages((cursor) =>
        listAwaitingReviewItems(api, auditId, cursor),
      ),
    [api],
  );
  const items = usePerCheckReads(
    itemChecks,
    queryKeys.inbox.awaitingItems,
    loadItems,
  );

  const failed = useQuery({
    queryKey: queryKeys.runs.list("failed", undefined),
    queryFn: () => listRuns(api, { state: "failed" }),
    ...POLLED,
  });
  const succeeded = useQuery({
    queryKey: queryKeys.runs.list("succeeded", undefined),
    queryFn: () => listRuns(api, { state: "succeeded" }),
    ...POLLED,
  });
  const active = useQuery({
    queryKey: queryKeys.runs.list(undefined, undefined, [], "active"),
    queryFn: () => listRuns(api, { lifecycle: "active" }),
    ...POLLED,
  });
  const queue = useQuery({
    queryKey: queryKeys.queue.control,
    queryFn: () => getOwnerQueueControl(api),
    ...POLLED,
  });

  const failedRecent = useMemo(
    () => recentRuns(failed.data?.items ?? NO_RUNS, now),
    [failed.data, now],
  );
  const succeededRecent = useMemo(
    () => recentRuns(succeeded.data?.items ?? NO_RUNS, now),
    [succeeded.data, now],
  );
  const waiting = useMemo(
    () => waitingRuns(active.data?.items ?? NO_RUNS),
    [active.data],
  );

  // Status of each listed Run: the cause of a failure, whether a waiting Run
  // needs a model retry, and the outputs of a finished one.
  const statusRuns = useMemo(
    () =>
      uniqueBy(
        [...waiting, ...failedRecent.shown, ...succeededRecent.shown],
        (run) => run.runId,
      ),
    [failedRecent.shown, succeededRecent.shown, waiting],
  );
  const statusQueries = useMemo(
    () =>
      statusRuns.map((run) => ({
        queryKey: queryKeys.runs.detail(run.runId),
        queryFn: () => getRun(api, run.runId),
        ...(run.state === "waiting" ? POLLED : SETTLED),
      })),
    [api, statusRuns],
  );
  const combineStatuses = useCallback(
    (results: UseQueryResult<RunStatus>[]): RunStatuses => {
      const byRun = new Map<string, RunStatus>();
      const failedRuns = new Set<string>();
      let pending = false;
      results.forEach((result, position) => {
        const run = statusRuns[position];
        if (run === undefined) return;
        if (result.data !== undefined) byRun.set(run.runId, result.data);
        if (result.isError) failedRuns.add(run.runId);
        if (result.isPending) pending = true;
      });
      return { byRun, failed: failedRuns, pending };
    },
    [statusRuns],
  );
  const runStatuses = useQueries({
    queries: statusQueries,
    combine: combineStatuses,
  });

  // The declared outputs of finished Runs' Workflow versions, to find the
  // primary result (UUS:102-103: never an arbitrary file instead).
  const identities = useMemo(
    () =>
      uniqueBy(
        succeededRecent.shown.flatMap((run) => {
          const identity = parseWorkflowIdentity(run.workflow);
          return identity === undefined ? [] : [identity];
        }),
        workflowKey,
      ),
    [succeededRecent.shown],
  );
  const contractQueries = useMemo(
    () =>
      identities.map((identity) => ({
        queryKey: queryKeys.workflows.detail(identity.name, identity.version),
        queryFn: ({ signal }: { signal: AbortSignal }) =>
          getWorkflow(api, identity.name, identity.version, signal),
        staleTime: Number.POSITIVE_INFINITY,
        refetchOnWindowFocus: false,
        retry: false,
      })),
    [api, identities],
  );
  const combineContracts = useCallback(
    (results: UseQueryResult<WorkflowResource>[]) => {
      const outputs = new Map<string, WorkflowOutputs>();
      results.forEach((result, position) => {
        const identity = identities[position];
        if (identity === undefined) return;
        if (result.data !== undefined) {
          try {
            outputs.set(
              workflowKey(identity),
              requireWorkflowOutputs(result.data, identity),
            );
          } catch {
            outputs.set(workflowKey(identity), null);
          }
        } else if (result.isError) outputs.set(workflowKey(identity), null);
      });
      return outputs;
    },
    [identities],
  );
  const workflowOutputs = useQueries({
    queries: contractQueries,
    combine: combineContracts,
  });

  const model = useMemo(
    () =>
      buildInbox({
        now,
        issues: issues.issues,
        decisions: decisions.decisions,
        checks: checks.checks,
        reports: readyReports,
        workspaces: workspaces.byCheck,
        failedRuns: failedRecent,
        succeededRuns: succeededRecent,
        waitingRuns: waiting,
        runStatuses: runStatuses.byRun,
      }),
    [
      checks.checks,
      decisions.decisions,
      failedRecent,
      issues.issues,
      now,
      readyReports,
      runStatuses.byRun,
      succeededRecent,
      waiting,
      workspaces.byCheck,
    ],
  );

  const indexFailed = checks.error !== null;
  const pending: Record<InboxSectionId, boolean> = {
    decide: issues.isPending || decisions.isPending,
    unblock:
      checks.isPending ||
      failed.isPending ||
      active.isPending ||
      workspaces.pending ||
      (waiting.length > 0 && runStatuses.pending),
    ready: checks.isPending || reportReads.pending || succeeded.isPending,
    running: checks.isPending || active.isPending,
  };
  const unavailable: Record<InboxSectionId, boolean> = {
    decide:
      issues.partial ||
      issues.error !== null ||
      decisions.partial ||
      decisions.error !== null,
    unblock:
      checks.partial ||
      indexFailed ||
      failed.isError ||
      active.isError ||
      workspaces.failed > 0,
    ready:
      checks.partial ||
      indexFailed ||
      reportReads.failed > 0 ||
      succeeded.isError,
    running: checks.partial || indexFailed || active.isError,
  };
  const incomplete =
    summary.partial ||
    Object.values(unavailable).some(Boolean) ||
    items.failed > 0 ||
    runStatuses.failed.size > 0 ||
    queue.isError;

  const refetchChecks = checks.refetch;
  const retry = useCallback(async () => {
    await Promise.all([
      // The project index, check pages and every failed per-check read.
      refetchChecks(),
      queryClient.refetchQueries({
        queryKey: queryKeys.inbox.all,
        type: "active",
        predicate: (query) => query.state.status === "error",
      }),
      queryClient.refetchQueries({
        queryKey: queryKeys.runs.all,
        type: "active",
        predicate: (query) =>
          query.state.status === "error" || query.queryKey[1] === "list",
      }),
      queryClient.refetchQueries({
        queryKey: queryKeys.queue.control,
        type: "active",
      }),
    ]);
  }, [queryClient, refetchChecks]);

  return {
    now,
    model,
    summary,
    issues,
    decisions,
    checks,
    reports: {
      reports: readyReports,
      pending: reportReads.pending,
      failed: reportReads.failed,
    },
    failedRuns: { query: failed, recent: failedRecent },
    succeededRuns: { query: succeeded, recent: succeededRecent },
    activeRuns: active,
    activeRunCount:
      active.data === undefined
        ? undefined
        : `${active.data.items.length}${active.data.page.hasMore ? "+" : ""}`,
    projectNames,
    runStatuses,
    workflowOutputs,
    workspaces,
    items,
    queue,
    pending,
    unavailable,
    incomplete,
    retry,
  };
}
