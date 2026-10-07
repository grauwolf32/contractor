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

import type { ArtifactMetadata } from "../../api/artifacts";
import {
  type Audit,
  type AuditFinding,
  type AuditFindingPage,
  type AuditState,
  listAuditFindings,
  listProjectAudits,
} from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import { CROSS_PROJECT_LIMITS } from "../../api/cross-project";
import { listProjectArtifacts } from "../../api/project-artifacts";
import { listProjectRuns } from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import { materialKind, MATERIAL_KIND_ORDER } from "./material-kinds";

/** How often the overview refreshes checks and Runs while it is open. */
const OVERVIEW_POLL_MS = 10_000;

/**
 * The project's newest checks: the first page, as the cross-project lists
 * read it (same key and request), so the overview shares the read the rail's
 * Inbox badge already polls. Decisions and check lifecycle actions (start,
 * pause, continue, stop, delete) refresh it through invalidateCrossProject,
 * so an overview that opens after one reads it again.
 */
export function useProjectChecks(projectId: string) {
  const api = usePublicAPI();
  return useQuery({
    queryKey: queryKeys.crossProject.audits(projectId, null),
    queryFn: () =>
      listProjectAudits(api, {
        projectId,
        limit: CROSS_PROJECT_LIMITS.auditsPerProject,
      }),
    staleTime: OVERVIEW_POLL_MS,
    refetchInterval: OVERVIEW_POLL_MS,
    refetchOnWindowFocus: false,
  });
}

/** Checks waiting for the user's decision: three, newest first. */
export function useWaitingChecks(projectId: string) {
  const api = usePublicAPI();
  return useQuery({
    queryKey: [...queryKeys.projects.audits.all(projectId), "waiting-review"],
    queryFn: () =>
      listProjectAudits(api, { projectId, state: "waiting_review", limit: 3 }),
    refetchInterval: OVERVIEW_POLL_MS,
  });
}

// Draft and deleting checks hold no possible issues (api/cross-project.ts).
const NO_ISSUES: readonly AuditState[] = ["draft", "deleting"];

export interface CheckIssues {
  audit: Audit;
  /** The Server's count of possible issues that need review. */
  total: number;
  /** The first page of them, oldest first. */
  items: readonly AuditFinding[];
}

export interface PossibleIssuesToReview {
  /** Per check that has been read, in the order of `checks`. */
  checks: CheckIssues[];
  /** Possible issues needing review in every check read. */
  total: number;
  /** Every check's read has settled. */
  settled: boolean;
  /** Some check's read failed: `total` is a lower bound. */
  partial: boolean;
  /** Checks with nothing to count yet, from this revision or an earlier one. */
  pendingAuditIds: string[];
  /** Checks whose read failed. */
  failedAuditIds: string[];
}

/**
 * The cache key of a check's possible issues that need review, without the
 * check revision: the key the cross-project reads (Issues, Inbox) use too.
 */
function toReviewKey(auditId: string) {
  return queryKeys.crossProject.findingsOf(auditId, "proposed", null, null);
}

/**
 * The newest cached page of possible issues to review of each given check
 * (by audit ID), from any earlier revision of the check. One pass over the
 * cross-project cache.
 */
function newestCachedPages(
  queryClient: QueryClient,
  audits: readonly Audit[],
): Map<string, AuditFindingPage> {
  const wanted = new Map(
    audits.map((audit) => [hashKey(toReviewKey(audit.auditId)), audit.auditId]),
  );
  const newest = new Map<string, { revision: number; data: unknown }>();
  for (const query of queryClient
    .getQueryCache()
    .findAll({ queryKey: queryKeys.crossProject.all })) {
    const revision = query.queryKey.at(-1);
    const data: unknown = query.state.data;
    if (typeof revision !== "number" || data === undefined) continue;
    const auditId = wanted.get(hashKey(query.queryKey.slice(0, -1)));
    if (auditId === undefined) continue;
    const known = newest.get(auditId);
    if (known === undefined || revision > known.revision)
      newest.set(auditId, { revision, data });
  }
  return new Map(
    [...newest].map(([auditId, { data }]) => [
      auditId,
      data as AuditFindingPage,
    ]),
  );
}

/**
 * Possible issues that need review (state `proposed`) in the given checks,
 * counted by the Server per check. Each read is the cross-project per-check
 * read (same key, keyed by the check's revision), so the projects list, the
 * overview, the Issues page and the Inbox share it: it never refetches while
 * the check is unchanged, and a check that changes moves to a new key. While
 * that new revision loads, the check's previous page stays counted, so the
 * count does not drop out on every poll of a running check.
 */
export function usePossibleIssuesToReview(
  checks: readonly Audit[],
): PossibleIssuesToReview {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const readable = useMemo(
    () => checks.filter((audit) => !NO_ISSUES.includes(audit.state)),
    [checks],
  );
  const queries = useMemo(() => {
    // Looked up when the checks change, which is when a revision changes.
    const previous =
      readable.length === 0
        ? undefined
        : newestCachedPages(queryClient, readable);
    return readable.map((audit) => {
      const placeholder = previous?.get(audit.auditId);
      return {
        queryKey: [...toReviewKey(audit.auditId), audit.revision],
        queryFn: () =>
          listAuditFindings(api, audit.auditId, { state: "proposed" }),
        staleTime: (
          query: Query<AuditFindingPage, Error, AuditFindingPage, QueryKey>,
        ) => (query.state.data === undefined ? 0 : ("static" as const)),
        refetchInterval: (
          query: Query<AuditFindingPage, Error, AuditFindingPage, QueryKey>,
        ) =>
          query.state.status === "error" ? CROSS_PROJECT_LIMITS.pollMs : false,
        refetchOnWindowFocus: false,
        retry: false,
        ...(placeholder === undefined ? {} : { placeholderData: placeholder }),
      };
    });
  }, [api, queryClient, readable]);
  const combine = useCallback(
    (results: UseQueryResult<AuditFindingPage>[]): PossibleIssuesToReview => {
      const read: CheckIssues[] = [];
      const pendingAuditIds: string[] = [];
      const failedAuditIds: string[] = [];
      let total = 0;
      results.forEach((result, position) => {
        const audit = readable[position];
        if (audit === undefined) return;
        if (result.isPending) pendingAuditIds.push(audit.auditId);
        if (result.isError) failedAuditIds.push(audit.auditId);
        const page = result.data;
        if (page === undefined) return;
        const count =
          Number.isSafeInteger(page.total) && page.total >= page.items.length
            ? page.total
            : page.items.length;
        total += count;
        read.push({
          audit,
          total: count,
          items: page.items.filter(
            (finding) => finding.auditId === audit.auditId,
          ),
        });
      });
      return {
        checks: read,
        total,
        settled: pendingAuditIds.length === 0,
        partial: failedAuditIds.length > 0,
        pendingAuditIds,
        failedAuditIds,
      };
    },
    [readable],
  );
  return useQueries({ queries, combine });
}

/** Up to five recent Runs of any state. */
export function useRecentRuns(projectId: string) {
  const api = usePublicAPI();
  const options = { limit: 5 };
  return useQuery({
    queryKey: queryKeys.projects.runView(projectId, options),
    queryFn: () => listProjectRuns(api, { projectId, ...options }),
    refetchInterval: OVERVIEW_POLL_MS,
  });
}

/** The three most recent successful Runs, for their results. */
export function useSuccessfulRuns(projectId: string) {
  const api = usePublicAPI();
  const options = { limit: 3, state: "succeeded" as const };
  return useQuery({
    queryKey: queryKeys.projects.runView(projectId, options),
    queryFn: () => listProjectRuns(api, { projectId, ...options }),
    refetchInterval: 15_000,
  });
}

export interface MaterialsSample {
  /** Current materials read, inputs first (source code, API spec, …). */
  items: ArtifactMetadata[];
  /** The project has more materials than were read. */
  more: boolean;
  isPending: boolean;
  /** Every read failed: nothing is known about the materials. */
  error: Error | null;
  /** Some read failed while others answered. */
  partial: boolean;
  isFetching: boolean;
  refetch: () => void;
}

/**
 * A bounded sample of the project's materials (S06: the overview never loads
 * the complete inventory): five current materials, and up to three source
 * archives and three API specs, which the check types need most.
 */
export function useMaterialsSample(projectId: string): MaterialsSample {
  const api = usePublicAPI();
  const base = queryKeys.projects.artifacts.all(projectId);
  const recent = useQuery({
    queryKey: [...base, "overview"],
    queryFn: () => listProjectArtifacts(api, { projectId, limit: 5 }),
  });
  const sources = useQuery({
    queryKey: [...base, "overview-sources"],
    queryFn: () =>
      listProjectArtifacts(api, { projectId, namespace: "sources", limit: 3 }),
  });
  const specs = useQuery({
    queryKey: [...base, "overview-openapi"],
    queryFn: () =>
      listProjectArtifacts(api, { projectId, namespace: "openapi", limit: 3 }),
  });
  const reads = [recent, sources, specs];
  const items = useMemo(() => {
    const unique = new Map<string, ArtifactMetadata>();
    for (const page of [sources.data, specs.data, recent.data]) {
      for (const item of page?.items ?? []) {
        if (!item.current) continue;
        unique.set(`${item.artifact.namespace}/${item.artifact.name}`, item);
      }
    }
    return [...unique.values()].sort(
      (left, right) =>
        MATERIAL_KIND_ORDER.indexOf(materialKind(left)) -
          MATERIAL_KIND_ORDER.indexOf(materialKind(right)) ||
        `${left.artifact.namespace}/${left.artifact.name}`.localeCompare(
          `${right.artifact.namespace}/${right.artifact.name}`,
        ),
    );
  }, [recent.data, sources.data, specs.data]);
  const failed = reads.filter((read) => read.error !== null);
  return {
    items,
    more: reads.some((read) => read.data?.page.hasMore === true),
    isPending: reads.some((read) => read.isPending),
    error: failed.length === reads.length ? (failed[0]?.error ?? null) : null,
    partial: failed.length > 0 && failed.length < reads.length,
    isFetching: reads.some((read) => read.isFetching),
    refetch: () => {
      for (const read of reads) void read.refetch();
    },
  };
}
