/**
 * Reads of the check page and the Checks list. Keys sit under the check's
 * detail key (queryKeys.audits.detail), so a decision or lifecycle change
 * that invalidates the check refreshes them too.
 */
import { useQuery, useQueries, useQueryClient } from "@tanstack/react-query";
import { useCallback, useMemo } from "react";

import {
  collectAuditPages,
  type AuditCollection,
} from "../../../api/audit-collections";
import {
  auditPollInterval,
  getAuditReport,
  getAuditWorkspace,
  listAuditFindings,
  listAuditReviews,
  type Audit,
  type AuditFinding,
  type AuditState,
  type AuditWorkspace,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { queryKeys } from "../../../api/query-keys";
import { useAuditCollection } from "./collections";
import { useAuditProjectionRefresh } from "./projection-refresh";

export const checkKeys = {
  /**
   * The check's current workspace counts (its page and its summary on
   * Checks), polled while the check can change. Also the prefix of every
   * workspace read of the check.
   */
  workspace: (auditId: string) =>
    [...queryKeys.audits.detail(auditId), "workspace"] as const,
  /**
   * Workspace counts of a listed check at one revision. A revision never
   * changes, so a read is kept until the list sees the check move on.
   */
  workspaceAt: (auditId: string, revision: number) =>
    [...queryKeys.audits.detail(auditId), "workspace", revision] as const,
  /** Every possible issue of the check, for the per-item counts. */
  findings: (auditId: string) =>
    [...queryKeys.audits.allFindings(auditId), "check-items"] as const,
  /** Pending review requests shown on the check's activity view. */
  waiting: (auditId: string) =>
    [...queryKeys.audits.allReviews(auditId), "check-activity"] as const,
};

/** Checks in these states can have a report. */
export const REPORT_STATES: readonly AuditState[] = [
  "waiting_review",
  "finalizing",
  "completed",
  "cancelled",
  "failed",
];

/** The newest cached workspace of a check, shown while a newer one loads. */
function previousWorkspace(
  queryClient: ReturnType<typeof useQueryClient>,
  auditId: string,
): AuditWorkspace | undefined {
  let newest: AuditWorkspace | undefined;
  for (const [, data] of queryClient.getQueriesData<AuditWorkspace>({
    queryKey: checkKeys.workspace(auditId),
  })) {
    if (
      data !== undefined &&
      (newest === undefined || data.auditRevision > newest.auditRevision)
    )
      newest = data;
  }
  return newest;
}

/**
 * The check's counts, read again every 5 s while it can change (S19 bounded
 * polling: the check page reads the check every second, and an active check
 * moves to a new revision about as often), right after a lifecycle change
 * (controls.tsx invalidates them) and once more when it stops changing.
 */
export function useCheckWorkspace(audit: Audit, enabled = true) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const queryKey = checkKeys.workspace(audit.auditId);
  const query = useQuery({
    queryKey,
    queryFn: () => getAuditWorkspace(api, audit.auditId),
    enabled,
    placeholderData: () => previousWorkspace(queryClient, audit.auditId),
    refetchInterval: auditPollInterval([audit], 5_000),
    refetchOnReconnect: true,
    refetchOnWindowFocus: false,
    retry: false,
  });
  useAuditProjectionRefresh(audit, queryKey, enabled);
  return query;
}

/**
 * Workspace counts of several listed checks (list rows), one read per
 * revision the list sees: the list itself is polled, so the reads are as
 * bounded as it is. Returns the counts by check ID.
 */
export function useCheckWorkspaces(
  audits: readonly Audit[],
): ReadonlyMap<string, AuditWorkspace> {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const queries = useMemo(
    () =>
      audits.map((audit) => ({
        queryKey: checkKeys.workspaceAt(audit.auditId, audit.revision),
        queryFn: () => getAuditWorkspace(api, audit.auditId),
        placeholderData: () => previousWorkspace(queryClient, audit.auditId),
        staleTime: Number.POSITIVE_INFINITY,
        refetchOnWindowFocus: false,
        retry: false,
      })),
    [api, audits, queryClient],
  );
  const combine = useCallback(
    (results: { data: AuditWorkspace | undefined }[]) => {
      const counts = new Map<string, AuditWorkspace>();
      results.forEach((result, position) => {
        const audit = audits[position];
        if (audit !== undefined && result.data !== undefined)
          counts.set(audit.auditId, result.data);
      });
      return counts;
    },
    [audits],
  );
  return useQueries({ queries, combine });
}

/**
 * Whether the check's issues and possible issues are read: a draft has none
 * and a check being deleted no longer serves them.
 */
export function findingsReadable(audit: Pick<Audit, "state">): boolean {
  return audit.state !== "draft" && audit.state !== "deleting";
}

/**
 * Every issue and possible issue of the check, in batches of the page cap,
 * for the counts and links on its items. Polled while the check can change.
 */
export function useCheckFindings(audit: Audit) {
  const api = usePublicAPI();
  const queryKey = checkKeys.findings(audit.auditId);
  const enabled = findingsReadable(audit);
  const query = useAuditCollection<AuditCollection<AuditFinding>>({
    queryKey,
    loadBatch: (cursor) =>
      collectAuditPages(
        (pageCursor) =>
          listAuditFindings(
            api,
            audit.auditId,
            pageCursor === undefined ? {} : { cursor: pageCursor },
          ),
        cursor === undefined ? {} : { cursor },
      ),
    enabled,
    refetchInterval: auditPollInterval([audit], 5_000),
  });
  useAuditProjectionRefresh(audit, queryKey, enabled);
  return query;
}

/**
 * The check's pending review requests (first page, oldest first), polled
 * while the check can change and read once more when it stops.
 */
export function useWaitingDecisions(audit: Audit, enabled = true) {
  const api = usePublicAPI();
  const queryKey = checkKeys.waiting(audit.auditId);
  const active = enabled && audit.state !== "draft";
  const query = useQuery({
    queryKey,
    queryFn: () => listAuditReviews(api, audit.auditId, { state: "pending" }),
    enabled: active,
    refetchInterval: auditPollInterval([audit], 5_000),
    refetchOnReconnect: true,
  });
  useAuditProjectionRefresh(audit, queryKey, active);
  return query;
}

/** The check's report, for checks that can have one. */
export function useCheckReport(audit: Audit, enabled = true) {
  const api = usePublicAPI();
  const queryKey = queryKeys.audits.report(audit.auditId);
  const active = enabled && REPORT_STATES.includes(audit.state);
  const query = useQuery({
    queryKey,
    queryFn: () => getAuditReport(api, audit.auditId),
    enabled: active,
    refetchInterval: auditPollInterval([audit], 5_000),
    refetchOnReconnect: true,
  });
  useAuditProjectionRefresh(audit, queryKey, active);
  return query;
}
