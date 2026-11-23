/**
 * Reads of the check page and the Checks list. Keys sit under the check's
 * detail key (queryKeys.audits.detail), so a decision or lifecycle change
 * that invalidates the check refreshes them too.
 */
import { useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useMemo } from "react";

import {
  auditPollInterval,
  getAuditWorkspace,
  type Audit,
  type AuditWorkspace,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { queryKeys } from "../../../api/query-keys";
import { pollInterval } from "../../../app/polling";
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
    [...queryKeys.audits.events(auditId), "check-activity"] as const,
};

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
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      getAuditWorkspace(api, audit.auditId, signal),
    enabled,
    placeholderData: () => previousWorkspace(queryClient, audit.auditId),
    refetchInterval: pollInterval<AuditWorkspace>({
      interval: 5000,
      settled: () => auditPollInterval([audit]) === false,
    }),
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
        queryFn: ({ signal }: { signal: AbortSignal }) =>
          getAuditWorkspace(api, audit.auditId, signal),
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
