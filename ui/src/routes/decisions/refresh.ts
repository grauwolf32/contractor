import type { QueryClient } from "@tanstack/react-query";

import { invalidateCrossProject } from "../../api/cross-project";
import { queryKeys } from "../../api/query-keys";

/**
 * Refetches everything a decision can change once the Server has answered,
 * whether it recorded the decision or refused it (no optimistic state):
 *
 * - the check and every read under its detail key: findings, reviews,
 *   workspace counts, report, items and provenance;
 * - project check lists, whose rows carry the check's state and revision;
 * - the cross-project lists of Inbox, Checks, Issues and Reports.
 *
 * The decision components know the check but not its project, so every
 * project's check list is marked stale; only mounted lists refetch.
 */
export async function refreshAfterDecision(
  queryClient: QueryClient,
  auditId: string,
): Promise<void> {
  await Promise.all([
    queryClient.invalidateQueries({
      queryKey: queryKeys.audits.detail(auditId),
    }),
    queryClient.invalidateQueries({
      queryKey: queryKeys.projects.all,
      predicate: (query) =>
        query.queryKey[1] === "detail" && query.queryKey[3] === "audits",
    }),
    invalidateCrossProject(queryClient),
  ]);
}
