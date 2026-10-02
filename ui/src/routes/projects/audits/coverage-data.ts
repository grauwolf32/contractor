import {
  collectAuditPages,
  type AuditCollection,
} from "../../../api/audit-collections";
import {
  auditPollInterval,
  listAuditCoverage,
  getAudit,
  type Audit,
  type AuditCoverageRow,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { queryKeys } from "../../../api/query-keys";

import { useSearchParams } from "react-router";
import { useQueryClient } from "@tanstack/react-query";
import { PublicAPIError } from "../../../api/error";
import { useAuditCollection } from "./collections";
import { useAuditProjectionRefresh } from "./projection-refresh";

interface CoverageBatch extends AuditCollection<AuditCoverageRow> {
  /** Audit revision every page of this batch was read under. */
  revision: number;
}

export function useAuditCoverage(audit: Audit) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [params] = useSearchParams();
  const expected = params.get("auditRevision");
  const queryKey = [
    ...queryKeys.audits.allCoverage(audit.auditId),
    // A revision pin identifies its own snapshot even if a new round starts.
    expected === null ? (audit.currentRoundId ?? null) : null,
    expected,
  ];
  const query = useAuditCollection<CoverageBatch>({
    queryKey,
    // A loaded pin is a snapshot. Keep it mounted even when another Audit
    // action invalidates the broader detail-query subtree.
    enabled:
      expected === null || queryClient.getQueryData(queryKey) === undefined,
    loadBatch: async (cursor, previous) => {
      const conflict = () =>
        new PublicAPIError({
          status: 409,
          code: "revision_conflict",
          message: "Audit changed; refresh the coverage context.",
        });
      // An unpinned first batch can straddle one revision bump. Retry that
      // whole batch once; pinned and continuation reads remain strict.
      const canRetry = expected === null && previous === undefined;
      for (let attempt = 0; attempt < (canRetry ? 2 : 1); attempt += 1) {
        const before = await getAudit(api, audit.auditId);
        if (expected !== null && String(before.revision) !== expected)
          throw conflict();
        if (previous !== undefined && previous.revision !== before.revision)
          throw conflict();
        const batch = await collectAuditPages(
          (pageCursor) =>
            listAuditCoverage(api, audit.auditId, {
              ...(pageCursor === undefined ? {} : { cursor: pageCursor }),
              ...(before.currentRoundId === undefined
                ? {}
                : { round: before.currentRoundId }),
            }),
          cursor === undefined ? {} : { cursor },
        );
        const after = await getAudit(api, audit.auditId);
        if (before.revision === after.revision)
          return { ...batch, revision: after.revision };
      }
      throw conflict();
    },
    refetchInterval:
      expected === null ? auditPollInterval([audit], 5_000) : false,
  });
  useAuditProjectionRefresh(audit, queryKey, expected === null);
  return query;
}
