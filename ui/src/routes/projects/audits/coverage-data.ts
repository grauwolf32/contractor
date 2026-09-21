import {
  collectAuditPages,
  type AuditCollection,
} from "../../../api/audit-collections";
import {
  auditNeedsPolling,
  listAuditCoverage,
  getAudit,
  type Audit,
  type AuditCoverageRow,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { queryKeys } from "../../../api/query-keys";

import { useSearchParams } from "react-router";
import { PublicAPIError } from "../../../api/error";
import { useAuditCollection } from "./collections";
import { useAuditProjectionRefresh } from "./projection-refresh";

interface CoverageBatch extends AuditCollection<AuditCoverageRow> {
  /** Audit revision every page of this batch was read under. */
  revision: number;
}

export function useAuditCoverage(audit: Audit) {
  const api = usePublicAPI();
  const [params] = useSearchParams();
  const expected = params.get("auditRevision");
  const queryKey = [
    ...queryKeys.audits.allCoverage(audit.auditId),
    audit.currentRoundId ?? null,
    expected,
  ];
  const query = useAuditCollection<CoverageBatch>({
    queryKey,
    loadBatch: async (cursor, previous) => {
      const before = await getAudit(api, audit.auditId);
      const conflict = () =>
        new PublicAPIError({
          status: 409,
          code: "revision_conflict",
          message: "Audit changed; refresh the coverage context.",
        });
      if (expected !== null && String(before.revision) !== expected)
        throw conflict();
      // A continuation must read under the revision of the batches before it.
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
      if (before.revision !== after.revision) throw conflict();
      return { ...batch, revision: after.revision };
    },
    refetchInterval: auditNeedsPolling(audit.state) ? 5_000 : false,
  });
  useAuditProjectionRefresh(audit, queryKey);
  return query;
}
