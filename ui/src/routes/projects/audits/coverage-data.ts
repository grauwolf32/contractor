import { useQuery } from "@tanstack/react-query";
import { collectAuditPages } from "../../../api/audit-collections";
import {
  auditNeedsPolling,
  listAuditCoverage,
  getAudit,
  type Audit,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { queryKeys } from "../../../api/query-keys";

import { useSearchParams } from "react-router";
import { PublicAPIError } from "../../../api/error";
import { useAuditProjectionRefresh } from "./projection-refresh";

export function useAuditCoverage(audit: Audit) {
  const api = usePublicAPI();
  const [params] = useSearchParams();
  const expected = params.get("auditRevision");
  const queryKey = [
    ...queryKeys.audits.allCoverage(audit.auditId),
    audit.currentRoundId ?? null,
    expected,
  ];
  const query = useQuery({
    queryKey,
    queryFn: async () => {
      const before = await getAudit(api, audit.auditId);
      const conflict = () =>
        new PublicAPIError({
          status: 409,
          code: "revision_conflict",
          message: "Audit changed; refresh the coverage context.",
        });
      if (expected !== null && String(before.revision) !== expected)
        throw conflict();
      const rows = await collectAuditPages((cursor) =>
        listAuditCoverage(api, audit.auditId, {
          ...(cursor === undefined ? {} : { cursor }),
          ...(before.currentRoundId === undefined
            ? {}
            : { round: before.currentRoundId }),
        }),
      );
      const after = await getAudit(api, audit.auditId);
      if (before.revision !== after.revision) throw conflict();
      return rows;
    },
    refetchInterval: auditNeedsPolling(audit.state) ? 5_000 : false,
    refetchOnReconnect: true,
  });
  useAuditProjectionRefresh(audit, queryKey);
  return query;
}
