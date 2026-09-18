import { useQuery } from "@tanstack/react-query";
import { collectAuditPages } from "../../../api/audit-collections";
import {
  auditNeedsPolling,
  listAuditCoverage,
  type Audit,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { queryKeys } from "../../../api/query-keys";

export function useAuditCoverage(audit: Audit) {
  const api = usePublicAPI();
  return useQuery({
    queryKey: [
      ...queryKeys.audits.allCoverage(audit.auditId),
      audit.currentRoundId ?? null,
    ],
    queryFn: () =>
      collectAuditPages((cursor) =>
        listAuditCoverage(api, audit.auditId, {
          ...(cursor === undefined ? {} : { cursor }),
          ...(audit.currentRoundId === undefined
            ? {}
            : { round: audit.currentRoundId }),
        }),
      ),
    refetchInterval: auditNeedsPolling(audit.state) ? 5_000 : false,
    refetchOnReconnect: true,
  });
}
