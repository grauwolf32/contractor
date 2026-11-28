import { useQuery, type UseQueryResult } from "@tanstack/react-query";

import {
  AUDIT_ID_PATTERN,
  auditPollInterval,
  getAudit,
  getAuditReport,
  type Audit,
  type AuditReport,
} from "../../api/audits";
import type { PublicAPI } from "../../api/client";
import { queryKeys } from "../../api/query-keys";
import { pollInterval } from "../../app/polling";
import { useAuditProjectionRefresh } from "../projects/audits/projection-refresh";

/** The check can no longer change on its own, so its reads stop polling. */
function settled(audit: Audit): boolean {
  return auditPollInterval([audit]) === false;
}

/**
 * The selected check of /reports/:auditId, read with the check page's key
 * and polled like it while the check can change; failed reads back off.
 * Idle without a valid ID.
 */
export function useReportCheck(
  api: PublicAPI,
  auditId: string | undefined,
): UseQueryResult<Audit> {
  return useQuery({
    queryKey: queryKeys.audits.detail(auditId ?? ""),
    queryFn: () => getAudit(api, auditId ?? ""),
    enabled: auditId !== undefined && AUDIT_ID_PATTERN.test(auditId),
    refetchInterval: pollInterval<Audit>({ interval: 1_000, settled }),
    refetchOnReconnect: true,
  });
}

/**
 * The check's report, read with the same key as the check page and the
 * decision components. It is polled while the check can still change (failed
 * reads back off), and read once more when polling stops.
 */
export function useAuditReport(
  audit: Audit,
  api: PublicAPI,
): UseQueryResult<AuditReport> {
  const queryKey = queryKeys.audits.report(audit.auditId);
  const report = useQuery({
    queryKey,
    queryFn: ({ signal }) => getAuditReport(api, audit.auditId, signal),
    refetchInterval: pollInterval<AuditReport>({
      interval: 1_000,
      settled: () => settled(audit),
    }),
    refetchOnReconnect: true,
  });
  useAuditProjectionRefresh(audit, queryKey);
  return report;
}
