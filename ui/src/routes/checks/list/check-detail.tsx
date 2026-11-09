import { useQuery } from "@tanstack/react-query";
import { auditPollInterval, getAudit, type Audit } from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { PublicAPIError } from "../../../api/error";
import { queryKeys } from "../../../api/query-keys";
import { pollInterval } from "../../../app/polling";
import { QueryView } from "../../../app/query-view";
import { DetailPane, EmptyState } from "../../../ui";
import { ResearchAuditDetail } from "../../projects/audits/research";

export function CheckDetail({ auditId }: { auditId: string }) {
  const api = usePublicAPI();
  const audit = useQuery({
    queryKey: queryKeys.audits.detail(auditId),
    queryFn: () => getAudit(api, auditId),
    refetchInterval: pollInterval<Audit>({
      interval: 1000,
      settled: (data) => auditPollInterval([data]) === false,
    }),
    refetchOnReconnect: true,
    retry: (count, error) =>
      !(error instanceof PublicAPIError && error.status === 404) && count < 2,
  });
  if (audit.error instanceof PublicAPIError && audit.error.status === 404)
    return (
      <DetailPane>
        <EmptyState title="This check no longer exists">
          It was deleted, or it belongs to someone else.
        </EmptyState>
      </DetailPane>
    );
  return (
    <QueryView query={audit} loading={<p role="status">Loading check…</p>}>
      {(current) => <ResearchAuditDetail audit={current} />}
    </QueryView>
  );
}
