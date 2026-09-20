import { useQueryClient, type QueryKey } from "@tanstack/react-query";
import { useEffect, useRef } from "react";

import { auditNeedsPolling, type Audit } from "../../../api/audits";

export function useAuditProjectionRefresh(
  audit: Audit,
  queryKey: QueryKey,
  enabled = true,
) {
  const queryClient = useQueryClient();
  const previous = useRef(audit);
  useEffect(() => {
    const before = previous.current;
    previous.current = audit;
    if (
      !enabled ||
      before.auditId !== audit.auditId ||
      (before.revision === audit.revision && before.state === audit.state) ||
      auditNeedsPolling(audit.state)
    ) {
      return;
    }
    // Finish with a fresh read when polling stops, including later retained
    // evidence changes. Cancel first so even an unfinished initial request
    // cannot be reused as the final snapshot. Keep the exact key (and URL pin)
    // captured here if navigation selects another query before this settles.
    const exactQuery = { queryKey, exact: true };
    void queryClient
      .cancelQueries(exactQuery)
      .then(() => queryClient.invalidateQueries(exactQuery));
  }, [audit, enabled, queryClient, queryKey]);
}
