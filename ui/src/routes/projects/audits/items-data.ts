import { collectAuditPages } from "../../../api/audit-collections";
import {
  auditNeedsPolling,
  listAuditItems,
  type Audit,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { queryKeys } from "../../../api/query-keys";
import { useAuditCollection } from "./collections";
import { useAuditProjectionRefresh } from "./projection-refresh";

/**
 * The Audit item (check) collection: attempts, produced artifacts and the
 * exact identity of every check. Bounded by the page cap and polled only
 * while the Audit is active; `enabled` lets a view read it lazily.
 */
export function useAuditItems(
  audit: Audit,
  api: ReturnType<typeof usePublicAPI>,
  enabled: boolean,
) {
  const queryKey = queryKeys.audits.allItems(audit.auditId);
  const query = useAuditCollection({
    queryKey,
    loadBatch: (cursor) =>
      collectAuditPages(
        (pageCursor) =>
          listAuditItems(
            api,
            audit.auditId,
            pageCursor === undefined ? {} : { cursor: pageCursor },
          ),
        cursor === undefined ? {} : { cursor },
      ),
    enabled,
    refetchInterval: auditNeedsPolling(audit.state) ? 1_000 : false,
  });
  useAuditProjectionRefresh(audit, queryKey, enabled);
  return query;
}
