import type { useAuditQueue } from "./queue-state";
import { PublicAPIError } from "../../../api/error";
import { ErrorNotice, formatTimestamp } from "../../artifacts/common";
import type { AuditFindingPage, AuditReviewPage } from "../../../api/audits";

export function AuditQueueError({
  error,
  onRefresh,
}: {
  error: Error;
  onRefresh: () => void;
}) {
  return (
    <div className="notice notice-error">
      <ErrorNotice error={error} />
      <p>
        {error instanceof PublicAPIError && error.status === 409
          ? "This Audit changed. Refresh the context to continue; no decision has been replayed."
          : "The requested subject or page is unavailable. It may have been removed or become inaccessible."}
      </p>
      <button type="button" className="secondary-button" onClick={onRefresh}>
        Refresh context
      </button>
    </div>
  );
}

export function AuditQueuePage({
  page,
  currentRevision,
  queue,
  onRefresh,
}: {
  page: AuditFindingPage | AuditReviewPage;
  currentRevision: number;
  queue: ReturnType<typeof useAuditQueue>;
  onRefresh: () => void;
}) {
  return (
    <div className="audit-queue-pagination">
      <p className="muted-copy">
        Showing {page.items.length} of {page.total ?? "unavailable"} matching
        records in this Audit
        {page.asOf === undefined
          ? " · Count snapshot unavailable"
          : ` · Revision ${page.auditRevision} · As of ${formatTimestamp(page.asOf)}`}
      </p>
      {page.auditRevision !== undefined &&
      page.auditRevision !== currentRevision ? (
        <p className="notice">
          This page is from an earlier Audit revision. Refresh the context
          before continuing.
        </p>
      ) : null}
      <div className="form-actions">
        <button type="button" className="secondary-button" onClick={onRefresh}>
          Refresh context
        </button>
        {queue.params.has("cursor") ? (
          <button
            type="button"
            className="secondary-button"
            onClick={queue.refresh}
          >
            First page
          </button>
        ) : null}
        <button
          type="button"
          className="secondary-button"
          disabled={!page.page.hasMore || page.page.nextCursor === undefined}
          onClick={() => queue.next(page)}
        >
          Next page
        </button>
      </div>
    </div>
  );
}
