import type { useAuditQueue } from "./queue-state";
import { PublicAPIError } from "../../../api/error";
import { ErrorNotice } from "../../../app/error-notice";
import { formatTimestamp } from "../../../app/format";
import type { AuditFindingPage, AuditReviewPage } from "../../../api/audits";
import { Button, TechnicalDetails } from "../../../ui";

import "../../issues/issues.css";

/** A page of a check's queue could not be read; nothing was replayed. */
export function AuditQueueError({
  error,
  onRefresh,
}: {
  error: Error;
  onRefresh: () => void;
}) {
  return (
    <div className="issues-queue-error">
      <ErrorNotice
        error={error}
        context="Could not load this view"
        onRetry={onRefresh}
        retryLabel="Refresh context"
      />
      <p className="issues-quiet">
        {error instanceof PublicAPIError && error.status === 409
          ? "This check changed. Refresh the context to continue; no decision has been replayed."
          : error instanceof PublicAPIError && error.status === 404
            ? "The requested subject or page is unavailable. It may have been removed or become inaccessible."
            : "Refresh the context to load this page again. Your current filters are retained."}
      </p>
    </div>
  );
}

/**
 * Where a page of a check's possible issues or reviews stands: how many of
 * the matching records it shows, whether it is from an earlier version of
 * the check, and the controls to refresh or page on. The check revision and
 * snapshot time are technical details.
 */
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
  const earlier =
    page.auditRevision !== undefined && page.auditRevision !== currentRevision;
  return (
    <div className="issues-queue">
      <div className="issues-queue-row">
        <p className="issues-queue-status">
          Showing {page.items.length} of {page.total ?? "unavailable"} matching
          records in this check
        </p>
        <div className="issues-queue-actions">
          <Button size="xs" onClick={onRefresh}>
            Refresh context
          </Button>
          {queue.params.has("cursor") ? (
            <Button size="xs" onClick={queue.refresh}>
              First page
            </Button>
          ) : null}
          {page.page.hasMore && page.page.nextCursor !== undefined ? (
            <Button size="xs" onClick={() => queue.next(page)}>
              Next page
            </Button>
          ) : null}
        </div>
      </div>
      {earlier ? (
        <p className="issues-notice" data-tone="warning">
          This page is from an earlier version of the check. Refresh the context
          before continuing.
        </p>
      ) : null}
      <TechnicalDetails summary="Snapshot details">
        <p className="issues-quiet">
          {page.asOf === undefined
            ? "Count snapshot unavailable"
            : `Check revision ${page.auditRevision} · As of ${formatTimestamp(page.asOf)}`}
        </p>
      </TechnicalDetails>
    </div>
  );
}
