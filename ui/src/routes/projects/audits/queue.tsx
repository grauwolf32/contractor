import { useSearchParams } from "react-router";
import { PublicAPIError } from "../../../api/error";
import { ErrorNotice, formatTimestamp } from "../../artifacts/common";
import type { AuditFindingPage, AuditReviewPage } from "../../../api/audits";

export function useAuditQueue() {
  const [params, setParams] = useSearchParams();
  function change(key: string, value: string) {
    const next = new URLSearchParams(params);
    next.delete("cursor");
    next.delete("auditRevision");
    if (value === "" || value === "all") next.delete(key);
    else next.set(key, value);
    setParams(next, { preventScrollReset: true });
  }
  function refresh() {
    const next = new URLSearchParams(params);
    next.delete("cursor");
    next.delete("auditRevision");
    setParams(next, { replace: true, preventScrollReset: true });
  }
  const revision = params.get("auditRevision");
  return {
    params,
    change,
    refresh,
    request: {
      ...(params.has("cursor") ? { cursor: params.get("cursor")! } : {}),
      ...(revision !== null &&
      /^[1-9][0-9]*$/.test(revision) &&
      Number.isSafeInteger(Number(revision))
        ? { auditRevision: Number(revision) }
        : {}),
    },
    next(page: AuditFindingPage | AuditReviewPage) {
      if (page.page.nextCursor === undefined) return;
      const next = new URLSearchParams(params);
      next.set("cursor", page.page.nextCursor);
      next.set("auditRevision", String(page.auditRevision));
      setParams(next, { preventScrollReset: true });
    },
  };
}

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
