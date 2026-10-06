import { useQuery, useQueryClient } from "@tanstack/react-query";

import { collectAuditPages } from "../../../api/audit-collections";
import {
  listAuditItems,
  listAuditReviews,
  type Audit,
  type AuditReviewRequest,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { queryKeys } from "../../../api/query-keys";
import { ContextLink } from "../../../app/context-navigation";
import { formatTimestamp } from "../../../app/format";
import { StateBadge } from "../../runs/components";
import { ActionDecision, DecisionRecord } from "../../decisions";
import { useAuditCollection } from "./collections";
import { LoadMoreControl } from "./load-more";
import { AuditQueueError, AuditQueuePage } from "./queue";
import { useAuditQueue } from "./queue-state";
import { AuditAnchor } from "./shared";

export function AuditReviews({ audit }: { audit: Audit }) {
  const queue = useAuditQueue();
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const state = queue.params.get("state") ?? "";
  const pendingOnly = state === "pending";
  const reviews = useQuery({
    queryKey: [
      ...queryKeys.audits.allReviews(audit.auditId),
      queue.params.toString(),
    ],
    queryFn: () =>
      listAuditReviews(api, audit.auditId, {
        ...queue.request,
        ...(["pending", "decided", "expired"].includes(state)
          ? { state: state as AuditReviewRequest["state"] }
          : {}),
      }),
  });
  const refresh = () => {
    queue.refresh();
    void queryClient.invalidateQueries({
      queryKey: queryKeys.audits.detail(audit.auditId),
    });
  };
  const items = useAuditCollection({
    queryKey: queryKeys.audits.allItems(audit.auditId),
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
    enabled:
      reviews.data?.items.some(
        (review) => review.subjectKind === "audit-item-action",
      ) ?? false,
    refetchInterval: false,
  });
  const subjects = new Map(
    items.items.map((item) => [item.itemId, item.subjectKey]),
  );
  const reviewLabels: Record<string, string> = {
    "requirement-applicability": "Requirement applicability",
    "active-check-approval": "Active check approval",
    "finding-triage": "Finding review",
    "report-acceptance": "Report acceptance",
  };
  if (reviews.isPending)
    return <p className="loading-copy">Loading reviews…</p>;
  if (reviews.error !== null)
    return <AuditQueueError error={reviews.error} onRefresh={refresh} />;
  const visibleReviews = reviews.data.items;
  return (
    <section className="panel audit-section-panel">
      <AuditAnchor />
      <div className="section-heading">
        <div>
          <p className="eyebrow">Decisions and review history</p>
          <h3>Human reviews</h3>
        </div>
      </div>
      <label className="audit-review-filter">
        Review state
        <select
          value={state || "all"}
          onChange={(event) => queue.change("state", event.target.value)}
        >
          <option value="all">All reviews</option>
          <option value="pending">Pending decisions</option>
          <option value="decided">Decided</option>
          <option value="expired">Expired</option>
        </select>
      </label>
      <AuditQueuePage
        page={reviews.data}
        currentRevision={audit.revision}
        queue={queue}
        onRefresh={refresh}
      />
      {items.error === null ? null : (
        <div className="notice">
          <p>
            Check names could not be loaded. Review identifiers are shown below.
          </p>
          <button
            className="secondary-button"
            type="button"
            onClick={() => void items.refetch()}
          >
            Retry check details
          </button>
        </div>
      )}
      <LoadMoreControl
        shown={items.items.length}
        noun="check names"
        truncated={items.truncated}
        loading={items.isLoadingMore}
        error={items.moreError}
        onLoadMore={items.loadMore}
        label="Load more check names"
      />
      {visibleReviews.length === 0 ? (
        <p className="muted-copy">
          {pendingOnly
            ? "No pending decisions. Choose All reviews to inspect previous decisions."
            : "No human review has been opened."}
        </p>
      ) : (
        <ol className="audit-review-history">
          {visibleReviews.map((review) => (
            <li
              className="audit-review-card"
              key={review.requestId}
              id={`review-${review.requestId}`}
            >
              <div className="audit-review-heading">
                <div>
                  <p className="eyebrow">
                    {reviewLabels[review.kind] ??
                      review.kind.replaceAll("-", " ")}
                  </p>
                  <h4>
                    {subjects.get(review.subjectId) ??
                      (review.subjectKind === "finding"
                        ? "Finding assessment"
                        : review.subjectKind === "audit-report"
                          ? "Audit report"
                          : "Review requested")}
                  </h4>
                </div>
                <StateBadge state={review.state} />
              </div>
              <p className="audit-review-meta">
                Requested {formatTimestamp(review.createdAt)}
              </p>
              {subjects.has(review.subjectId) ? (
                <ContextLink
                  returnLabel="Audit reviews"
                  to={`/projects/${encodeURIComponent(audit.projectId)}/audits/${encodeURIComponent(audit.auditId)}/coverage#check-${encodeURIComponent(review.subjectId)}`}
                >
                  View check →
                </ContextLink>
              ) : null}
              {review.subjectKind === "finding" ? (
                <ContextLink
                  returnLabel="Audit reviews"
                  to={`/projects/${encodeURIComponent(audit.projectId)}/audits/${encodeURIComponent(audit.auditId)}/findings?finding=${encodeURIComponent(review.findingId ?? review.subjectId)}&review=${encodeURIComponent(review.requestId)}`}
                >
                  Review finding →
                </ContextLink>
              ) : null}
              {review.subjectKind === "audit-report" ? (
                <ContextLink
                  returnLabel="Audit reviews"
                  to={`/projects/${encodeURIComponent(audit.projectId)}/audits/${encodeURIComponent(audit.auditId)}/report?review=${encodeURIComponent(review.requestId)}`}
                >
                  Review report →
                </ContextLink>
              ) : null}
              <details className="audit-record-details">
                <summary>Review details</summary>
                <dl className="metadata-grid">
                  <div>
                    <dt>Type</dt>
                    <dd>{review.kind}</dd>
                  </div>
                  <div>
                    <dt>Subject revision</dt>
                    <dd>{review.subjectRevision}</dd>
                  </div>
                  <div>
                    <dt>Review ID</dt>
                    <dd>
                      <code>{review.requestId}</code>
                    </dd>
                  </div>
                  <div>
                    <dt>Subject ID</dt>
                    <dd>
                      <code>{review.findingId ?? review.subjectId}</code>
                    </dd>
                  </div>
                </dl>
              </details>
              {review.decision === undefined ? (
                <>
                  {review.state === "pending" ? null : (
                    <p className="muted-copy">No decision recorded.</p>
                  )}
                  {review.state === "pending" &&
                  review.subjectKind === "audit-item-action" ? (
                    <div className="decisions-inline">
                      <ActionDecision auditId={audit.auditId} review={review} />
                    </div>
                  ) : null}
                </>
              ) : (
                <DecisionRecord decision={review.decision} />
              )}
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
