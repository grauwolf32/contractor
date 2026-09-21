import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { collectAuditPages } from "../../../api/audit-collections";
import {
  decideAuditAction,
  listAuditItems,
  listAuditReviews,
  type Audit,
  type AuditReviewAction,
  type AuditReviewRequest,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { queryKeys } from "../../../api/query-keys";
import { ContextLink } from "../../../app/context-navigation";
import { MutationDraftKeyring } from "../../../mutations/idempotency";
import { formatTimestamp } from "../../artifacts/common";
import { StateBadge } from "../../runs/components";
import { AuditMutationNotice } from "./controls";
import { useAuditCollection } from "./collections";
import { LoadMoreControl } from "./load-more";
import { AuditQueueError, AuditQueuePage } from "./queue";
import { useAuditQueue } from "./queue-state";
import { AuditAnchor, AuditMarkdown } from "./shared";

export function ActionReviewControls({
  audit,
  review,
}: {
  audit: Audit;
  review: AuditReviewRequest;
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [rationale, setRationale] = useState("");
  const [keyring] = useState(
    () =>
      new MutationDraftKeyring<Record<string, string | number>>(
        "audit-action-review",
      ),
  );
  async function invalidate(): Promise<void> {
    await Promise.all([
      queryClient.invalidateQueries({
        queryKey: queryKeys.audits.detail(audit.auditId),
      }),
      queryClient.invalidateQueries({
        queryKey: queryKeys.audits.reviews(audit.auditId),
      }),
      queryClient.invalidateQueries({
        queryKey: queryKeys.audits.items(audit.auditId),
      }),
      queryClient.invalidateQueries({
        queryKey: queryKeys.audits.report(audit.auditId),
      }),
    ]);
  }
  const decision = useMutation({
    mutationFn: (action: AuditReviewAction) => {
      const trimmed = rationale.trim();
      const draft = {
        auditId: audit.auditId,
        requestId: review.requestId,
        revision: review.revision,
        action,
        rationale: trimmed,
      };
      return decideAuditAction(api, {
        auditId: audit.auditId,
        requestId: review.requestId,
        expectedRevision: review.revision,
        idempotencyKey: keyring.keyFor(draft),
        action,
        rationale: trimmed,
      });
    },
    onSuccess: async () => {
      setRationale("");
      await invalidate();
    },
    onError: invalidate,
  });
  const unavailable = rationale.trim().length === 0 || decision.isPending;
  return (
    <div className="audit-finding-review-actions">
      <label>
        Rationale
        <textarea
          value={rationale}
          maxLength={65_536}
          rows={3}
          onChange={(event) => setRationale(event.target.value)}
          placeholder="Explain your decision. Markdown is supported."
        />
      </label>
      <div className="button-row">
        {review.requestedActions.includes("approve") ? (
          <button
            type="button"
            disabled={unavailable}
            onClick={() => decision.mutate("approve")}
          >
            Approve exact subject
          </button>
        ) : null}
        {review.requestedActions.includes("reject") ? (
          <button
            className="secondary-button"
            type="button"
            disabled={unavailable}
            onClick={() => decision.mutate("reject")}
          >
            Reject
          </button>
        ) : null}
        {review.requestedActions.includes("not_applicable") ? (
          <button
            className="secondary-button"
            type="button"
            disabled={unavailable}
            onClick={() => decision.mutate("not_applicable")}
          >
            Mark not applicable
          </button>
        ) : null}
      </div>
      {decision.error === null ? null : (
        <AuditMutationNotice error={decision.error} />
      )}
    </div>
  );
}

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
                  to={`/projects/${encodeURIComponent(audit.projectId)}/audits/${encodeURIComponent(audit.auditId)}/checks#check-${encodeURIComponent(review.subjectId)}`}
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
                    <ActionReviewControls audit={audit} review={review} />
                  ) : null}
                </>
              ) : (
                <>
                  <span>
                    {review.decision.action ?? review.decision.verdict}
                    {review.decision.severity === undefined
                      ? ""
                      : ` · ${review.decision.severity}`}
                    {" · "}
                    {review.decision.actorId}
                  </span>
                  <AuditMarkdown source={review.decision.rationale} />
                </>
              )}
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
