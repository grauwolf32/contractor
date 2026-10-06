import { useQuery } from "@tanstack/react-query";
import { useId } from "react";

import { collectAuditPages } from "../../api/audit-collections";
import { listAuditReviews, type AuditReviewRequest } from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import { RecordedTime } from "../../app/recorded-time";
import { reviewStateLabel } from "../../app/vocabulary";
import { StatusChip } from "../../ui";
import { DecisionRecord } from "../decisions";

import "./issues.css";

function when(review: AuditReviewRequest): number {
  const parsed = Date.parse(review.decision?.createdAt ?? review.createdAt);
  return Number.isFinite(parsed) ? parsed : Number.NEGATIVE_INFINITY;
}

/**
 * Every review request of one possible issue, newest first: recorded
 * decisions with who, when and why, and requests still open or expired.
 */
export function DecisionHistory({
  auditId,
  findingId,
}: {
  auditId: string;
  findingId: string;
}) {
  const api = usePublicAPI();
  const headingId = useId();
  const history = useQuery({
    queryKey: queryKeys.issues.history(auditId, findingId),
    queryFn: () =>
      collectAuditPages((cursor) =>
        listAuditReviews(api, auditId, {
          finding: findingId,
          ...(cursor === undefined ? {} : { cursor }),
        }),
      ),
    retry: false,
  });
  const reviews =
    history.data === undefined
      ? []
      : history.data.items
          .filter((review) => review.kind === "finding-triage")
          .sort((left, right) => when(right) - when(left));
  return (
    <section className="issues-history" aria-labelledby={headingId}>
      <h3 id={headingId} className="issues-heading">
        Decisions
      </h3>
      {history.data === undefined ? (
        history.error === null ? (
          <p className="issues-quiet" role="status">
            Loading decisions…
          </p>
        ) : (
          <div className="issues-notice" data-tone="error" role="alert">
            <p>The decisions could not be loaded: {history.error.message}</p>
            <button
              type="button"
              className="ui-btn"
              data-size="xs"
              disabled={history.isFetching}
              onClick={() => void history.refetch()}
            >
              Try again
            </button>
          </div>
        )
      ) : reviews.length === 0 ? (
        <p className="issues-quiet">No decision has been recorded yet.</p>
      ) : (
        <>
          <ol className="issues-history-list">
            {reviews.map((review) => {
              if (review.decision !== undefined)
                return (
                  <li key={review.requestId}>
                    <DecisionRecord decision={review.decision} />
                  </li>
                );
              const state = reviewStateLabel(review.state);
              return (
                <li key={review.requestId}>
                  <p className="issues-history-request">
                    <StatusChip tone={state.tone} size="sm">
                      {review.state === "expired"
                        ? "Expired without a decision"
                        : state.label}
                    </StatusChip>
                    <span className="issues-quiet">
                      Review requested <RecordedTime value={review.createdAt} />
                    </span>
                  </p>
                </li>
              );
            })}
          </ol>
          {history.data.truncated ? (
            <p className="issues-quiet">
              This possible issue has more review requests than shown here.
            </p>
          ) : null}
        </>
      )}
    </section>
  );
}
