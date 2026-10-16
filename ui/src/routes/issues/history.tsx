import { useQuery } from "@tanstack/react-query";
import { useId } from "react";

import { collectAuditPages } from "../../api/audit-collections";
import { listAuditReviews, type AuditReviewRequest } from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import { QueryView } from "../../app/query-view";
import { RecordedTime } from "../../app/recorded-time";
import { reviewStateLabel } from "../../app/vocabulary";
import { StatusChip } from "../../ui";
import { DecisionRecord } from "../decisions";

import "./issues.css";

function when(review: AuditReviewRequest): number {
  const parsed = Date.parse(review.decision?.createdAt ?? review.createdAt);
  return Number.isFinite(parsed) ? parsed : Number.NEGATIVE_INFINITY;
}

/** The possible issue's triage requests, newest first. */
function triage(reviews: readonly AuditReviewRequest[]): AuditReviewRequest[] {
  return reviews
    .filter((review) => review.kind === "finding-triage")
    .sort((left, right) => when(right) - when(left));
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
  return (
    <section className="issues-history" aria-labelledby={headingId}>
      <h3 id={headingId} className="issues-heading">
        Decisions
      </h3>
      <QueryView
        query={history}
        loading={
          <p className="ui-caption" role="status">
            Loading decisions…
          </p>
        }
        errorContext="The decisions could not be loaded."
        isEmpty={(data) => triage(data.items).length === 0}
        empty={<p className="ui-caption">No decision has been recorded yet.</p>}
      >
        {(data) => (
          <>
            <ol className="issues-history-list">
              {triage(data.items).map((review) => {
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
                      <span className="ui-caption">
                        Review requested{" "}
                        <RecordedTime value={review.createdAt} />
                      </span>
                    </p>
                  </li>
                );
              })}
            </ol>
            {data.truncated ? (
              <p className="ui-caption">
                This possible issue has more review requests than shown here.
              </p>
            ) : null}
          </>
        )}
      </QueryView>
    </section>
  );
}
