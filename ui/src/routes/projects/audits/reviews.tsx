import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useId } from "react";

import {
  listAuditReviews,
  type Audit,
  type AuditItem,
  type AuditReviewRequest,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { queryKeys } from "../../../api/query-keys";
import { ContextLink } from "../../../app/context-navigation";
import { RecordedTime } from "../../../app/recorded-time";
import {
  capitalize,
  itemNoun,
  REVIEW_STATE_LABELS,
  reviewKindLabel,
  reviewStateLabel,
  type AuditReviewState,
  type ItemKind,
} from "../../../app/vocabulary";
import {
  FilterChips,
  IdChip,
  StatusChip,
  TechnicalDetails,
  type FilterChipOption,
} from "../../../ui";
import { ActionDecision, DecisionRecord } from "../../decisions";
import { itemHash, sectionPath } from "./check-links";
import type { AuditCollectionQuery } from "./collections";
import { LoadMoreControl } from "./load-more";
import { AuditQueueError, AuditQueuePage } from "./queue";
import { useAuditQueue } from "./queue-state";

type StateFilter = AuditReviewState | "all";

const STATES = Object.keys(REVIEW_STATE_LABELS) as AuditReviewState[];

const STATE_OPTIONS: readonly FilterChipOption<StateFilter>[] = [
  { value: "all", label: "All" },
  ...STATES.map((state) => ({
    value: state,
    label: REVIEW_STATE_LABELS[state].label,
  })),
];

function parseState(value: string | null): StateFilter {
  return STATES.find((state) => state === value) ?? "all";
}

/** What a review request is about, in words. */
function subjectTitle(
  review: AuditReviewRequest,
  subjects: ReadonlyMap<string, string>,
): string {
  const known = subjects.get(review.subjectId);
  if (known !== undefined) return known;
  switch (review.subjectKind) {
    case "finding":
      return "Possible issue";
    case "audit-report":
      return "Check report";
    default:
      return "Review requested";
  }
}

function ReviewCard({
  audit,
  review,
  kind,
  subjects,
}: {
  audit: Audit;
  review: AuditReviewRequest;
  kind: ItemKind;
  subjects: ReadonlyMap<string, string>;
}) {
  const heading = useId();
  const state = reviewStateLabel(review.state);
  const findingId = review.findingId ?? review.subjectId;
  return (
    <li
      className="checks-review"
      id={`review-${review.requestId}`}
      aria-labelledby={heading}
    >
      <div className="checks-review-heading">
        <div>
          <p className="checks-eyebrow">{reviewKindLabel(review.kind)}</p>
          <h3 id={heading}>{subjectTitle(review, subjects)}</h3>
        </div>
        <StatusChip tone={state.tone} size="sm">
          {state.label}
        </StatusChip>
      </div>
      <p className="checks-review-meta">
        Requested <RecordedTime value={review.createdAt} />
        {review.expiresAt === undefined || review.state !== "pending" ? null : (
          <>
            {" "}
            · expires <RecordedTime value={review.expiresAt} />
          </>
        )}
      </p>
      <p className="checks-review-links">
        {review.subjectKind === "audit-item-action" &&
        subjects.has(review.subjectId) ? (
          <ContextLink
            returnLabel="Check decisions"
            to={`${sectionPath(audit.projectId, audit.auditId, "coverage")}${itemHash(review.subjectId)}`}
          >
            View {itemNoun(kind, 1)} →
          </ContextLink>
        ) : null}
        {review.subjectKind === "finding" ? (
          <ContextLink
            returnLabel="Check decisions"
            to={`${sectionPath(audit.projectId, audit.auditId, "findings")}?finding=${encodeURIComponent(findingId)}&review=${encodeURIComponent(review.requestId)}`}
          >
            Review possible issue →
          </ContextLink>
        ) : null}
        {review.subjectKind === "audit-report" ? (
          <ContextLink
            returnLabel="Check decisions"
            to={`${sectionPath(audit.projectId, audit.auditId, "report")}?review=${encodeURIComponent(review.requestId)}`}
          >
            Review report →
          </ContextLink>
        ) : null}
      </p>
      {review.subjectKind === "audit-item-action" &&
      (review.state === "pending" || review.decision !== undefined) ? (
        // Stays mounted once decided (it shows the decision then), so its
        // "Decision recorded" announcement is read out.
        <div className="decisions-inline">
          <ActionDecision auditId={audit.auditId} review={review} />
        </div>
      ) : review.decision !== undefined ? (
        <DecisionRecord decision={review.decision} />
      ) : review.state === "pending" ? null : (
        <p className="checks-quiet">No decision recorded.</p>
      )}
      <TechnicalDetails summary="Review details">
        <dl className="checks-facts">
          <div>
            <dt>Type</dt>
            <dd>{reviewKindLabel(review.kind)}</dd>
          </div>
          <div>
            <dt>Subject revision</dt>
            <dd>{review.subjectRevision}</dd>
          </div>
          <div>
            <dt>Review ID</dt>
            <dd>
              <IdChip value={review.requestId} label="review ID" />
            </dd>
          </div>
          <div>
            <dt>Subject ID</dt>
            <dd>
              <IdChip value={findingId} label="subject ID" />
            </dd>
          </div>
        </dl>
      </TechnicalDetails>
    </li>
  );
}

/**
 * The check's decisions: every review request (active test approvals,
 * requirement applicability, possible issues, report acceptance), filtered
 * by state and read in pinned Server pages. Item decisions are made in
 * place; possible issues and the report open their own pages with a return
 * link to this queue, its filters and cursor (S19:1835-1838).
 */
export function AuditReviews({
  audit,
  kind,
  subjects,
  items,
}: {
  audit: Audit;
  kind: ItemKind;
  /** Plain names of the check's items by item ID. */
  subjects: ReadonlyMap<string, string>;
  items: AuditCollectionQuery<AuditItem>;
}) {
  const queue = useAuditQueue();
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const state = parseState(queue.params.get("state"));
  const reviews = useQuery({
    queryKey: [
      ...queryKeys.audits.allReviews(audit.auditId),
      queue.params.toString(),
    ],
    queryFn: () =>
      listAuditReviews(api, audit.auditId, {
        ...queue.request,
        ...(state === "all" ? {} : { state }),
      }),
  });
  const refresh = () => {
    queue.refresh();
    void queryClient.invalidateQueries({
      queryKey: queryKeys.audits.detail(audit.auditId),
    });
  };
  const needsItems =
    reviews.data?.items.some(
      (review) => review.subjectKind === "audit-item-action",
    ) ?? false;
  return (
    <div className="checks-section">
      <div className="checks-section-heading">
        <h2>Decisions</h2>
        <p className="checks-quiet">
          What the check asked you to decide, and what was decided.
        </p>
      </div>
      <FilterChips
        label="Filter decisions by state"
        options={STATE_OPTIONS}
        value={state}
        onChange={(value) => queue.change("state", value)}
      />
      {reviews.isPending ? (
        <p className="checks-quiet" role="status">
          Loading decisions…
        </p>
      ) : reviews.error !== null ? (
        <AuditQueueError error={reviews.error} onRefresh={refresh} />
      ) : (
        <>
          <AuditQueuePage
            page={reviews.data}
            currentRevision={audit.revision}
            queue={queue}
            onRefresh={refresh}
          />
          {needsItems && items.error !== null ? (
            <div className="checks-notice" data-tone="warning">
              <p>
                {capitalize(itemNoun(kind, 2))} could not be loaded, so their
                identifiers are shown instead.
              </p>
              <button
                className="ui-btn"
                data-size="xs"
                type="button"
                onClick={() => void items.refetch()}
              >
                Try again
              </button>
            </div>
          ) : null}
          {needsItems ? (
            <LoadMoreControl
              shown={items.items.length}
              noun={`${itemNoun(kind, 1)} names`}
              truncated={items.truncated}
              loading={items.isLoadingMore}
              error={items.moreError}
              onLoadMore={items.loadMore}
              label={`Load more ${itemNoun(kind, 1)} names`}
            />
          ) : null}
          {reviews.data.items.length === 0 ? (
            <p className="checks-quiet">
              {state === "pending"
                ? "Nothing is waiting for you. Choose All to see earlier decisions."
                : "No decisions have been requested yet."}
            </p>
          ) : (
            <ol className="checks-reviews">
              {reviews.data.items.map((review) => (
                <ReviewCard
                  key={review.requestId}
                  audit={audit}
                  review={review}
                  kind={kind}
                  subjects={subjects}
                />
              ))}
            </ol>
          )}
        </>
      )}
    </div>
  );
}
