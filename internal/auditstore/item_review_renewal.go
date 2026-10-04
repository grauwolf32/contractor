package auditstore

import (
	"context"
	"errors"
	"fmt"
	"math"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/randomid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// RenewExpiredItemReview repairs one exact item approval under a live
// controller claim. Database time decides expiry; the old approval is never
// extended, even when it was already approved before Resume.
func (s *PostgresStore) RenewExpiredItemReview(
	ctx context.Context, claim ControllerClaim, expectedAuditRevision uint64,
) (bool, error) {
	if err := validateClaimIdentity(claim); err != nil {
		return false, err
	}
	if expectedAuditRevision == 0 || expectedAuditRevision > math.MaxInt64 {
		return false, ErrInvalid
	}
	var changed bool
	var err error
	switch db := s.db.(type) {
	case *pgxpool.Pool:
		err = persistencepostgres.InTxWithRetry(ctx, db, pgx.TxOptions{}, func(tx pgx.Tx) error {
			changed, err = renewExpiredItemReview(ctx, tx, claim, expectedAuditRevision)
			return err
		})
	case pgx.Tx:
		changed, err = renewExpiredItemReview(ctx, db, claim, expectedAuditRevision)
	default:
		err = errors.New("renew Audit item review: PostgreSQL transaction support is required")
	}
	return changed, err
}

func renewExpiredItemReview(
	ctx context.Context, tx pgx.Tx, claim ControllerClaim, expectedAuditRevision uint64,
) (bool, error) {
	var auditID string
	err := tx.QueryRow(ctx, `
SELECT audit_id FROM audit_controller_claims
 WHERE audit_id=$1 AND holder_id=$2 AND epoch=$3
   AND expires_at > clock_timestamp()
 FOR UPDATE`, claim.AuditID, claim.HolderID, claim.Epoch).Scan(&auditID)
	if errors.Is(err, pgx.ErrNoRows) {
		return false, ErrClaimLost
	}
	if err != nil {
		return false, fmt.Errorf("lock Audit claim for review renewal: %w", err)
	}

	var deadline *time.Time
	err = tx.QueryRow(ctx, `
SELECT deadline_at FROM audits
 WHERE audit_id=$1 AND revision=$2
   AND state IN ('active','waiting_review') AND dispatch_state='open'
   AND (deadline_at IS NULL OR deadline_at > clock_timestamp())
 FOR UPDATE`, auditID, int64(expectedAuditRevision)).Scan(&deadline)
	if errors.Is(err, pgx.ErrNoRows) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("lock Audit for review renewal: %w", err)
	}

	renewed, err := renewExpiredItemReviews(ctx, tx, auditID, deadline, 1)
	return renewed != 0, err
}

// expiredItemReview is an unfinished item whose exact review expired without
// a live replacement, with the request whose authority it last held.
type expiredItemReview struct {
	itemID, kind, digest, requestID, state string
}

// renewExpiredItemReviews replaces the expired authority of up to limit
// unfinished items in item order, or of every such item when limit is zero.
// The caller holds the Audit row lock and decides the fresh requests' expiry.
// Database time decides expiry, and an old approval is never extended: a
// still-pending or approved request is expired, a fresh exact-subject request
// is created, the item returns to awaiting review, and each review.expired and
// review.requested event advances the Audit revision by one.
func renewExpiredItemReviews(
	ctx context.Context, tx pgx.Tx, auditID string, deadline *time.Time, limit int,
) (int, error) {
	var rowLimit *int
	if limit > 0 {
		rowLimit = &limit
	}
	rows, err := tx.Query(ctx, `
SELECT item.item_id, item.approval_kind, item.approval_subject_digest,
       review.request_id, review.state
  FROM audit_items AS item
  JOIN audit_review_requests AS review
    ON review.audit_id=item.audit_id
   AND review.subject_kind='audit-item-action'
   AND review.subject_id=item.item_id
   AND review.kind=item.approval_kind
   AND review.subject_revision=1
   AND review.subject_digest=item.approval_subject_digest
 WHERE item.audit_id=$1 AND item.state IN ('ready','awaiting_review')
   AND item.approval_kind <> 'none'
   AND (review.state='expired' OR (
       review.state IN ('pending','decided')
       AND review.expires_at <= clock_timestamp()
   ))
   AND (review.state <> 'decided' OR EXISTS (
       SELECT 1 FROM audit_review_decisions AS decision
        WHERE decision.request_id=review.request_id AND decision.action='approve'
   ))
   AND NOT EXISTS (
       SELECT 1 FROM audit_review_requests AS live
        WHERE live.audit_id=item.audit_id
          AND live.subject_kind='audit-item-action'
          AND live.subject_id=item.item_id
          AND live.kind=item.approval_kind
          AND live.subject_revision=1
          AND live.subject_digest=item.approval_subject_digest
          AND (live.expires_at IS NULL OR live.expires_at > clock_timestamp())
          AND (live.state='pending' OR (live.state='decided' AND EXISTS (
              SELECT 1 FROM audit_review_decisions AS decision
               WHERE decision.request_id=live.request_id AND decision.action='approve'
          )))
   )
 ORDER BY item.ordinal, item.item_id,
          CASE review.state WHEN 'pending' THEN 0 WHEN 'decided' THEN 1 ELSE 2 END,
          review.created_at DESC, review.request_id DESC
 LIMIT $2
 FOR UPDATE OF review`, auditID, rowLimit)
	if err != nil {
		return 0, fmt.Errorf("find expired Audit item reviews: %w", err)
	}
	// An item may hold several expired requests; its first row names the
	// request whose authority it last held.
	candidates := make([]expiredItemReview, 0)
	for rows.Next() {
		var candidate expiredItemReview
		if err := rows.Scan(&candidate.itemID, &candidate.kind, &candidate.digest,
			&candidate.requestID, &candidate.state); err != nil {
			rows.Close()
			return 0, fmt.Errorf("scan expired Audit item review: %w", err)
		}
		if len(candidates) == 0 || candidates[len(candidates)-1].itemID != candidate.itemID {
			candidates = append(candidates, candidate)
		}
	}
	if err := rows.Err(); err != nil {
		return 0, fmt.Errorf("find expired Audit item reviews: %w", err)
	}
	if candidates, err = lockRenewableItems(ctx, tx, auditID, candidates); err != nil || len(candidates) == 0 {
		return 0, err
	}

	expiring := make([]string, 0, len(candidates))
	for _, candidate := range candidates {
		if candidate.state != "expired" {
			expiring = append(expiring, candidate.requestID)
		}
	}
	expiredRevisions := make(map[string]int64, len(expiring))
	if len(expiring) != 0 {
		rows, err := tx.Query(ctx, `
UPDATE audit_review_requests
   SET state='expired', revision=revision+1,
       updated_at=GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
 WHERE request_id=ANY($1::text[]) AND state IN ('pending','decided')
RETURNING request_id, revision`, expiring)
		if err != nil {
			return 0, fmt.Errorf("expire Audit item reviews: %w", err)
		}
		for rows.Next() {
			var requestID string
			var revision int64
			if err := rows.Scan(&requestID, &revision); err != nil {
				rows.Close()
				return 0, fmt.Errorf("expire Audit item reviews: %w", err)
			}
			expiredRevisions[requestID] = revision
		}
		if err := rows.Err(); err != nil {
			return 0, fmt.Errorf("expire Audit item reviews: %w", err)
		}
		if len(expiredRevisions) != len(expiring) {
			return 0, fmt.Errorf("expire Audit item reviews: %d of %d locked requests changed",
				len(expiring)-len(expiredRevisions), len(expiring))
		}
	}

	requestIDs := make([]string, len(candidates))
	itemIDs := make([]string, len(candidates))
	kinds := make([]string, len(candidates))
	digests := make([]string, len(candidates))
	actions := make([]string, len(candidates))
	for index, candidate := range candidates {
		if requestIDs[index], err = randomid.New("review-renew-"); err != nil {
			return 0, fmt.Errorf("create Audit item review identity: %w", err)
		}
		itemIDs[index], kinds[index], digests[index] = candidate.itemID, candidate.kind, candidate.digest
		actions[index] = itemReviewActions(candidate.kind)
	}
	if _, err := tx.Exec(ctx, `
INSERT INTO audit_review_requests (
    request_id,audit_id,finding_id,subject_kind,subject_id,kind,
    subject_revision,subject_digest,requested_actions,state,expires_at,
    idempotency_key,request_digest
)
SELECT renewal.request_id,$1,NULL,'audit-item-action',renewal.item_id,renewal.kind,
       1,renewal.digest,renewal.actions::jsonb,'pending',$2,
       renewal.request_id,renewal.digest
  FROM unnest($3::text[],$4::text[],$5::text[],$6::text[],$7::text[])
       WITH ORDINALITY AS renewal(request_id,item_id,kind,digest,actions,position)
 ORDER BY renewal.position`,
		auditID, deadline, requestIDs, itemIDs, kinds, digests, actions); err != nil {
		return 0, fmt.Errorf("request renewed Audit item reviews: %w", err)
	}
	if _, err := tx.Exec(ctx, `
UPDATE audit_items
   SET state='awaiting_review',
       updated_at=GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
 WHERE audit_id=$1 AND item_id=ANY($2::text[]) AND state IN ('ready','awaiting_review')`,
		auditID, itemIDs); err != nil {
		return 0, fmt.Errorf("reset Audit items for renewed reviews: %w", err)
	}

	eventCount := int64(len(candidates) + len(expiredRevisions))
	var sequence int64
	err = tx.QueryRow(ctx, `
UPDATE audits
   SET revision=revision+$2, next_event_sequence=next_event_sequence+$2,
       updated_at=GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
 WHERE audit_id=$1
RETURNING next_event_sequence-$2`, auditID, eventCount).Scan(&sequence)
	if err != nil {
		return 0, fmt.Errorf("advance Audit review events: %w", err)
	}
	sequences := make([]int64, 0, eventCount)
	eventKinds := make([]string, 0, eventCount)
	entities := make([]string, 0, eventCount)
	entityRevisions := make([]int64, 0, eventCount)
	subjects := make([]string, 0, eventCount)
	reviewKinds := make([]string, 0, eventCount)
	appendEvent := func(kind, entityID string, entityRevision int64, candidate expiredItemReview) {
		sequences = append(sequences, sequence)
		sequence++
		eventKinds = append(eventKinds, kind)
		entities = append(entities, entityID)
		entityRevisions = append(entityRevisions, entityRevision)
		subjects = append(subjects, candidate.itemID)
		reviewKinds = append(reviewKinds, candidate.kind)
	}
	for index, candidate := range candidates {
		if revision, expired := expiredRevisions[candidate.requestID]; expired {
			appendEvent("review.expired", candidate.requestID, revision, candidate)
		}
		appendEvent("review.requested", requestIDs[index], 1, candidate)
	}
	if _, err := tx.Exec(ctx, `
INSERT INTO audit_events (audit_id,sequence_number,kind,entity_id,entity_revision,summary)
SELECT $1,event.sequence_number,event.kind,event.entity_id,event.entity_revision,
       jsonb_build_object('subjectKind','audit-item-action','subjectId',event.subject_id,'kind',event.review_kind)
  FROM unnest($2::bigint[],$3::text[],$4::text[],$5::bigint[],$6::text[],$7::text[])
       AS event(sequence_number,kind,entity_id,entity_revision,subject_id,review_kind)`,
		auditID, sequences, eventKinds, entities, entityRevisions, subjects, reviewKinds); err != nil {
		return 0, fmt.Errorf("record renewed Audit item reviews: %w", err)
	}
	return len(candidates), nil
}

// lockRenewableItems locks the candidates' items and keeps those that are
// still unfinished with the same exact approval subject.
func lockRenewableItems(
	ctx context.Context, tx pgx.Tx, auditID string, candidates []expiredItemReview,
) ([]expiredItemReview, error) {
	if len(candidates) == 0 {
		return nil, nil
	}
	itemIDs := make([]string, len(candidates))
	for index, candidate := range candidates {
		itemIDs[index] = candidate.itemID
	}
	rows, err := tx.Query(ctx, `
SELECT item_id,state,approval_kind,approval_subject_digest FROM audit_items
 WHERE audit_id=$1 AND item_id=ANY($2::text[])
 ORDER BY ordinal, item_id
 FOR UPDATE`, auditID, itemIDs)
	if err != nil {
		return nil, fmt.Errorf("lock Audit items for review renewal: %w", err)
	}
	type lockedItem struct {
		state, kind string
		digest      *string
	}
	locked := make(map[string]lockedItem, len(candidates))
	for rows.Next() {
		var itemID string
		var item lockedItem
		if err := rows.Scan(&itemID, &item.state, &item.kind, &item.digest); err != nil {
			rows.Close()
			return nil, fmt.Errorf("lock Audit items for review renewal: %w", err)
		}
		locked[itemID] = item
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("lock Audit items for review renewal: %w", err)
	}
	renewable := candidates[:0]
	for _, candidate := range candidates {
		item, exists := locked[candidate.itemID]
		if exists && (item.state == "ready" || item.state == "awaiting_review") &&
			item.kind == candidate.kind && item.digest != nil && *item.digest == candidate.digest {
			renewable = append(renewable, candidate)
		}
	}
	return renewable, nil
}

// itemReviewActions lists the decisions a fresh item review request offers.
func itemReviewActions(kind string) string {
	if kind == "requirement-applicability" {
		return `["approve","reject","not_applicable"]`
	}
	return `["approve","reject"]`
}
