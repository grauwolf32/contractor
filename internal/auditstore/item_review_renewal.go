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

	var itemID, kind, digest, oldRequestID, oldState string
	err = tx.QueryRow(ctx, `
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
 LIMIT 1
 FOR UPDATE OF review`, auditID).Scan(&itemID, &kind, &digest, &oldRequestID, &oldState)
	if errors.Is(err, pgx.ErrNoRows) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("find expired Audit item review: %w", err)
	}
	var itemState, lockedKind string
	var lockedDigest *string
	err = tx.QueryRow(ctx, `
SELECT state,approval_kind,approval_subject_digest FROM audit_items
 WHERE audit_id=$1 AND item_id=$2 FOR UPDATE`, auditID, itemID).
		Scan(&itemState, &lockedKind, &lockedDigest)
	if err != nil {
		return false, fmt.Errorf("lock Audit item for review renewal: %w", err)
	}
	if (itemState != "ready" && itemState != "awaiting_review") ||
		lockedKind != kind || lockedDigest == nil || *lockedDigest != digest {
		return false, nil
	}

	var expiredRevision *int64
	if oldState != "expired" {
		var revision int64
		err = tx.QueryRow(ctx, `
UPDATE audit_review_requests
   SET state='expired', revision=revision+1,
       updated_at=GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
 WHERE request_id=$1 AND state IN ('pending','decided')
RETURNING revision`, oldRequestID).Scan(&revision)
		if err != nil {
			return false, fmt.Errorf("expire Audit item review: %w", err)
		}
		expiredRevision = &revision
	}

	requestID, err := randomid.New("review-renew-")
	if err != nil {
		return false, fmt.Errorf("create Audit item review identity: %w", err)
	}
	requestedActions := `["approve","reject"]`
	if kind == "requirement-applicability" {
		requestedActions = `["approve","reject","not_applicable"]`
	}
	_, err = tx.Exec(ctx, `
INSERT INTO audit_review_requests (
    request_id,audit_id,finding_id,subject_kind,subject_id,kind,
    subject_revision,subject_digest,requested_actions,state,expires_at,
    idempotency_key,request_digest
) VALUES ($1,$2,NULL,'audit-item-action',$3,$4,1,$5,$6::jsonb,'pending',$7,$1,$5)`,
		requestID, auditID, itemID, kind, digest, requestedActions, deadline)
	if err != nil {
		return false, fmt.Errorf("request renewed Audit item review: %w", err)
	}
	_, err = tx.Exec(ctx, `
UPDATE audit_items
   SET state='awaiting_review',
       updated_at=GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
 WHERE audit_id=$1 AND item_id=$2 AND state IN ('ready','awaiting_review')`, auditID, itemID)
	if err != nil {
		return false, fmt.Errorf("reset Audit item for renewed review: %w", err)
	}

	eventCount := int64(1)
	if expiredRevision != nil {
		eventCount++
	}
	var firstSequence int64
	err = tx.QueryRow(ctx, `
UPDATE audits
   SET revision=revision+$2, next_event_sequence=next_event_sequence+$2,
       updated_at=GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
 WHERE audit_id=$1
RETURNING next_event_sequence-$2`, auditID, eventCount).Scan(&firstSequence)
	if err != nil {
		return false, fmt.Errorf("advance Audit review events: %w", err)
	}
	if expiredRevision != nil {
		_, err = tx.Exec(ctx, `
INSERT INTO audit_events (audit_id,sequence_number,kind,entity_id,entity_revision,summary)
VALUES ($1,$2,'review.expired',$3,$4,
        jsonb_build_object('subjectKind','audit-item-action','subjectId',$5::text,'kind',$6::text))`,
			auditID, firstSequence, oldRequestID, *expiredRevision, itemID, kind)
		if err != nil {
			return false, fmt.Errorf("record expired Audit item review: %w", err)
		}
		firstSequence++
	}
	_, err = tx.Exec(ctx, `
INSERT INTO audit_events (audit_id,sequence_number,kind,entity_id,entity_revision,summary)
VALUES ($1,$2,'review.requested',$3,1,
        jsonb_build_object('subjectKind','audit-item-action','subjectId',$4::text,'kind',$5::text))`,
		auditID, firstSequence, requestID, itemID, kind)
	if err != nil {
		return false, fmt.Errorf("record renewed Audit item review: %w", err)
	}
	return true, nil
}
