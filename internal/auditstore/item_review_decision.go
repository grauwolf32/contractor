package auditstore

import (
	"context"
	"errors"

	"github.com/jackc/pgx/v5"
)

// ApplyItemReviewDecision revalidates and updates the locked exact item subject.
func (s *PostgresStore) ApplyItemReviewDecision(
	ctx context.Context, auditID, itemID, kind, digest string,
	action string, rationale string,
) error {
	var state ItemState
	var approvalKind ItemApprovalKind
	var approvalDigest *string
	err := s.db.QueryRow(ctx, `
SELECT state, approval_kind, approval_subject_digest
  FROM audit_items
 WHERE audit_id = $1 AND item_id = $2
 FOR UPDATE`, auditID, itemID).Scan(&state, &approvalKind, &approvalDigest)
	if errors.Is(err, pgx.ErrNoRows) {
		return ErrNotFound
	}
	if err != nil {
		return err
	}
	if state != ItemAwaitingReview || approvalDigest == nil ||
		string(approvalKind) != kind || *approvalDigest != digest {
		return ErrPrecondition
	}
	if action == "approve" {
		tag, err := s.db.Exec(ctx, `
UPDATE audit_items
   SET state = 'ready',
       updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
 WHERE audit_id = $1 AND item_id = $2 AND state = 'awaiting_review'`, auditID, itemID)
		if err != nil {
			return err
		}
		if tag.RowsAffected() != 1 {
			return ErrPrecondition
		}
		return nil
	}
	if action == "not_applicable" {
		if approvalKind != ItemApprovalApplicability {
			return ErrPrecondition
		}
		tag, err := s.db.Exec(ctx, `
UPDATE audit_items
   SET state = 'settled', final_disposition = 'not-applicable',
       updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond'),
       coverage_status = 'not-applicable', coverage_completed = '[]'::jsonb,
       coverage_gaps = '[]'::jsonb, coverage_rationale = $3,
       coverage_updated_at = GREATEST(clock_timestamp(), coverage_updated_at + interval '1 microsecond')
 WHERE audit_id = $1 AND item_id = $2 AND state = 'awaiting_review'`, auditID, itemID, rationale)
		if err != nil {
			return err
		}
		if tag.RowsAffected() != 1 {
			return ErrPrecondition
		}
		return nil
	}
	if action != "reject" {
		return ErrInvalid
	}
	tag, err := s.db.Exec(ctx, rejectAwaitingItemSQL, auditID, itemID)
	if err != nil {
		return err
	}
	if tag.RowsAffected() != 1 {
		return ErrPrecondition
	}
	return nil
}
