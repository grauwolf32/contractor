package auditstore

import (
	"context"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/jackc/pgx/v5"
)

// ResumeRenewalParams identifies the paused Audit whose expired item reviews
// Resume renews. DeadlineAt is the deadline Resume applies, which also bounds
// the fresh requests; nil leaves them without expiry.
type ResumeRenewalParams struct {
	OwnerID          string
	AuditID          string
	ExpectedRevision uint64
	DeadlineAt       *time.Time
}

// RenewItemReviewsForResume renews every unfinished item review that expired
// while the Audit was paused, inside the caller's Resume transaction. It uses
// the Controller's renewal, so each renewal records review.expired and
// review.requested events with one Audit revision per event. It returns the
// Audit revision that the subsequent Resume transition must expect.
func (s *PostgresStore) RenewItemReviewsForResume(ctx context.Context, params ResumeRenewalParams) (uint64, error) {
	if err := validateText("ownerID", params.OwnerID, 256, true); err != nil {
		return 0, err
	}
	if err := validateID("auditID", params.AuditID); err != nil {
		return 0, err
	}
	if params.ExpectedRevision == 0 || params.ExpectedRevision > math.MaxInt64 {
		return 0, invalidf("Audit resume revision is invalid")
	}
	tx, ok := s.db.(pgx.Tx)
	if !ok {
		return 0, errors.New("renew Audit item reviews for Resume: the Resume transaction is required")
	}
	var revision int64
	err := tx.QueryRow(ctx, `
SELECT revision FROM audits
 WHERE audit_id=$1 AND owner_id=$2 AND state='paused'
 FOR UPDATE`, params.AuditID, params.OwnerID).Scan(&revision)
	if errors.Is(err, pgx.ErrNoRows) {
		return 0, ErrPrecondition
	}
	if err != nil {
		return 0, fmt.Errorf("lock paused Audit for review renewal: %w", err)
	}
	if uint64(revision) != params.ExpectedRevision {
		return 0, ErrPrecondition
	}
	if _, err := renewExpiredItemReviews(ctx, tx, params.AuditID, params.DeadlineAt, 0); err != nil {
		return 0, err
	}
	if err := tx.QueryRow(ctx, `SELECT revision FROM audits WHERE audit_id=$1`, params.AuditID).
		Scan(&revision); err != nil {
		return 0, fmt.Errorf("read renewed Audit revision: %w", err)
	}
	return uint64(revision), nil
}
