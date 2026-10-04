package auditstore

import (
	"context"
	"errors"
	"fmt"
)

// Claim leases a bounded set of Audits with reconcilable work. A paused Audit
// only observes and collects the Runs it already submitted, so it is claimed
// only while such an execution exists; cancel and delete move it out of
// paused and make it claimable again.
func (s *PostgresStore) Claim(ctx context.Context, params ClaimParams) ([]ControllerClaim, error) {
	if err := validateClaimParams(params); err != nil {
		return nil, err
	}
	rows, err := s.db.Query(ctx, claimAuditSQL, params.HolderID, params.Lease.Milliseconds(), params.Limit)
	if err != nil {
		return nil, fmt.Errorf("claim Audits: %w", err)
	}
	defer rows.Close()
	result := make([]ControllerClaim, 0, params.Limit)
	for rows.Next() {
		claim, scanErr := scanClaim(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan Audit claim: %w", scanErr)
		}
		result = append(result, claim)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate Audit claims: %w", err)
	}
	return result, nil
}

func (s *PostgresStore) ReleaseClaim(ctx context.Context, claim ControllerClaim) error {
	if err := validateClaimIdentity(claim); err != nil {
		return err
	}
	tag, err := s.db.Exec(ctx, `
UPDATE audit_controller_claims
   SET holder_id = NULL, claimed_at = NULL, expires_at = NULL
 WHERE audit_id = $1 AND holder_id = $2 AND epoch = $3`,
		claim.AuditID, claim.HolderID, claim.Epoch,
	)
	if err != nil {
		return fmt.Errorf("release Audit claim: %w", err)
	}
	if tag.RowsAffected() != 1 {
		return ErrClaimLost
	}
	return nil
}

func scanClaim(row scanner) (ControllerClaim, error) {
	var result ControllerClaim
	var epoch int64
	if err := row.Scan(
		&result.AuditID, &result.HolderID, &epoch,
		&result.ClaimedAt, &result.ExpiresAt,
	); err != nil {
		return ControllerClaim{}, err
	}
	if epoch <= 0 || !result.ExpiresAt.After(result.ClaimedAt) {
		return ControllerClaim{}, errors.New("stored Audit claim is invalid")
	}
	result.Epoch = uint64(epoch)
	return result, nil
}
