package auditstore

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
)

func (s *PostgresStore) Claim(ctx context.Context, params ClaimParams) ([]ControllerClaim, error) {
	if err := validateClaimParams(params); err != nil {
		return nil, err
	}
	rows, err := s.db.Query(ctx, `
WITH candidates AS (
    SELECT claim.audit_id
      FROM audit_controller_claims AS claim
      JOIN audits AS audit USING (audit_id)
     WHERE audit.state IN (
               'active', 'waiting_review', 'paused', 'finalizing', 'cancelling', 'deleting'
           )
       AND (claim.holder_id IS NULL OR claim.expires_at <= clock_timestamp())
     ORDER BY claim.epoch, audit.updated_at, audit.audit_id
     FOR UPDATE OF claim SKIP LOCKED
     LIMIT $3
), claimed AS (
    UPDATE audit_controller_claims AS claim
       SET epoch = epoch + 1,
           holder_id = $1,
           claimed_at = clock_timestamp(),
           expires_at = clock_timestamp() + $2::bigint * interval '1 millisecond'
      FROM candidates
     WHERE claim.audit_id = candidates.audit_id
    RETURNING claim.audit_id, claim.holder_id, claim.epoch,
              claim.claimed_at, claim.expires_at
)
SELECT audit_id, holder_id, epoch, claimed_at, expires_at
  FROM claimed
 ORDER BY claimed_at, audit_id`, params.HolderID, params.Lease.Milliseconds(), params.Limit)
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

func (s *PostgresStore) RenewClaim(
	ctx context.Context,
	claim ControllerClaim,
	lease time.Duration,
) (ControllerClaim, error) {
	if err := validateClaimIdentity(claim); err != nil {
		return ControllerClaim{}, err
	}
	if lease < time.Second || lease > 5*time.Minute {
		return ControllerClaim{}, invalidf("claim lease is invalid")
	}
	renewed, err := scanClaim(s.db.QueryRow(ctx, `
UPDATE audit_controller_claims
   SET claimed_at = clock_timestamp(),
       expires_at = clock_timestamp() + $4::bigint * interval '1 millisecond'
 WHERE audit_id = $1 AND holder_id = $2 AND epoch = $3
   AND expires_at > clock_timestamp()
RETURNING audit_id, holder_id, epoch, claimed_at, expires_at`,
		claim.AuditID, claim.HolderID, claim.Epoch, lease.Milliseconds(),
	))
	if errors.Is(err, pgx.ErrNoRows) {
		return ControllerClaim{}, ErrClaimLost
	}
	if err != nil {
		return ControllerClaim{}, fmt.Errorf("renew Audit claim: %w", err)
	}
	return renewed, nil
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
