package pgstore

import (
	"context"
	"errors"
	"fmt"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

// auditClaims is the Audit controller's claim table, leased and checked
// through the shared controller claim helper (PER-R24).
var auditClaims = persistencepostgres.NewControllerClaims("audit_controller_claims", "audit_id")

// Claim leases a bounded set of Audits with reconcilable work. A paused Audit
// only observes and collects the Runs it already submitted, so it is claimed
// only while such an execution exists; cancel and delete move it out of
// paused and make it claimable again.
func (s *ClaimStore) Claim(ctx context.Context, params auditstore.ClaimParams) ([]auditstore.ControllerClaim, error) {
	if err := validateClaimParams(params); err != nil {
		return nil, err
	}
	rows, err := s.db.Query(ctx, claimAuditSQL, params.HolderID, params.Lease.Milliseconds(), params.Limit)
	if err != nil {
		return nil, fmt.Errorf("claim Audits: %w", err)
	}
	defer rows.Close()
	result := make([]auditstore.ControllerClaim, 0, params.Limit)
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

func (s *ClaimStore) ReleaseClaim(ctx context.Context, claim auditstore.ControllerClaim) error {
	if err := validateClaimIdentity(claim); err != nil {
		return err
	}
	err := auditClaims.Release(ctx, s.db, claim.AuditID, claim.HolderID, int64(claim.Epoch))
	if errors.Is(err, persistencepostgres.ErrControllerClaimLost) {
		return auditstore.ErrClaimLost
	}
	if err != nil {
		return fmt.Errorf("release Audit claim: %w", err)
	}
	return nil
}

func scanClaim(row scanner) (auditstore.ControllerClaim, error) {
	claim, err := persistencepostgres.ScanControllerClaim(row)
	if err != nil {
		return auditstore.ControllerClaim{}, err
	}
	return auditstore.ControllerClaim{
		AuditID: claim.SubjectID, HolderID: claim.HolderID, Epoch: uint64(claim.Epoch),
		ClaimedAt: claim.ClaimedAt, ExpiresAt: claim.ExpiresAt,
	}, nil
}
