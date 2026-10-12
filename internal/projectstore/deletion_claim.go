package projectstore

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
)

// DeletionClaim is one Project deletion controller's lease on a deleting
// Project in one deletion phase. Every change made under the claim checks
// that the Project still holds it in that phase; a lost claim is
// ErrDeletionClaimLost.
type DeletionClaim struct {
	ProjectID string
	OwnerID   string
	ClaimID   string
	Phase     DeletionPhase
}

// ClaimDeletionParams lease a deleting Project to ClaimID for Duration, or for
// PurgeDuration while the Project is in a purge phase, whose steps run longer.
type ClaimDeletionParams struct {
	ClaimID       string
	Duration      time.Duration
	PurgeDuration time.Duration
}

// PostgresTxStore is a PostgresStore bound to the caller's transaction. Its
// own methods take row locks that must last until the caller commits, so they
// exist only on a store that cannot be backed by a pool.
type PostgresTxStore struct {
	*PostgresStore
	tx pgx.Tx
}

// NewPostgresTxStore binds a PostgresStore to the caller's transaction tx.
func NewPostgresTxStore(tx pgx.Tx) *PostgresTxStore {
	return &PostgresTxStore{PostgresStore: NewPostgresStore(tx), tx: tx}
}

// ClaimDeletion leases the deleting Project that requested deletion first
// among those whose deletion claim is absent or expired, skipping a Project
// another transaction has locked. It reports false when none is claimable.
func (s *PostgresStore) ClaimDeletion(
	ctx context.Context, params ClaimDeletionParams,
) (DeletionClaim, bool, error) {
	if strings.TrimSpace(params.ClaimID) == "" || params.Duration.Microseconds() <= 0 ||
		params.PurgeDuration.Microseconds() <= 0 {
		return DeletionClaim{}, false, invalid("Project deletion claim is invalid")
	}
	claim := DeletionClaim{ClaimID: params.ClaimID}
	err := s.db.QueryRow(ctx, claimProjectDeletionSQL,
		params.ClaimID, params.Duration.Microseconds(), params.PurgeDuration.Microseconds(),
		DeletionPurgingRuns, DeletionPurgingArtifacts,
	).Scan(&claim.ProjectID, &claim.OwnerID, &claim.Phase)
	if errors.Is(err, pgx.ErrNoRows) {
		return DeletionClaim{}, false, nil
	}
	if err != nil {
		return DeletionClaim{}, false, fmt.Errorf("claim Project deletion: %w", err)
	}
	if !claim.Phase.Valid() {
		return DeletionClaim{}, false, errors.New("claimed Project deletion phase is invalid")
	}
	return claim, true, nil
}

// AdvanceDeletion moves the claimed Project from the claim's phase to next.
func (s *PostgresStore) AdvanceDeletion(ctx context.Context, claim DeletionClaim, next DeletionPhase) error {
	if err := validateDeletionClaim(claim); err != nil {
		return err
	}
	if !next.Valid() {
		return invalid("Project deletion phase is invalid")
	}
	tag, err := s.db.Exec(ctx, `
UPDATE projects
SET deletion_phase = $4,
    revision = revision + 1,
    updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
WHERE project_id = $1 AND owner_id = $2
  AND lifecycle_state = 'deleting' AND deletion_phase = $3
  AND deletion_claim_id = $5`,
		claim.ProjectID, claim.OwnerID, claim.Phase, next, claim.ClaimID,
	)
	if err != nil {
		return fmt.Errorf("advance Project deletion phase: %w", err)
	}
	if tag.RowsAffected() != 1 {
		return ErrDeletionClaimLost
	}
	return nil
}

// ReleaseDeletionClaim ends the claim, so any controller may claim the
// Project again.
func (s *PostgresStore) ReleaseDeletionClaim(ctx context.Context, claim DeletionClaim) error {
	if err := validateDeletionClaim(claim); err != nil {
		return err
	}
	tag, err := s.db.Exec(ctx, `
UPDATE projects
SET deletion_claim_id = NULL,
    deletion_claimed_at = NULL,
    deletion_claim_expires_at = NULL
WHERE project_id = $1 AND owner_id = $2 AND deletion_claim_id = $3`,
		claim.ProjectID, claim.OwnerID, claim.ClaimID,
	)
	if err != nil {
		return fmt.Errorf("release Project deletion claim: %w", err)
	}
	if tag.RowsAffected() != 1 {
		return ErrDeletionClaimLost
	}
	return nil
}

// DeferDeletionClaim keeps the claim until delay from now, so a Project that
// cannot progress yields to other deleting Projects while a restarted
// controller can still claim it once the claim expires.
func (s *PostgresStore) DeferDeletionClaim(ctx context.Context, claim DeletionClaim, delay time.Duration) error {
	if err := validateDeletionClaim(claim); err != nil {
		return err
	}
	if delay < 0 {
		return invalid("Project deletion claim delay is invalid")
	}
	tag, err := s.db.Exec(ctx, `
UPDATE projects
SET deletion_claim_expires_at = GREATEST(
        clock_timestamp() + ($4::bigint * interval '1 microsecond'),
        deletion_claimed_at + interval '1 microsecond'
    )
WHERE project_id = $1 AND owner_id = $2 AND deletion_claim_id = $3`,
		claim.ProjectID, claim.OwnerID, claim.ClaimID, delay.Microseconds(),
	)
	if err != nil {
		return fmt.Errorf("defer Project deletion claim: %w", err)
	}
	if tag.RowsAffected() != 1 {
		return ErrDeletionClaimLost
	}
	return nil
}

// LockDeletionClaim locks the claimed Project FOR UPDATE until the caller's
// transaction ends, while it is deleting in the claim's phase under the
// claim. Work that the claim authorizes in that transaction cannot then race
// a phase change or another controller.
func (s *PostgresTxStore) LockDeletionClaim(ctx context.Context, claim DeletionClaim) error {
	if err := validateDeletionClaim(claim); err != nil {
		return err
	}
	var locked bool
	err := s.tx.QueryRow(ctx, `
SELECT true
FROM projects
WHERE project_id = $1 AND owner_id = $2
  AND lifecycle_state = 'deleting' AND deletion_phase = $3
  AND deletion_claim_id = $4
FOR UPDATE`, claim.ProjectID, claim.OwnerID, claim.Phase, claim.ClaimID).Scan(&locked)
	if errors.Is(err, pgx.ErrNoRows) {
		return ErrDeletionClaimLost
	}
	if err != nil {
		return fmt.Errorf("lock claimed Project deletion: %w", err)
	}
	return nil
}

// DeleteClaimedProject deletes the Project row that LockDeletionClaim locked
// in the same transaction: the last step of a Project purge, after its Runs,
// Audits, finding receipts and Artifacts are gone.
func (s *PostgresTxStore) DeleteClaimedProject(ctx context.Context, claim DeletionClaim) error {
	if err := validateDeletionClaim(claim); err != nil {
		return err
	}
	tag, err := s.tx.Exec(ctx, `DELETE FROM projects WHERE project_id = $1 AND owner_id = $2`,
		claim.ProjectID, claim.OwnerID,
	)
	if err != nil {
		return fmt.Errorf("delete Project: %w", err)
	}
	if tag.RowsAffected() != 1 {
		return ErrDeletionClaimLost
	}
	return nil
}

func validateDeletionClaim(claim DeletionClaim) error {
	if err := validateIdentity(claim.OwnerID, claim.ProjectID); err != nil {
		return err
	}
	if strings.TrimSpace(claim.ClaimID) == "" || !claim.Phase.Valid() {
		return invalid("Project deletion claim is invalid")
	}
	return nil
}
