package evalstore

import (
	"context"
	"errors"
	"time"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/jackc/pgx/v5"
)

type Claim struct {
	ExperimentID, HolderID string
	Epoch                  int64
	ExpiresAt              time.Time
}

// Claim polls only work that can advance an Eval. A paused Eval with no
// command or dirty projection needs no tick until its wall deadline passes.
// That deadline is compared with the stable statement timestamp: a volatile
// clock cannot be an index condition, so every poll would read every pause.
func (s *Store) Claim(ctx context.Context, holder string, lease time.Duration, limit int) ([]Claim, error) {
	if !resourceID.MatchString(holder) || lease < time.Second || lease > 5*time.Minute || limit < 1 || limit > evaldomain.MaxPageSize {
		return nil, evaldomain.Failure("eval_invalid")
	}
	rows, err := s.db.Query(ctx, claimStatement, holder, lease.Milliseconds(), limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]Claim, 0)
	for rows.Next() {
		var c Claim
		if err = rows.Scan(&c.ExperimentID, &c.HolderID, &c.Epoch, &c.ExpiresAt); err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, rows.Err()
}

func (s *Store) checkClaim(ctx context.Context, id string, c Claim) error {
	if s.tx == nil {
		return ErrTransaction
	}
	if c.ExperimentID != id || c.HolderID == "" || c.Epoch < 1 {
		return ErrClaimLost
	}
	var ok bool
	err := s.db.QueryRow(ctx, `
SELECT TRUE FROM eval_controller_claims
WHERE experiment_id = $1 AND holder_id = $2 AND epoch = $3 AND expires_at > clock_timestamp()
FOR UPDATE
`, id, c.HolderID, c.Epoch).Scan(&ok)
	if errors.Is(err, pgx.ErrNoRows) {
		return ErrClaimLost
	}
	return err
}

func (s *Store) ReleaseClaim(ctx context.Context, c Claim) error {
	tag, err := s.db.Exec(ctx, `
UPDATE eval_controller_claims
SET holder_id = NULL,expires_at = NULL
WHERE experiment_id = $1
    AND holder_id = $2
    AND epoch = $3
`, c.ExperimentID, c.HolderID, c.Epoch)
	if err == nil && tag.RowsAffected() != 1 {
		return ErrClaimLost
	}
	return err
}

// ClaimTargetExists distinguishes a stale lease from successful experiment purge.
func (s *Store) ClaimTargetExists(ctx context.Context, claim Claim) (bool, error) {
	var exists bool
	err := s.db.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM eval_experiments WHERE experiment_id=$1)`, claim.ExperimentID).Scan(&exists)
	return exists, err
}
