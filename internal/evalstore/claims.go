package evalstore

import (
	"context"
	"errors"
	"time"

	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/evaldomain"
)

type Claim struct {
	ExperimentID, HolderID string
	Epoch                  int64
	ExpiresAt              time.Time
}

// evalClaims is the Eval controller's claim table, leased and checked through
// the shared controller claim helper (PER-R24).
var evalClaims = pg.NewControllerClaims("eval_controller_claims", "experiment_id")

// Claim polls only work that can advance an Eval. A paused Eval with no
// command or dirty projection needs no tick until its wall deadline passes.
// That deadline is compared with the stable statement timestamp: a volatile
// clock cannot be an index condition, so every poll would read every pause.
func (s *Store) Claim(ctx context.Context, holder string, lease time.Duration, limit int) ([]Claim, error) {
	if !contracts.ValidResourceID(holder) || !pg.ValidControllerLease(lease) || limit < 1 || limit > evaldomain.MaxPageSize {
		return nil, evaldomain.Failure("eval_invalid")
	}
	rows, err := s.db.Query(ctx, claimStatement, holder, lease.Milliseconds(), limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]Claim, 0)
	for rows.Next() {
		c, err := pg.ScanControllerClaim(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, Claim{ExperimentID: c.SubjectID, HolderID: c.HolderID, Epoch: c.Epoch, ExpiresAt: c.ExpiresAt})
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
	err := evalClaims.Lock(ctx, s.tx, id, c.HolderID, c.Epoch)
	if errors.Is(err, pg.ErrControllerClaimLost) {
		return ErrClaimLost
	}
	return err
}

func (s *Store) ReleaseClaim(ctx context.Context, c Claim) error {
	err := evalClaims.Release(ctx, s.db, c.ExperimentID, c.HolderID, c.Epoch)
	if errors.Is(err, pg.ErrControllerClaimLost) {
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
