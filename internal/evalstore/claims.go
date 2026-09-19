package evalstore

import (
	"context"
	"errors"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/jackc/pgx/v5"
	"time"
)

type Claim struct {
	ExperimentID, HolderID string
	Epoch                  int64
	ExpiresAt              time.Time
}

func (s *Store) Claim(ctx context.Context, holder string, lease time.Duration, limit int) ([]Claim, error) {
	if !resourceID.MatchString(holder) || lease < time.Second || lease > 5*time.Minute || limit < 1 || limit > 100 {
		return nil, evaldomain.Failure("eval_invalid")
	}
	rows, err := s.db.Query(ctx, `WITH candidates AS (
 SELECT c.experiment_id FROM eval_controller_claims c JOIN eval_experiments e USING(experiment_id)
 WHERE (e.state IN ('preparing','running','settling','pausing','paused','cancelling','interrupted') OR EXISTS (SELECT 1 FROM eval_commands cmd WHERE cmd.experiment_id=e.experiment_id AND cmd.state IN ('accepted','running'))) AND (c.holder_id IS NULL OR c.expires_at<=clock_timestamp())
 ORDER BY c.epoch,e.updated_at,e.experiment_id FOR UPDATE OF c SKIP LOCKED LIMIT $3
 ) UPDATE eval_controller_claims c SET epoch=epoch+1,holder_id=$1,expires_at=clock_timestamp()+$2::bigint*interval '1 millisecond'
 FROM candidates x WHERE c.experiment_id=x.experiment_id RETURNING c.experiment_id,c.holder_id,c.epoch,c.expires_at`, holder, lease.Milliseconds(), limit)
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
	err := s.db.QueryRow(ctx, `SELECT true FROM eval_controller_claims WHERE experiment_id=$1 AND holder_id=$2 AND epoch=$3 AND expires_at>clock_timestamp() FOR UPDATE`, id, c.HolderID, c.Epoch).Scan(&ok)
	if errors.Is(err, pgx.ErrNoRows) {
		return ErrClaimLost
	}
	return err
}
func (s *Store) RenewClaim(ctx context.Context, c Claim, lease time.Duration) (Claim, error) {
	if lease < time.Second || lease > 5*time.Minute {
		return Claim{}, evaldomain.Failure("eval_invalid")
	}
	err := s.db.QueryRow(ctx, `UPDATE eval_controller_claims SET expires_at=clock_timestamp()+$4::bigint*interval '1 millisecond' WHERE experiment_id=$1 AND holder_id=$2 AND epoch=$3 AND expires_at>clock_timestamp() RETURNING expires_at`, c.ExperimentID, c.HolderID, c.Epoch, lease.Milliseconds()).Scan(&c.ExpiresAt)
	if errors.Is(err, pgx.ErrNoRows) {
		err = ErrClaimLost
	}
	return c, err
}
func (s *Store) ReleaseClaim(ctx context.Context, c Claim) error {
	tag, err := s.db.Exec(ctx, `UPDATE eval_controller_claims SET holder_id=NULL,expires_at=NULL WHERE experiment_id=$1 AND holder_id=$2 AND epoch=$3`, c.ExperimentID, c.HolderID, c.Epoch)
	if err == nil && tag.RowsAffected() != 1 {
		return ErrClaimLost
	}
	return err
}
