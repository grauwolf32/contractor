package scanstore

import (
	"context"
	"github.com/grauwolf32/contractor/internal/artifacts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"time"
)

func (s *Store) PurgeExpired(ctx context.Context, now time.Time) error {
	return persistencepostgres.Within(ctx, s.db, pgx.TxOptions{}, func(tx pgx.Tx) error {
		var owner, id string
		err := tx.QueryRow(ctx, expiredSweepSQL, now.Add(-Retention)).Scan(&owner, &id)
		if err == pgx.ErrNoRows {
			return nil
		}
		if err != nil {
			return err
		}
		_, err = tx.Exec(ctx, `UPDATE scan_sweeps SET purging=true WHERE sweep_id=$1`, id)
		if err != nil {
			return err
		}
		return purgeMemberBatch(ctx, tx, owner, id)
	})
}

// purgeMemberBatch holds the terminal parent lock for the whole transaction.
// A partial cleanup leaves purging=true for the next bounded maintenance tick.
func purgeMemberBatch(ctx context.Context, tx pgx.Tx, owner, id string) error {
	tag, err := tx.Exec(ctx, purgeMemberBatchSQL, id, TickMembers)
	if err != nil {
		return err
	}
	if tag.RowsAffected() == TickMembers {
		return nil
	}
	purger, err := artifacts.NewPostgresPurger(tx)
	if err != nil {
		return err
	}
	if err := purger.PurgeSweepInput(ctx, owner, id); err != nil {
		return err
	}
	_, err = tx.Exec(ctx, `DELETE FROM scan_sweeps WHERE sweep_id=$1 AND purging`, id)
	return err
}
