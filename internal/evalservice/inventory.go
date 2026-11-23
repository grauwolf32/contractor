package evalservice

import (
	"context"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

func (s *Service) Executions(ctx context.Context, owner, id, member, after string, limit int, revision *int64) (evalstore.InventoryPage, error) {
	var out evalstore.InventoryPage
	if !validPage(-1, limit) {
		return out, evaldomain.Failure("eval_invalid")
	}
	err := pg.InTx(ctx, s.pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead, AccessMode: pgx.ReadOnly}, func(tx pgx.Tx) error {
		st := evalstore.NewTxStore(tx, s.storage)
		if err := refusePurging(ctx, st, owner, id); err != nil {
			return err
		}
		var err error
		out, err = st.InventoryPage(ctx, owner, id, member, after, limit, revision)
		return err
	})
	return out, err
}

// refusePurging answers a member-level read of an experiment whose purge may
// already have removed some of its rows with eval_project_deleting, never with
// a partial result (EVM-R244).
func refusePurging(ctx context.Context, st *evalstore.Store, owner, id string) error {
	purging, err := st.Purging(ctx, owner, id)
	if err == nil && purging {
		err = evaldomain.Failure("eval_project_deleting")
	}
	return err
}
