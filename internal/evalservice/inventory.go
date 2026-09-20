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
		var err error
		out, err = evalstore.NewTxStore(tx).InventoryPage(ctx, owner, id, member, after, limit, revision)
		return err
	})
	return out, err
}
