package postgres

import (
	"context"
	"github.com/jackc/pgx/v5"
)

// AfterCommit schedules best-effort effects only after a definite commit.
// False means the caller owns an unwrapped transaction; it must arrange its
// own cleanup. Never run effects after rollback or an ambiguous commit error.
func AfterCommit(tx pgx.Tx, effect func()) bool {
	current, ok := tx.(*commitTx)
	if !ok {
		return false
	}
	current.effects = append(current.effects, effect)
	return true
}

type commitTx struct {
	pgx.Tx
	effects []func()
}

func (tx *commitTx) Commit(ctx context.Context) error {
	err := tx.Tx.Commit(ctx)
	if err == nil {
		for _, f := range tx.effects {
			f()
		}
	}
	tx.effects = nil
	return err
}
func (tx *commitTx) Rollback(ctx context.Context) error { tx.effects = nil; return tx.Tx.Rollback(ctx) }
