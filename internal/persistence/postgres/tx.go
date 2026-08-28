package postgres

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

// DBTX is implemented by pgx pools and transactions. Repositories accept this
// narrow interface so a caller can compose RunStore and ArtifactStore methods
// in one explicit transaction without either repository nesting transactions.
type DBTX interface {
	Exec(context.Context, string, ...any) (pgconn.CommandTag, error)
	Query(context.Context, string, ...any) (pgx.Rows, error)
	QueryRow(context.Context, string, ...any) pgx.Row
}

// InTx runs fn in one caller-visible transaction. Repositories themselves do
// not call this helper; callers decide the atomic boundary and construct their
// transaction-scoped stores from the provided pgx.Tx.
func InTx(
	ctx context.Context,
	pool *pgxpool.Pool,
	options pgx.TxOptions,
	fn func(pgx.Tx) error,
) (err error) {
	tx, err := pool.BeginTx(ctx, options)
	if err != nil {
		return fmt.Errorf("begin PostgreSQL transaction: %w", err)
	}
	defer func() {
		rollbackCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
		rollbackErr := tx.Rollback(rollbackCtx)
		if rollbackErr != nil && rollbackErr != pgx.ErrTxClosed && err == nil {
			err = fmt.Errorf("rollback PostgreSQL transaction: %w", rollbackErr)
		}
	}()

	if err := fn(tx); err != nil {
		return err
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit PostgreSQL transaction: %w", err)
	}
	return nil
}
