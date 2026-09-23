package postgres

import (
	"context"
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
	tx, err := BeginTx(ctx, pool, options)
	if err != nil {
		return WrapError("begin PostgreSQL transaction", err)
	}
	defer func() {
		rollbackCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
		rollbackErr := tx.Rollback(rollbackCtx)
		if rollbackErr != nil && rollbackErr != pgx.ErrTxClosed && err == nil {
			err = WrapError("rollback PostgreSQL transaction", rollbackErr)
		}
	}()

	if err := ApplyTransactionBudget(ctx, tx); err != nil {
		return err
	}
	if err := fn(tx); err != nil {
		return err
	}
	if err := tx.Commit(ctx); err != nil {
		return WrapError("commit PostgreSQL transaction", err)
	}
	return nil
}

// BeginTx starts a transaction for a caller that must own and classify its
// commit itself. Like InTx, the transaction runs AfterCommit effects only after
// a definite commit; the caller must Commit or Rollback it.
func BeginTx(ctx context.Context, pool *pgxpool.Pool, options pgx.TxOptions) (pgx.Tx, error) {
	tx, err := pool.BeginTx(ctx, options)
	if err != nil {
		return nil, err
	}
	return &commitTx{Tx: tx}, nil
}
