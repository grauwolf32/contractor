package postgres

import (
	"context"
	"errors"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

// SavepointQueryRow is QueryRow for a statement whose failure its caller
// recovers from with further statements on db, such as a replay read after a
// unique violation. In a caller's transaction a failed statement aborts the
// whole transaction, so there the statement runs under a savepoint that a
// database error rolls back to; on a pool every statement already is its own
// transaction. The statement runs when the row is scanned.
func SavepointQueryRow(ctx context.Context, db DBTX, sql string, args ...any) pgx.Row {
	return savepointRow{ctx: ctx, db: db, sql: sql, args: args}
}

type savepointRow struct {
	ctx  context.Context
	db   DBTX
	sql  string
	args []any
}

func (r savepointRow) Scan(dest ...any) error {
	tx, ok := r.db.(pgx.Tx)
	if !ok {
		return r.db.QueryRow(r.ctx, r.sql, r.args...).Scan(dest...)
	}
	savepoint, err := tx.Begin(r.ctx)
	if err != nil {
		return WrapError("begin PostgreSQL savepoint", err)
	}
	err = savepoint.QueryRow(r.ctx, r.sql, r.args...).Scan(dest...)
	var postgresError *pgconn.PgError
	if errors.As(err, &postgresError) {
		// Only a database error aborts the transaction. If the rollback to
		// the savepoint fails too, the transaction is unusable, and that is
		// the failure the caller must see instead of starting its recovery.
		rollbackCtx, cancel := context.WithTimeout(context.WithoutCancel(r.ctx), 5*time.Second)
		defer cancel()
		if rollbackErr := savepoint.Rollback(rollbackCtx); rollbackErr != nil {
			return WrapError("roll back to PostgreSQL savepoint", rollbackErr)
		}
		return err
	}
	// No rows or a scan error leave the statement's effects in place, as they
	// would be on a pool.
	if releaseErr := savepoint.Commit(r.ctx); releaseErr != nil {
		return WrapError("release PostgreSQL savepoint", releaseErr)
	}
	return err
}
