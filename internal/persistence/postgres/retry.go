package postgres

import (
	"context"
	"errors"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const MaxTransactionAttempts = 3

// InTxWithRetry is opt-in, for database-only operations whose entire callback
// can safely be replayed. Each attempt uses a fresh transaction and snapshot;
// callbacks must reset any captured result state on entry. Never use it around
// external effects. InTx finishes rollback before the next attempt begins.
// Only PostgreSQL's definite serialization/deadlock aborts are retried, never
// business CAS conflicts, timeouts, EOF or ambiguous commit/transport failures.
func InTxWithRetry(ctx context.Context, pool *pgxpool.Pool, options pgx.TxOptions, fn func(pgx.Tx) error) error {
	return retryTransaction(ctx, func() error { return InTx(ctx, pool, options, fn) })
}

func retryTransaction(ctx context.Context, attempt func() error) error {
	for index := 0; ; index++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		err := attempt()
		if err == nil || index+1 == MaxTransactionAttempts || !IsTransactionConflict(err) {
			return err
		}
		timer := time.NewTimer(time.Duration(index+1) * 10 * time.Millisecond)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
}

// IsTransactionConflict identifies server-confirmed transaction aborts. An
// unclassified transport/commit error must never be assumed rolled back.
func IsTransactionConflict(err error) bool {
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	switch SQLState(err) {
	case "40001", "40P01":
		return true
	default:
		return false
	}
}
