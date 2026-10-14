package controlplane

import (
	"context"
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
)

type failedPlacementLockTx struct {
	pgx.Tx
	cancelled bool
}

func (tx failedPlacementLockTx) QueryRow(ctx context.Context, _ string, _ ...any) pgx.Row {
	if tx.cancelled {
		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		return tx.Tx.QueryRow(cancelled, `SELECT 1`)
	}
	return tx.Tx.QueryRow(ctx, `SELECT placement_lock_failure()`)
}

func TestPlacementLockPreservesDatabaseFailure(t *testing.T) {
	ctx := t.Context()
	pool := isolatedPlacementPool(t, ctx)
	fixture := newPlacementFixture(t, ctx, pool, nil)
	if _, err := pool.Exec(ctx, `CREATE FUNCTION placement_lock_failure() RETURNS text LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'private SQL payload' USING ERRCODE = 'serialization_failure';
END $$`); err != nil {
		t.Fatal(err)
	}
	for _, cancelled := range []bool{false, true} {
		attempts := 0
		err := persistencepostgres.InTxWithRetry(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			attempts++
			return fixture.allocator.pinReservations(ctx, failedPlacementLockTx{Tx: tx, cancelled: cancelled},
				fixture.request(), nil, nil, nil,
				map[string]runtimeconfig.ResolvedRuntimeConfig{}, map[string]reporting.PerformanceCollectionPolicy{})
		})
		if cancelled {
			if !errors.Is(err, context.Canceled) || attempts != 1 {
				t.Fatalf("cancelled lock: attempts=%d, error=%v", attempts, err)
			}
		} else if !persistencepostgres.IsTransactionConflict(err) || attempts != 3 {
			t.Fatalf("conflicting lock: attempts=%d, error=%v", attempts, err)
		}
	}
}
