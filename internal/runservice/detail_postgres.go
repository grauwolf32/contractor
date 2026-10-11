package runservice

import (
	"context"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	plannersession "github.com/grauwolf32/contractor/internal/planner/session"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// PostgresRunDetailTransaction reads each Run detail in one read-only
// REPEATABLE READ transaction of pool. Every read observes the snapshot of
// the first one, so a transition committed between two reads is invisible to
// all of them. The transaction takes no row locks and writes nothing, so it
// cannot fail with a serialization conflict and is never retried. Every
// reader uses the transaction's connection, never a second one from pool.
func PostgresRunDetailTransaction(pool *pgxpool.Pool) RunDetailTransaction {
	return func(ctx context.Context, fn func(RunDetailStores) error) error {
		if pool == nil {
			return fmt.Errorf("%w: Run detail database is missing", ErrNotConfigured)
		}
		return persistencepostgres.InTx(ctx, pool, pgx.TxOptions{
			IsoLevel: pgx.RepeatableRead, AccessMode: pgx.ReadOnly,
		}, func(tx pgx.Tx) error {
			runs := runstore.NewPostgresStore(tx)
			plans, err := plannersession.New(runs, plannersession.Options{})
			if err != nil {
				return err
			}
			return fn(RunDetailStores{
				Runs: runs, Deletion: rundeletion.New(tx),
				Artifacts: artifacts.NewService(artifacts.NewPostgresRepository(tx)),
				Plans:     plans, Metrics: savepointStageMetrics{tx: tx},
				Resources: telemetry.NewRepository(tx),
				Recovery: func(ctx context.Context, runID string) (*gatewayrecovery.Status, error) {
					return gatewayrecovery.ReadStatus(ctx, tx, runID)
				},
			})
		})
	}
}

// savepointStageMetrics reads the optional StageMetrics inside a savepoint. A
// failed statement aborts the whole PostgreSQL transaction; rolling back to
// the savepoint keeps the snapshot usable for the durable Run state.
type savepointStageMetrics struct {
	tx pgx.Tx
}

func (m savepointStageMetrics) GetStageMetricsBatch(
	ctx context.Context, ids []string,
) (map[string]telemetry.StageMetricsRecord, error) {
	var result map[string]telemetry.StageMetricsRecord
	err := pgx.BeginFunc(ctx, m.tx, func(savepoint pgx.Tx) error {
		var err error
		result, err = telemetry.NewRepository(savepoint).GetStageMetricsBatch(ctx, ids)
		return err
	})
	return result, err
}
