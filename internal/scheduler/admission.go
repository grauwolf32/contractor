package scheduler

import (
	"context"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

// AdmitStage starts the wall-clock budget only after resource placement succeeds.
// The Run/Stage lock order agrees with cancellation and all finalization commits.
func (p *PostgresPersistence) AdmitStage(ctx context.Context, runID, stageID string) (runstore.StageExecution, error) {
	var stage runstore.StageExecution
	err := persistencepostgres.InTx(ctx, p.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := lockRunState(ctx, tx, runID, runstore.RunPending, runstore.RunRunning, runstore.RunWaiting); err != nil {
			return err
		}
		if err := lockStageForRun(ctx, tx, stageID, runID); err != nil {
			return err
		}
		store := runstore.NewPostgresStore(tx)
		run, err := store.GetRun(ctx, runID)
		if err != nil {
			return err
		}
		if run.State == runstore.RunPending {
			if err := store.LockRunQueueAdmission(ctx, runID); err != nil {
				return err
			}
			if _, err := store.TransitionRun(ctx, runID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"}); err != nil {
				return err
			}
		}
		if _, err := tx.Exec(ctx, `UPDATE stage_executions SET admitted_at=COALESCE(admitted_at,clock_timestamp()) WHERE stage_execution_id=$1`, stageID); err != nil {
			return err
		}
		stage, err = store.GetStageExecution(ctx, stageID)
		return err
	})
	return stage, err
}
