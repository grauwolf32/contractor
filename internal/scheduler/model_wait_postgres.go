package scheduler

import (
	"context"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

// ReconcileModelWait is an idempotent Scheduler projection, fenced by its
// claim and exact Stage. No memory lock is held during the transaction.
func (p *PostgresPersistence) ReconcileModelWait(ctx context.Context, runID, stageID, code string) error {
	return persistencepostgres.InTx(ctx, p.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		store := runstore.NewPostgresTxStore(tx, p.storage)
		state, err := lockClaimedRun(ctx, store, runID, abortRunStates...)
		if err != nil {
			return err
		}
		stage, err := store.LockStageState(ctx, runID, stageID)
		if err != nil {
			return err
		}
		if state == runstore.RunCancelling || stage.Terminal() {
			return nil
		}
		if code == "" {
			return store.LeaveWaiting(ctx, runID)
		}
		if state == runstore.RunPending && stage == runstore.StagePreparing {
			return store.RecordAdmissionWait(ctx, runID, code)
		}
		if stage == runstore.StageRunning {
			return store.EnterWaiting(ctx, runID, code)
		}
		return nil
	})
}
