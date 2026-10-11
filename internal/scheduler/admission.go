package scheduler

import (
	"context"
	"fmt"

	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

// AdmitStage starts the wall-clock budget only after resource placement succeeds.
// The Run/Stage lock order agrees with cancellation and all finalization commits.
func (p *PostgresPersistence) AdmitStage(ctx context.Context, runID, stageID string) (runstore.StageExecution, error) {
	var stage runstore.StageExecution
	err := persistencepostgres.InTx(ctx, p.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := lockRunState(ctx, tx, runID, progressionRunStates...); err != nil {
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

// DiscardUnadmittedPlacement undoes the durable placement of a Stage whose
// admission Queue Pause deferred. The Stage was never admitted, so no Runtime
// prepared its allocations; deleting their rows together with the live
// reservation lets the next claim after resume place it again instead of
// finding pinned rows without grants (SCH-R152). Run and Stage are locked in
// the order every commit point uses.
func (p *PostgresPersistence) DiscardUnadmittedPlacement(
	ctx context.Context, runID, stageID string, discard func() error,
) error {
	if discard == nil {
		return fmt.Errorf("placement discard is required")
	}
	return persistencepostgres.InTx(ctx, p.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := lockRunState(ctx, tx, runID, runstore.RunPending); err != nil {
			return err
		}
		if err := lockStageForRun(ctx, tx, stageID, runID); err != nil {
			return err
		}
		var unadmitted bool
		if err := tx.QueryRow(ctx, `
SELECT state='preparing' AND admitted_at IS NULL FROM stage_executions
WHERE stage_execution_id=$1`, stageID).Scan(&unadmitted); err != nil {
			return err
		}
		if !unadmitted {
			return &runstore.StateConflictError{Resource: "StageExecution", ID: stageID, Expected: "unadmitted preparing"}
		}
		if _, err := tx.Exec(ctx, `
DELETE FROM stage_allocations
WHERE stage_execution_id=$1 AND release_attempted_at IS NULL AND release_completed_at IS NULL`, stageID); err != nil {
			return err
		}
		// Admission may have reserved a recovery probe for this placement.
		if err := gatewayrecovery.ReleaseAdmission(ctx, tx, runID); err != nil {
			return err
		}
		return discard()
	})
}
