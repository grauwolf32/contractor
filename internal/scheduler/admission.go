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
		store := runstore.NewPostgresTxStore(tx)
		state, err := store.LockRunState(ctx, runID, progressionRunStates...)
		if err != nil {
			return err
		}
		if _, err := store.LockStageState(ctx, runID, stageID); err != nil {
			return err
		}
		if state == runstore.RunPending {
			if err := store.LockRunQueueAdmission(ctx, runID); err != nil {
				return err
			}
			if _, err := store.TransitionRun(ctx, runID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"}); err != nil {
				return err
			}
		}
		stage, err = store.MarkStageAdmitted(ctx, stageID)
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
		store := runstore.NewPostgresTxStore(tx)
		if _, err := store.LockRunState(ctx, runID, runstore.RunPending); err != nil {
			return err
		}
		if err := store.DiscardUnadmittedAllocations(ctx, runID, stageID); err != nil {
			return err
		}
		// Admission may have reserved a recovery probe for this placement.
		if err := gatewayrecovery.ReleaseAdmission(ctx, tx, runID); err != nil {
			return err
		}
		return discard()
	})
}
