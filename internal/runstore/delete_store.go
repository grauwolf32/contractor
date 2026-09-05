package runstore

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// DeleteReleasedTerminalRun permanently removes one owner-visible terminal
// Run and its Run-owned data. A pool-backed store creates the transaction; a
// transaction-backed store participates in its caller's transaction so Project
// lifecycle deletion can compose this operation atomically.
func (s *PostgresStore) DeleteReleasedTerminalRun(
	ctx context.Context,
	ownerID string,
	runID string,
) error {
	if err := validateOpaque("ownerID", ownerID); err != nil {
		return err
	}
	if err := validateOpaque("runID", runID); err != nil {
		return err
	}
	switch db := s.db.(type) {
	case *pgxpool.Pool:
		return persistencepostgres.InTx(ctx, db, pgx.TxOptions{}, func(tx pgx.Tx) error {
			return deleteReleasedTerminalRun(ctx, tx, ownerID, runID)
		})
	case pgx.Tx:
		return deleteReleasedTerminalRun(ctx, db, ownerID, runID)
	default:
		return fmt.Errorf("delete WorkflowRun: PostgreSQL transaction support is required")
	}
}

func deleteReleasedTerminalRun(ctx context.Context, tx pgx.Tx, ownerID, runID string) error {
	var state WorkflowRunState
	err := tx.QueryRow(ctx, `
SELECT state
FROM workflow_runs
WHERE run_id = $1 AND owner_id = $2
FOR UPDATE`, runID, ownerID).Scan(&state)
	if errors.Is(err, pgx.ErrNoRows) {
		return fmt.Errorf("delete WorkflowRun %q: %w", runID, ErrNotFound)
	}
	if err != nil {
		return fmt.Errorf("lock WorkflowRun %q for deletion: %w", runID, err)
	}
	if !RunLifecycleTerminal.Includes(state) {
		return &RunNotDeletableError{RunID: runID, Reason: RunNotTerminal}
	}

	var pendingRelease bool
	if err := tx.QueryRow(ctx, `
SELECT EXISTS (
    SELECT 1
    FROM stage_executions AS execution
    JOIN stage_allocations AS allocation
      ON allocation.stage_execution_id = execution.stage_execution_id
    WHERE execution.run_id = $1
      AND allocation.release_completed_at IS NULL
)`, runID).Scan(&pendingRelease); err != nil {
		return fmt.Errorf("check WorkflowRun %q allocation release: %w", runID, err)
	}
	if pendingRelease {
		return &RunNotDeletableError{RunID: runID, Reason: RunAllocationReleasePending}
	}

	purger, err := artifacts.NewPostgresPurger(tx)
	if err != nil {
		return fmt.Errorf("prepare WorkflowRun %q Artifact purge: %w", runID, err)
	}
	if err := purger.PurgeRun(ctx, runID); err != nil {
		return fmt.Errorf("purge WorkflowRun %q Artifacts: %w", runID, err)
	}
	tag, err := tx.Exec(ctx, `
DELETE FROM workflow_runs
WHERE run_id = $1 AND owner_id = $2`, runID, ownerID)
	if err != nil {
		return fmt.Errorf("delete WorkflowRun %q: %w", runID, err)
	}
	if tag.RowsAffected() != 1 {
		return fmt.Errorf("delete WorkflowRun %q: %w", runID, ErrConflict)
	}
	return nil
}
