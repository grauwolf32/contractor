package runstore

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
)

// MarkStageAdmitted records the admission of StageExecution
// stageExecutionID and returns the Stage. Its first admission sets
// admitted_at, which starts the Stage's wall-clock budget (RUN-R28); a later
// one keeps it. The caller holds the Run and Stage row locks of the
// admission transaction (LockRunState, LockStageState).
func (s *PostgresStore) MarkStageAdmitted(ctx context.Context, stageExecutionID string) (StageExecution, error) {
	if err := validateOpaque("stageExecutionID", stageExecutionID); err != nil {
		return StageExecution{}, err
	}
	result, err := scanStageExecution(s.db.QueryRow(ctx, markStageAdmittedSQL, stageExecutionID))
	if errors.Is(err, pgx.ErrNoRows) {
		return StageExecution{}, fmt.Errorf("admit StageExecution %q: %w", stageExecutionID, ErrNotFound)
	}
	if err != nil {
		return StageExecution{}, fmt.Errorf("admit StageExecution %q: %w", stageExecutionID, err)
	}
	return result, nil
}

// DiscardUnadmittedAllocations deletes the unreleased allocation rows of
// StageExecution stageExecutionID of Run runID, whose admission Queue Pause
// deferred. The Stage must still be preparing and never admitted: then no
// Runtime prepared those allocations, and the next claim after resume places
// the Stage again instead of finding pinned rows without grants (SCH-R152).
// It locks the Stage row after the caller locked the Run row (LockRunState);
// a Stage that was admitted or left preparing is a StateConflictError.
func (s *PostgresTxStore) DiscardUnadmittedAllocations(ctx context.Context, runID, stageExecutionID string) error {
	if _, err := s.LockStageState(ctx, runID, stageExecutionID); err != nil {
		return err
	}
	var unadmitted bool
	if err := s.tx.QueryRow(ctx, `
SELECT state = 'preparing' AND admitted_at IS NULL FROM stage_executions
WHERE stage_execution_id = $1`, stageExecutionID).Scan(&unadmitted); err != nil {
		return fmt.Errorf("check admission of StageExecution %q: %w", stageExecutionID, err)
	}
	if !unadmitted {
		return &StateConflictError{Resource: "StageExecution", ID: stageExecutionID, Expected: "unadmitted preparing"}
	}
	if _, err := s.tx.Exec(ctx, `
DELETE FROM stage_allocations
WHERE stage_execution_id = $1 AND release_attempted_at IS NULL AND release_completed_at IS NULL`,
		stageExecutionID); err != nil {
		return fmt.Errorf("discard unadmitted allocations of StageExecution %q: %w", stageExecutionID, err)
	}
	return nil
}
