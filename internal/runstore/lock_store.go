package runstore

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"github.com/jackc/pgx/v5"
)

// LockRunState locks the row of Run runID until the caller's transaction
// ends and returns the Run's state. Every transaction that changes a Run or
// its StageExecutions together with other state takes this lock first and
// LockStageState after it, so they serialize on the Run row in one order
// (RUN-R29). With expected states the Run must be in one of them, or
// LockRunState returns a StateConflictError; a missing Run is ErrNotFound.
func (s *PostgresTxStore) LockRunState(
	ctx context.Context, runID string, expected ...WorkflowRunState,
) (WorkflowRunState, error) {
	if err := validateOpaque("runID", runID); err != nil {
		return "", err
	}
	var state WorkflowRunState
	err := s.tx.QueryRow(ctx, `SELECT state FROM workflow_runs WHERE run_id = $1 FOR UPDATE`, runID).Scan(&state)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", fmt.Errorf("lock WorkflowRun %q: %w", runID, ErrNotFound)
	}
	if err != nil {
		return "", fmt.Errorf("lock WorkflowRun %q: %w", runID, err)
	}
	if !state.Valid() {
		return "", fmt.Errorf("lock WorkflowRun %q: stored state %q is not a WorkflowRun state", runID, state)
	}
	if len(expected) > 0 && !slices.Contains(expected, state) {
		return "", &StateConflictError{Resource: "WorkflowRun", ID: runID, Expected: fmt.Sprint(expected)}
	}
	return state, nil
}

// LockStageState locks the row of StageExecution stageExecutionID until the
// caller's transaction ends and returns its state. The caller locked the row
// of its Run runID first (LockRunState). The Stage must belong to that Run
// and, with expected states, be in one of them; otherwise LockStageState
// returns a StateConflictError. A missing Stage is ErrNotFound.
func (s *PostgresTxStore) LockStageState(
	ctx context.Context, runID, stageExecutionID string, expected ...StageExecutionState,
) (StageExecutionState, error) {
	if err := validateOpaque("runID", runID); err != nil {
		return "", err
	}
	if err := validateOpaque("stageExecutionID", stageExecutionID); err != nil {
		return "", err
	}
	var actualRunID string
	var state StageExecutionState
	err := s.tx.QueryRow(ctx, `
SELECT run_id, state FROM stage_executions WHERE stage_execution_id = $1 FOR UPDATE`,
		stageExecutionID).Scan(&actualRunID, &state)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", fmt.Errorf("lock StageExecution %q: %w", stageExecutionID, ErrNotFound)
	}
	if err != nil {
		return "", fmt.Errorf("lock StageExecution %q: %w", stageExecutionID, err)
	}
	if actualRunID != runID {
		return "", &StateConflictError{Resource: "StageExecution Run", ID: stageExecutionID, Expected: runID}
	}
	if len(expected) > 0 && !slices.Contains(expected, state) {
		return "", &StateConflictError{Resource: "StageExecution", ID: stageExecutionID, Expected: fmt.Sprint(expected)}
	}
	return state, nil
}
