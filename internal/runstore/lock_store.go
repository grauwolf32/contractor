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
	state, _, err := s.lockRunOwner(ctx, runID, "", expected...)
	return state, err
}

// LockClaimedRunState locks the Run and checks the exact Scheduler claim in
// the same read. Expiry alone does not revoke an unchanged live owner's claim
// (SCH-R120); replacement or release does. The caller retains this lock until
// every lifecycle mutation commits or rolls back.
func (s *PostgresTxStore) LockClaimedRunState(
	ctx context.Context, runID, claimID string, expected ...WorkflowRunState,
) (WorkflowRunState, error) {
	if err := validateOpaque("claimID", claimID); err != nil {
		return "", err
	}
	state, _, err := s.lockRunOwner(ctx, runID, claimID, expected...)
	return state, err
}

// lockRunOwner is LockRunState that also returns the Run's owner from the
// same locking read, so a caller that authorizes the owner under the Run row
// lock does not read the row again.
func (s *PostgresTxStore) lockRunOwner(
	ctx context.Context, runID, claimID string, expected ...WorkflowRunState,
) (WorkflowRunState, string, error) {
	if err := validateOpaque("runID", runID); err != nil {
		return "", "", err
	}
	var state WorkflowRunState
	var ownerID string
	var actualClaimID *string
	err := s.tx.QueryRow(ctx, `SELECT state, owner_id, scheduler_claim_id FROM workflow_runs WHERE run_id = $1 FOR UPDATE`, runID).
		Scan(&state, &ownerID, &actualClaimID)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", "", fmt.Errorf("lock WorkflowRun %q: %w", runID, ErrNotFound)
	}
	if err != nil {
		return "", "", fmt.Errorf("lock WorkflowRun %q: %w", runID, err)
	}
	if !state.Valid() {
		return "", "", fmt.Errorf("lock WorkflowRun %q: stored state %q is not a WorkflowRun state", runID, state)
	}
	if len(expected) > 0 && !slices.Contains(expected, state) {
		return "", "", &StateConflictError{Resource: "WorkflowRun", ID: runID, Expected: fmt.Sprint(expected)}
	}
	if claimID != "" && (actualClaimID == nil || *actualClaimID != claimID) {
		return "", "", &StateConflictError{Resource: "WorkflowRun claim", ID: runID, Expected: claimID}
	}
	return state, ownerID, nil
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
