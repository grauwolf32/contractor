package runstore

import (
	"context"

	"github.com/grauwolf32/contractor/internal/config/resolved"
)

// LockStageOperation fences a Server operation against Run cancellation, claim
// loss and Stage progression. Callers retain both locks until their mutation
// commits. All reads use this transaction; no connection is acquired beside it.
func (s *PostgresTxStore) LockStageOperation(ctx context.Context, runID, stageID, claimID string) (StageExecution, error) {
	if validateOpaque("claimID", claimID) != nil {
		return StageExecution{}, ErrInvalid
	}
	if _, err := s.LockRunState(ctx, runID, RunRunning); err != nil {
		return StageExecution{}, err
	}
	var owned bool
	if err := s.tx.QueryRow(ctx, `SELECT COALESCE(scheduler_claim_id=$2 AND scheduler_claim_expires_at>clock_timestamp(),false) FROM workflow_runs WHERE run_id=$1`, runID, claimID).Scan(&owned); err != nil {
		return StageExecution{}, err
	}
	if !owned {
		return StageExecution{}, ErrConflict
	}
	if _, err := s.LockStageState(ctx, runID, stageID, StagePreparing, StageRunning); err != nil {
		return StageExecution{}, err
	}
	execution, err := s.GetStageExecution(ctx, stageID)
	if err != nil {
		return StageExecution{}, err
	}
	stage, err := resolved.DecodeResolvedStageSnapshot(execution.StageSpecSnapshot)
	if err != nil || stage.Operation == nil || execution.AdmittedAt == nil {
		return StageExecution{}, ErrInvalid
	}
	return execution, nil
}

// StartStageOperation uses the ordinary running state without creating a
// Planner session. A recovered running operation keeps its original admission.
func (s *PostgresTxStore) StartStageOperation(ctx context.Context, runID, stageID, claimID string) (StageExecution, error) {
	execution, err := s.LockStageOperation(ctx, runID, stageID, claimID)
	if err != nil || execution.State == StageRunning {
		return execution, err
	}
	_, err = s.tx.Exec(ctx, `UPDATE stage_executions SET state='running', state_reason_code='server_operation_started', state_reason_message='', updated_at=clock_timestamp() WHERE stage_execution_id=$1`, stageID)
	if err != nil {
		return StageExecution{}, err
	}
	return s.GetStageExecution(ctx, stageID)
}
