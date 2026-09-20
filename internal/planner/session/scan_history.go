package session

import (
	"context"
	"fmt"
	"sort"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

type ScanHistoryStore interface {
	GetRun(context.Context, string) (runstore.WorkflowRun, error)
	ListStageExecutions(context.Context, string) ([]runstore.StageExecution, error)
	GetPlannerSession(context.Context, string) (runstore.PlannerSession, error)
}

func (s *Service) ReadAuditScanHistory(ctx context.Context, runID string) ([]planner.ScanAttempt, error) {
	store, ok := s.store.(ScanHistoryStore)
	if !ok {
		return nil, fmt.Errorf("Audit scan history store is unavailable")
	}
	return ReadAuditScanHistory(ctx, store, runID)
}

// ReadAuditScanHistory is shared by dispatch recovery and terminal collection.
// Reading errors must never be interpreted as an empty, retryable history.
func ReadAuditScanHistory(ctx context.Context, store ScanHistoryStore, runID string) ([]planner.ScanAttempt, error) {
	run, err := store.GetRun(ctx, runID)
	if err != nil {
		return nil, err
	}
	if run.RunID != runID || run.PublicationMode != runstore.PublicationAuditManaged || run.AuditExecutionID == nil {
		return nil, fmt.Errorf("Audit scan requires an Audit-managed Run")
	}
	workflow, err := config.DecodeResolvedWorkflowSnapshot(run.WorkflowSnapshot)
	if err != nil {
		return nil, err
	}
	if workflow.AuditTask == nil || workflow.AuditTask.Contract != contracts.AuditTaskOpenAPIScanV1 {
		return nil, fmt.Errorf("Run does not declare a scan task executor")
	}
	stages, err := store.ListStageExecutions(ctx, runID)
	if err != nil {
		return nil, err
	}
	result := []planner.ScanAttempt{}
	for _, stage := range stages {
		if stage.RunID != runID {
			return nil, fmt.Errorf("scan history Run identity differs")
		}
		// Ordinary preparation/finalization stages do not contribute scanner
		// outcomes. Recovery follows the executor declared in the frozen Run.
		if stage.StageName != workflow.AuditTask.Stage {
			continue
		}
		spec, err := config.DecodeResolvedStageSnapshot(stage.StageSpecSnapshot)
		if err != nil {
			return nil, err
		}
		if spec.AuditScan == nil {
			return nil, fmt.Errorf("scan task executor Stage has no scan configuration")
		}
		attempt := planner.ScanAttempt{StageExecutionID: stage.StageExecutionID, StageName: stage.StageName, State: planner.ScanState{Jobs: []planner.ScanJobRecord{}}}
		switch stage.State {
		case runstore.StageSucceeded, runstore.StageFailed, runstore.StageInterrupted, runstore.StageCancelled:
			attempt.Terminal = true
		}
		if stage.PlannerSessionID != nil {
			stored, err := store.GetPlannerSession(ctx, *stage.PlannerSessionID)
			if err != nil {
				return nil, err
			}
			if stored.SessionID != *stage.PlannerSessionID || stored.StageExecutionID != stage.StageExecutionID || stored.StateSchemaVersion != contracts.APIVersion ||
				stage.PlannerInvocationID == nil || stored.InvocationID != *stage.PlannerInvocationID {
				return nil, fmt.Errorf("scan history session identity differs")
			}
			state, err := decodeState(stored.State)
			if err != nil {
				return nil, err
			}
			if state.Scan != nil {
				attempt.State = cloneScanState(state.Scan.State)
			}
		}
		result = append(result, attempt)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].StageExecutionID < result[j].StageExecutionID })
	return result, nil
}
