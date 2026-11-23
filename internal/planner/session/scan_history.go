package session

import (
	"context"
	"fmt"
	"sort"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

type ScanHistoryStore interface {
	GetRun(context.Context, string) (runstore.WorkflowRun, error)
	ListStageLineage(context.Context, string, string) ([]runstore.StageLineage, error)
	ReadPlannerSessions(context.Context, []string, func(runstore.PlannerSession) error) error
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
	if run.RunID != runID {
		return nil, fmt.Errorf("scan history Run identity differs")
	}
	workflow, err := resolved.DecodeResolvedWorkflowSnapshot(run.WorkflowSnapshot)
	if err != nil {
		return nil, err
	}
	executor := ""
	for name, stage := range workflow.Stages {
		if stage.ScanPreparation != nil {
			if executor != "" {
				return nil, fmt.Errorf("multiple prepared scan Stages")
			}
			executor = name
		}
	}
	if executor == "" {
		return nil, fmt.Errorf("Run has no prepared scan Stage")
	}
	stages, err := store.ListStageLineage(ctx, runID, executor)
	if err != nil {
		return nil, err
	}
	executors := make([]runstore.StageLineage, 0, len(stages))
	sessionIDs := make([]string, 0, len(stages))
	expected := make(map[string]runstore.StageLineage)
	for _, stage := range stages {
		if stage.RunID != runID {
			return nil, fmt.Errorf("scan history Run identity differs")
		}
		// Ordinary preparation/finalization stages do not contribute scanner
		// outcomes. Recovery follows the executor declared in the frozen Run.
		if stage.StageName != executor {
			continue
		}
		executors = append(executors, stage)
		if stage.PlannerSessionID != nil {
			if _, duplicate := expected[*stage.PlannerSessionID]; duplicate {
				return nil, fmt.Errorf("scan history session identity differs")
			}
			sessionIDs = append(sessionIDs, *stage.PlannerSessionID)
			expected[*stage.PlannerSessionID] = stage
		}
	}
	// Every attempt's session comes from one read, whatever the number of
	// attempts in the Run's bounded Stage history.
	scanStates := make(map[string]*planner.ScanState, len(sessionIDs))
	err = store.ReadPlannerSessions(ctx, sessionIDs, func(stored runstore.PlannerSession) error {
		stage, found := expected[stored.SessionID]
		if !found || stored.StageExecutionID != stage.StageExecutionID || stored.StateSchemaVersion != contracts.APIVersion ||
			stage.PlannerInvocationID == nil || stored.InvocationID != *stage.PlannerInvocationID {
			return fmt.Errorf("scan history session identity differs")
		}
		state, err := decodeState(stored.State)
		if err != nil {
			return err
		}
		scanStates[stored.SessionID] = nil
		if state.Scan != nil {
			scan := cloneScanState(state.Scan.State)
			scanStates[stored.SessionID] = &scan
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	result := []planner.ScanAttempt{}
	for _, stage := range executors {
		attempt := planner.ScanAttempt{
			StageExecutionID: stage.StageExecutionID, StageName: stage.StageName,
			State: planner.ScanState{Jobs: []planner.ScanJobRecord{}}, Terminal: stage.State.Terminal(),
		}
		if stage.PlannerSessionID != nil {
			scan, found := scanStates[*stage.PlannerSessionID]
			if !found {
				return nil, fmt.Errorf("load scan Planner session: %w", runstore.ErrNotFound)
			}
			if scan != nil {
				attempt.State = *scan
			}
		}
		result = append(result, attempt)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].StageExecutionID < result[j].StageExecutionID })
	return result, nil
}
