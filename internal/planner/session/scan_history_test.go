package session

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

type scanHistoryTestStore struct {
	*fencedScanStore
	run     runstore.WorkflowRun
	stages  []runstore.StageExecution
	readErr error
}

func (s *scanHistoryTestStore) GetRun(context.Context, string) (runstore.WorkflowRun, error) {
	return s.run, s.readErr
}

func (s *scanHistoryTestStore) ListStageExecutions(context.Context, string) ([]runstore.StageExecution, error) {
	return s.stages, s.readErr
}

func newScanHistoryFixture(t *testing.T) *scanHistoryTestStore {
	t.Helper()
	snapshot, err := config.Load("../../../configs", config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("audit-openapi-sqlmap-scan@1")
	if err != nil {
		t.Fatal(err)
	}
	ordinary, err := snapshot.Workflow("artifact-copy@2")
	if err != nil {
		t.Fatal(err)
	}
	before := ordinary.Stages[ordinary.EntryStage]
	before.Context = config.StageContext{Artifacts: map[string]config.ContextArtifact{}}
	before.WorkflowOutputs = map[string]string{}
	before.On.Succeeded = config.TransitionAction{Kind: config.TransitionNext, NextStage: "check-request"}
	scan := workflow.Stages["scan"]
	workflow.Stages = map[string]config.ResolvedStage{"prepare": before, "check-request": scan}
	workflow.EntryStage = "prepare"
	workflow.AuditTask.Stage = "check-request"
	if err := config.ValidateWorkflowGraph(workflow); err != nil {
		t.Fatal(err)
	}
	encode := func(value any) []byte {
		t.Helper()
		data, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		return data
	}
	store, service := newScanTestService(t)
	store.execution.RunID, store.execution.StageName = "audit-run", "check-request"
	store.execution.StageSpecSnapshot = encode(scan)
	started := beginScanTest(t, service)
	if err := service.InitializeScan(t.Context(), started.Identity, scanTestPlan("assigned-request")); err != nil {
		t.Fatal(err)
	}
	claimScanTest(t, service, started.Identity, "assigned-request")
	// Model an interrupted Stage whose scanner acknowledgement was lost.
	store.execution.State = runstore.StageInterrupted
	executionID := "audit-execution"
	return &scanHistoryTestStore{
		fencedScanStore: store,
		run:             runstore.WorkflowRun{RunID: "audit-run", PublicationMode: runstore.PublicationAuditManaged, AuditExecutionID: &executionID, WorkflowSnapshot: encode(workflow)},
		stages: []runstore.StageExecution{
			{RunID: "audit-run", StageExecutionID: "before-scan", StageName: "prepare", State: runstore.StageSucceeded, StageSpecSnapshot: encode(before)},
			store.execution,
		},
	}
}

func TestScanHistoryFollowsDeclaredExecutorWithinComposedWorkflow(t *testing.T) {
	store := newScanHistoryFixture(t)
	history, err := ReadAuditScanHistory(t.Context(), store, store.run.RunID)
	if err != nil || len(history) != 1 {
		t.Fatalf("scan history: %+v, %v", history, err)
	}
	if history[0].StageName != "check-request" || !history[0].Terminal || !planner.ScanAttemptNeedsRecovery(history[0]) {
		t.Fatalf("executor outcome was lost: %+v", history[0])
	}
	history[0].State.Jobs[0].Status = planner.ScanJobFailed
	again, err := ReadAuditScanHistory(t.Context(), store, store.run.RunID)
	if err != nil || again[0].State.Jobs[0].Status != planner.ScanJobStarted {
		t.Fatal("read-only projection changed the durable journal", err)
	}
}

func TestScanHistoryCannotTurnInvalidAuthorityIntoEmptyHistory(t *testing.T) {
	for name, change := range map[string]func(*scanHistoryTestStore){
		"read failure":              func(s *scanHistoryTestStore) { s.readErr = errors.New("database unavailable") },
		"ordinary Run":              func(s *scanHistoryTestStore) { s.run.AuditExecutionID = nil },
		"invalid Workflow snapshot": func(s *scanHistoryTestStore) { s.run.WorkflowSnapshot = []byte(`{}`) },
		"invalid scan snapshot":     func(s *scanHistoryTestStore) { s.stages[1].StageSpecSnapshot = []byte(`{}`) },
		"foreign Stage":             func(s *scanHistoryTestStore) { s.stages[1].RunID = "other-run" },
		"lost session":              func(s *scanHistoryTestStore) { s.session.SessionID = "other-session" },
	} {
		t.Run(name, func(t *testing.T) {
			store := newScanHistoryFixture(t)
			change(store)
			if _, err := ReadAuditScanHistory(t.Context(), store, store.run.RunID); err == nil {
				t.Fatal("invalid history became retryable")
			}
		})
	}
}
