package session

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
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
	snapshot, err := configload.Load("../../../testdata/configs-scan")
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("audit-openapi-sqlmap-scan@1")
	if err != nil {
		t.Fatal(err)
	}
	// Any Stage without an auditScan executor is an ordinary Stage here.
	ordinary, err := snapshot.Workflow("nuclei-target@1")
	if err != nil {
		t.Fatal(err)
	}
	before := ordinary.Stages[ordinary.EntryStage]
	before.Context = resolved.StageContext{Artifacts: map[string]resolved.ContextArtifact{}}
	before.WorkflowOutputs = map[string]string{}
	before.On.Succeeded = resolved.TransitionAction{Kind: resolved.TransitionNext, NextStage: "check-request"}
	scan := workflow.Stages["scan"]
	workflow.Stages = map[string]resolved.ResolvedStage{"prepare": before, "check-request": scan}
	// The composed Workflow declares the parameter its ordinary Stage consumes.
	workflow.Parameters = ordinary.Parameters
	workflow.EntryStage = "prepare"
	workflow.AuditTask.Stage = "check-request"
	if err := resolved.ValidateWorkflowGraph(workflow); err != nil {
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

// batchCountingScanStore serves many scan attempts and records every session
// read, so the history read cannot fall back to one read per attempt.
type batchCountingScanStore struct {
	*scanHistoryTestStore
	sessions map[string]runstore.PlannerSession
	batches  [][]string
}

func (s *batchCountingScanStore) GetPlannerSessions(
	_ context.Context, sessionIDs []string,
) (map[string]runstore.PlannerSession, error) {
	s.batches = append(s.batches, append([]string(nil), sessionIDs...))
	result := make(map[string]runstore.PlannerSession, len(sessionIDs))
	for _, id := range sessionIDs {
		if session, ok := s.sessions[id]; ok {
			result[id] = session
		}
	}
	return result, nil
}

func TestScanHistoryReadsEveryAttemptSessionInOneBatch(t *testing.T) {
	const attempts = 64
	fixture := newScanHistoryFixture(t)
	executor := fixture.stages[1]
	store := &batchCountingScanStore{
		scanHistoryTestStore: fixture, sessions: make(map[string]runstore.PlannerSession, attempts),
	}
	stages := []runstore.StageExecution{fixture.stages[0]}
	for index := range attempts {
		stageID := fmt.Sprintf("scan-attempt-%03d", index)
		sessionID, invocationID := "session-"+stageID, "invocation-"+stageID
		stage := executor
		stage.StageExecutionID, stage.PlannerSessionID, stage.PlannerInvocationID = stageID, &sessionID, &invocationID
		stages = append(stages, stage)
		session := fixture.session
		session.SessionID, session.StageExecutionID, session.InvocationID = sessionID, stageID, invocationID
		store.sessions[sessionID] = session
	}
	fixture.stages = stages
	history, err := ReadAuditScanHistory(t.Context(), store, fixture.run.RunID)
	if err != nil || len(history) != attempts {
		t.Fatalf("scan history = %d attempts, %v; want %d", len(history), err, attempts)
	}
	if len(store.batches) != 1 || len(store.batches[0]) != attempts {
		t.Fatalf("session reads = %d batches, want one batch of %d sessions", len(store.batches), attempts)
	}
	delete(store.sessions, "session-scan-attempt-007")
	if _, err := ReadAuditScanHistory(t.Context(), store, fixture.run.RunID); !errors.Is(err, runstore.ErrNotFound) {
		t.Fatalf("history with a lost attempt session = %v, want ErrNotFound", err)
	}
}
