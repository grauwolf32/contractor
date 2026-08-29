package session

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestServiceCreatesOneInvocationAndRecoversCompletion(t *testing.T) {
	store := &memoryStore{execution: runstore.StageExecution{
		StageExecutionID: "stage-1", State: runstore.StagePreparing,
	}}
	sequence := 0
	service, err := New(store, Options{NewID: func(prefix string) (string, error) {
		sequence++
		return prefix + string(rune('0'+sequence)), nil
	}})
	if err != nil {
		t.Fatal(err)
	}

	started, err := service.Begin(context.Background(), "stage-1")
	if err != nil || !started.Invoke || started.Completion != nil {
		t.Fatalf("Begin = (%+v, %v)", started, err)
	}
	if _, err := service.Begin(context.Background(), "stage-1"); !errors.Is(err, planner.ErrInvocationInProgress) {
		t.Fatalf("concurrent Begin error = %v", err)
	}
	revision := "input-r1"
	facts := planner.RequestFacts{
		Binding: "builder", ObjectiveDigest: "sha256:" + strings.Repeat("a", 64),
		InstructionsDigest: "sha256:" + strings.Repeat("b", 64),
		ParameterNames:     []string{"mode"},
		Artifacts: map[string]contracts.ArtifactRef{
			"source": {Namespace: "inputs", Name: "source", Revision: &revision},
		},
	}
	if err := service.RecordRequest(context.Background(), started.Identity, facts); err != nil {
		t.Fatal(err)
	}
	if err := service.RecordRequest(context.Background(), started.Identity, facts); err != nil {
		t.Fatalf("idempotent RecordRequest: %v", err)
	}
	resultRevision := "result-r1"
	result := contracts.StageContentResult{
		APIVersion: contracts.APIVersion, Outcome: contracts.StageSucceeded, Summary: "done",
		Artifacts: map[string]contracts.ArtifactRef{
			"report": {Namespace: "builder", Name: "report", Revision: &resultRevision},
		},
	}
	completion := planner.Completion{Result: &result}
	if err := service.Complete(context.Background(), started.Identity, completion); err != nil {
		t.Fatal(err)
	}
	if err := service.Complete(context.Background(), started.Identity, completion); err != nil {
		t.Fatalf("idempotent Complete: %v", err)
	}
	if len(store.events) != 2 || store.events[0].SequenceNumber != 1 ||
		store.events[1].SequenceNumber != 2 {
		t.Fatalf("events = %+v", store.events)
	}
	if strings.Contains(string(store.events[0].Event), "strict") ||
		strings.Contains(string(store.events[0].Event), "Build a report") {
		t.Fatalf("request event retained semantic text or values: %s", store.events[0].Event)
	}

	recovered, err := service.Begin(context.Background(), "stage-1")
	if err != nil || recovered.Invoke || recovered.Completion == nil ||
		!reflect.DeepEqual(*recovered.Completion.Result, result) {
		t.Fatalf("recovered Begin = (%+v, %v)", recovered, err)
	}
	if store.startCalls != 1 {
		t.Fatalf("StartPlanner calls = %d", store.startCalls)
	}
}

type memoryStore struct {
	execution  runstore.StageExecution
	session    runstore.PlannerSession
	events     []runstore.AppendPlannerEventParams
	startCalls int
}

func (s *memoryStore) GetStageExecution(
	context.Context, string,
) (runstore.StageExecution, error) {
	return s.execution, nil
}

func (s *memoryStore) StartPlanner(
	_ context.Context, params runstore.StartPlannerParams,
) error {
	s.startCalls++
	if s.execution.State != runstore.StagePreparing {
		return &runstore.StateConflictError{
			Resource: "StageExecution", ID: params.StageExecutionID,
			Expected: string(runstore.StagePreparing),
		}
	}
	s.execution.State = runstore.StageRunning
	s.execution.PlannerSessionID = &params.SessionID
	s.execution.PlannerInvocationID = &params.InvocationID
	s.session = runstore.PlannerSession{
		SessionID: params.SessionID, StageExecutionID: params.StageExecutionID,
		InvocationID: params.InvocationID, StateSchemaVersion: params.StateSchemaVersion,
		State: append(json.RawMessage(nil), params.InitialState...),
	}
	return nil
}

func (s *memoryStore) GetPlannerSession(
	_ context.Context, sessionID string,
) (runstore.PlannerSession, error) {
	if s.session.SessionID != sessionID {
		return runstore.PlannerSession{}, runstore.ErrNotFound
	}
	result := s.session
	result.State = append(json.RawMessage(nil), s.session.State...)
	return result, nil
}

func (s *memoryStore) AppendPlannerEvent(
	_ context.Context, params runstore.AppendPlannerEventParams,
) error {
	if params.SequenceNumber != int64(len(s.events)+1) {
		return runstore.ErrConflict
	}
	s.events = append(s.events, params)
	s.session.StateSchemaVersion = params.NewStateSchemaVersion
	s.session.State = append(json.RawMessage(nil), params.NewState...)
	return nil
}
