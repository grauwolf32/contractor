package session

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"google.golang.org/adk/model"
	adksession "google.golang.org/adk/session"
	"google.golang.org/genai"
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
		Bindings: []string{"builder"}, ObjectiveDigest: "sha256:" + strings.Repeat("a", 64),
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
	if len(store.events) != 3 || store.events[0].SequenceNumber != 1 ||
		store.events[1].SequenceNumber != 2 || store.events[2].SequenceNumber != 3 {
		t.Fatalf("events = %+v", store.events)
	}
	if strings.Contains(string(store.events[1].Event), "strict") ||
		strings.Contains(string(store.events[1].Event), "Build a report") {
		t.Fatalf("request event retained semantic text or values: %s", store.events[1].Event)
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

func TestADKSessionPersistsOnlyBoundedRedactedEventFacts(t *testing.T) {
	const secret = "sk-provider-error-and-tool-argument"
	const memoryContent = "memory-content-retention-canary"
	const memoryDescription = "memory-description-retention-canary"
	const memoryTag = "memory-tag-retention-canary"
	store := &memoryStore{execution: runstore.StageExecution{
		StageExecutionID: "stage-1", State: runstore.StagePreparing,
	}}
	sequence := 0
	service, err := New(store, Options{NewID: func(prefix string) (string, error) {
		sequence++
		return fmt.Sprintf("%s%d", prefix, sequence), nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	started, err := service.Begin(t.Context(), "stage-1")
	if err != nil {
		t.Fatal(err)
	}
	if err := service.RecordRequest(t.Context(), started.Identity, planner.RequestFacts{
		Bindings: []string{"builder"}, ObjectiveDigest: "sha256:" + strings.Repeat("a", 64),
		InstructionsDigest: "sha256:" + strings.Repeat("b", 64),
		ParameterNames:     []string{}, Artifacts: map[string]contracts.ArtifactRef{},
	}); err != nil {
		t.Fatal(err)
	}
	adk, err := service.NewADKSession(t.Context(), started.Identity, ADKOptions{
		AppName: "contractor_streamline", UserID: "stage-1",
		AllowedTools: []string{"finish", "write_memory", "read_memory"},
	})
	if err != nil {
		t.Fatal(err)
	}
	created, err := adk.Create(t.Context(), &adksession.CreateRequest{
		AppName: "contractor_streamline", UserID: "stage-1", SessionID: started.Identity.SessionID,
	})
	if err != nil {
		t.Fatal(err)
	}
	event := adksession.NewEvent("adk-invocation")
	event.Author = "streamline_planner"
	event.LLMResponse = model.LLMResponse{
		Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
			genai.NewPartFromText("provider payload " + secret),
			genai.NewPartFromFunctionCall("finish", map[string]any{"summary": secret}),
			genai.NewPartFromFunctionCall("write_memory", map[string]any{
				"name": "safe_note", "content": memoryContent,
				"description": memoryDescription, "tags": []any{memoryTag},
			}),
			genai.NewPartFromFunctionResponse("read_memory", map[string]any{
				"name": "safe_note", "content": memoryContent,
				"description": memoryDescription, "tags": []any{memoryTag},
			}),
			genai.NewPartFromFunctionCall(secret, map[string]any{"token": secret}),
		}},
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount: 11, CandidatesTokenCount: 7,
		},
	}
	if err := adk.AppendEvent(t.Context(), created.Session, event); err != nil {
		t.Fatal(err)
	}
	if len(store.events) != 3 {
		t.Fatalf("events = %+v", store.events)
	}
	persisted := string(store.events[2].Event)
	runPersisted := string(store.events[2].RunEvent.Data)
	for _, canary := range []string{secret, memoryContent, memoryDescription, memoryTag} {
		if strings.Contains(persisted, canary) || strings.Contains(runPersisted, canary) ||
			strings.Contains(string(store.session.State), canary) {
			t.Fatalf("ADK facts retained %q: event=%s run=%s", canary, persisted, runPersisted)
		}
	}
	if strings.Contains(persisted, "summary") ||
		!strings.Contains(persisted, `"functionCalls":["finish","write_memory","unknown"]`) ||
		!strings.Contains(persisted, `"functionResults":["read_memory"]`) {
		t.Fatalf("unsafe or incomplete ADK facts: event=%s run=%s", persisted, runPersisted)
	}
	state, err := decodeState(store.session.State)
	if err != nil || state.ADKEventCount != 1 || state.ADKInputTokens != 11 || state.ADKOutputTokens != 7 {
		t.Fatalf("state = (%+v, %v)", state, err)
	}
}

func TestServicePersistsDeterministicTypedPlanAndFacts(t *testing.T) {
	store, service, identity := runningTestService(t)
	controller, err := planner.NewPlannerPlanController("Global objective is joined from the Stage")
	if err != nil {
		t.Fatal(err)
	}

	before := controller.Snapshot()
	first, planErr := controller.AddSubtask("Inspect", "Read the source")
	if planErr != nil {
		t.Fatal(planErr)
	}
	recordPlan(t, service, identity, before, first, planner.PlannerEventPlanChanged)
	recordFact(t, service, identity, planner.PlannerFact{
		Kind: planner.PlannerEventCurrentChanged, Key: "current:1",
		PlanRevision: first.Revision, SubtaskID: first.CurrentSubtaskID,
	})

	before = controller.Snapshot()
	second, planErr := controller.AddSubtask("Review", "Check the result")
	if planErr != nil {
		t.Fatal(planErr)
	}
	recordPlan(t, service, identity, before, second, planner.PlannerEventPlanChanged)

	before = controller.Snapshot()
	claim, planErr := controller.ClaimCurrentSubtask("0", "reviewer")
	if planErr != nil {
		t.Fatal(planErr)
	}
	recordFact(t, service, identity, planner.PlannerFact{
		Kind: planner.PlannerEventDispatchSelected, Key: "selected:" + claim.CallID,
		PlanRevision: before.Revision, SubtaskID: claim.Subtask.ID,
		CallID: claim.CallID, WorkerName: "reviewer",
	})
	started := controller.Snapshot()
	recordPlan(t, service, identity, before, started, planner.PlannerEventDispatchStarted)

	before = controller.Snapshot()
	completed, planErr := controller.CompleteDispatch(claim.CallID, contracts.StageSucceeded)
	if planErr != nil {
		t.Fatal(planErr)
	}
	recordPlan(t, service, identity, before, completed, planner.PlannerEventDispatchCompleted)
	recordFact(t, service, identity, planner.PlannerFact{
		Kind: planner.PlannerEventCurrentChanged, Key: "current:4",
		PlanRevision: completed.Revision, SubtaskID: completed.CurrentSubtaskID,
	})

	loaded, ok, err := service.LoadPlan(t.Context(), identity)
	if err != nil || !ok || loaded.Revision != 4 || loaded.CurrentSubtaskID != "1" ||
		loaded.ActiveDispatch != nil || len(loaded.Subtasks) != 2 ||
		loaded.Subtasks[0].Status != planner.PlannerSubtaskSucceeded ||
		loaded.Subtasks[1].Status != planner.PlannerSubtaskPending {
		t.Fatalf("loaded plan = (%+v, %t, %v)", loaded, ok, err)
	}
	wantKinds := []runstore.RunEventKind{
		runstore.RunEventPlannerStarted,
		runstore.RunEventPlannerRequestRecorded,
		runstore.RunEventPlannerPlanChanged,
		runstore.RunEventPlannerCurrentChanged,
		runstore.RunEventPlannerPlanChanged,
		runstore.RunEventPlannerDispatchSelected,
		runstore.RunEventPlannerDispatchStarted,
		runstore.RunEventPlannerDispatchCompleted,
		runstore.RunEventPlannerCurrentChanged,
	}
	if got := store.runEventKinds(); !reflect.DeepEqual(got, wantKinds) {
		t.Fatalf("Run event kinds = %v, want %v", got, wantKinds)
	}
	for index, event := range store.events {
		if event.SequenceNumber != int64(index+1) {
			t.Fatalf("Planner event %d sequence = %d", index, event.SequenceNumber)
		}
	}
	encoded, err := json.Marshal(loaded)
	if err != nil || strings.Contains(string(encoded), "Global objective") {
		t.Fatalf("durable plan copied global objective: %s (%v)", encoded, err)
	}
}

func TestServiceConcurrentPlanCompareAllowsOneWinner(t *testing.T) {
	store, service, identity := runningTestService(t)
	left := planner.PlannerPlanProjection{
		Revision: 1, CurrentSubtaskID: "0", Subtasks: []planner.PlannerSubtask{{
			ID: "0", Objective: "left", Instructions: "left", Status: planner.PlannerSubtaskPending,
		}},
	}
	right := planner.PlannerPlanProjection{
		Revision: 1, CurrentSubtaskID: "0", Subtasks: []planner.PlannerSubtask{{
			ID: "0", Objective: "right", Instructions: "right", Status: planner.PlannerSubtaskPending,
		}},
	}
	start := make(chan struct{})
	errorsByCall := make(chan error, 2)
	for _, projection := range []planner.PlannerPlanProjection{left, right} {
		projection := projection
		go func() {
			<-start
			errorsByCall <- service.RecordPlan(t.Context(), identity, planner.PlannerPlanTransition{
				Kind: planner.PlannerEventPlanChanged, ExpectedRevision: 0, Plan: projection,
			})
		}()
	}
	close(start)
	var succeeded, conflicted int
	for range 2 {
		err := <-errorsByCall
		switch {
		case err == nil:
			succeeded++
		case errors.Is(err, runstore.ErrConflict):
			conflicted++
		default:
			t.Fatalf("unexpected concurrent append error: %v", err)
		}
	}
	if succeeded != 1 || conflicted != 1 || len(store.snapshotEvents()) != 3 {
		t.Fatalf("succeeded=%d conflicted=%d events=%d", succeeded, conflicted, len(store.snapshotEvents()))
	}
	loaded, ok, err := service.LoadPlan(t.Context(), identity)
	if err != nil || !ok || loaded.Revision != 1 ||
		(loaded.Subtasks[0].Objective != "left" && loaded.Subtasks[0].Objective != "right") {
		t.Fatalf("winning plan = (%+v, %t, %v)", loaded, ok, err)
	}

	transition := planner.PlannerPlanTransition{
		Kind: planner.PlannerEventPlanChanged, ExpectedRevision: 0, Plan: loaded,
	}
	if err := service.RecordPlan(t.Context(), identity, transition); err != nil {
		t.Fatalf("idempotent exact plan append: %v", err)
	}
	transition.Kind = planner.PlannerEventDispatchStarted
	if err := service.RecordPlan(t.Context(), identity, transition); !errors.Is(err, runstore.ErrConflict) {
		t.Fatalf("different event kind reused committed revision: %v", err)
	}
	if len(store.snapshotEvents()) != 3 {
		t.Fatalf("idempotent/conflicting retries appended events: %+v", store.snapshotEvents())
	}
}

func runningTestService(t *testing.T) (*memoryStore, *Service, planner.SessionIdentity) {
	t.Helper()
	store := &memoryStore{execution: runstore.StageExecution{
		StageExecutionID: "stage-plan", State: runstore.StagePreparing,
	}}
	service, err := New(store, Options{})
	if err != nil {
		t.Fatal(err)
	}
	started, err := service.Begin(t.Context(), "stage-plan")
	if err != nil {
		t.Fatal(err)
	}
	if err := service.RecordRequest(t.Context(), started.Identity, planner.RequestFacts{
		Bindings: []string{"reviewer"}, ObjectiveDigest: "sha256:" + strings.Repeat("a", 64),
		InstructionsDigest: "sha256:" + strings.Repeat("b", 64),
		ParameterNames:     []string{}, Artifacts: map[string]contracts.ArtifactRef{},
	}); err != nil {
		t.Fatal(err)
	}
	return store, service, started.Identity
}

func recordPlan(
	t *testing.T,
	service *Service,
	identity planner.SessionIdentity,
	before planner.PlannerPlan,
	after planner.PlannerPlan,
	kind planner.PlannerEventKind,
) {
	t.Helper()
	if err := service.RecordPlan(t.Context(), identity, planner.PlannerPlanTransition{
		Kind: kind, ExpectedRevision: before.Revision, Plan: after.Projection(),
	}); err != nil {
		t.Fatal(err)
	}
}

func recordFact(t *testing.T, service *Service, identity planner.SessionIdentity, fact planner.PlannerFact) {
	t.Helper()
	if err := service.RecordFact(t.Context(), identity, fact); err != nil {
		t.Fatal(err)
	}
	if err := service.RecordFact(t.Context(), identity, fact); err != nil {
		t.Fatalf("idempotent Planner fact: %v", err)
	}
}

type memoryStore struct {
	mu         sync.Mutex
	execution  runstore.StageExecution
	session    runstore.PlannerSession
	events     []runstore.AppendPlannerEventParams
	startCalls int
}

func (s *memoryStore) GetStageExecution(
	context.Context, string,
) (runstore.StageExecution, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.execution, nil
}

func (s *memoryStore) StartPlanner(
	_ context.Context, params runstore.StartPlannerParams,
) error {
	s.mu.Lock()
	defer s.mu.Unlock()
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
		State: append(json.RawMessage(nil), params.InitialState...), NextEventSequence: 2,
	}
	s.events = append(s.events, runstore.AppendPlannerEventParams{
		EventID: params.EventID, SessionID: params.SessionID,
		StageExecutionID: params.StageExecutionID, InvocationID: params.InvocationID,
		SequenceNumber:     1,
		EventSchemaVersion: params.EventSchemaVersion, Event: append(json.RawMessage(nil), params.Event...),
		NewStateSchemaVersion: params.StateSchemaVersion,
		NewState:              append(json.RawMessage(nil), params.InitialState...),
		RunEvent:              params.RunEvent,
	})
	return nil
}

func (s *memoryStore) GetPlannerSession(
	_ context.Context, sessionID string,
) (runstore.PlannerSession, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
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
	s.mu.Lock()
	defer s.mu.Unlock()
	if params.SequenceNumber != s.session.NextEventSequence {
		return runstore.ErrConflict
	}
	s.events = append(s.events, params)
	s.session.StateSchemaVersion = params.NewStateSchemaVersion
	s.session.State = append(json.RawMessage(nil), params.NewState...)
	s.session.NextEventSequence++
	return nil
}

func (s *memoryStore) snapshotEvents() []runstore.AppendPlannerEventParams {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]runstore.AppendPlannerEventParams(nil), s.events...)
}

func (s *memoryStore) runEventKinds() []runstore.RunEventKind {
	events := s.snapshotEvents()
	result := make([]runstore.RunEventKind, 0, len(events))
	for _, event := range events {
		result = append(result, event.RunEvent.Kind)
	}
	return result
}
