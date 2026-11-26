package streamline

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/contracts/runlabels"
	plannermemory "github.com/grauwolf32/contractor/internal/memory"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/planner"
	plannersession "github.com/grauwolf32/contractor/internal/planner/session"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"github.com/jackc/pgx/v5/pgxpool"
)

type markerProbeSessions struct {
	*plannersession.Service
	mu      sync.Mutex
	markers int
	lostAck bool
}

func (s *markerProbeSessions) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.markers
}

const compactorPolicyID = "compactor-test"

func compactionInvocation(profile string) planner.Invocation {
	invocation := testInvocation("builder")
	invocation.Stage.Planner = resolved.PlannerRef{PlannerID: profile, Version: "1"}
	main := invocation.ModelAccess.ModelPolicy
	main.ContextWindowTokens, main.MaxOutputTokens = 4096, 512
	invocation.ModelAccess.ModelPolicy = main
	compactor := contracts.ResolvedModelPolicy{
		Ref: contracts.ModelPolicyRef{
			PolicyID: compactorPolicyID, Version: "1", Digest: "sha256:" + strings.Repeat("e", 64),
		},
		Model: "fake-compactor", ContextWindowTokens: 131_072, MaxOutputTokens: 1024,
		MaxModelCalls: 1, MaxTotalTokens: 140_000,
	}
	invocation.Stage.PlannerContextManagement = &resolved.PlannerContextManagement{
		Strategy: resolved.PlannerContextCompact,
		Instructions: contracts.ResolvedInstructions{
			Ref: "instructions/compact.md", Text: "Keep decisions and open work.",
			Digest: "sha256:" + strings.Repeat("d", 64),
		},
		ContextWindowRatio: 0.75, KeepRecentGroups: 1, MaxCompactions: 8,
	}
	invocation.Stage.ExecutionConfig.Planner = &resolved.ResolvedConsumerExecutionConfig{
		ModelPolicy: main, CompactorModelPolicy: &compactor,
		Origins: resolved.ExecutionConfigOrigins{CompactorModelPolicy: "workflow.executionConfig.planner"},
	}
	return invocation
}

func compactionInspector() *fakeInspector {
	return &fakeInspector{mediaTypes: map[string]string{
		refKey(exactRef("builder", "review", "report-r1")): "application/json",
		refKey(exactRef("builder", "review", "report-r2")): "application/json",
	}}
}

func installRequiredEventGuard(t *testing.T, ctx context.Context, pool *pgxpool.Pool, phase, fault string) {
	t.Helper()
	prefix := ""
	if fault == "unavailable" {
		prefix = "NEW."
	}
	predicate := fmt.Sprintf("%skind = 'planner.activity' AND %sdata #> '{activity,functionResults}' ? 'finish'", prefix, prefix)
	if phase == "ordinary" {
		predicate = fmt.Sprintf("%skind = 'planner.activity' AND %sdata #>> '{activity,author}' IN ('streamline_planner', 'router_planner')", prefix, prefix)
	}
	if fault == "invalid" {
		if _, err := pool.Exec(ctx, "ALTER TABLE workflow_run_events ADD CONSTRAINT v419_required_event_rejected CHECK (NOT ("+predicate+"))"); err != nil {
			t.Fatal(err)
		}
		return
	}
	// This is a PostgreSQL-returned, injected connection-failure condition. It
	// does not disconnect or stop a shared PostgreSQL service.
	statement := fmt.Sprintf(`CREATE FUNCTION v419_required_event_outage() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF %s THEN
        RAISE EXCEPTION 'controlled required event outage' USING ERRCODE = '%s';
    END IF;
    RETURN NEW;
END $$`, predicate, persistencepostgres.SQLStateConnectionFailure)
	if _, err := pool.Exec(ctx, statement); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `CREATE TRIGGER v419_required_event_outage BEFORE INSERT ON workflow_run_events FOR EACH ROW EXECUTE FUNCTION v419_required_event_outage()`); err != nil {
		t.Fatal(err)
	}
}

func appendProbeInvocation(profile string) planner.Invocation {
	invocation := testInvocation("builder")
	invocation.Stage.Planner = resolved.PlannerRef{PlannerID: profile, Version: "1"}
	invocation.Stage.Result.Artifacts = map[string]resolved.ArtifactSlot{}
	invocation.ModelAccess.ModelPolicy.MaxModelCalls = 4
	return invocation
}

func appendProbeWorkers() *fakeWorkerInvoker {
	return &fakeWorkerInvoker{results: map[string]stageresult.StageContentResult{
		"builder": stageResult("Completed bounded work", map[string]contracts.ArtifactRef{}),
	}}
}

func migrationBoundarySession(t *testing.T, store plannersession.Store) *plannersession.Service {
	t.Helper()
	sessions, err := plannersession.New(store, plannersession.Options{})
	if err != nil {
		t.Fatal(err)
	}
	return sessions
}

type fakeWorkerInvoker struct {
	mu          sync.Mutex
	results     map[string]stageresult.StageContentResult
	completions map[string]contracts.WorkerCompletion
	failures    map[string][]error
	calls       []workerInvocation
}

func (w *fakeWorkerInvoker) Invoke(
	ctx context.Context,
	binding string,
	_ contracts.WorkerHandle,
	request contracts.StageContentRequest,
) (contracts.WorkerCompletion, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	deadline, _ := ctx.Deadline()
	w.calls = append(w.calls, workerInvocation{binding: binding, request: request, deadline: deadline})
	if failures := w.failures[binding]; len(failures) > 0 {
		w.failures[binding] = failures[1:]
		return contracts.WorkerCompletion{}, failures[0]
	}
	if completion, ok := w.completions[binding]; ok {
		if completion.Result != nil {
			result := planner.CloneWorkerResult(*completion.Result)
			completion.Result = &result
		}
		if completion.Failure != nil {
			failure := *completion.Failure
			completion.Failure = &failure
		}
		return completion, nil
	}
	result, ok := w.results[binding]
	if !ok {
		return contracts.WorkerCompletion{}, errors.New("unexpected Worker")
	}
	return workerCompletion(request.SubtaskID, result.Summary, result.Artifacts), nil
}

func (w *fakeWorkerInvoker) bindingCalls() []string {
	w.mu.Lock()
	defer w.mu.Unlock()
	result := make([]string, 0, len(w.calls))
	for _, call := range w.calls {
		result = append(result, call.binding)
	}
	return result
}

type fakeInspector struct {
	mediaTypes map[string]string
}

func (i *fakeInspector) Inspect(
	_ context.Context, _ string, ref contracts.ArtifactRef,
) (planner.ArtifactMetadata, error) {
	if refKey(ref) == refKey(exactRef("inputs", "source", "source-r1")) {
		return planner.ArtifactMetadata{MediaType: "text/plain"}, nil
	}
	mediaType, ok := i.mediaTypes[refKey(ref)]
	if !ok {
		return planner.ArtifactMetadata{}, artifacts.ErrArtifactNotFound
	}
	return planner.ArtifactMetadata{MediaType: mediaType}, nil
}

func testInvocation(bindings ...string) planner.Invocation {
	sourceRevision := "source-r1"
	stageAgents := make(map[string]resolved.ResolvedAgentBinding, len(bindings))
	workers := make(map[string]contracts.WorkerHandle, len(bindings))
	for _, name := range bindings {
		templateRef := contracts.AgentTemplateRef{
			TemplateID: name, Version: "1", Digest: "sha256:" + strings.Repeat("a", 64),
		}
		runtimeRef := contracts.WorkerRuntimeRef{RuntimeID: "adk", Version: "1"}
		stageAgents[name] = resolved.ResolvedAgentBinding{
			Namespace: name,
			Template: contracts.ResolvedAgentTemplate{
				Ref: templateRef, Runtime: runtimeRef, Description: "Worker " + name,
			},
		}
		workers[name] = contracts.WorkerHandle{
			AllocationID: "allocation-" + name, AgentTemplateRef: templateRef,
			WorkerRuntimeRef: runtimeRef, AgentCard: map[string]any{"name": name},
			LeaseExpiresAt: time.Now().Add(5 * time.Second),
		}
	}
	return planner.Invocation{
		RuntimeExecutor:  &contracts.WorkerHandle{AllocationID: "allocation-planner", WorkerRuntimeRef: contracts.WorkerRuntimeRef{RuntimeID: "planner", Version: "1"}},
		StageExecutionID: "stage-1", RunID: "run-1", Deadline: time.Now().Add(time.Minute),
		ModelAccess: &planner.ModelAccess{
			ModelPolicy: contracts.ResolvedModelPolicy{
				Ref:   contracts.ModelPolicyRef{PolicyID: "planner-test", Version: "1", Digest: "sha256:" + strings.Repeat("b", 64)},
				Model: "fake-streamline-model", MaxOutputTokens: 1024,
				MaxModelCalls: 32, MaxTotalTokens: 200_000, MaxWorkerCalls: 64,
			},
			LLMGateway: llmgateway.ResolvedLLMGatewayConfig{
				Ref:      llmgateway.LLMGatewayConfigRef{GatewayID: "planner-test", Version: "1", Digest: "sha256:" + strings.Repeat("c", 64)},
				Protocol: llmgateway.OpenAICompatibleProtocol, URL: "https://gateway.example/v1",
			},
		},
		Stage: resolved.ResolvedStage{
			Objective:    "Produce a reviewed report",
			Instructions: contracts.ResolvedInstructions{Text: "Use the Workers carefully."},
			Planner:      resolved.PlannerRef{PlannerID: "streamline", Version: "1"},
			Agents:       stageAgents,
			Context: resolved.StageContext{Artifacts: map[string]resolved.ContextArtifact{
				"source": {Namespace: "inputs", Name: "source", Required: true},
			}},
			Result: resolved.StageResultContract{Artifacts: map[string]resolved.ArtifactSlot{
				"report": {Required: true, MediaTypes: []string{"application/json"}},
			}},
		},
		Context: planner.StageContext{
			Parameters: map[string]string{"mode": "strict"},
			Artifacts: map[string]*contracts.ArtifactRef{
				"source": pointerRef(exactRef("inputs", "source", sourceRevision)),
			},
		},
		Workers: workers,
	}
}

func stageResult(summary string, resultArtifacts map[string]contracts.ArtifactRef) stageresult.StageContentResult {
	return stageresult.StageContentResult{
		APIVersion: contracts.APIVersion, Outcome: stageresult.StageSucceeded,
		Summary: summary, Artifacts: resultArtifacts,
	}
}

func workerCompletion(
	subtaskID string,
	result string,
	resultArtifacts map[string]contracts.ArtifactRef,
) contracts.WorkerCompletion {
	return contracts.WorkerCompletion{
		APIVersion: contracts.APIVersion,
		Result: &contracts.WorkerResult{
			SubtaskID: subtaskID, Result: result,
			Observations: contracts.WorkerObservations{
				Profile: contracts.WorkerObservationProfileLeanV1,
				Tools:   map[string]contracts.ToolObservationCount{},
			},
			Artifacts: resultArtifacts, Summarized: false,
		},
		InvocationID: "worker-invocation-1", StateRevision: 2,
	}
}

func assertPlannerCode(t *testing.T, err error, code string) {
	t.Helper()
	var value *planner.Error
	if !errors.As(err, &value) || value.Code != code {
		t.Fatalf("error = %v, want Planner code %q", err, code)
	}
}

func createPreparingStreamlineStage(t *testing.T, ctx context.Context, runs *runstore.PostgresStore) {
	t.Helper()
	if _, err := runstoretest.CreateRun(ctx, runs, runstore.CreateRunParams{
		LaneClass: resolved.LaneClassModel,
		RunID:     "run-1", OwnerID: "owner", WorkflowName: "test", WorkflowVersion: "1",
		WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{}`),
		RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatal(err)
	}
	for _, step := range [][2]runstore.WorkflowRunState{
		{runstore.RunInitializing, runstore.RunPending}, {runstore.RunPending, runstore.RunRunning},
	} {
		if _, err := runs.TransitionRun(ctx, "run-1", step[0], step[1], runstore.Reason{Code: "test"}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := runs.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{
		LaneClass:        resolved.LaneClassModel,
		StageExecutionID: "stage-1", RunID: "run-1", StageName: "review", Attempt: 1,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: json.RawMessage(`{"objective":"build","session":"isolated"}`),
		StageContextSchemaVersion: contracts.APIVersion,
		StageContext:              runstore.StageContextSnapshot{Parameters: map[string]string{}, Artifacts: map[string]runstore.PinnedContextArtifact{}},
	}); err != nil {
		t.Fatal(err)
	}
}

type streamlineMetadataReader func(context.Context, contracts.ArtifactRef) (artifacts.Metadata, error)

func newStreamlineInspector(runID string, read streamlineMetadataReader) (*planner.ArtifactServiceInspector, error) {
	scope, err := artifacts.RunScope(runID)
	if err != nil {
		return nil, err
	}
	return planner.NewArtifactServiceInspector(artifacts.NewService(streamlineMetadataRepository{scope: scope, read: read}))
}

func streamlineTestTelemetry(
	t *testing.T, invocation planner.Invocation,
) (telemetry.PlannerTelemetry, <-chan []byte) {
	t.Helper()
	payload := make(chan []byte, 1)
	collector := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		const maxFixtureOTLPBytes = 65 << 20
		body, err := io.ReadAll(io.LimitReader(request.Body, maxFixtureOTLPBytes+1))
		if err != nil || len(body) > maxFixtureOTLPBytes {
			t.Error("invalid or oversized OTLP fixture payload")
			response.WriteHeader(http.StatusRequestEntityTooLarge)
			return
		}
		payload <- body
		response.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(collector.Close)
	registry, err := telemetry.NewBuiltinPlannerAdapterRegistry()
	if err != nil {
		t.Fatal(err)
	}
	adapter, err := registry.Create(telemetry.PlannerAdapterOTLPHTTP, telemetry.PlannerAdapterSettings{
		Endpoint: collector.URL + "/v1/traces", Headers: map[string]contracts.SecretString{},
		FlushTimeout: time.Second, RunMetadataLabels: runlabels.RunMetadataLabels{},
		Resource: telemetry.PlannerResource{
			RunID: invocation.RunID, StageExecutionID: invocation.StageExecutionID,
			PlannerRef: resolved.StreamlinePlanner,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(adapter.Close)
	return adapter, payload
}

func (unavailableWorkerStateReader) ReadWorkerState(
	context.Context,
	contracts.WorkerHandle,
	string,
) (reporting.WorkerStateReadResult, error) {
	return reporting.WorkerStateReadResult{}, &reporting.WorkerStateReadError{
		Code: "runtime_unavailable", Retryable: true,
	}
}

func pointerRef(value contracts.ArtifactRef) *contracts.ArtifactRef { return &value }

func materialArgs(namespace, name string) map[string]any {
	return map[string]any{"namespace": namespace, "name": name}
}

func cloneTestPlanProjection(value planner.PlannerPlanProjection) planner.PlannerPlanProjection {
	result := value
	result.Subtasks = append([]planner.PlannerSubtask(nil), value.Subtasks...)
	if value.ActiveDispatch != nil {
		active := *value.ActiveDispatch
		result.ActiveDispatch = &active
	}
	return result
}

func (s *markerProbeSessions) RecordActivity(ctx context.Context, id planner.SessionIdentity, activity planner.Activity) error {
	if err := s.Service.RecordActivity(ctx, id, activity); err != nil {
		return err
	}
	if len(activity.FunctionCalls) == 0 && len(activity.FunctionResults) == 0 && activity.InputTokens == 0 && activity.OutputTokens == 0 {
		s.mu.Lock()
		s.markers++
		s.mu.Unlock()
		if s.lostAck {
			return fmt.Errorf("marker acknowledgement lost")
		}
	}
	return nil
}

type plannerMemoryArtifactStore struct {
	mu             sync.Mutex
	bindings       map[string]plannerMemoryArtifact
	nextRevision   int
	clock          time.Time
	semanticWrites int
	conflictNext   bool
}

func newPlannerMemoryArtifactStore() *plannerMemoryArtifactStore {
	return &plannerMemoryArtifactStore{
		bindings: make(map[string]plannerMemoryArtifact),
		clock:    time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC),
	}
}

func (s *plannerMemoryArtifactStore) LoadAll(
	_ context.Context,
	binding plannermemory.Binding,
) ([]artifacts.ReadResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	prefix := binding.RunID + "/" + binding.Namespace + "/" + plannermemory.ArtifactNamePrefix
	result := make([]artifacts.ReadResult, 0)
	for key, stored := range s.bindings {
		if !strings.HasPrefix(key, prefix) {
			continue
		}
		revision := stored.revision
		result = append(result, artifacts.ReadResult{
			Ref: artifacts.ArtifactRef{
				Namespace: binding.Namespace, Name: strings.TrimPrefix(key, binding.RunID+"/"+binding.Namespace+"/"),
				Revision: &revision,
			},
			Payload: artifacts.Payload{
				MediaType: stored.payload.MediaType, Data: append([]byte(nil), stored.payload.Data...),
			},
			BindingCreatedAt: stored.bindingCreatedAt, RevisionCreatedAt: stored.revisionCreatedAt,
		})
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Ref.Name < result[j].Ref.Name })
	if len(result) > plannermemory.MaximumNotes+1 {
		result = result[:plannermemory.MaximumNotes+1]
	}
	return result, nil
}

func (s *plannerMemoryArtifactStore) Read(
	_ context.Context,
	binding plannermemory.Binding,
	ref artifacts.ArtifactRef,
) (artifacts.ReadResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	stored, ok := s.bindings[memoryStoreKey(binding, ref.Name)]
	if !ok {
		return artifacts.ReadResult{}, artifacts.ErrArtifactNotFound
	}
	revision := stored.revision
	return artifacts.ReadResult{
		Ref: artifacts.ArtifactRef{
			Namespace: binding.Namespace, Name: ref.Name, Revision: &revision,
		},
		Payload: artifacts.Payload{
			MediaType: stored.payload.MediaType, Data: append([]byte(nil), stored.payload.Data...),
		},
		BindingCreatedAt: stored.bindingCreatedAt, RevisionCreatedAt: stored.revisionCreatedAt,
	}, nil
}

func (s *plannerMemoryArtifactStore) Write(
	_ context.Context,
	binding plannermemory.Binding,
	target artifacts.ArtifactRef,
	payload artifacts.Payload,
	expectedRevision *string,
) (artifacts.WriteResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.conflictNext {
		s.conflictNext = false
		return artifacts.WriteResult{}, artifacts.ErrArtifactConflict
	}
	key := memoryStoreKey(binding, target.Name)
	current, exists := s.bindings[key]
	if expectedRevision == nil && exists || expectedRevision != nil && (!exists || current.revision != *expectedRevision) {
		return artifacts.WriteResult{}, artifacts.ErrArtifactConflict
	}
	s.nextRevision++
	s.clock = s.clock.Add(time.Microsecond)
	createdAt := current.bindingCreatedAt
	if !exists {
		createdAt = s.clock
	}
	stored := plannerMemoryArtifact{
		payload: artifacts.Payload{
			MediaType: payload.MediaType, Data: append([]byte(nil), payload.Data...),
		},
		revision:          "revision-" + time.Unix(int64(s.nextRevision), 0).UTC().Format("150405"),
		bindingCreatedAt:  createdAt,
		revisionCreatedAt: s.clock,
	}
	s.bindings[key] = stored
	s.semanticWrites++
	revision := stored.revision
	return artifacts.WriteResult{
		Ref: artifacts.ArtifactRef{
			Namespace: binding.Namespace, Name: target.Name, Revision: &revision,
		},
		MediaType: payload.MediaType, Size: int64(len(payload.Data)),
		BindingCreatedAt: createdAt, RevisionCreatedAt: stored.revisionCreatedAt,
	}, nil
}

var _ plannermemory.Store = (*plannerMemoryArtifactStore)(nil)

type streamlineMetadataRepository struct {
	artifacts.Repository
	scope artifacts.Scope
	read  streamlineMetadataReader
}

func (r streamlineMetadataRepository) Metadata(
	ctx context.Context, scope artifacts.Scope, ref contracts.ArtifactRef,
) (artifacts.Metadata, error) {
	if scope != r.scope {
		return artifacts.Metadata{}, artifacts.ErrArtifactNotFound
	}
	return r.read(ctx, ref)
}

type plannerMemoryArtifact struct {
	payload           artifacts.Payload
	revision          string
	bindingCreatedAt  time.Time
	revisionCreatedAt time.Time
}

func memoryStoreKey(binding plannermemory.Binding, name string) string {
	return binding.RunID + "/" + binding.Namespace + "/" + name
}

type fakeSessions struct {
	identity      planner.SessionIdentity
	recovered     *planner.Completion
	completion    *planner.Completion
	request       planner.RequestFacts
	plan          *planner.PlannerPlanProjection
	transitions   []planner.PlannerPlanTransition
	facts         []planner.PlannerFact
	completeCalls int
	completeHook  func(context.Context, planner.Completion) error
}

func newFakeSessions() *fakeSessions {
	return &fakeSessions{identity: planner.SessionIdentity{
		SessionID: "planner-session", StageExecutionID: "stage-1", InvocationID: "planner-invocation",
	}}
}

func (s *fakeSessions) Begin(context.Context, string) (planner.SessionStart, error) {
	if s.recovered != nil {
		return planner.SessionStart{Identity: s.identity, Completion: s.recovered}, nil
	}
	return planner.SessionStart{Identity: s.identity, Invoke: true}, nil
}

func (s *fakeSessions) RecordRequest(
	_ context.Context, _ planner.SessionIdentity, facts planner.RequestFacts,
) error {
	s.request = facts
	return nil
}

func (s *fakeSessions) Complete(
	ctx context.Context, _ planner.SessionIdentity, completion planner.Completion,
) error {
	s.completeCalls++
	if s.completeHook != nil {
		if err := s.completeHook(ctx, completion); err != nil {
			return err
		}
	}
	copy := completion
	s.completion = &copy
	return nil
}

func (s *fakeSessions) RecordPlan(
	_ context.Context,
	_ planner.SessionIdentity,
	transition planner.PlannerPlanTransition,
) error {
	var previous *planner.PlannerPlanProjection
	if s.plan != nil {
		copy := cloneTestPlanProjection(*s.plan)
		previous = &copy
	}
	if err := planner.ValidatePlannerPlanTransition(previous, transition.Plan, transition.Kind); err != nil {
		return err
	}
	if previous == nil && transition.ExpectedRevision != 0 ||
		previous != nil && transition.ExpectedRevision != previous.Revision {
		return runstore.ErrConflict
	}
	copy := cloneTestPlanProjection(transition.Plan)
	s.plan = &copy
	s.transitions = append(s.transitions, transition)
	return nil
}

func (s *fakeSessions) RecordFact(
	_ context.Context,
	_ planner.SessionIdentity,
	fact planner.PlannerFact,
) error {
	s.facts = append(s.facts, fact)
	return nil
}

type workerInvocation struct {
	binding  string
	request  contracts.StageContentRequest
	deadline time.Time
}

type unavailableWorkerStateReader struct{}

func exactRef(namespace, name, revision string) contracts.ArtifactRef {
	return contracts.ArtifactRef{Namespace: namespace, Name: name, Revision: &revision}
}

func refKey(ref contracts.ArtifactRef) string {
	if ref.Revision == nil {
		return ref.Namespace + "/" + ref.Name
	}
	return ref.Namespace + "/" + ref.Name + "@" + *ref.Revision
}
