package planner

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

func TestPassthroughPlannerInvokesOnceAndRecoversRecordedResult(t *testing.T) {
	revision := "result-r1"
	candidate := contracts.StageContentResult{
		APIVersion: contracts.APIVersion,
		Outcome:    contracts.StageSucceeded,
		Summary:    "report created",
		Artifacts: map[string]contracts.ArtifactRef{
			"report": {Namespace: "builder", Name: "report", Revision: &revision},
		},
	}
	sessions := &memorySessions{}
	worker := &recordingWorker{result: candidate}
	inspector := &fakeInspector{mediaTypes: map[string]string{
		"builder/report/result-r1": "application/json",
	}}
	factory, err := NewPassthroughFactory(sessions, worker, inspector)
	if err != nil {
		t.Fatal(err)
	}
	registry, err := NewRegistry(factory)
	if err != nil {
		t.Fatal(err)
	}
	invocation := testInvocation()
	binding := invocation.Stage.Agents["builder"]
	binding.Template.Toolsets = []contracts.ToolsetSelection{{
		Ref:   contracts.ToolsetRef{ToolsetID: "memory-tools", Version: "1"},
		Tools: []string{"read_memory", "write_memory"},
	}}
	invocation.Stage.Agents["builder"] = binding
	telemetryAdapter, telemetryPayload := passthroughTestTelemetry(t, invocation)
	invocation.Instrumentation = telemetryAdapter.Instrumentation()

	first, err := registry.Create(PassthroughRef, invocation)
	if err != nil {
		t.Fatal(err)
	}
	got, err := first.Run(context.Background())
	if err != nil {
		t.Fatalf("first Run: %v", err)
	}
	if !reflect.DeepEqual(got, candidate) {
		t.Fatalf("result = %+v, want %+v", got, candidate)
	}
	if worker.calls != 1 || sessions.beginCalls != 1 || sessions.requestCalls != 1 ||
		sessions.completeCalls != 1 {
		t.Fatalf(
			"calls worker=%d begin=%d request=%d complete=%d",
			worker.calls, sessions.beginCalls, sessions.requestCalls, sessions.completeCalls,
		)
	}
	if worker.binding != "builder" || worker.handle.AllocationID != "allocation-1" {
		t.Fatalf("Worker address = (%q, %q)", worker.binding, worker.handle.AllocationID)
	}
	if !worker.deadline.Equal(invocation.Deadline) {
		t.Fatalf("Worker deadline = %s, want Stage deadline %s", worker.deadline, invocation.Deadline)
	}
	if worker.request.Objective != invocation.Stage.Objective ||
		worker.request.Instructions != invocation.Stage.Instructions.Text ||
		worker.request.Parameters["mode"] != "strict" ||
		len(worker.request.Artifacts) != 1 || worker.request.Artifacts["source"].Revision == nil {
		t.Fatalf("StageContentRequest = %+v", worker.request)
	}
	if len(sessions.facts.ParameterNames) != 1 || sessions.facts.ParameterNames[0] != "mode" ||
		sessions.facts.ObjectiveDigest == invocation.Stage.Objective ||
		sessions.facts.InstructionsDigest == invocation.Stage.Instructions.Text {
		t.Fatalf("request facts are not redacted/bounded: %+v", sessions.facts)
	}
	report, ok := first.(ReportProvider).ExecutionReport()
	if !ok || !report.Complete || len(report.ToolCalls) != 1 ||
		report.ToolCalls[0].Outcome != contracts.ToolCallSucceeded ||
		report.Metrics.Tools["a2a.invoke"].Calls == nil ||
		*report.Metrics.Tools["a2a.invoke"].Calls != 1 {
		t.Fatalf("Planner execution report = (%+v, %t)", report, ok)
	}
	if export := telemetryAdapter.Flush(t.Context()); !export.Succeeded {
		t.Fatalf("Passthrough telemetry export = %+v", export)
	}
	payload := <-telemetryPayload
	if !bytes.Contains(payload, []byte(telemetry.PlannerSpanSession)) ||
		!bytes.Contains(payload, []byte(telemetry.PlannerSpanWorker)) ||
		bytes.Contains(payload, []byte(invocation.Stage.Objective)) ||
		bytes.Contains(payload, []byte(invocation.Stage.Instructions.Text)) {
		t.Fatalf("Passthrough OTLP payload is incomplete or unsafe: %q", payload)
	}

	recovered, err := registry.Create(PassthroughRef, invocation)
	if err != nil {
		t.Fatal(err)
	}
	got, err = recovered.Run(context.Background())
	if err != nil || !reflect.DeepEqual(got, candidate) {
		t.Fatalf("recovered Run = (%+v, %v)", got, err)
	}
	if worker.calls != 1 || sessions.beginCalls != 2 || sessions.completeCalls != 1 {
		t.Fatalf("recovery invoked work twice: worker=%d sessions=%+v", worker.calls, sessions)
	}
}

func passthroughTestTelemetry(
	t *testing.T, invocation Invocation,
) (telemetry.PlannerTelemetry, <-chan []byte) {
	t.Helper()
	payload := make(chan []byte, 1)
	collector := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		body, _ := io.ReadAll(io.LimitReader(request.Body, 3<<20))
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
		FlushTimeout: time.Second,
		Resource: telemetry.PlannerResource{
			RunID: invocation.RunID, StageExecutionID: invocation.StageExecutionID,
			PlannerRef: PassthroughRef,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(adapter.Close)
	return adapter, payload
}

func TestPassthroughPlannerRejectsInvalidArtifactResults(t *testing.T) {
	tests := []struct {
		name       string
		mutate     func(*contracts.StageContentResult)
		mediaTypes map[string]string
		wantCode   string
	}{
		{
			name: "unversioned",
			mutate: func(result *contracts.StageContentResult) {
				ref := result.Artifacts["report"]
				ref.Revision = nil
				result.Artifacts["report"] = ref
			},
			wantCode: "invalid_worker_result",
		},
		{
			name: "unknown slot",
			mutate: func(result *contracts.StageContentResult) {
				result.Artifacts["unknown"] = result.Artifacts["report"]
				delete(result.Artifacts, "report")
			},
			wantCode: "result_contract_violation",
		},
		{
			name: "missing required slot",
			mutate: func(result *contracts.StageContentResult) {
				delete(result.Artifacts, "report")
			},
			wantCode: "result_contract_violation",
		},
		{
			name:   "incompatible media type",
			mutate: func(*contracts.StageContentResult) {},
			mediaTypes: map[string]string{
				"builder/report/result-r1": "text/plain",
			},
			wantCode: "result_contract_violation",
		},
		{
			name: "reserved Memory binding",
			mutate: func(result *contracts.StageContentResult) {
				ref := result.Artifacts["report"]
				ref.Name = "memory.report"
				result.Artifacts["report"] = ref
			},
			mediaTypes: map[string]string{
				"builder/memory.report/result-r1": "application/json",
			},
			wantCode: "result_contract_violation",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			revision := "result-r1"
			candidate := contracts.StageContentResult{
				APIVersion: contracts.APIVersion,
				Outcome:    contracts.StageSucceeded, Summary: "result",
				Artifacts: map[string]contracts.ArtifactRef{
					"report": {Namespace: "builder", Name: "report", Revision: &revision},
				},
			}
			test.mutate(&candidate)
			sessions := &memorySessions{}
			worker := &recordingWorker{result: candidate}
			inspector := &fakeInspector{mediaTypes: test.mediaTypes}
			factory, _ := NewPassthroughFactory(sessions, worker, inspector)
			instance, err := factory.Create(testInvocation())
			if err != nil {
				t.Fatal(err)
			}

			result, err := instance.Run(context.Background())
			var plannerError *Error
			if !errors.As(err, &plannerError) || plannerError.Code != test.wantCode {
				t.Fatalf("Run = (%+v, %v), want %s", result, err, test.wantCode)
			}
			if sessions.completion == nil || sessions.completion.Failure == nil ||
				sessions.completion.Result != nil {
				t.Fatalf("invalid candidate was not recorded as a Planner failure: %+v", sessions.completion)
			}
		})
	}
}

func TestPassthroughPlannerRecordsTypedFailureAndDoesNotRetryInvocation(t *testing.T) {
	sessions := &memorySessions{}
	worker := &recordingWorker{err: NewError(
		"worker_input_required", "Worker requires unsupported input", false, nil,
	)}
	factory, _ := NewPassthroughFactory(sessions, worker, &fakeInspector{})
	invocation := testInvocation()
	instance, _ := factory.Create(invocation)

	_, err := instance.Run(context.Background())
	var plannerError *Error
	if !errors.As(err, &plannerError) || plannerError.Code != "worker_input_required" {
		t.Fatalf("Run error = %v", err)
	}
	report, ok := instance.(ReportProvider).ExecutionReport()
	if !ok || len(report.ToolCalls) != 1 ||
		report.ToolCalls[0].Outcome != contracts.ToolCallFailed ||
		len(report.Errors) != 1 || report.Errors[0].Code != "worker_input_required" {
		t.Fatalf("failed Planner report = (%+v, %t)", report, ok)
	}
	recovered, _ := factory.Create(invocation)
	_, err = recovered.Run(context.Background())
	if !errors.As(err, &plannerError) || plannerError.Code != "worker_input_required" ||
		worker.calls != 1 {
		t.Fatalf("recovery error/calls = (%v, %d)", err, worker.calls)
	}
}

func TestPassthroughPlannerDeadlineIsBoundedAndPersisted(t *testing.T) {
	sessions := &memorySessions{}
	worker := &recordingWorker{waitForContext: true}
	factory, _ := NewPassthroughFactory(sessions, worker, &fakeInspector{})
	invocation := testInvocation()
	invocation.Deadline = time.Now().Add(25 * time.Millisecond)
	instance, _ := factory.Create(invocation)

	started := time.Now()
	_, err := instance.Run(context.Background())
	var plannerError *Error
	if !errors.As(err, &plannerError) || plannerError.Code != "worker_deadline_exceeded" {
		t.Fatalf("Run error = %v", err)
	}
	if elapsed := time.Since(started); elapsed > time.Second {
		t.Fatalf("deadline took %s", elapsed)
	}
	if sessions.completion == nil || sessions.completion.Failure == nil ||
		sessions.completion.Failure.Code != "worker_deadline_exceeded" {
		t.Fatalf("deadline completion = %+v", sessions.completion)
	}
}

func TestPassthroughFactoryRequiresOneMatchingWorker(t *testing.T) {
	factory, _ := NewPassthroughFactory(&memorySessions{}, &recordingWorker{}, &fakeInspector{})
	invocation := testInvocation()
	invocation.Workers["second"] = invocation.Workers["builder"]
	if _, err := factory.Create(invocation); err == nil {
		t.Fatal("factory accepted multiple Workers")
	}

	invocation = testInvocation()
	invocation.Stage.Planner = workflowconfig.PlannerRef{PlannerID: "streamline", Version: "1"}
	if _, err := factory.Create(invocation); err == nil {
		t.Fatal("factory accepted a Stage configured for another PlannerFactory")
	}
}

func testInvocation() Invocation {
	inputRevision := "input-r1"
	templateRef := contracts.AgentTemplateRef{
		TemplateID: "artifact_builder", Version: "1", Digest: "sha256:" + repeat("a", 64),
	}
	runtimeRef := contracts.WorkerRuntimeRef{RuntimeID: "adk", Version: "1"}
	return Invocation{
		StageExecutionID: "stage-execution-1",
		RunID:            "run-1",
		Stage: workflowconfig.ResolvedStage{
			Objective: "Build a report",
			Instructions: contracts.ResolvedInstructions{
				Ref: "instructions/report.md", Digest: "sha256:" + repeat("b", 64),
				Text: "Read the source and write the report.",
			},
			Planner: workflowconfig.PlannerRef{PlannerID: "passthrough", Version: "1"},
			Agents: map[string]workflowconfig.ResolvedAgentBinding{
				"builder": {
					Template:  contracts.ResolvedAgentTemplate{Ref: templateRef, Runtime: runtimeRef},
					Namespace: "builder",
				},
			},
			Context: workflowconfig.StageContext{Artifacts: map[string]workflowconfig.ContextArtifact{
				"source": {Namespace: "inputs", Name: "source", Required: true},
				"notes":  {Namespace: "inputs", Name: "notes", Required: false},
			}},
			Result: workflowconfig.StageResultContract{Artifacts: map[string]workflowconfig.ArtifactSlot{
				"report": {Required: true, MediaTypes: []string{"application/json"}},
			}},
		},
		Context: StageContext{
			Parameters: map[string]string{"mode": "strict"},
			Artifacts: map[string]*contracts.ArtifactRef{
				"source": {Namespace: "inputs", Name: "source", Revision: &inputRevision},
				"notes":  nil,
			},
		},
		Workers: map[string]contracts.WorkerHandle{
			"builder": {
				AllocationID: "allocation-1", AgentTemplateRef: templateRef,
				WorkerRuntimeRef: runtimeRef, AgentCard: map[string]any{"name": "builder"},
				LeaseExpiresAt: time.Now().Add(5 * time.Second),
			},
		},
		Deadline: time.Now().Add(30 * time.Second),
	}
}

type memorySessions struct {
	beginCalls    int
	requestCalls  int
	completeCalls int
	facts         RequestFacts
	completion    *Completion
}

func (s *memorySessions) Begin(_ context.Context, stageExecutionID string) (SessionStart, error) {
	s.beginCalls++
	identity := SessionIdentity{
		SessionID: "session-1", StageExecutionID: stageExecutionID, InvocationID: "invocation-1",
	}
	if s.completion != nil {
		return SessionStart{Identity: identity, Completion: s.completion}, nil
	}
	return SessionStart{Identity: identity, Invoke: true}, nil
}

func (s *memorySessions) RecordRequest(
	_ context.Context, _ SessionIdentity, facts RequestFacts,
) error {
	s.requestCalls++
	s.facts = facts
	return nil
}

func (s *memorySessions) Complete(
	_ context.Context, _ SessionIdentity, completion Completion,
) error {
	s.completeCalls++
	s.completion = &completion
	return nil
}

type recordingWorker struct {
	calls          int
	binding        string
	handle         contracts.WorkerHandle
	request        contracts.StageContentRequest
	deadline       time.Time
	result         contracts.StageContentResult
	err            error
	waitForContext bool
}

func (w *recordingWorker) Invoke(
	ctx context.Context,
	binding string,
	handle contracts.WorkerHandle,
	request contracts.StageContentRequest,
) (contracts.StageContentResult, error) {
	w.calls++
	w.binding, w.handle, w.request = binding, handle, request
	w.deadline, _ = ctx.Deadline()
	if w.waitForContext {
		<-ctx.Done()
		return contracts.StageContentResult{}, ctx.Err()
	}
	return w.result, w.err
}

type fakeInspector struct {
	mediaTypes map[string]string
}

func (i *fakeInspector) Inspect(
	_ context.Context, _ string, ref contracts.ArtifactRef,
) (ArtifactMetadata, error) {
	if ref.Revision == nil {
		return ArtifactMetadata{}, errors.New("unversioned")
	}
	mediaType, ok := i.mediaTypes[ref.Namespace+"/"+ref.Name+"/"+*ref.Revision]
	if !ok {
		return ArtifactMetadata{}, errors.New("not found")
	}
	return ArtifactMetadata{MediaType: mediaType}, nil
}

func repeat(value string, count int) string {
	result := ""
	for range count {
		result += value
	}
	return result
}
