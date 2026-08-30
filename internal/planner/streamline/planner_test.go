package streamline

import (
	"context"
	"encoding/json"
	"errors"
	"iter"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	plannersession "github.com/grauwolf32/contractor/internal/planner/session"
	"google.golang.org/adk/model"
	adksession "google.golang.org/adk/session"
	"google.golang.org/genai"
)

func TestStreamlineCallsFixedWorkersSequentiallyThenFinishes(t *testing.T) {
	draftRevision, reportRevision := "draft-r1", "report-r1"
	model := &scriptedModel{steps: []modelStep{
		functionStep("worker_analyzer", map[string]any{
			"objective": "Analyze the input", "instructions": "Produce a draft",
			"parameters": map[string]any{"mode": "strict"},
			"artifacts": map[string]any{
				"source": artifactArgs("inputs", "source", "source-r1"),
			},
		}),
		functionStep("worker_reviewer", map[string]any{
			"objective": "Review the draft", "instructions": "Produce the final report",
			"parameters": map[string]any{"mode": "strict"},
			"artifacts": map[string]any{
				"draft": artifactArgs("analysis", "draft", draftRevision),
			},
		}),
		functionStep(finishToolName, map[string]any{
			"outcome": string(contracts.StageSucceeded), "summary": "reviewed", "artifacts": map[string]any{
				"report": artifactArgs("review", "report", reportRevision),
			},
		}),
	}}
	workers := &fakeWorkerInvoker{results: map[string]contracts.StageContentResult{
		"analyzer": stageResult("draft ready", map[string]contracts.ArtifactRef{
			"draft": exactRef("analysis", "draft", draftRevision),
		}),
		"reviewer": stageResult("report ready", map[string]contracts.ArtifactRef{
			"report": exactRef("review", "report", reportRevision),
		}),
	}}
	sessions := newFakeSessions()
	inspector := &fakeInspector{mediaTypes: map[string]string{
		refKey(exactRef("inputs", "source", "source-r1")):    "text/plain",
		refKey(exactRef("analysis", "draft", draftRevision)): "application/json",
		refKey(exactRef("review", "report", reportRevision)): "application/json",
	}}
	factory := mustFactory(t, sessions, workers, inspector, model, Limits{})
	instance, err := factory.Create(testInvocation("analyzer", "reviewer"))
	if err != nil {
		t.Fatal(err)
	}

	result, err := instance.Run(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if result.Outcome != contracts.StageSucceeded || result.Artifacts["report"].Revision == nil ||
		*result.Artifacts["report"].Revision != reportRevision {
		t.Fatalf("result = %+v", result)
	}
	if got := workers.bindingCalls(); !reflect.DeepEqual(got, []string{"analyzer", "reviewer"}) {
		t.Fatalf("Worker calls = %v", got)
	}
	if workers.calls[0].request.Parameters["mode"] != "strict" ||
		workers.calls[1].request.Artifacts["draft"].Revision == nil ||
		*workers.calls[1].request.Artifacts["draft"].Revision != draftRevision {
		t.Fatalf("Worker requests did not preserve structured context: %+v", workers.calls)
	}
	for _, call := range workers.calls {
		if !call.deadline.Equal(instance.(*streamlinePlanner).invocation.Deadline) {
			t.Fatalf("Worker %q deadline = %s, want Stage deadline %s",
				call.binding, call.deadline, instance.(*streamlinePlanner).invocation.Deadline)
		}
	}
	if sessions.completion == nil || sessions.completion.Result == nil {
		t.Fatal("Planner completion was not recorded")
	}
	report, ok := instance.(planner.ReportProvider).ExecutionReport()
	if !ok || report.Metrics.ModelCalls == nil || *report.Metrics.ModelCalls != 3 ||
		report.Metrics.TotalTokens == nil || *report.Metrics.TotalTokens != 60 {
		t.Fatalf("report = %+v", report)
	}
}

func TestStreamlineRejectsUnknownToolAndInvalidFinishThenCorrects(t *testing.T) {
	revision := "report-r1"
	model := &scriptedModel{steps: []modelStep{
		functionStep("worker_not_prepared", map[string]any{
			"objective": "should not run", "instructions": "none",
		}),
		functionStep(finishToolName, map[string]any{
			"outcome": string(contracts.StageSucceeded), "summary": "invalid", "artifacts": map[string]any{
				"undeclared": artifactArgs("builder", "report", revision),
			},
		}),
		functionStep(finishToolName, map[string]any{
			"outcome": string(contracts.StageSucceeded), "summary": "valid", "artifacts": map[string]any{
				"report": artifactArgs("builder", "report", revision),
			},
		}),
	}}
	workers := &fakeWorkerInvoker{}
	sessions := newFakeSessions()
	inspector := &fakeInspector{mediaTypes: map[string]string{
		refKey(exactRef("builder", "report", revision)): "application/json",
	}}
	factory := mustFactory(t, sessions, workers, inspector, model, Limits{})
	instance, err := factory.Create(testInvocation("builder"))
	if err != nil {
		t.Fatal(err)
	}
	result, err := instance.Run(t.Context())
	if err != nil || result.Summary != "valid" {
		t.Fatalf("Run = (%+v, %v)", result, err)
	}
	if len(workers.calls) != 0 {
		t.Fatalf("unknown Worker caused side effects: %+v", workers.calls)
	}
	report, _ := instance.(planner.ReportProvider).ExecutionReport()
	if report.Metrics.ModelCalls == nil || *report.Metrics.ModelCalls != 3 ||
		len(report.Errors) == 0 || report.Errors[0].Code != "planner_tool_selection_invalid" {
		t.Fatalf("report = %+v", report)
	}
}

func TestStreamlineFinishesWithFailedCandidate(t *testing.T) {
	model := &scriptedModel{steps: []modelStep{functionStep(finishToolName, map[string]any{
		"outcome": string(contracts.StageFailed), "summary": "analysis could not be completed",
		"artifacts": map[string]any{},
		"error": map[string]any{
			"code": "insufficient_evidence", "message": "required evidence was unavailable", "retryable": false,
		},
	})}}
	sessions := newFakeSessions()
	instance, err := mustFactory(
		t, sessions, &fakeWorkerInvoker{}, &fakeInspector{}, model, Limits{},
	).Create(testInvocation("builder"))
	if err != nil {
		t.Fatal(err)
	}
	result, err := instance.Run(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if result.Outcome != contracts.StageFailed || result.Error == nil ||
		result.Error.Code != "insufficient_evidence" || result.Error.Retryable {
		t.Fatalf("failed result = %+v", result)
	}
	if sessions.completion == nil || sessions.completion.Result == nil ||
		sessions.completion.Result.Outcome != contracts.StageFailed {
		t.Fatalf("completion = %+v", sessions.completion)
	}
}

func TestStreamlineRejectsInvalidFinishShapesThenAcceptsCorrection(t *testing.T) {
	model := &scriptedModel{steps: []modelStep{
		functionStep(finishToolName, map[string]any{
			"outcome": string(contracts.StageSucceeded), "summary": "invalid success",
			"artifacts": map[string]any{},
			"error":     map[string]any{"code": "unexpected", "message": "must be rejected", "retryable": false},
		}),
		functionStep(finishToolName, map[string]any{
			"outcome": string(contracts.StageFailed), "summary": "missing error", "artifacts": map[string]any{},
		}),
		functionStep(finishToolName, map[string]any{
			"outcome": "unknown", "summary": "unknown outcome", "artifacts": map[string]any{},
		}),
		functionStep(finishToolName, map[string]any{
			"outcome": string(contracts.StageFailed), "summary": "valid failure", "artifacts": map[string]any{},
			"error": map[string]any{"code": "not_completed", "message": "work remains", "retryable": true},
		}),
	}}
	instance, err := mustFactory(
		t, newFakeSessions(), &fakeWorkerInvoker{}, &fakeInspector{}, model, Limits{},
	).Create(testInvocation("builder"))
	if err != nil {
		t.Fatal(err)
	}
	result, err := instance.Run(t.Context())
	if err != nil || result.Outcome != contracts.StageFailed || result.Summary != "valid failure" {
		t.Fatalf("Run = (%+v, %v)", result, err)
	}
	report, _ := instance.(planner.ReportProvider).ExecutionReport()
	metrics := report.Metrics.Tools[finishToolName]
	if metrics.Calls == nil || *metrics.Calls != 4 || metrics.Failed == nil || *metrics.Failed != 3 ||
		metrics.Succeeded == nil || *metrics.Succeeded != 1 {
		t.Fatalf("finish metrics = %+v", metrics)
	}
	for _, call := range report.ToolCalls {
		outcome, _ := call.Arguments["outcome"].(string)
		if outcome != "succeeded" && outcome != "failed" && outcome != "invalid" {
			t.Fatalf("unsafe outcome persisted: %+v", call.Arguments)
		}
	}
}

func TestStreamlineHasNoModelFacingEscalateTool(t *testing.T) {
	instance, err := mustFactory(
		t, newFakeSessions(), &fakeWorkerInvoker{}, &fakeInspector{}, &scriptedModel{}, Limits{},
	).Create(testInvocation("builder"))
	if err != nil {
		t.Fatal(err)
	}
	streamline := instance.(*streamlinePlanner)
	tools, allowed, err := streamline.buildTools(newExecutionState(streamline.limits))
	if err != nil {
		t.Fatal(err)
	}
	for _, current := range tools {
		if current.Name() == "escalate" {
			t.Fatal("model-facing escalate tool is present")
		}
	}
	if _, exists := allowed["escalate"]; exists {
		t.Fatal("model-facing escalate tool is allowlisted")
	}
	if _, exists := allowed[finishToolName]; !exists || strings.Contains(streamline.systemInstruction(), "call escalate") {
		t.Fatalf("completion tools=%v instruction=%q", allowed, streamline.systemInstruction())
	}
}

func TestStreamlineStopsWhenModelCallBudgetIsExhausted(t *testing.T) {
	model := &scriptedModel{fallback: textStep("not finished")}
	sessions := newFakeSessions()
	factory := mustFactory(
		t, sessions, &fakeWorkerInvoker{}, &fakeInspector{}, model,
		Limits{MaxModelCalls: 2, MaxTokens: 1_000, MaxWorkerCalls: 4, MaxWallTime: time.Minute},
	)
	instance, err := factory.Create(testInvocation("builder"))
	if err != nil {
		t.Fatal(err)
	}
	_, err = instance.Run(t.Context())
	assertPlannerCode(t, err, "planner_model_call_limit")
	if model.callCount() != 2 || sessions.completion == nil || sessions.completion.Failure == nil ||
		sessions.completion.Failure.Code != "planner_model_call_limit" {
		t.Fatalf("calls=%d completion=%+v", model.callCount(), sessions.completion)
	}
}

func TestStreamlineStopsWhenTokenBudgetIsExceeded(t *testing.T) {
	model := &scriptedModel{fallback: func(*model.LLMRequest) (*model.LLMResponse, error) {
		response, _ := textStep("not finished")(nil)
		response.UsageMetadata = &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount: 40, CandidatesTokenCount: 20, TotalTokenCount: 60,
		}
		return response, nil
	}}
	factory := mustFactory(
		t, newFakeSessions(), &fakeWorkerInvoker{}, &fakeInspector{}, model,
		Limits{MaxModelCalls: 8, MaxTokens: 100, MaxWorkerCalls: 4, MaxWallTime: time.Minute},
	)
	instance, err := factory.Create(testInvocation("builder"))
	if err != nil {
		t.Fatal(err)
	}
	_, err = instance.Run(t.Context())
	assertPlannerCode(t, err, "planner_token_limit")
	if model.callCount() != 2 {
		t.Fatalf("model calls = %d", model.callCount())
	}
}

func TestStreamlineStopsBeforeWorkerCallBeyondBudget(t *testing.T) {
	model := &scriptedModel{steps: []modelStep{
		functionStep("worker_builder", map[string]any{
			"objective": "first", "instructions": "first call",
			"parameters": map[string]any{}, "artifacts": map[string]any{},
		}),
		functionStep("worker_builder", map[string]any{
			"objective": "second", "instructions": "must be rejected",
			"parameters": map[string]any{}, "artifacts": map[string]any{},
		}),
	}}
	workers := &fakeWorkerInvoker{results: map[string]contracts.StageContentResult{
		"builder": stageResult("intermediate", map[string]contracts.ArtifactRef{}),
	}}
	factory := mustFactory(
		t, newFakeSessions(), workers, &fakeInspector{}, model,
		Limits{MaxModelCalls: 8, MaxTokens: 1_000, MaxWorkerCalls: 1, MaxWallTime: time.Minute},
	)
	instance, err := factory.Create(testInvocation("builder"))
	if err != nil {
		t.Fatal(err)
	}
	_, err = instance.Run(t.Context())
	assertPlannerCode(t, err, "planner_worker_call_limit")
	if len(workers.calls) != 1 {
		t.Fatalf("Worker calls = %d, want one actual call", len(workers.calls))
	}
}

func TestStreamlineWallDeadlineStopsBlockedModel(t *testing.T) {
	factory := mustFactory(
		t, newFakeSessions(), &fakeWorkerInvoker{}, &fakeInspector{}, blockingModel{},
		Limits{MaxModelCalls: 8, MaxTokens: 1_000, MaxWorkerCalls: 4, MaxWallTime: 5 * time.Millisecond},
	)
	instance, err := factory.Create(testInvocation("builder"))
	if err != nil {
		t.Fatal(err)
	}
	_, err = instance.Run(t.Context())
	assertPlannerCode(t, err, "planner_deadline_exceeded")
}

func TestStreamlineRecoveryDoesNotInvokeModelOrWorker(t *testing.T) {
	revision := "report-r1"
	recovered := stageResult("already done", map[string]contracts.ArtifactRef{
		"report": exactRef("builder", "report", revision),
	})
	sessions := newFakeSessions()
	sessions.recovered = &planner.Completion{Result: &recovered}
	model := &scriptedModel{fallback: func(*model.LLMRequest) (*model.LLMResponse, error) {
		return nil, errors.New("must not run")
	}}
	workers := &fakeWorkerInvoker{}
	inspector := &fakeInspector{mediaTypes: map[string]string{
		refKey(recovered.Artifacts["report"]): "application/json",
	}}
	factory := mustFactory(t, sessions, workers, inspector, model, Limits{})
	instance, err := factory.Create(testInvocation("builder"))
	if err != nil {
		t.Fatal(err)
	}
	result, err := instance.Run(t.Context())
	if err != nil || result.Summary != "already done" || model.callCount() != 0 ||
		len(workers.calls) != 0 || sessions.adkCalls != 0 {
		t.Fatalf("recovery result=%+v err=%v model=%d workers=%d adk=%d", result, err, model.callCount(), len(workers.calls), sessions.adkCalls)
	}
}

func TestStreamlineProviderErrorIsSafe(t *testing.T) {
	const secret = "sk-provider-secret-never-persist"
	model := &scriptedModel{fallback: func(*model.LLMRequest) (*model.LLMResponse, error) {
		return nil, errors.New("provider echoed " + secret)
	}}
	sessions := newFakeSessions()
	factory := mustFactory(t, sessions, &fakeWorkerInvoker{}, &fakeInspector{}, model, Limits{})
	instance, err := factory.Create(testInvocation("builder"))
	if err != nil {
		t.Fatal(err)
	}
	_, err = instance.Run(t.Context())
	assertPlannerCode(t, err, "planner_gateway_unavailable")
	report, _ := instance.(planner.ReportProvider).ExecutionReport()
	encoded, marshalErr := json.Marshal(struct {
		Error      string                    `json:"error"`
		Completion *planner.Completion       `json:"completion"`
		Report     contracts.ExecutionReport `json:"report"`
	}{Error: err.Error(), Completion: sessions.completion, Report: report})
	if marshalErr != nil || strings.Contains(string(encoded), secret) {
		t.Fatalf("provider secret leaked: %s (%v)", encoded, marshalErr)
	}
}

func TestStreamlineRejectsResponseWithoutTokenUsage(t *testing.T) {
	model := &scriptedModel{fallback: func(*model.LLMRequest) (*model.LLMResponse, error) {
		return &model.LLMResponse{Content: genai.NewContentFromText("not finished", genai.RoleModel)}, nil
	}}
	factory := mustFactory(t, newFakeSessions(), &fakeWorkerInvoker{}, &fakeInspector{}, model, Limits{})
	instance, err := factory.Create(testInvocation("builder"))
	if err != nil {
		t.Fatal(err)
	}
	_, err = instance.Run(t.Context())
	assertPlannerCode(t, err, "planner_gateway_invalid_response")
}

type modelStep func(*model.LLMRequest) (*model.LLMResponse, error)

type scriptedModel struct {
	mu       sync.Mutex
	steps    []modelStep
	fallback modelStep
	calls    int
}

type blockingModel struct{}

func (blockingModel) Name() string { return "blocking-model" }

func (blockingModel) GenerateContent(
	ctx context.Context, _ *model.LLMRequest, _ bool,
) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		<-ctx.Done()
		yield(nil, ctx.Err())
	}
}

func (*scriptedModel) Name() string { return "fake-streamline-model" }

func (m *scriptedModel) GenerateContent(
	_ context.Context, request *model.LLMRequest, _ bool,
) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.mu.Lock()
		index := m.calls
		m.calls++
		var step modelStep
		if index < len(m.steps) {
			step = m.steps[index]
		} else {
			step = m.fallback
		}
		m.mu.Unlock()
		if step == nil {
			yield(nil, errors.New("unexpected model call"))
			return
		}
		response, err := step(request)
		yield(response, err)
	}
}

func (m *scriptedModel) callCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.calls
}

func functionStep(name string, args map[string]any) modelStep {
	return func(*model.LLMRequest) (*model.LLMResponse, error) {
		content := genai.NewContentFromFunctionCall(name, args, genai.RoleModel)
		content.Parts[0].FunctionCall.ID = "call-" + name
		return &model.LLMResponse{
			Content: content,
			UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount: 10, CandidatesTokenCount: 10, TotalTokenCount: 20,
			},
		}, nil
	}
}

func textStep(value string) modelStep {
	return func(*model.LLMRequest) (*model.LLMResponse, error) {
		return &model.LLMResponse{
			Content: genai.NewContentFromText(value, genai.RoleModel),
			UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount: 10, CandidatesTokenCount: 10, TotalTokenCount: 20,
			},
		}, nil
	}
}

type fakeSessions struct {
	identity   planner.SessionIdentity
	recovered  *planner.Completion
	completion *planner.Completion
	request    planner.RequestFacts
	adkCalls   int
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
	_ context.Context, _ planner.SessionIdentity, completion planner.Completion,
) error {
	copy := completion
	s.completion = &copy
	return nil
}

func (s *fakeSessions) NewADKSession(
	_ context.Context,
	_ planner.SessionIdentity,
	_ plannersession.ADKOptions,
) (adksession.Service, error) {
	s.adkCalls++
	return adksession.InMemoryService(), nil
}

type workerInvocation struct {
	binding  string
	request  contracts.StageContentRequest
	deadline time.Time
}

type fakeWorkerInvoker struct {
	mu      sync.Mutex
	results map[string]contracts.StageContentResult
	calls   []workerInvocation
}

func (w *fakeWorkerInvoker) Invoke(
	ctx context.Context,
	binding string,
	_ contracts.WorkerHandle,
	request contracts.StageContentRequest,
) (contracts.StageContentResult, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	deadline, _ := ctx.Deadline()
	w.calls = append(w.calls, workerInvocation{binding: binding, request: request, deadline: deadline})
	result, ok := w.results[binding]
	if !ok {
		return contracts.StageContentResult{}, errors.New("unexpected Worker")
	}
	return result, nil
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
	mediaType, ok := i.mediaTypes[refKey(ref)]
	if !ok {
		return planner.ArtifactMetadata{}, errors.New("artifact absent")
	}
	return planner.ArtifactMetadata{MediaType: mediaType}, nil
}

func mustFactory(
	t *testing.T,
	sessions *fakeSessions,
	workers *fakeWorkerInvoker,
	inspector *fakeInspector,
	llm model.LLM,
	limits Limits,
) *Factory {
	t.Helper()
	factory, err := NewFactory(sessions, sessions, workers, inspector, llm, limits)
	if err != nil {
		t.Fatal(err)
	}
	return factory
}

func testInvocation(bindings ...string) planner.Invocation {
	sourceRevision := "source-r1"
	stageAgents := make(map[string]workflowconfig.ResolvedAgentBinding, len(bindings))
	workers := make(map[string]contracts.WorkerHandle, len(bindings))
	for _, name := range bindings {
		templateRef := contracts.AgentTemplateRef{
			TemplateID: name, Version: "1", Digest: "sha256:" + strings.Repeat("a", 64),
		}
		runtimeRef := contracts.WorkerRuntimeRef{RuntimeID: "adk", Version: "1"}
		stageAgents[name] = workflowconfig.ResolvedAgentBinding{
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
		StageExecutionID: "stage-1", RunID: "run-1", Deadline: time.Now().Add(time.Minute),
		Stage: workflowconfig.ResolvedStage{
			Objective:    "Produce a reviewed report",
			Instructions: contracts.ResolvedInstructions{Text: "Use the Workers carefully."},
			Planner:      workflowconfig.PlannerRef{PlannerID: "streamline", Version: "1"},
			Agents:       stageAgents,
			Context: workflowconfig.StageContext{Artifacts: map[string]workflowconfig.ContextArtifact{
				"source": {Namespace: "inputs", Name: "source", Required: true},
			}},
			Result: workflowconfig.StageResultContract{Artifacts: map[string]workflowconfig.ArtifactSlot{
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

func stageResult(summary string, artifacts map[string]contracts.ArtifactRef) contracts.StageContentResult {
	return contracts.StageContentResult{
		APIVersion: contracts.APIVersion, Outcome: contracts.StageSucceeded,
		Summary: summary, Artifacts: artifacts,
	}
}

func exactRef(namespace, name, revision string) contracts.ArtifactRef {
	return contracts.ArtifactRef{Namespace: namespace, Name: name, Revision: &revision}
}

func pointerRef(value contracts.ArtifactRef) *contracts.ArtifactRef { return &value }

func artifactArgs(namespace, name, revision string) map[string]any {
	return map[string]any{"namespace": namespace, "name": name, "revision": revision}
}

func refKey(ref contracts.ArtifactRef) string {
	if ref.Revision == nil {
		return ref.Namespace + "/" + ref.Name
	}
	return ref.Namespace + "/" + ref.Name + "@" + *ref.Revision
}

func assertPlannerCode(t *testing.T, err error, code string) {
	t.Helper()
	var value *planner.Error
	if !errors.As(err, &value) || value.Code != code {
		t.Fatalf("error = %v, want Planner code %q", err, code)
	}
}

var _ planner.SessionService = (*fakeSessions)(nil)
var _ ADKSessionFactory = (*fakeSessions)(nil)
var _ model.LLM = (*scriptedModel)(nil)
