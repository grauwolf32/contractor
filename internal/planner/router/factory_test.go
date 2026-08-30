package router

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

func TestRouterSelectsExactWorkerWithDeterministicPromptAndContext(t *testing.T) {
	revision := "report-r1"
	llm := &scriptedModel{steps: []modelStep{
		functionStep("add_subtask", map[string]any{
			"objective": "Build the API document", "instructions": "Use the exact source",
		}),
		functionStep("execute_current_subtask", map[string]any{
			"subtask_id": "0", "worker_name": "reviewer",
		}),
		functionStep("finish", map[string]any{
			"outcome": "succeeded", "summary": "complete",
			"artifacts": map[string]any{
				"report": artifactArgs("review", "report", revision),
			},
		}),
	}}
	workers := &fakeWorkerInvoker{results: map[string]contracts.StageContentResult{
		"reviewer": successfulResult("review complete"),
	}}
	instance := mustPlanner(t, llm, workers, testInvocation())
	result, err := instance.Run(t.Context())
	if err != nil || result.Outcome != contracts.StageSucceeded {
		t.Fatalf("Run = (%+v, %v)", result, err)
	}
	if got := workers.bindingCalls(); !reflect.DeepEqual(got, []string{"reviewer"}) {
		t.Fatalf("Worker calls = %v", got)
	}
	call := workers.calls[0]
	if call.handle.AllocationID != "allocation-reviewer" ||
		call.request.Objective != "Build the API document" ||
		call.request.Instructions != "Use the exact source" ||
		call.request.Parameters["mode"] != "strict" ||
		call.request.Artifacts["source"].Revision == nil ||
		*call.request.Artifacts["source"].Revision != "source-r1" {
		t.Fatalf("routed request = %+v handle=%+v", call.request, call.handle)
	}

	request := llm.firstRequest()
	instruction := contentText(request.Config.SystemInstruction)
	builder := "- builder: Builds exact API descriptions"
	reviewer := "- reviewer: Reviews API descriptions"
	if strings.Index(instruction, builder) < 0 || strings.Index(instruction, reviewer) < 0 ||
		strings.Index(instruction, builder) > strings.Index(instruction, reviewer) {
		t.Fatalf("Available agents are not stable and lexical:\n%s", instruction)
	}
	encodedRequest, marshalErr := json.Marshal(request)
	if marshalErr != nil {
		t.Fatal(marshalErr)
	}
	for _, forbidden := range []string{
		"allocation-builder", "allocation-reviewer", "runtime-placement.invalid", "runtime-secret",
	} {
		if strings.Contains(string(encodedRequest), forbidden) {
			t.Fatalf("model request leaked physical placement %q: %s", forbidden, encodedRequest)
		}
	}
	assertRouterExecuteSchema(t, request)

	report, ok := instance.(planner.ReportProvider).ExecutionReport()
	if !ok {
		t.Fatal("Router does not expose an execution report")
	}
	found := false
	for _, toolCall := range report.ToolCalls {
		if toolCall.Tool == "execute_current_subtask" {
			found = toolCall.Arguments["binding"] == "reviewer"
		}
	}
	if !found {
		t.Fatalf("Router did not record the selected logical Worker: %+v", report.ToolCalls)
	}
}

func TestRouterRejectsUnknownAndStaleSelectionBeforeWorker(t *testing.T) {
	revision := "report-r1"
	llm := &scriptedModel{steps: []modelStep{
		functionStep("add_subtask", map[string]any{
			"objective": "Build", "instructions": "Execute once",
		}),
		functionStep("execute_current_subtask", map[string]any{
			"subtask_id": "0", "worker_name": "unknown",
		}),
		functionStep("execute_current_subtask", map[string]any{
			"subtask_id": "1", "worker_name": "reviewer",
		}),
		functionStep("execute_current_subtask", map[string]any{
			"subtask_id": "0", "worker_name": "builder",
		}),
		functionStep("finish", map[string]any{
			"outcome": "succeeded", "summary": "complete",
			"artifacts": map[string]any{
				"report": artifactArgs("review", "report", revision),
			},
		}),
	}}
	workers := &fakeWorkerInvoker{results: map[string]contracts.StageContentResult{
		"builder": successfulResult("built"),
	}}
	instance := mustPlanner(t, llm, workers, testInvocation())
	result, err := instance.Run(t.Context())
	if err != nil || result.Outcome != contracts.StageSucceeded {
		t.Fatalf("Run = (%+v, %v)", result, err)
	}
	if got := workers.bindingCalls(); !reflect.DeepEqual(got, []string{"builder"}) {
		t.Fatalf("invalid selection crossed A2A: %v", got)
	}
	report, _ := instance.(planner.ReportProvider).ExecutionReport()
	metrics := report.Metrics.Tools["execute_current_subtask"]
	if metrics.Calls == nil || *metrics.Calls != 2 || metrics.Failed == nil || *metrics.Failed != 1 {
		t.Fatalf("execute metrics = %+v", metrics)
	}
}

func TestRouterRejectsParallelToolCalls(t *testing.T) {
	revision := "report-r1"
	llm := &scriptedModel{steps: []modelStep{
		parallelFunctionStep(
			functionCall("add_subtask", map[string]any{"objective": "first", "instructions": "first"}),
			functionCall("add_subtask", map[string]any{"objective": "second", "instructions": "second"}),
		),
		functionStep("add_subtask", map[string]any{"objective": "only", "instructions": "execute"}),
		functionStep("execute_current_subtask", map[string]any{
			"subtask_id": "0", "worker_name": "builder",
		}),
		functionStep("finish", map[string]any{
			"outcome": "succeeded", "summary": "complete",
			"artifacts": map[string]any{
				"report": artifactArgs("review", "report", revision),
			},
		}),
	}}
	workers := &fakeWorkerInvoker{results: map[string]contracts.StageContentResult{
		"builder": successfulResult("built"),
	}}
	instance := mustPlanner(t, llm, workers, testInvocation())
	result, err := instance.Run(t.Context())
	if err != nil || result.Outcome != contracts.StageSucceeded || len(workers.calls) != 1 {
		t.Fatalf("Run=(%+v,%v) Worker calls=%d", result, err, len(workers.calls))
	}
	report, _ := instance.(planner.ReportProvider).ExecutionReport()
	if len(report.Errors) == 0 || report.Errors[0].Code != "planner_tool_selection_invalid" {
		t.Fatalf("parallel selection was not recorded: %+v", report.Errors)
	}
}

func TestRouterFactoryRejectsEmptyBindingsBeforeSideEffects(t *testing.T) {
	llm := &scriptedModel{}
	workers := &fakeWorkerInvoker{}
	sessions := newFakeSessions()
	factory, err := NewFactory(sessions, sessions, workers, fakeInspector{}, llm, DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	if factory.Ref() != planner.RouterRef {
		t.Fatalf("Router factory ref = %q", factory.Ref())
	}
	invocation := testInvocation()
	invocation.Stage.Agents = map[string]workflowconfig.ResolvedAgentBinding{}
	invocation.Workers = map[string]contracts.WorkerHandle{}
	if _, err := factory.Create(invocation); err == nil || !strings.Contains(err.Error(), "one or more matched") {
		t.Fatalf("Create error = %v", err)
	}
	if llm.callCount() != 0 || len(workers.calls) != 0 || sessions.adkCalls != 0 {
		t.Fatal("invalid Router invocation caused side effects")
	}
}

func assertRouterExecuteSchema(t *testing.T, request *model.LLMRequest) {
	t.Helper()
	var declaration *genai.FunctionDeclaration
	for _, group := range request.Config.Tools {
		for _, current := range group.FunctionDeclarations {
			if current.Name == "execute_current_subtask" {
				declaration = current
			}
		}
	}
	if declaration == nil {
		t.Fatal("execute_current_subtask declaration is absent")
	}
	encoded, err := json.Marshal(declaration.ParametersJsonSchema)
	if err != nil {
		t.Fatal(err)
	}
	schema := string(encoded)
	for _, required := range []string{`"subtask_id"`, `"worker_name"`, `"enum":["builder","reviewer"]`, `"additionalProperties":false`} {
		if !strings.Contains(schema, required) {
			t.Fatalf("Router execute schema lacks %s: %s", required, schema)
		}
	}
	for _, forbidden := range []string{"objective", "instructions", "parameters", "artifacts", "endpoint", "allocation"} {
		if strings.Contains(schema, `"`+forbidden+`"`) {
			t.Fatalf("Router execute schema exposes %q: %s", forbidden, schema)
		}
	}
}

func mustPlanner(
	t *testing.T, llm model.LLM, workers *fakeWorkerInvoker, invocation planner.Invocation,
) planner.Planner {
	t.Helper()
	sessions := newFakeSessions()
	factory, err := NewFactory(sessions, sessions, workers, fakeInspector{}, llm, DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	instance, err := factory.Create(invocation)
	if err != nil {
		t.Fatal(err)
	}
	return instance
}

type modelStep func(*model.LLMRequest) (*model.LLMResponse, error)

type scriptedModel struct {
	mu       sync.Mutex
	steps    []modelStep
	calls    int
	requests []*model.LLMRequest
}

func (*scriptedModel) Name() string { return "fake-router-model" }

func (m *scriptedModel) GenerateContent(
	_ context.Context, request *model.LLMRequest, _ bool,
) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.mu.Lock()
		index := m.calls
		m.calls++
		m.requests = append(m.requests, request)
		var step modelStep
		if index < len(m.steps) {
			step = m.steps[index]
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

func (m *scriptedModel) firstRequest() *model.LLMRequest {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.requests) == 0 {
		return nil
	}
	return m.requests[0]
}

func functionStep(name string, args map[string]any) modelStep {
	return parallelFunctionStep(functionCall(name, args))
}

func functionCall(name string, args map[string]any) *genai.Part {
	part := genai.NewPartFromFunctionCall(name, args)
	part.FunctionCall.ID = "call-" + name
	return part
}

func parallelFunctionStep(parts ...*genai.Part) modelStep {
	return func(*model.LLMRequest) (*model.LLMResponse, error) {
		return &model.LLMResponse{
			Content: genai.NewContentFromParts(parts, genai.RoleModel),
			UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount: 10, CandidatesTokenCount: 10, TotalTokenCount: 20,
			},
		}, nil
	}
}

func contentText(content *genai.Content) string {
	if content == nil {
		return ""
	}
	var result strings.Builder
	for _, part := range content.Parts {
		if part != nil {
			result.WriteString(part.Text)
		}
	}
	return result.String()
}

type fakeSessions struct {
	identity planner.SessionIdentity
	complete *planner.Completion
	adkCalls int
}

func newFakeSessions() *fakeSessions {
	return &fakeSessions{identity: planner.SessionIdentity{
		SessionID: "router-session", StageExecutionID: "stage-router", InvocationID: "router-invocation",
	}}
}

func (s *fakeSessions) Begin(context.Context, string) (planner.SessionStart, error) {
	return planner.SessionStart{Identity: s.identity, Invoke: true}, nil
}

func (*fakeSessions) RecordRequest(context.Context, planner.SessionIdentity, planner.RequestFacts) error {
	return nil
}

func (s *fakeSessions) Complete(
	_ context.Context, _ planner.SessionIdentity, completion planner.Completion,
) error {
	cloned := completion
	s.complete = &cloned
	return nil
}

func (s *fakeSessions) NewADKSession(
	_ context.Context, _ planner.SessionIdentity, _ plannersession.ADKOptions,
) (adksession.Service, error) {
	s.adkCalls++
	return adksession.InMemoryService(), nil
}

type workerInvocation struct {
	binding string
	handle  contracts.WorkerHandle
	request contracts.StageContentRequest
}

type fakeWorkerInvoker struct {
	mu      sync.Mutex
	results map[string]contracts.StageContentResult
	calls   []workerInvocation
}

func (w *fakeWorkerInvoker) Invoke(
	_ context.Context,
	binding string,
	handle contracts.WorkerHandle,
	request contracts.StageContentRequest,
) (contracts.StageContentResult, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.calls = append(w.calls, workerInvocation{binding: binding, handle: handle, request: request})
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

type fakeInspector struct{}

func (fakeInspector) Inspect(
	_ context.Context, runID string, ref contracts.ArtifactRef,
) (planner.ArtifactMetadata, error) {
	if runID != "run-router" || ref.Revision == nil {
		return planner.ArtifactMetadata{}, errors.New("artifact outside RunScope")
	}
	switch ref.Namespace + "/" + ref.Name + "@" + *ref.Revision {
	case "inputs/source@source-r1":
		return planner.ArtifactMetadata{MediaType: "text/plain"}, nil
	case "review/report@report-r1":
		return planner.ArtifactMetadata{MediaType: "application/json"}, nil
	default:
		return planner.ArtifactMetadata{}, errors.New("artifact absent")
	}
}

func testInvocation() planner.Invocation {
	bindings := []struct {
		name, description string
	}{
		{name: "reviewer", description: "Reviews API descriptions"},
		{name: "builder", description: "Builds exact API descriptions"},
	}
	agents := make(map[string]workflowconfig.ResolvedAgentBinding, len(bindings))
	workers := make(map[string]contracts.WorkerHandle, len(bindings))
	for index, binding := range bindings {
		digestCharacter := "a"
		if index > 0 {
			digestCharacter = "b"
		}
		template := contracts.AgentTemplateRef{
			TemplateID: binding.name, Version: "1", Digest: "sha256:" + strings.Repeat(digestCharacter, 64),
		}
		runtime := contracts.WorkerRuntimeRef{RuntimeID: "adk", Version: "1"}
		agents[binding.name] = workflowconfig.ResolvedAgentBinding{
			Namespace: binding.name,
			Template: contracts.ResolvedAgentTemplate{
				Ref: template, Runtime: runtime, Description: binding.description,
			},
		}
		workers[binding.name] = contracts.WorkerHandle{
			AllocationID:     "allocation-" + binding.name,
			AgentTemplateRef: template, WorkerRuntimeRef: runtime,
			AgentCard: map[string]any{
				"name": binding.name, "url": "https://runtime-placement.invalid/" + binding.name,
				"token": "runtime-secret",
			},
			LeaseExpiresAt: time.Now().Add(time.Minute),
		}
	}
	revision := "source-r1"
	return planner.Invocation{
		StageExecutionID: "stage-router", RunID: "run-router", Deadline: time.Now().Add(time.Minute),
		Stage: workflowconfig.ResolvedStage{
			Objective:    "Build an API description",
			Instructions: contracts.ResolvedInstructions{Text: "Route work by immutable capability."},
			Planner:      workflowconfig.PlannerRef{PlannerID: "router", Version: "1"},
			Agents:       agents,
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
				"source": {Namespace: "inputs", Name: "source", Revision: &revision},
			},
		},
		Workers: workers,
	}
}

func successfulResult(summary string) contracts.StageContentResult {
	return contracts.StageContentResult{
		APIVersion: contracts.APIVersion, Outcome: contracts.StageSucceeded,
		Summary: summary, Artifacts: map[string]contracts.ArtifactRef{},
	}
}

func artifactArgs(namespace, name, revision string) map[string]any {
	return map[string]any{"namespace": namespace, "name": name, "revision": revision}
}

var _ planner.SessionService = (*fakeSessions)(nil)
var _ ADKSessionFactory = (*fakeSessions)(nil)
var _ model.LLM = (*scriptedModel)(nil)
