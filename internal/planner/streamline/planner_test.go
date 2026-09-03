package streamline

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"iter"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	plannersession "github.com/grauwolf32/contractor/internal/planner/session"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"google.golang.org/adk/model"
	adksession "google.golang.org/adk/session"
	"google.golang.org/genai"
)

func TestStreamlineCallsSingleWorkerWithStoredSubtaskAndCompleteContext(t *testing.T) {
	reportRevision := "report-r1"
	model := &scriptedModel{steps: []modelStep{
		addSubtaskStep("Analyze the input", "Produce the final report"),
		functionStep(executeCurrentSubtaskToolName, map[string]any{"subtask_id": "0"}),
		functionStep(finishToolName, map[string]any{
			"outcome": string(contracts.StageSucceeded), "summary": "complete", "artifacts": map[string]any{
				"report": artifactArgs("review", "report", reportRevision),
			},
		}),
	}}
	workers := &fakeWorkerInvoker{results: map[string]contracts.StageContentResult{
		"builder": stageResult("report ready", map[string]contracts.ArtifactRef{
			"report": exactRef("review", "report", reportRevision),
		}),
	}}
	sessions := newFakeSessions()
	inspector := &fakeInspector{mediaTypes: map[string]string{
		refKey(exactRef("inputs", "source", "source-r1")):    "text/plain",
		refKey(exactRef("review", "report", reportRevision)): "application/json",
	}}
	factory := mustFactory(t, sessions, workers, inspector, model, Limits{})
	invocation := testInvocation("builder")
	telemetryAdapter, telemetryPayload := streamlineTestTelemetry(t, invocation)
	invocation.Instrumentation = telemetryAdapter.Instrumentation()
	instance, err := factory.Create(invocation)
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
	if got := workers.bindingCalls(); !reflect.DeepEqual(got, []string{"builder"}) {
		t.Fatalf("Worker calls = %v", got)
	}
	if workers.calls[0].request.SubtaskID != "0" ||
		workers.calls[0].request.Parameters["mode"] != "strict" ||
		workers.calls[0].request.Objective != "Analyze the input" ||
		workers.calls[0].request.Instructions != "Produce the final report" ||
		workers.calls[0].request.Artifacts["source"].Revision == nil ||
		*workers.calls[0].request.Artifacts["source"].Revision != "source-r1" {
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
	if export := telemetryAdapter.Flush(t.Context()); !export.Succeeded {
		t.Fatalf("Streamline telemetry export = %+v", export)
	}
	payload := <-telemetryPayload
	for _, expected := range []telemetry.PlannerSpanName{
		telemetry.PlannerSpanSession, telemetry.PlannerSpanModel, telemetry.PlannerSpanWorker,
		telemetry.PlannerSpanSubtask, telemetry.PlannerSpanFinish,
	} {
		if !bytes.Contains(payload, []byte(expected)) {
			t.Fatalf("Streamline OTLP payload lacks %s: %q", expected, payload)
		}
	}
	for _, forbidden := range []string{
		invocation.Stage.Objective, invocation.Stage.Instructions.Text,
		"Analyze the input", "Produce the final report", "report ready",
	} {
		if bytes.Contains(payload, []byte(forbidden)) {
			t.Fatalf("Streamline OTLP payload exposed %q: %q", forbidden, payload)
		}
	}
	if sessions.plan == nil || sessions.plan.Revision != 3 ||
		sessions.plan.Subtasks[0].Status != planner.PlannerSubtaskSucceeded ||
		len(sessions.transitions) != 3 || len(sessions.facts) != 4 ||
		sessions.transitions[0].Kind != planner.PlannerEventPlanChanged ||
		sessions.transitions[1].Kind != planner.PlannerEventDispatchStarted ||
		sessions.transitions[2].Kind != planner.PlannerEventDispatchCompleted ||
		sessions.facts[1].Kind != planner.PlannerEventDispatchSelected ||
		sessions.facts[1].WorkerName != "builder" ||
		sessions.facts[3].Kind != planner.PlannerEventFinishRequested {
		t.Fatalf("durable plan transitions=%+v facts=%+v plan=%+v", sessions.transitions, sessions.facts, sessions.plan)
	}
	encodedPlanFacts, marshalErr := json.Marshal(struct {
		Plan        *planner.PlannerPlanProjection
		Transitions []planner.PlannerPlanTransition
		Facts       []planner.PlannerFact
	}{sessions.plan, sessions.transitions, sessions.facts})
	if marshalErr != nil || strings.Contains(string(encodedPlanFacts), "report ready") {
		t.Fatalf("Worker response leaked into durable plan facts: %s (%v)", encodedPlanFacts, marshalErr)
	}
	report, ok := instance.(planner.ReportProvider).ExecutionReport()
	if !ok || report.Metrics.ModelCalls == nil || *report.Metrics.ModelCalls != 3 ||
		report.Metrics.TotalTokens == nil || *report.Metrics.TotalTokens != 60 {
		t.Fatalf("report = %+v", report)
	}
}

func streamlineTestTelemetry(
	t *testing.T, invocation planner.Invocation,
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
		FlushTimeout: time.Second, RunMetadataLabels: contracts.RunMetadataLabels{},
		Resource: telemetry.PlannerResource{
			RunID: invocation.RunID, StageExecutionID: invocation.StageExecutionID,
			PlannerRef: Ref,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(adapter.Close)
	return adapter, payload
}

func TestStreamlineRejectsUnknownToolAndInvalidFinishThenCorrects(t *testing.T) {
	revision := "report-r1"
	model := &scriptedModel{steps: []modelStep{
		functionStep("worker_not_prepared", map[string]any{
			"subtask_id": "0",
		}),
		addSubtaskStep("Build the report", "Produce the declared report"),
		functionStep(executeCurrentSubtaskToolName, map[string]any{"subtask_id": "0"}),
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
	workers := &fakeWorkerInvoker{results: map[string]contracts.StageContentResult{
		"builder": stageResult("report prepared", map[string]contracts.ArtifactRef{}),
	}}
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
	if got := workers.bindingCalls(); !reflect.DeepEqual(got, []string{"builder"}) {
		t.Fatalf("Worker calls = %+v", got)
	}
	report, _ := instance.(planner.ReportProvider).ExecutionReport()
	if report.Metrics.ModelCalls == nil || *report.Metrics.ModelCalls != 5 ||
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

func TestStreamlineWorkerFailureFailsDispatchButPlannerStillOwnsFinish(t *testing.T) {
	model := &scriptedModel{steps: []modelStep{
		addSubtaskStep("Analyze the source", "Return the bounded finding"),
		functionStep(executeCurrentSubtaskToolName, map[string]any{"subtask_id": "0"}),
		functionStep(finishToolName, map[string]any{
			"outcome": string(contracts.StageFailed), "summary": "analysis could not be completed",
			"artifacts": map[string]any{},
			"error": map[string]any{
				"code": "insufficient_evidence", "message": "required evidence was unavailable",
				"retryable": false,
			},
		}),
	}}
	workers := &fakeWorkerInvoker{completions: map[string]contracts.WorkerCompletion{
		"builder": {
			APIVersion: contracts.APIVersion,
			Failure: &contracts.WorkerFailure{
				Code: "worker_budget_exhausted", Message: "Worker budget was exhausted", Retryable: true,
			},
			InvocationID: "worker-failed-dispatch", StateRevision: 4,
		},
	}}
	sessions := newFakeSessions()
	instance, err := mustFactory(
		t, sessions, workers, &fakeInspector{}, model, Limits{},
	).Create(testInvocation("builder"))
	if err != nil {
		t.Fatal(err)
	}
	result, err := instance.Run(t.Context())
	if err != nil || result.Outcome != contracts.StageFailed || result.Error == nil ||
		result.Error.Code != "insufficient_evidence" {
		t.Fatalf("Run = (%+v, %v)", result, err)
	}
	if sessions.plan == nil || sessions.plan.Subtasks[0].Status != planner.PlannerSubtaskFailed {
		t.Fatalf("failed Worker dispatch plan = %+v", sessions.plan)
	}
	report, _ := instance.(planner.ReportProvider).ExecutionReport()
	found := false
	for _, call := range report.ToolCalls {
		if call.Tool == executeCurrentSubtaskToolName && call.Error != nil {
			found = call.Error.Code == "worker_budget_exhausted"
		}
	}
	if !found {
		t.Fatalf("WorkerFailure was not recorded as a failed dispatch: %+v", report.ToolCalls)
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

func TestStreamlineExposesExactSingleWorkerToolContract(t *testing.T) {
	instance, err := mustFactory(
		t, newFakeSessions(), &fakeWorkerInvoker{}, &fakeInspector{}, &scriptedModel{}, Limits{},
	).Create(testInvocation("builder"))
	if err != nil {
		t.Fatal(err)
	}
	streamline := instance.(*streamlinePlanner)
	tools, allowed, err := streamline.buildTools(
		newExecutionState(streamline.limits), streamline.sessions.(*fakeSessions).identity,
	)
	if err != nil {
		t.Fatal(err)
	}
	names := make([]string, 0, len(tools))
	var executeDeclaration *genai.FunctionDeclaration
	var listDeclaration *genai.FunctionDeclaration
	var usageDeclaration *genai.FunctionDeclaration
	for _, current := range tools {
		names = append(names, current.Name())
		if current.Name() == executeCurrentSubtaskToolName || current.Name() == listSubtasksToolName ||
			current.Name() == getWorkerToolUsageToolName {
			provider, ok := current.(interface {
				Declaration() *genai.FunctionDeclaration
			})
			if !ok {
				t.Fatalf("tool %s (%T) does not expose a declaration", current.Name(), current)
			}
			if current.Name() == listSubtasksToolName {
				listDeclaration = provider.Declaration()
			} else if current.Name() == getWorkerToolUsageToolName {
				usageDeclaration = provider.Declaration()
			} else {
				executeDeclaration = provider.Declaration()
			}
		}
	}
	wantNames := []string{
		addSubtaskToolName, listSubtasksToolName, executeCurrentSubtaskToolName,
		getWorkerToolUsageToolName, finishToolName,
	}
	if !reflect.DeepEqual(names, wantNames) || len(allowed) != len(wantNames) {
		t.Fatalf("tools = %v allowed = %v, want exactly %v", names, allowed, wantNames)
	}
	for _, name := range wantNames {
		if _, exists := allowed[name]; !exists {
			t.Fatalf("%s is not allowlisted", name)
		}
	}
	if executeDeclaration == nil {
		t.Fatal("execute_current_subtask declaration is absent")
	}
	if listDeclaration == nil {
		t.Fatal("list_subtasks declaration is absent")
	}
	if usageDeclaration == nil {
		t.Fatal("get_worker_tool_usage declaration is absent")
	}
	listSchema, err := json.Marshal(listDeclaration.ParametersJsonSchema)
	if err != nil {
		t.Fatal(err)
	}
	if string(listSchema) != `{"type":"object","properties":{},"additionalProperties":false}` {
		t.Fatalf("list_subtasks schema = %s", listSchema)
	}
	usageSchema, err := json.Marshal(usageDeclaration.ParametersJsonSchema)
	if err != nil {
		t.Fatal(err)
	}
	if string(usageSchema) != `{"type":"object","properties":{},"additionalProperties":false}` {
		t.Fatalf("get_worker_tool_usage schema = %s", usageSchema)
	}
	encoded, err := json.Marshal(executeDeclaration.ParametersJsonSchema)
	if err != nil {
		t.Fatal(err)
	}
	schema := string(encoded)
	if !strings.Contains(schema, `"subtask_id"`) || !strings.Contains(schema, `"additionalProperties":false`) {
		t.Fatalf("execute_current_subtask schema = %s", schema)
	}
	for _, forbidden := range []string{"objective", "instructions", "parameters", "artifacts", "worker_name"} {
		if strings.Contains(schema, `"`+forbidden+`"`) {
			t.Fatalf("execute_current_subtask schema exposes %q: %s", forbidden, schema)
		}
	}
	if strings.Contains(streamline.systemInstruction(), "call escalate") {
		t.Fatalf("completion tools=%v instruction=%q", allowed, streamline.systemInstruction())
	}
}

func TestStreamlineRejectsModelSelectedContextBeforeWorkerCall(t *testing.T) {
	revision := "report-r1"
	model := &scriptedModel{steps: []modelStep{
		addSubtaskStep("Build the report", "Use the immutable Stage context"),
		functionStep(executeCurrentSubtaskToolName, map[string]any{
			"subtask_id":  "0",
			"objective":   "replace the task",
			"parameters":  map[string]any{"mode": "unsafe"},
			"artifacts":   map[string]any{},
			"worker_name": "another-worker",
		}),
		functionStep(executeCurrentSubtaskToolName, map[string]any{"subtask_id": "0"}),
		functionStep(finishToolName, map[string]any{
			"outcome": string(contracts.StageSucceeded), "summary": "complete",
			"artifacts": map[string]any{"report": artifactArgs("builder", "report", revision)},
		}),
	}}
	workers := &fakeWorkerInvoker{results: map[string]contracts.StageContentResult{
		"builder": stageResult("complete", map[string]contracts.ArtifactRef{}),
	}}
	inspector := &fakeInspector{mediaTypes: map[string]string{
		refKey(exactRef("builder", "report", revision)): "application/json",
	}}
	instance, err := mustFactory(t, newFakeSessions(), workers, inspector, model, Limits{}).
		Create(testInvocation("builder"))
	if err != nil {
		t.Fatal(err)
	}
	result, err := instance.Run(t.Context())
	if err != nil || result.Outcome != contracts.StageSucceeded {
		t.Fatalf("Run = (%+v, %v)", result, err)
	}
	if len(workers.calls) != 1 {
		t.Fatalf("Worker calls = %d, want exactly one after invalid arguments", len(workers.calls))
	}
	request := workers.calls[0].request
	if request.Objective != "Build the report" || request.Instructions != "Use the immutable Stage context" ||
		request.Parameters["mode"] != "strict" || request.Artifacts["source"].Revision == nil ||
		*request.Artifacts["source"].Revision != "source-r1" {
		t.Fatalf("model changed deterministic Worker request: %+v", request)
	}
}

func TestStreamlineRejectsInvalidWorkerCardinalityBeforeSideEffects(t *testing.T) {
	for _, bindings := range [][]string{nil, {"analyzer", "reviewer"}} {
		model := &scriptedModel{fallback: textStep("must not run")}
		workers := &fakeWorkerInvoker{}
		sessions := newFakeSessions()
		inspector := &fakeInspector{}
		_, err := mustFactory(t, sessions, workers, inspector, model, Limits{}).
			Create(testInvocation(bindings...))
		if err == nil || !strings.Contains(err.Error(), "requires exactly one logical Agent binding") {
			t.Fatalf("Create(%v) error = %v", bindings, err)
		}
		if sessions.adkCalls != 0 || model.callCount() != 0 || len(workers.calls) != 0 {
			t.Fatalf("invalid factory caused side effects: sessions=%d model=%d Workers=%d",
				sessions.adkCalls, model.callCount(), len(workers.calls))
		}
	}
}

func TestConfiguredFactoryBuildsEachPlannerFromInvocationModelAccess(t *testing.T) {
	sessions := newFakeSessions()
	workers := &fakeWorkerInvoker{}
	inspector := &fakeInspector{}
	models := []*scriptedModel{{}, {}}
	var accesses []planner.ModelAccess
	factory, err := NewConfiguredFactory(
		sessions, sessions, workers, inspector, unavailableWorkerStateReader{},
		func(access planner.ModelAccess) (model.LLM, error) {
			accesses = append(accesses, access)
			return models[len(accesses)-1], nil
		},
		Limits{MaxWallTime: 19 * time.Second},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := factory.Create(testInvocation("builder")); err == nil ||
		!strings.Contains(err.Error(), "requires resolved Planner model access") {
		t.Fatalf("missing ModelAccess error = %v", err)
	}

	for index, values := range []struct {
		model, gateway, credential, token string
		modelCalls, workerCalls, tokens   int
	}{
		{"planner-small", "small", "credential-small", "secret-small", 3, 4, 5000},
		{"planner-strong", "strong", "credential-strong", "secret-strong", 7, 9, 15000},
	} {
		credential := contracts.LLMCredentialRef{CredentialID: values.credential}
		access := planner.ModelAccess{
			ModelPolicy: contracts.ResolvedModelPolicy{
				Ref: contracts.ModelPolicyRef{
					PolicyID: values.model, Version: "1", Digest: "sha256:" + strings.Repeat(string(rune('a'+index)), 64),
				},
				Model: values.model, MaxOutputTokens: 2048 + index,
				MaxModelCalls: values.modelCalls, MaxWorkerCalls: values.workerCalls,
				MaxTotalTokens: values.tokens,
			},
			LLMGateway: contracts.ResolvedLLMGatewayConfig{
				Ref: contracts.LLMGatewayConfigRef{
					GatewayID: values.gateway, Version: "1", Digest: "sha256:" + strings.Repeat(string(rune('c'+index)), 64),
				},
				Protocol: contracts.OpenAICompatibleProtocol,
				URL:      "https://" + values.gateway + ".example/v1",
			},
			Credential: &credential,
			Token:      contracts.NewSecretString(values.token),
		}
		invocation := testInvocation("builder")
		invocation.ModelAccess = &access
		created, err := factory.Create(invocation)
		if err != nil {
			t.Fatal(err)
		}
		instance := created.(*streamlinePlanner)
		if instance.model != models[index] || instance.limits.MaxModelCalls != values.modelCalls ||
			instance.limits.MaxWorkerCalls != values.workerCalls ||
			instance.limits.MaxTokens != int64(values.tokens) ||
			instance.limits.MaxWallTime != 19*time.Second {
			t.Fatalf("configured Planner %d = model:%T limits:%+v", index, instance.model, instance.limits)
		}
	}
	if len(accesses) != 2 || accesses[0].ModelPolicy.Model == accesses[1].ModelPolicy.Model ||
		accesses[0].LLMGateway.URL == accesses[1].LLMGateway.URL ||
		accesses[0].Token.Reveal() == accesses[1].Token.Reveal() {
		t.Fatalf("per-invocation model access = %+v", accesses)
	}
}

func TestStreamlineRejectsPendingSuccessfulFinishThenCompletesSubtask(t *testing.T) {
	revision := "report-r1"
	model := &scriptedModel{steps: []modelStep{
		addSubtaskStep("Build the report", "Produce the declared output"),
		functionStep(finishToolName, map[string]any{
			"outcome": string(contracts.StageSucceeded), "summary": "premature",
			"artifacts": map[string]any{"report": artifactArgs("builder", "report", revision)},
		}),
		functionStep(executeCurrentSubtaskToolName, map[string]any{"subtask_id": "0"}),
		functionStep(finishToolName, map[string]any{
			"outcome": string(contracts.StageSucceeded), "summary": "complete",
			"artifacts": map[string]any{"report": artifactArgs("builder", "report", revision)},
		}),
	}}
	workers := &fakeWorkerInvoker{results: map[string]contracts.StageContentResult{
		"builder": stageResult("work complete", map[string]contracts.ArtifactRef{}),
	}}
	inspector := &fakeInspector{mediaTypes: map[string]string{
		refKey(exactRef("builder", "report", revision)): "application/json",
	}}
	instance, err := mustFactory(t, newFakeSessions(), workers, inspector, model, Limits{}).
		Create(testInvocation("builder"))
	if err != nil {
		t.Fatal(err)
	}
	result, err := instance.Run(t.Context())
	if err != nil || result.Summary != "complete" || len(workers.calls) != 1 {
		t.Fatalf("Run=(%+v,%v) Worker calls=%d", result, err, len(workers.calls))
	}
	report, _ := instance.(planner.ReportProvider).ExecutionReport()
	finishMetrics := report.Metrics.Tools[finishToolName]
	if finishMetrics.Calls == nil || *finishMetrics.Calls != 2 ||
		finishMetrics.Failed == nil || *finishMetrics.Failed != 1 ||
		finishMetrics.Succeeded == nil || *finishMetrics.Succeeded != 1 {
		t.Fatalf("finish metrics = %+v", finishMetrics)
	}
}

func TestStreamlineRejectsStaleSubtaskBeforeWorkerSideEffect(t *testing.T) {
	revision := "report-r1"
	model := &scriptedModel{steps: []modelStep{
		addSubtaskStep("First", "Execute first"),
		addSubtaskStep("Second", "Execute second"),
		functionStep(listSubtasksToolName, map[string]any{}),
		functionStep(executeCurrentSubtaskToolName, map[string]any{"subtask_id": "1"}),
		functionStep(executeCurrentSubtaskToolName, map[string]any{"subtask_id": "0"}),
		functionStep(executeCurrentSubtaskToolName, map[string]any{"subtask_id": "1"}),
		functionStep(finishToolName, map[string]any{
			"outcome": string(contracts.StageSucceeded), "summary": "complete",
			"artifacts": map[string]any{"report": artifactArgs("builder", "report", revision)},
		}),
	}}
	workers := &fakeWorkerInvoker{results: map[string]contracts.StageContentResult{
		"builder": stageResult("work complete", map[string]contracts.ArtifactRef{}),
	}}
	inspector := &fakeInspector{mediaTypes: map[string]string{
		refKey(exactRef("builder", "report", revision)): "application/json",
	}}
	instance, err := mustFactory(t, newFakeSessions(), workers, inspector, model, Limits{}).
		Create(testInvocation("builder"))
	if err != nil {
		t.Fatal(err)
	}
	result, err := instance.Run(t.Context())
	if err != nil || result.Outcome != contracts.StageSucceeded {
		t.Fatalf("Run = (%+v, %v)", result, err)
	}
	if got := workers.bindingCalls(); !reflect.DeepEqual(got, []string{"builder", "builder"}) {
		t.Fatalf("Worker calls = %v", got)
	}
	if workers.calls[0].request.SubtaskID != "0" || workers.calls[1].request.SubtaskID != "1" {
		t.Fatalf("Worker subtask IDs = %q, %q", workers.calls[0].request.SubtaskID, workers.calls[1].request.SubtaskID)
	}
	streamline := instance.(*streamlinePlanner)
	plan := streamline.plan.Snapshot()
	if plan.Revision != 6 || plan.CurrentSubtaskID != "" || plan.ActiveDispatch != nil ||
		plan.Subtasks[0].Status != planner.PlannerSubtaskSucceeded ||
		plan.Subtasks[1].Status != planner.PlannerSubtaskSucceeded {
		t.Fatalf("plan = %+v", plan)
	}
	report, _ := instance.(planner.ReportProvider).ExecutionReport()
	workerMetrics := report.Metrics.Tools[executeCurrentSubtaskToolName]
	if workerMetrics.Calls == nil || *workerMetrics.Calls != 3 ||
		workerMetrics.Failed == nil || *workerMetrics.Failed != 1 {
		t.Fatalf("Worker metrics = %+v", workerMetrics)
	}
}

func TestStreamlineWorkerErrorClearsClaimAndAllowsNextSubtask(t *testing.T) {
	revision := "report-r1"
	model := &scriptedModel{steps: []modelStep{
		addSubtaskStep("First", "This dispatch will fail"),
		addSubtaskStep("Second", "Produce the report"),
		functionStep(executeCurrentSubtaskToolName, map[string]any{"subtask_id": "0"}),
		functionStep(executeCurrentSubtaskToolName, map[string]any{"subtask_id": "1"}),
		functionStep(finishToolName, map[string]any{
			"outcome": string(contracts.StageSucceeded), "summary": "recovered",
			"artifacts": map[string]any{"report": artifactArgs("builder", "report", revision)},
		}),
	}}
	workers := &fakeWorkerInvoker{
		failures: map[string][]error{"builder": {errors.New("temporary Worker failure")}},
		results: map[string]contracts.StageContentResult{
			"builder": stageResult("report complete", map[string]contracts.ArtifactRef{}),
		},
	}
	inspector := &fakeInspector{mediaTypes: map[string]string{
		refKey(exactRef("builder", "report", revision)): "application/json",
	}}
	instance, err := mustFactory(t, newFakeSessions(), workers, inspector, model, Limits{}).
		Create(testInvocation("builder"))
	if err != nil {
		t.Fatal(err)
	}
	result, err := instance.Run(t.Context())
	if err != nil || result.Outcome != contracts.StageSucceeded || len(workers.calls) != 2 {
		t.Fatalf("Run=(%+v,%v) Worker calls=%d", result, err, len(workers.calls))
	}
	plan := instance.(*streamlinePlanner).plan.Snapshot()
	if plan.ActiveDispatch != nil || plan.CurrentSubtaskID != "" ||
		plan.Subtasks[0].Status != planner.PlannerSubtaskFailed ||
		plan.Subtasks[1].Status != planner.PlannerSubtaskSucceeded {
		t.Fatalf("plan retained a stale claim after Worker error: %+v", plan)
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
		addSubtaskStep("First", "Run the first call"),
		addSubtaskStep("Second", "Run the second call"),
		functionStep(executeCurrentSubtaskToolName, map[string]any{"subtask_id": "0"}),
		functionStep(executeCurrentSubtaskToolName, map[string]any{"subtask_id": "1"}),
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

func addSubtaskStep(objective, instructions string) modelStep {
	return functionStep(addSubtaskToolName, map[string]any{
		"objective": objective, "instructions": instructions,
	})
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
	identity    planner.SessionIdentity
	recovered   *planner.Completion
	completion  *planner.Completion
	request     planner.RequestFacts
	plan        *planner.PlannerPlanProjection
	transitions []planner.PlannerPlanTransition
	facts       []planner.PlannerFact
	adkCalls    int
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

func (s *fakeSessions) LoadPlan(
	context.Context,
	planner.SessionIdentity,
) (planner.PlannerPlanProjection, bool, error) {
	if s.plan == nil {
		return planner.PlannerPlanProjection{}, false, nil
	}
	return cloneTestPlanProjection(*s.plan), true, nil
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
	mu          sync.Mutex
	results     map[string]contracts.StageContentResult
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
		return planner.CloneWorkerCompletion(completion), nil
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
	factory, err := NewFactory(
		sessions, sessions, workers, inspector, unavailableWorkerStateReader{}, llm, limits,
	)
	if err != nil {
		t.Fatal(err)
	}
	return factory
}

type unavailableWorkerStateReader struct{}

func (unavailableWorkerStateReader) ReadWorkerState(
	context.Context,
	contracts.WorkerHandle,
	string,
) (planner.WorkerStateReadResult, error) {
	return planner.WorkerStateReadResult{}, &planner.WorkerStateReadError{
		Code: "runtime_unavailable", Retryable: true,
	}
}

var _ planner.WorkerStateReader = unavailableWorkerStateReader{}

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

func workerCompletion(
	subtaskID string,
	result string,
	artifacts map[string]contracts.ArtifactRef,
) contracts.WorkerCompletion {
	return contracts.WorkerCompletion{
		APIVersion: contracts.APIVersion,
		Result: &contracts.WorkerResult{
			SubtaskID: subtaskID, Result: result,
			Observations: contracts.WorkerObservations{
				Profile: contracts.WorkerObservationProfileLeanV1,
				Tools:   map[string]contracts.ToolObservationCount{},
			},
			Artifacts: artifacts, Summarized: false,
		},
		InvocationID: "worker-invocation-1", StateRevision: 2,
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

func cloneTestPlanProjection(value planner.PlannerPlanProjection) planner.PlannerPlanProjection {
	result := value
	result.Subtasks = append([]planner.PlannerSubtask(nil), value.Subtasks...)
	if value.ActiveDispatch != nil {
		active := *value.ActiveDispatch
		result.ActiveDispatch = &active
	}
	return result
}

var _ planner.PlanSessionService = (*fakeSessions)(nil)
var _ ADKSessionFactory = (*fakeSessions)(nil)
var _ model.LLM = (*scriptedModel)(nil)
