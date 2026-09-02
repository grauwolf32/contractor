package streamline

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/planner/stateview"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
)

func TestStateToolsExposeExactProfileSpecificSchemas(t *testing.T) {
	reader := &plannerStateTestReader{}
	streamline := newStateTestPlanner(
		t, streamlineProfile, stateToolInvocation("streamline", "builder"), reader,
		&fakeWorkerInvoker{}, &scriptedModel{},
	)
	streamlineTools := plannerTools(t, streamline)
	t.Cleanup(streamline.stateViews.Close)
	for _, name := range stateToolNames {
		current, exists := streamlineTools[name]
		if !exists {
			t.Fatalf("Streamline State tool %q is absent", name)
		}
		schema := stateInputSchema(t, current)
		if _, exposed := schema.Properties["worker_name"]; exposed || len(schema.Required) != 0 {
			t.Fatalf("Streamline %s schema exposes Worker routing: %+v", name, schema)
		}
		assertStateSchemaSurface(t, name, schema)
	}

	routerInvocation := stateToolInvocation("router", "builder", "reviewer")
	reviewer := routerInvocation.Stage.Agents["reviewer"]
	// Structural code analysis uses the workspace but does not represent a
	// model-observed content read in the Runtime tracker.
	reviewer.Template.Toolsets = []contracts.ToolsetSelection{{
		Ref:   contracts.ToolsetRef{ToolsetID: "code-analysis", Version: "1"},
		Tools: []string{"list_symbols"},
	}}
	routerInvocation.Stage.Agents["reviewer"] = reviewer
	router := newStateTestPlanner(
		t, routerProfile, routerInvocation, reader, &fakeWorkerInvoker{}, &scriptedModel{},
	)
	routerTools := plannerTools(t, router)
	t.Cleanup(router.stateViews.Close)
	for _, name := range stateToolNames {
		current, exists := routerTools[name]
		if !exists {
			t.Fatalf("Router State tool %q is absent", name)
		}
		schema := stateInputSchema(t, current)
		worker := schema.Properties["worker_name"]
		if worker == nil || !reflect.DeepEqual(schema.Required, []string{"worker_name"}) {
			t.Fatalf("Router %s lacks required worker_name: %+v", name, schema)
		}
		want := []any{"builder"}
		if name == getWorkerToolUsageToolName {
			want = []any{"builder", "reviewer"}
		}
		if !reflect.DeepEqual(worker.Enum, want) {
			t.Fatalf("Router %s worker enum = %v, want %v", name, worker.Enum, want)
		}
		assertStateSchemaSurface(t, name, schema)
	}
}

func TestStreamlineStateToolsProjectNewestCompletionWithoutDurablePaths(t *testing.T) {
	const (
		firstPath  = "src/main.py"
		secondPath = "README.md"
		unreadPath = "docs/unread.md"
	)
	revision := "report-r1"
	reader := &plannerStateTestReader{states: map[string]contracts.AgentStateSnapshot{
		"allocation-builder": plannerStateSnapshot(t, "worker-invocation-1", "0", 2),
	}}
	var opaqueCursor string
	modelValue := &scriptedModel{steps: []modelStep{
		addSubtaskStep("Inspect the workspace", "Return the report"),
		functionStep(executeCurrentSubtaskToolName, map[string]any{"subtask_id": "0"}),
		functionStep(listReadFilesToolName, map[string]any{"cursor": "", "limit": 1}),
		func(request *model.LLMRequest) (*model.LLMResponse, error) {
			page := requireStateResult(t, request, listReadFilesToolName, "page")
			assertStatePaths(t, page, []string{firstPath})
			opaqueCursor, _ = page["nextCursor"].(string)
			if opaqueCursor == "" {
				t.Fatal("first read page has no cursor")
			}
			return functionStep(listReadFilesToolName, map[string]any{
				"cursor": opaqueCursor, "limit": 1,
			})(request)
		},
		func(request *model.LLMRequest) (*model.LLMResponse, error) {
			page := requireStateResult(t, request, listReadFilesToolName, "page")
			assertStatePaths(t, page, []string{secondPath})
			if _, exists := page["nextCursor"]; exists {
				t.Fatalf("last read page retained a cursor: %+v", page)
			}
			return functionStep(listUnreadFilesToolName, map[string]any{"limit": 100})(request)
		},
		func(request *model.LLMRequest) (*model.LLMResponse, error) {
			page := requireStateResult(t, request, listUnreadFilesToolName, "page")
			assertStatePaths(t, page, []string{unreadPath})
			return functionStep(getWorkspaceCoverageToolName, map[string]any{})(request)
		},
		func(request *model.LLMRequest) (*model.LLMResponse, error) {
			coverage := requireStateResult(t, request, getWorkspaceCoverageToolName, "coverage")
			if coverage["scopedFiles"] != float64(3) || coverage["readFiles"] != float64(2) ||
				coverage["unreadFiles"] != float64(1) || coverage["scopeComplete"] != true ||
				coverage["detailComplete"] != true {
				t.Fatalf("coverage projection = %+v", coverage)
			}
			return functionStep(getWorkerToolUsageToolName, map[string]any{})(request)
		},
		func(request *model.LLMRequest) (*model.LLMResponse, error) {
			usage := requireStateResult(t, request, getWorkerToolUsageToolName, "usage")
			if usage["modelCalls"] != float64(1) || usage["toolCalls"] != float64(3) ||
				usage["toolErrors"] != float64(1) {
				t.Fatalf("tool usage projection = %+v", usage)
			}
			return functionStep(finishToolName, map[string]any{
				"outcome": string(contracts.StageSucceeded), "summary": "complete",
				"artifacts": map[string]any{
					"report": artifactArgs("builder", "report", revision),
				},
			})(request)
		},
	}}
	workers := &fakeWorkerInvoker{results: map[string]contracts.StageContentResult{
		"builder": stageResult("complete", map[string]contracts.ArtifactRef{
			"report": exactRef("builder", "report", revision),
		}),
	}}
	sessions := newFakeSessions()
	inspector := &fakeInspector{mediaTypes: map[string]string{
		refKey(exactRef("builder", "report", revision)): "application/json",
	}}
	factory, err := NewFactory(
		sessions, sessions, workers, inspector, reader, modelValue,
		Limits{MaxModelCalls: 12},
	)
	if err != nil {
		t.Fatal(err)
	}
	instance, err := factory.Create(stateToolInvocation("streamline", "builder"))
	if err != nil {
		t.Fatal(err)
	}
	result, err := instance.Run(t.Context())
	if err != nil || result.Outcome != contracts.StageSucceeded {
		t.Fatalf("Run = (%+v, %v)", result, err)
	}
	calls := reader.callsSnapshot()
	if len(calls) != 5 {
		t.Fatalf("Worker State reads = %+v", calls)
	}
	for index, call := range calls {
		if call.allocationID != "allocation-builder" {
			t.Fatalf("State read routed to %q", call.allocationID)
		}
		if index == 0 && call.ifNoneMatch != "" || index > 0 && call.ifNoneMatch != stateTestETag(2) {
			t.Fatalf("State conditional read %d = %q", index, call.ifNoneMatch)
		}
	}
	report, _ := instance.(planner.ReportProvider).ExecutionReport()
	retained, marshalErr := json.Marshal(struct {
		Report contracts.ExecutionReport
		Plan   *planner.PlannerPlanProjection
		Facts  []planner.PlannerFact
	}{report, sessions.plan, sessions.facts})
	if marshalErr != nil {
		t.Fatal(marshalErr)
	}
	for _, forbidden := range []string{
		firstPath, secondPath, unreadPath, opaqueCursor,
		"allocation-builder", "worker-invocation-1", stateTestETag(2),
	} {
		if forbidden != "" && strings.Contains(string(retained), forbidden) {
			t.Fatalf("durable Planner projection retained %q: %s", forbidden, retained)
		}
	}
}

func TestStateProjectionFailureDoesNotRewriteCompletedSubtask(t *testing.T) {
	revision := "report-r1"
	reader := &plannerStateTestReader{err: errors.New("private-runtime-secret")}
	modelValue := &scriptedModel{steps: []modelStep{
		addSubtaskStep("Build the report", "Use the source"),
		functionStep(executeCurrentSubtaskToolName, map[string]any{"subtask_id": "0"}),
		functionStep(getWorkerToolUsageToolName, map[string]any{}),
		func(request *model.LLMRequest) (*model.LLMResponse, error) {
			response := requireFunctionResponse(t, request, getWorkerToolUsageToolName)
			failure, _ := response["error"].(map[string]any)
			if response["ok"] != false || failure["code"] != stateview.CodeUnavailable {
				t.Fatalf("closed State failure = %+v", response)
			}
			return functionStep(finishToolName, map[string]any{
				"outcome": string(contracts.StageSucceeded), "summary": "complete",
				"artifacts": map[string]any{
					"report": artifactArgs("builder", "report", revision),
				},
			})(request)
		},
	}}
	workers := &fakeWorkerInvoker{results: map[string]contracts.StageContentResult{
		"builder": stageResult("complete", map[string]contracts.ArtifactRef{
			"report": exactRef("builder", "report", revision),
		}),
	}}
	sessions := newFakeSessions()
	factory, err := NewFactory(
		sessions, sessions, workers,
		&fakeInspector{mediaTypes: map[string]string{
			refKey(exactRef("builder", "report", revision)): "application/json",
		}},
		reader, modelValue, Limits{},
	)
	if err != nil {
		t.Fatal(err)
	}
	instance, err := factory.Create(testInvocation("builder"))
	if err != nil {
		t.Fatal(err)
	}
	result, err := instance.Run(t.Context())
	if err != nil || result.Outcome != contracts.StageSucceeded || sessions.plan == nil ||
		len(sessions.plan.Subtasks) != 1 ||
		sessions.plan.Subtasks[0].Status != planner.PlannerSubtaskSucceeded {
		t.Fatalf("State failure rewrote semantic execution: result=%+v err=%v plan=%+v", result, err, sessions.plan)
	}
	report, _ := instance.(planner.ReportProvider).ExecutionReport()
	encoded, _ := json.Marshal(report)
	if strings.Contains(string(encoded), "private-runtime-secret") {
		t.Fatalf("State failure leaked transport cause: %s", encoded)
	}
}

func TestStateProjectionFailureUsesClosedCodes(t *testing.T) {
	for _, test := range []struct {
		inputCode      string
		inputRetryable bool
		wantCode       string
		wantRetryable  bool
	}{
		{stateview.CodeUnavailable, false, stateview.CodeUnavailable, false},
		{stateview.CodeChanged, false, stateview.CodeChanged, true},
		{stateview.CodeWorkspaceIncomplete, true, stateview.CodeWorkspaceIncomplete, false},
		{"private_runtime_secret", false, stateview.CodeUnavailable, true},
	} {
		failure := stateToolFailure(&stateview.Error{
			Code: test.inputCode, Retryable: test.inputRetryable,
		})
		if failure.Code != test.wantCode || failure.Retryable != test.wantRetryable ||
			strings.Contains(failure.Message, "private_runtime_secret") {
			t.Fatalf("stateToolFailure(%q, %t) = %+v", test.inputCode, test.inputRetryable, failure)
		}
	}
}

func TestRouterStateToolReadsOnlySelectedLogicalWorker(t *testing.T) {
	reader := &plannerStateTestReader{states: map[string]contracts.AgentStateSnapshot{
		"allocation-builder":  plannerStateSnapshot(t, "worker-invocation-1", "0", 2),
		"allocation-reviewer": plannerStateSnapshot(t, "worker-invocation-1", "0", 2),
	}}
	modelValue := &scriptedModel{steps: []modelStep{
		addSubtaskStep("Inspect", "Use builder"),
		functionStep(executeCurrentSubtaskToolName, map[string]any{
			"subtask_id": "0", "worker_name": "builder",
		}),
		functionStep(getWorkerToolUsageToolName, map[string]any{"worker_name": "builder"}),
		func(request *model.LLMRequest) (*model.LLMResponse, error) {
			_ = requireStateResult(t, request, getWorkerToolUsageToolName, "usage")
			return functionStep(finishToolName, map[string]any{
				"outcome": string(contracts.StageSucceeded), "summary": "complete",
				"artifacts": map[string]any{},
			})(request)
		},
	}}
	workers := &fakeWorkerInvoker{results: map[string]contracts.StageContentResult{
		"builder": stageResult("complete", map[string]contracts.ArtifactRef{}),
	}}
	invocation := stateToolInvocation("router", "builder", "reviewer")
	invocation.Stage.Result.Artifacts = map[string]workflowconfig.ArtifactSlot{}
	plannerValue := newStateTestPlanner(
		t, routerProfile, invocation, reader, workers, modelValue,
	)
	result, err := plannerValue.Run(t.Context())
	if err != nil || result.Outcome != contracts.StageSucceeded {
		t.Fatalf("Router Run = (%+v, %v)", result, err)
	}
	if calls := reader.callsSnapshot(); len(calls) != 1 || calls[0].allocationID != "allocation-builder" {
		t.Fatalf("Router State calls = %+v", calls)
	}
}

func assertStateSchemaSurface(
	t *testing.T,
	operation string,
	schema *jsonschema.Schema,
) {
	t.Helper()
	allowed := map[string]struct{}{"worker_name": {}}
	if operation == listReadFilesToolName || operation == listUnreadFilesToolName {
		allowed["cursor"] = struct{}{}
		allowed["limit"] = struct{}{}
	}
	for name := range schema.Properties {
		if _, ok := allowed[name]; !ok {
			t.Fatalf("%s schema exposes undeclared argument %q", operation, name)
		}
	}
	encoded, _ := json.Marshal(schema)
	if !strings.Contains(string(encoded), `"additionalProperties":false`) {
		t.Fatalf("%s schema is open: %s", operation, encoded)
	}
	if operation == listReadFilesToolName || operation == listUnreadFilesToolName {
		cursor, limit := schema.Properties["cursor"], schema.Properties["limit"]
		if cursor == nil || cursor.Type != "string" || cursor.MaxLength == nil || *cursor.MaxLength != 128 ||
			string(cursor.Default) != `""` || limit == nil || limit.Type != "integer" ||
			limit.Minimum == nil || *limit.Minimum != 1 || limit.Maximum == nil ||
			*limit.Maximum != 100 || string(limit.Default) != `100` {
			t.Fatalf("%s pagination schema is not exact: %s", operation, encoded)
		}
	} else if schema.Properties["cursor"] != nil || schema.Properties["limit"] != nil {
		t.Fatalf("%s unexpectedly exposes pagination: %s", operation, encoded)
	}
	for _, forbidden := range []string{"allocation", "invocation", "revision", "endpoint", "state_key"} {
		if strings.Contains(string(encoded), forbidden) {
			t.Fatalf("%s schema exposes %q: %s", operation, forbidden, encoded)
		}
	}
}

func stateInputSchema(t *testing.T, current tool.Tool) *jsonschema.Schema {
	t.Helper()
	schema, ok := toolDeclaration(t, current).ParametersJsonSchema.(*jsonschema.Schema)
	if !ok {
		t.Fatalf("State tool %s has unexpected schema %T", current.Name(), toolDeclaration(t, current).ParametersJsonSchema)
	}
	return schema
}

func requireStateResult(
	t *testing.T,
	request *model.LLMRequest,
	operation string,
	field string,
) map[string]any {
	t.Helper()
	response := requireFunctionResponse(t, request, operation)
	if response["ok"] != true {
		t.Fatalf("%s response = %+v", operation, response)
	}
	result, ok := response[field].(map[string]any)
	if !ok {
		t.Fatalf("%s %s projection = %+v", operation, field, response[field])
	}
	encoded, _ := json.Marshal(response)
	for _, forbidden := range []string{"allocationId", "invocationId", "stateRevision", "runtimeUrl"} {
		if strings.Contains(string(encoded), forbidden) {
			t.Fatalf("%s result exposes %q: %s", operation, forbidden, encoded)
		}
	}
	return result
}

func assertStatePaths(t *testing.T, page map[string]any, want []string) {
	t.Helper()
	raw, ok := page["files"].([]any)
	if !ok {
		t.Fatalf("State page files = %+v", page["files"])
	}
	got := make([]string, len(raw))
	for index := range raw {
		got[index], _ = raw[index].(string)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("State page files = %v, want %v", got, want)
	}
}

func newStateTestPlanner(
	t *testing.T,
	profile plannerProfile,
	invocation planner.Invocation,
	reader planner.WorkerStateReader,
	workers *fakeWorkerInvoker,
	llm model.LLM,
) *streamlinePlanner {
	t.Helper()
	sessions := newFakeSessions()
	factory, err := newFactory(
		profile, sessions, sessions, workers, &fakeInspector{}, reader,
		nil, llm, nil, Limits{MaxModelCalls: 12},
	)
	if err != nil {
		t.Fatal(err)
	}
	instance, err := factory.Create(invocation)
	if err != nil {
		t.Fatal(err)
	}
	return instance.(*streamlinePlanner)
}

func stateToolInvocation(plannerID string, bindings ...string) planner.Invocation {
	invocation := testInvocation(bindings...)
	invocation.Stage.Planner = workflowconfig.PlannerRef{PlannerID: plannerID, Version: "1"}
	invocation.Stage.Context.Workspace = &workflowconfig.WorkspaceContext{
		Mode:    contracts.WorkspaceModeDirect,
		Sources: []workflowconfig.WorkspaceSource{{Artifact: "source", Target: ""}},
	}
	for _, name := range bindings {
		binding := invocation.Stage.Agents[name]
		binding.Template.Toolsets = []contracts.ToolsetSelection{{
			Ref:   contracts.ToolsetRef{ToolsetID: "filesystem", Version: "1"},
			Tools: []string{"read_file"},
		}}
		invocation.Stage.Agents[name] = binding
	}
	return invocation
}

func plannerStateSnapshot(
	t *testing.T,
	invocationID string,
	subtaskID string,
	revision uint64,
) contracts.AgentStateSnapshot {
	t.Helper()
	payload, err := os.ReadFile(filepath.Join(
		"..", "..", "..", "api", "testdata", "v1alpha1", "valid", "agent-state-snapshot.json",
	))
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := contracts.DecodeStrict[contracts.AgentStateSnapshot](payload)
	if err != nil {
		t.Fatal(err)
	}
	snapshot.State.StateRevision = revision
	snapshot.State.CurrentInvocation = nil
	completed := snapshot.State.LastCompletedInvocation
	completed.InvocationID = invocationID
	completed.SubtaskID = subtaskID
	completed.Metrics.ToolCalls = 3
	completed.Metrics.ToolErrors = 1
	completed.Metrics.Tools = map[string]contracts.WorkerStateInvocationToolMetric{
		"grep":      {Calls: 1, Failures: 1},
		"read_file": {Calls: 2},
	}
	completed.Workspace.ScopePaths = []string{"README.md", "docs/unread.md", "src/main.py"}
	completed.Workspace.Interactions = []contracts.WorkerStateWorkspaceInteraction{
		{Path: "src/main.py", FirstOrdinal: 1, LastOrdinal: 1, ReadCalls: 1},
		{Path: "README.md", FirstOrdinal: 2, LastOrdinal: 2, ReadCalls: 1},
	}
	if err := snapshot.Validate(); err != nil {
		t.Fatal(err)
	}
	return snapshot
}

type plannerStateReadCall struct {
	allocationID string
	ifNoneMatch  string
}

type plannerStateTestReader struct {
	mu     sync.Mutex
	states map[string]contracts.AgentStateSnapshot
	err    error
	calls  []plannerStateReadCall
}

func (r *plannerStateTestReader) ReadWorkerState(
	_ context.Context,
	handle contracts.WorkerHandle,
	ifNoneMatch string,
) (planner.WorkerStateReadResult, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.calls = append(r.calls, plannerStateReadCall{
		allocationID: handle.AllocationID, ifNoneMatch: ifNoneMatch,
	})
	if r.err != nil {
		return planner.WorkerStateReadResult{}, r.err
	}
	snapshot, exists := r.states[handle.AllocationID]
	if !exists {
		return planner.WorkerStateReadResult{}, &planner.WorkerStateReadError{
			Code: "runtime_unavailable", Retryable: true,
		}
	}
	etag := stateTestETag(snapshot.State.StateRevision)
	if ifNoneMatch == etag {
		return planner.WorkerStateReadResult{ETag: etag, NotModified: true}, nil
	}
	return planner.WorkerStateReadResult{Snapshot: &snapshot, ETag: etag}, nil
}

func (r *plannerStateTestReader) callsSnapshot() []plannerStateReadCall {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]plannerStateReadCall(nil), r.calls...)
}

func stateTestETag(revision uint64) string {
	return `"contractor-agent-state-v1-` + strconv.FormatUint(revision, 10) + `"`
}

var _ planner.WorkerStateReader = (*plannerStateTestReader)(nil)
