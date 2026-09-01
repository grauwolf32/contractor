package streamline

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/grauwolf32/contractor/internal/artifacts"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	plannermemory "github.com/grauwolf32/contractor/internal/memory"
	"github.com/grauwolf32/contractor/internal/planner"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

func TestPlannerMemorySchemasMirrorExactSelections(t *testing.T) {
	store := newPlannerMemoryArtifactStore()
	streamlineInvocation := memoryInvocation("streamline", map[string][]string{
		"builder": {
			plannermemory.ToolListMemories,
			plannermemory.ToolReadMemory,
			plannermemory.ToolWriteMemory,
		},
	})
	streamline := newMemoryPlanner(t, streamlineProfile, store, streamlineInvocation, &scriptedModel{})
	streamlineTools := plannerTools(t, streamline)
	if got, want := memoryToolNames(streamlineTools), []string{
		plannermemory.ToolListMemories,
		plannermemory.ToolReadMemory,
		plannermemory.ToolWriteMemory,
	}; !reflect.DeepEqual(got, want) {
		t.Fatalf("Streamline Memory tools = %v, want %v", got, want)
	}
	for _, name := range memoryToolNames(streamlineTools) {
		schema := toolSchema(t, streamlineTools[name])
		if strings.Contains(schema, `"worker_name"`) || strings.Contains(schema, `"namespace"`) ||
			strings.Contains(schema, `"revision"`) {
			t.Fatalf("Streamline %s schema exposes adapter state: %s", name, schema)
		}
	}
	writeSchema := toolSchema(t, streamlineTools[plannermemory.ToolWriteMemory])
	if strings.Contains(writeSchema, `"uniqueItems":true`) || strings.Contains(writeSchema, `"maxItems":3`) {
		t.Fatalf("write_memory schema rejects tags that its codec must normalize: %s", writeSchema)
	}

	routerInvocation := memoryInvocation("router", map[string][]string{
		"builder": plannermemory.OperationNames(),
		"reviewer": {
			plannermemory.ToolListMemories,
			plannermemory.ToolReadMemory,
			plannermemory.ToolSearchMemory,
		},
	})
	router := newMemoryPlanner(t, routerProfile, store, routerInvocation, &scriptedModel{})
	routerTools := plannerTools(t, router)
	if got, want := memoryToolNames(routerTools), plannermemory.OperationNames(); !reflect.DeepEqual(got, want) {
		t.Fatalf("Router Memory tools = %v, want %v", got, want)
	}
	for _, operation := range plannermemory.OperationNames() {
		declaration := toolDeclaration(t, routerTools[operation])
		schema, ok := declaration.ParametersJsonSchema.(*jsonschema.Schema)
		if !ok {
			t.Fatalf("%s schema type = %T", operation, declaration.ParametersJsonSchema)
		}
		worker := schema.Properties["worker_name"]
		got := make([]string, len(worker.Enum))
		for index, value := range worker.Enum {
			got[index], _ = value.(string)
		}
		want := []string{"builder"}
		if operation == plannermemory.ToolListMemories || operation == plannermemory.ToolReadMemory ||
			operation == plannermemory.ToolSearchMemory {
			want = []string{"builder", "reviewer"}
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("%s worker_name enum = %v, want %v", operation, got, want)
		}
	}
}

func TestPlannerAndWorkerShareOneLogicalMemoryNamespace(t *testing.T) {
	const (
		bodyCanary        = "planner-memory-body-canary"
		descriptionCanary = "planner-memory-description-canary"
		tagCanary         = "planner-memory-tag-canary"
	)
	store := newPlannerMemoryArtifactStore()
	invocation := memoryInvocation("streamline", map[string][]string{
		"builder": plannermemory.OperationNames(),
	})
	workerMemory, err := plannermemory.NewNamespace(store, plannermemory.Binding{
		RunID: invocation.RunID, StageExecutionID: invocation.StageExecutionID,
		Namespace: invocation.Stage.Agents["builder"].Namespace,
	})
	if err != nil {
		t.Fatal(err)
	}
	worker := &memoryWorkerInvoker{memory: workerMemory, expected: bodyCanary}
	model := &scriptedModel{steps: []modelStep{
		functionStep(plannermemory.ToolWriteMemory, map[string]any{
			"name": "shared_note", "content": bodyCanary,
			"description": descriptionCanary, "tags": []any{tagCanary, tagCanary},
		}),
		func(request *model.LLMRequest) (*model.LLMResponse, error) {
			response := requireFunctionResponse(t, request, plannermemory.ToolWriteMemory)
			assertFullMemoryResult(t, response, bodyCanary, descriptionCanary, []string{tagCanary})
			return addSubtaskStep("Update the shared note", "Append the Worker observation")(request)
		},
		functionStep(executeCurrentSubtaskToolName, map[string]any{"subtask_id": "0"}),
		functionStep(plannermemory.ToolReadMemory, map[string]any{"name": "shared_note"}),
		func(request *model.LLMRequest) (*model.LLMResponse, error) {
			response := requireFunctionResponse(t, request, plannermemory.ToolReadMemory)
			assertFullMemoryResult(
				t, response, bodyCanary+"\nworker update", descriptionCanary, []string{tagCanary},
			)
			return functionStep(finishToolName, map[string]any{
				"outcome": string(contracts.StageSucceeded), "summary": "shared Memory observed",
				"artifacts": map[string]any{},
			})(request)
		},
	}}
	sessions := newFakeSessions()
	factory, err := NewFactoryWithMemory(
		sessions, sessions, worker, &fakeInspector{}, store, model, Limits{},
	)
	if err != nil {
		t.Fatal(err)
	}
	instance, err := factory.Create(invocation)
	if err != nil {
		t.Fatal(err)
	}
	result, err := instance.Run(t.Context())
	if err != nil || result.Outcome != contracts.StageSucceeded || !worker.updated {
		t.Fatalf("Planner Run = (%+v, %v), Worker updated=%v", result, err, worker.updated)
	}
	report, ok := instance.(planner.ReportProvider).ExecutionReport()
	if !ok {
		t.Fatal("Planner report is unavailable")
	}
	retained, _ := json.Marshal(struct {
		Report      contracts.ExecutionReport
		Request     planner.RequestFacts
		Plan        *planner.PlannerPlanProjection
		Transitions []planner.PlannerPlanTransition
		Facts       []planner.PlannerFact
	}{report, sessions.request, sessions.plan, sessions.transitions, sessions.facts})
	for _, forbidden := range []string{bodyCanary, descriptionCanary, tagCanary, "revision-"} {
		if strings.Contains(string(retained), forbidden) {
			t.Fatalf("durable Planner diagnostics retained %q: %s", forbidden, retained)
		}
	}
	if store.semanticWrites != 2 {
		t.Fatalf("semantic Memory writes = %d, want Planner write plus Worker append", store.semanticWrites)
	}
}

func TestPlannerMemoryOwnsBoundedArgumentAndRoutingErrors(t *testing.T) {
	tests := []struct {
		name      string
		profile   plannerProfile
		bindings  map[string][]string
		operation string
		arguments map[string]any
		wantCode  string
		seed      bool
		conflict  bool
	}{
		{
			name: "unknown Router Worker", profile: routerProfile,
			bindings:  map[string][]string{"builder": {plannermemory.ToolWriteMemory}},
			operation: plannermemory.ToolWriteMemory,
			arguments: map[string]any{
				"worker_name": "stale", "name": "safe_note", "content": "must-not-write",
			},
			wantCode: plannermemory.CodeForbidden,
		},
		{
			name: "unknown Router Worker wins over malformed note", profile: routerProfile,
			bindings:  map[string][]string{"builder": {plannermemory.ToolWriteMemory}},
			operation: plannermemory.ToolWriteMemory,
			arguments: map[string]any{
				"worker_name": "stale", "name": 42, "content": "must-not-write",
				"unexpected": true,
			},
			wantCode: plannermemory.CodeForbidden,
		},
		{
			name: "extra Streamline argument", profile: streamlineProfile,
			bindings:  map[string][]string{"builder": {plannermemory.ToolWriteMemory}},
			operation: plannermemory.ToolWriteMemory,
			arguments: map[string]any{
				"name": "safe_note", "content": "must-not-write", "namespace": "other",
			},
			wantCode: plannermemory.CodeInvalid,
		},
		{
			name: "hidden CAS conflict", profile: streamlineProfile,
			bindings:  map[string][]string{"builder": {plannermemory.ToolWriteMemory}},
			operation: plannermemory.ToolWriteMemory,
			arguments: map[string]any{"name": "safe_note", "content": "replacement"},
			wantCode:  plannermemory.CodeChanged, seed: true, conflict: true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			store := newPlannerMemoryArtifactStore()
			invocation := memoryInvocation(strings.TrimSuffix(test.profile.ref, "@1"), test.bindings)
			if test.seed {
				bound, err := plannermemory.NewNamespace(store, plannermemory.Binding{
					RunID: invocation.RunID, StageExecutionID: invocation.StageExecutionID,
					Namespace: invocation.Stage.Agents["builder"].Namespace,
				})
				if err != nil {
					t.Fatal(err)
				}
				if _, err := bound.WriteMemory(t.Context(), "safe_note", "original", "", nil); err != nil {
					t.Fatal(err)
				}
			}
			beforeWrites := store.semanticWrites
			store.conflictNext = test.conflict
			model := &scriptedModel{steps: []modelStep{
				functionStep(test.operation, test.arguments),
				func(request *model.LLMRequest) (*model.LLMResponse, error) {
					response := requireFunctionResponse(t, request, test.operation)
					failure, ok := response["error"].(map[string]any)
					if !ok || failure["code"] != test.wantCode {
						t.Fatalf("Memory failure = %+v, want %s", response, test.wantCode)
					}
					return functionStep(finishToolName, map[string]any{
						"outcome": string(contracts.StageFailed), "summary": "expected rejection",
						"artifacts": map[string]any{},
						"error": map[string]any{
							"code": "expected_rejection", "message": "bounded test failure", "retryable": false,
						},
					})(request)
				},
			}}
			instance := newMemoryPlanner(t, test.profile, store, invocation, model)
			result, err := instance.Run(t.Context())
			if err != nil || result.Outcome != contracts.StageFailed {
				t.Fatalf("Planner Run = (%+v, %v)", result, err)
			}
			if store.semanticWrites != beforeWrites {
				t.Fatalf("rejected Memory call changed store: before=%d after=%d", beforeWrites, store.semanticWrites)
			}
		})
	}
}

func TestPlannerMemoryDependencyFailsBeforeModelFactory(t *testing.T) {
	invocation := memoryInvocation("streamline", map[string][]string{
		"builder": {plannermemory.ToolReadMemory},
	})
	modelCalls := 0
	sessions := newFakeSessions()
	factory, err := NewConfiguredFactory(
		sessions, sessions, &fakeWorkerInvoker{}, &fakeInspector{},
		func(planner.ModelAccess) (model.LLM, error) {
			modelCalls++
			return &scriptedModel{}, nil
		},
		Limits{},
	)
	if err != nil {
		t.Fatal(err)
	}
	invocation.ModelAccess = &planner.ModelAccess{}
	if _, err := factory.Create(invocation); err == nil || !strings.Contains(err.Error(), "Memory store") {
		t.Fatalf("Create error = %v, want missing Memory store", err)
	}
	if modelCalls != 0 || sessions.adkCalls != 0 {
		t.Fatalf("dependency failure caused side effects: model=%d adk=%d", modelCalls, sessions.adkCalls)
	}
}

func TestPlannerMemoryCallConsumesTheOrdinaryModelTurnBudget(t *testing.T) {
	const bodyCanary = "memory-budget-body-canary"
	store := newPlannerMemoryArtifactStore()
	invocation := memoryInvocation("streamline", map[string][]string{
		"builder": {plannermemory.ToolWriteMemory},
	})
	model := &scriptedModel{steps: []modelStep{
		functionStep(plannermemory.ToolWriteMemory, map[string]any{
			"name": "budget_note", "content": bodyCanary,
		}),
	}}
	sessions := newFakeSessions()
	factory, err := newFactory(
		streamlineProfile, sessions, sessions, &fakeWorkerInvoker{}, &fakeInspector{}, store,
		model, nil,
		Limits{MaxModelCalls: 1, MaxTokens: 1_000, MaxWorkerCalls: 4, MaxWallTime: time.Minute},
	)
	if err != nil {
		t.Fatal(err)
	}
	instance, err := factory.Create(invocation)
	if err != nil {
		t.Fatal(err)
	}
	_, err = instance.Run(t.Context())
	assertPlannerCode(t, err, "planner_model_call_limit")
	if model.callCount() != 1 || store.semanticWrites != 1 || sessions.completion == nil ||
		sessions.completion.Failure == nil ||
		sessions.completion.Failure.Code != "planner_model_call_limit" {
		t.Fatalf(
			"model calls=%d writes=%d completion=%+v",
			model.callCount(), store.semanticWrites, sessions.completion,
		)
	}
	report, ok := instance.(planner.ReportProvider).ExecutionReport()
	if !ok || report.Metrics.ModelCalls == nil || *report.Metrics.ModelCalls != 1 {
		t.Fatalf("Planner report = (%+v, %t)", report, ok)
	}
	metric := report.Metrics.Tools[plannermemory.ToolWriteMemory]
	if metric.Calls == nil || *metric.Calls != 1 || metric.Succeeded == nil || *metric.Succeeded != 1 {
		t.Fatalf("Memory metrics = %+v", metric)
	}
	encoded, _ := json.Marshal(report)
	if strings.Contains(string(encoded), bodyCanary) {
		t.Fatalf("Planner report retained Memory body: %s", encoded)
	}
}

func newMemoryPlanner(
	t *testing.T,
	profile plannerProfile,
	store plannermemory.Store,
	invocation planner.Invocation,
	llm model.LLM,
) *streamlinePlanner {
	t.Helper()
	sessions := newFakeSessions()
	factory, err := newFactory(
		profile, sessions, sessions, &fakeWorkerInvoker{}, &fakeInspector{}, store,
		llm, nil, Limits{},
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

func memoryInvocation(plannerID string, selections map[string][]string) planner.Invocation {
	names := make([]string, 0, len(selections))
	for name := range selections {
		names = append(names, name)
	}
	sort.Strings(names)
	invocation := testInvocation(names...)
	invocation.Stage.Planner = workflowconfig.PlannerRef{PlannerID: plannerID, Version: "1"}
	invocation.Stage.Result.Artifacts = map[string]workflowconfig.ArtifactSlot{}
	for _, name := range names {
		binding := invocation.Stage.Agents[name]
		binding.Template.Toolsets = []contracts.ToolsetSelection{{
			Ref:   contracts.ToolsetRef{ToolsetID: "memory-tools", Version: "1"},
			Tools: append([]string(nil), selections[name]...),
		}}
		invocation.Stage.Agents[name] = binding
	}
	return invocation
}

func plannerTools(t *testing.T, value *streamlinePlanner) map[string]tool.Tool {
	t.Helper()
	tools, _, err := value.buildTools(newExecutionState(value.limits), newFakeSessions().identity)
	if err != nil {
		t.Fatal(err)
	}
	result := make(map[string]tool.Tool, len(tools))
	for _, current := range tools {
		result[current.Name()] = current
	}
	return result
}

func memoryToolNames(tools map[string]tool.Tool) []string {
	result := make([]string, 0, len(plannermemory.OperationNames()))
	for _, operation := range plannermemory.OperationNames() {
		if _, present := tools[operation]; present {
			result = append(result, operation)
		}
	}
	return result
}

func toolDeclaration(t *testing.T, value tool.Tool) *genai.FunctionDeclaration {
	t.Helper()
	provider, ok := value.(interface {
		Declaration() *genai.FunctionDeclaration
	})
	if !ok {
		t.Fatalf("tool %T has no declaration", value)
	}
	return provider.Declaration()
}

func toolSchema(t *testing.T, value tool.Tool) string {
	t.Helper()
	encoded, err := json.Marshal(toolDeclaration(t, value).ParametersJsonSchema)
	if err != nil {
		t.Fatal(err)
	}
	return string(encoded)
}

func requireFunctionResponse(
	t *testing.T,
	request *model.LLMRequest,
	name string,
) map[string]any {
	t.Helper()
	for contentIndex := len(request.Contents) - 1; contentIndex >= 0; contentIndex-- {
		content := request.Contents[contentIndex]
		if content == nil {
			continue
		}
		for partIndex := len(content.Parts) - 1; partIndex >= 0; partIndex-- {
			part := content.Parts[partIndex]
			if part != nil && part.FunctionResponse != nil && part.FunctionResponse.Name == name {
				return part.FunctionResponse.Response
			}
		}
	}
	t.Fatalf("request has no %s FunctionResponse", name)
	return nil
}

func assertFullMemoryResult(
	t *testing.T,
	response map[string]any,
	content string,
	description string,
	tags []string,
) {
	t.Helper()
	wantKeys := []string{"content", "created_at", "description", "name", "ordinal", "tags", "updated_at"}
	gotKeys := make([]string, 0, len(response))
	for name := range response {
		gotKeys = append(gotKeys, name)
	}
	sort.Strings(gotKeys)
	if !reflect.DeepEqual(gotKeys, wantKeys) || response["content"] != content ||
		response["description"] != description {
		t.Fatalf("full Memory result = %+v", response)
	}
	gotTags, _ := response["tags"].([]any)
	if len(gotTags) != len(tags) {
		t.Fatalf("Memory tags = %+v, want %v", gotTags, tags)
	}
	for index := range tags {
		if gotTags[index] != tags[index] {
			t.Fatalf("Memory tags = %+v, want %v", gotTags, tags)
		}
	}
	encoded, _ := json.Marshal(response)
	for _, forbidden := range []string{"artifact", "namespace", "revision", "note"} {
		if strings.Contains(string(encoded), `"`+forbidden+`"`) {
			t.Fatalf("Memory result exposes %q: %s", forbidden, encoded)
		}
	}
}

type memoryWorkerInvoker struct {
	memory   *plannermemory.Namespace
	expected string
	updated  bool
}

func (w *memoryWorkerInvoker) Invoke(
	ctx context.Context,
	_ string,
	_ contracts.WorkerHandle,
	_ contracts.StageContentRequest,
) (contracts.StageContentResult, error) {
	note, err := w.memory.ReadMemory(ctx, "shared_note")
	if err != nil {
		return contracts.StageContentResult{}, err
	}
	if note.Content != w.expected {
		return contracts.StageContentResult{}, errors.New("Worker observed unexpected Memory content")
	}
	if _, err := w.memory.AppendMemory(ctx, "shared_note", "worker update"); err != nil {
		return contracts.StageContentResult{}, err
	}
	w.updated = true
	return stageResult("updated shared Memory", map[string]contracts.ArtifactRef{}), nil
}

type plannerMemoryArtifact struct {
	payload           artifacts.Payload
	revision          string
	bindingCreatedAt  time.Time
	revisionCreatedAt time.Time
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

func (s *plannerMemoryArtifactStore) List(
	_ context.Context,
	binding plannermemory.Binding,
) ([]artifacts.ArtifactRef, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	prefix := binding.RunID + "/" + binding.Namespace + "/"
	result := make([]artifacts.ArtifactRef, 0)
	for key := range s.bindings {
		if strings.HasPrefix(key, prefix) {
			result = append(result, artifacts.ArtifactRef{
				Namespace: binding.Namespace, Name: strings.TrimPrefix(key, prefix),
			})
		}
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Name < result[j].Name })
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

func memoryStoreKey(binding plannermemory.Binding, name string) string {
	return binding.RunID + "/" + binding.Namespace + "/" + name
}

var _ plannermemory.Store = (*plannerMemoryArtifactStore)(nil)
var _ planner.WorkerInvoker = (*memoryWorkerInvoker)(nil)
