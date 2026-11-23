package streamline

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync/atomic"
	"testing"

	plannermemory "github.com/grauwolf32/contractor/internal/memory"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/planner/stateview"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	adksession "google.golang.org/adk/v2/session"
	"google.golang.org/genai"
)

func migrationFailedFinish(request *model.LLMRequest) (*model.LLMResponse, error) {
	return functionStep(finishToolName, map[string]any{
		"outcome": "failed", "summary": "bounded characterization", "artifacts": map[string]any{},
		"error": map[string]any{"code": "characterization_stop", "message": "test stop", "retryable": false},
	})(request)
}

func migrationPlanner(t *testing.T, profile, family string, llm model.LLM) (*streamlinePlanner, *plannerMemoryArtifactStore) {
	t.Helper()
	selected := streamlineProfile
	if profile == "router" {
		selected = routerProfile
	}
	if family == "memory" {
		store := newPlannerMemoryArtifactStore()
		invocation := memoryInvocation(profile, map[string][]string{"builder": plannermemory.OperationNames()})
		return newMemoryPlanner(t, selected, store, invocation, llm), store
	}
	invocation := testInvocation("builder")
	invocation.Stage.Planner.PlannerID = profile
	if family == "state" {
		invocation = stateToolInvocation(profile, "builder")
	}
	sessions := newFakeSessions()
	factory, err := newFactory(selected, sessions, sessions, &fakeWorkerInvoker{}, &fakeInspector{}, &plannerStateTestReader{}, nil, testModelFactory(llm), Limits{})
	if err != nil {
		t.Fatal(err)
	}
	instance, err := factory.Create(invocation)
	if err != nil {
		t.Fatal(err)
	}
	return instance.(*streamlinePlanner), nil
}

func TestGoADKMigrationSchemas(t *testing.T) {
	for _, profile := range []string{"streamline", "router"} {
		t.Run(profile, func(t *testing.T) {
			for _, family := range []string{"core", "memory", "state"} {
				t.Run(family, func(t *testing.T) {
					schemas := map[string]json.RawMessage{}
					llm := &scriptedModel{steps: []modelStep{func(request *model.LLMRequest) (*model.LLMResponse, error) {
						if request.Config == nil {
							return nil, errors.New("missing real Runner config")
						}
						for _, packed := range request.Config.Tools {
							for _, declaration := range packed.FunctionDeclarations {
								encoded, err := json.Marshal(declaration.ParametersJsonSchema)
								if err != nil {
									return nil, err
								}
								schemas[declaration.Name] = encoded
							}
						}
						return migrationFailedFinish(request)
					}}}
					instance, _ := migrationPlanner(t, profile, family, llm)
					result, err := instance.Run(t.Context())
					if err != nil || result.Outcome != stageresult.StageFailed {
						t.Fatalf("Run = (%+v,%v)", result, err)
					}
					names := []string{addSubtaskToolName, listSubtasksToolName, executeCurrentSubtaskToolName, finishToolName}
					if family == "memory" {
						names = plannermemory.OperationNames()
					}
					if family == "state" {
						names = append([]string(nil), stateToolNames[:]...)
					}
					for _, name := range names {
						var schema struct {
							Type       string                     `json:"type"`
							Properties map[string]json.RawMessage `json:"properties"`
							Required   []string                   `json:"required"`
						}
						if err := json.Unmarshal(schemas[name], &schema); err != nil || schema.Type != "object" {
							t.Fatalf("%s schema = %s (%v)", name, schemas[name], err)
						}
						if family != "core" || name == executeCurrentSubtaskToolName {
							_, routed := schema.Properties["worker_name"]
							if routed != (profile == "router") {
								t.Fatalf("%s worker_name exposed=%v for %s", name, routed, profile)
							}
							if routed {
								var worker struct {
									Type string   `json:"type"`
									Enum []string `json:"enum"`
								}
								if err := json.Unmarshal(schema.Properties["worker_name"], &worker); err != nil || worker.Type != "string" || !reflect.DeepEqual(worker.Enum, []string{"builder"}) {
									t.Fatalf("%s worker binding schema = %s", name, schema.Properties["worker_name"])
								}
							}
						}
						for _, forbidden := range []string{"namespace", "expected_revision", "runtime_agent_id"} {
							if _, found := schema.Properties[forbidden]; found {
								t.Fatalf("%s exposes %s", name, forbidden)
							}
						}
					}
					if llm.callCount() != 1 {
						t.Fatal("schema inspection caused another model operation")
					}
				})
			}
		})
	}
}

func TestGoADKMigrationRawArguments(t *testing.T) {
	rows := []struct {
		name, family, tool, code string
		args                     map[string]any
		executionError           bool
	}{
		{name: "core-extra", family: "core", tool: addSubtaskToolName, args: map[string]any{"objective": "test", "instructions": "test", "unexpected": true}, executionError: true},
		{name: "core-context", family: "core", tool: addSubtaskToolName, args: map[string]any{"objective": "test", "instructions": "test", "tool_context": "injected"}, executionError: true},
		{name: "core-nested-revision", family: "core", tool: finishToolName, args: map[string]any{"outcome": "succeeded", "summary": "test", "artifacts": map[string]any{"report": map[string]any{"namespace": "builder", "name": "report", "revision": "r1"}}}, executionError: true},
		{name: "core-nested-extra", family: "core", tool: finishToolName, args: map[string]any{"outcome": "succeeded", "summary": "test", "artifacts": map[string]any{"report": map[string]any{"namespace": "builder", "name": "report", "revision": "r1", "unexpected": true}}}, executionError: true},
		{name: "core-number", family: "core", tool: addSubtaskToolName, args: map[string]any{"objective": 42, "instructions": "test"}, executionError: true},
		{name: "core-bool", family: "core", tool: addSubtaskToolName, args: map[string]any{"objective": true, "instructions": "test"}, executionError: true},
		{name: "core-array", family: "core", tool: addSubtaskToolName, args: map[string]any{"objective": []any{"test"}, "instructions": "test"}, executionError: true},
		{name: "memory-number", family: "memory", tool: plannermemory.ToolWriteMemory, args: map[string]any{"name": 42, "content": "must not write"}, code: plannermemory.CodeInvalid},
		{name: "memory-null", family: "memory", tool: plannermemory.ToolWriteMemory, args: map[string]any{"name": "safe_note", "content": nil}, code: plannermemory.CodeInvalid},
		{name: "memory-extra", family: "memory", tool: plannermemory.ToolWriteMemory, args: map[string]any{"name": "safe_note", "content": "must not write", "namespace": "forbidden"}, code: plannermemory.CodeInvalid},
		{name: "state-fraction", family: "state", tool: listReadFilesToolName, args: map[string]any{"limit": 1.5}, code: stateview.CodeRequestInvalid},
		{name: "state-bool", family: "state", tool: listReadFilesToolName, args: map[string]any{"limit": true}, code: stateview.CodeRequestInvalid},
		{name: "state-null", family: "state", tool: listReadFilesToolName, args: map[string]any{"limit": nil}, code: stateview.CodeRequestInvalid},
		{name: "state-extra", family: "state", tool: listReadFilesToolName, args: map[string]any{"limit": 1, "unexpected": true}, code: stateview.CodeRequestInvalid},
	}
	for _, profile := range []string{"streamline", "router"} {
		t.Run(profile, func(t *testing.T) {
			for _, row := range rows {
				t.Run(row.name, func(t *testing.T) {
					arguments := make(map[string]any, len(row.args)+1)
					for k, v := range row.args {
						arguments[k] = v
					}
					if profile == "router" && row.family != "core" {
						arguments["worker_name"] = "builder"
					}
					var failureCode string
					llm := &scriptedModel{steps: []modelStep{
						functionStep(row.tool, arguments),
						func(request *model.LLMRequest) (*model.LLMResponse, error) {
							result := requireFunctionResponse(t, request, row.tool)
							if row.executionError {
								diagnostic, ok := result["error"].(string)
								if !ok || diagnostic == "" {
									return nil, fmt.Errorf("native argument rejection shape %T", result["error"])
								}
								failureCode = "native_argument_rejected"
								return migrationFailedFinish(request)
							}
							failure, ok := result["error"].(map[string]any)
							if !ok {
								return nil, errors.New("invalid arguments produced no owned tool failure")
							}
							failureCode, _ = failure["code"].(string)
							return migrationFailedFinish(request)
						},
					}}
					instance, store := migrationPlanner(t, profile, row.family, llm)
					result, err := instance.Run(t.Context())
					if row.executionError {
						if err != nil || result.Outcome != stageresult.StageFailed || failureCode != "native_argument_rejected" || llm.callCount() != 2 {
							t.Fatalf("malformed native args = (%+v,%v), calls=%d", result, err, llm.callCount())
						}
					} else if err != nil || result.Outcome != stageresult.StageFailed || failureCode != row.code || llm.callCount() != 2 {
						t.Fatalf("owned argument rejection = (%+v,%v), code=%s calls=%d", result, err, failureCode, llm.callCount())
					}
					if len(instance.plan.Snapshot().Subtasks) != 0 {
						t.Fatal("malformed args changed the plan")
					}
					if store != nil && store.semanticWrites != 0 {
						t.Fatal("malformed args changed Memory")
					}
				})
			}
		})
	}
}

type migrationCallbackModel struct {
	calls    int
	streams  []bool
	generate func(context.Context, *model.LLMRequest) (*model.LLMResponse, error)
}

func (*migrationCallbackModel) Name() string { return "migration-model" }
func (m *migrationCallbackModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.calls++
		m.streams = append(m.streams, stream)
		response, err := m.generate(ctx, req)
		yield(response, err)
	}
}

func TestGoADKMigrationModelCallbacks(t *testing.T) {
	for _, selected := range []string{"success", "before-rejection", "provider-error", "rejected-response", "after-rejection"} {
		t.Run(selected, func(t *testing.T) {
			before, after, onError, charged := 0, 0, 0, int64(0)
			sentinel := planner.NewError("planner_gateway_unavailable", "safe characterization", true, nil)
			llm := &migrationCallbackModel{generate: func(context.Context, *model.LLMRequest) (*model.LLMResponse, error) {
				if selected == "provider-error" {
					return nil, sentinel
				}
				response, _ := textStep("terminal native text")(nil)
				if selected == "rejected-response" {
					return nil, &gatewayResponseRejected{usage: response.UsageMetadata}
				}
				return response, nil
			}}
			root, err := llmagent.New(llmagent.Config{Name: "migration_agent", Model: llm,
				BeforeModelCallbacks: []llmagent.BeforeModelCallback{func(agent.Context, *model.LLMRequest) (*model.LLMResponse, error) {
					before++
					if selected == "before-rejection" {
						return nil, sentinel
					}
					return nil, nil
				}},
				AfterModelCallbacks: []llmagent.AfterModelCallback{func(_ agent.Context, response *model.LLMResponse, callErr error) (*model.LLMResponse, error) {
					after++
					if callErr != nil {
						return nil, fmt.Errorf("unexpected after error: %w", callErr)
					}
					charged += int64(response.UsageMetadata.TotalTokenCount)
					if selected == "after-rejection" {
						return nil, sentinel
					}
					return nil, nil
				}},
				OnModelErrorCallbacks: []llmagent.OnModelErrorCallback{func(_ agent.Context, _ *model.LLMRequest, modelErr error) (*model.LLMResponse, error) {
					onError++
					var rejected *gatewayResponseRejected
					if errors.As(modelErr, &rejected) {
						charged += int64(rejected.usage.TotalTokenCount)
					}
					return nil, modelErr
				}},
			})
			if err != nil {
				t.Fatal(err)
			}
			service := adksession.InMemoryService()
			if _, err := service.Create(t.Context(), &adksession.CreateRequest{AppName: "migration", UserID: "owner", SessionID: "session"}); err != nil {
				t.Fatal(err)
			}
			actual, err := runner.New(runner.Config{AppName: "migration", Agent: root, SessionService: service})
			if err != nil {
				t.Fatal(err)
			}
			var iterationErr error
			for _, err := range actual.Run(t.Context(), "owner", "session", genai.NewContentFromText("bounded test input", genai.RoleUser), agent.RunConfig{}) {
				if err != nil {
					iterationErr = err
					break
				}
			}
			if before != 1 {
				t.Fatalf("before callbacks=%d", before)
			}
			expectedCalls := 1
			if selected == "before-rejection" {
				expectedCalls = 0
			}
			if llm.calls != expectedCalls {
				t.Fatalf("provider calls=%d want%d", llm.calls, expectedCalls)
			}
			for _, stream := range llm.streams {
				if stream {
					t.Fatal("nonstreaming Planner requested streaming")
				}
			}
			expectedAfter, expectedError, expectedCharge := 0, 0, int64(0)
			switch selected {
			case "success", "after-rejection":
				expectedAfter = 1
				expectedCharge = 20
			case "provider-error":
				expectedError = 1
			case "rejected-response":
				expectedError = 1
				expectedCharge = 20
			}
			if after != expectedAfter || onError != expectedError || charged != expectedCharge {
				t.Fatalf("callbacks/charge = %d/%d/%d", after, onError, charged)
			}
			if selected == "success" {
				if iterationErr != nil {
					t.Fatal(iterationErr)
				}
			} else if selected == "rejected-response" {
				var rejected *gatewayResponseRejected
				if !errors.As(iterationErr, &rejected) {
					t.Fatalf("typed rejected response lost: %T", iterationErr)
				}
			} else {
				var failure *planner.Error
				if !errors.As(iterationErr, &failure) || failure.Failure.Code != "planner_gateway_unavailable" {
					t.Fatalf("typed planner failure lost: %v", iterationErr)
				}
			}
		})
	}
}

func TestGoADKMigrationHTTPAdmission(t *testing.T) {
	for _, profile := range []string{"streamline", "router"} {
		t.Run(profile, func(t *testing.T) {
			for _, selected := range []string{"success", "model-budget", "rejected-usage"} {
				t.Run(selected, func(t *testing.T) {
					var calls atomic.Int32
					var wireError atomic.Bool
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						calls.Add(1)
						var request map[string]any
						if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
							wireError.Store(true)
							http.Error(w, "fixture input", 400)
							return
						}
						if request["parallel_tool_calls"] != false || request["stream"] == true {
							wireError.Store(true)
						}
						tools, ok := request["tools"].([]any)
						if !ok || len(tools) < 4 {
							wireError.Store(true)
						}
						finish, arguments := "tool_calls", `{"outcome":"failed","summary":"bounded gateway","artifacts":{},"error":{"code":"characterization_stop","message":"test stop","retryable":false}}`
						message := map[string]any{"role": "assistant", "tool_calls": []any{map[string]any{"id": "wire-finish", "type": "function", "function": map[string]any{"name": "finish", "arguments": arguments}}}}
						if selected == "model-budget" {
							finish = "stop"
							message = map[string]any{"role": "assistant", "content": "nonterminal planning text"}
						}
						if selected == "rejected-usage" {
							finish = "length"
						}
						w.Header().Set("Content-Type", "application/json")
						_ = json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"finish_reason": finish, "message": message}}, "usage": map[string]any{"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}})
					}))
					defer server.Close()
					llm, err := NewOpenAICompatibleModel(GatewaySettings{URL: server.URL + "/v1", Model: "planner-model", HTTPClient: server.Client()})
					if err != nil {
						t.Fatal(err)
					}
					instance, _ := migrationPlanner(t, profile, "core", llm)
					instance.limits.MaxModelCalls = 1
					result, err := instance.Run(t.Context())
					if wireError.Load() || calls.Load() != 1 {
						t.Fatalf("HTTP schema/nonstreaming/call bound failed: calls%d invalid%v", calls.Load(), wireError.Load())
					}
					if selected == "success" {
						if err != nil || result.Outcome != stageresult.StageFailed {
							t.Fatalf("Run=(%+v,%v)", result, err)
						}
					} else {
						want := "planner_model_call_limit"
						if selected == "rejected-usage" {
							want = "planner_gateway_invalid_response"
						}
						if err == nil || planner.FailureFrom(err).Code != want {
							t.Fatalf("failure=%v want%s", err, want)
						}
					}
					report, ok := instance.ExecutionReport()
					if !ok || report.Metrics.ModelCalls == nil || *report.Metrics.ModelCalls != 1 || report.Metrics.InputTokens == nil || *report.Metrics.InputTokens != 7 || report.Metrics.OutputTokens == nil || *report.Metrics.OutputTokens != 3 {
						t.Fatalf("admission/provider/usage accounting=%+v", report.Metrics)
					}
				})
			}
		})
	}
}
