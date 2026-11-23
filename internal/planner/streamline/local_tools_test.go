package streamline

import (
	"reflect"
	"strings"
	"testing"

	plannermemory "github.com/grauwolf32/contractor/internal/memory"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

func TestLocalToolsPreserveRequestPacking(t *testing.T) {
	for _, profile := range []plannerProfile{streamlineProfile, routerProfile} {
		t.Run(profile.ref, func(t *testing.T) {
			invocation := memoryInvocation(strings.TrimSuffix(profile.ref, "@1"), map[string][]string{
				"builder": {plannermemory.ToolListMemories},
			})
			instance := newMemoryPlanner(t, profile, newPlannerMemoryArtifactStore(), invocation, &scriptedModel{})
			t.Cleanup(instance.stateViews.Close)
			tools := plannerTools(t, instance)
			for _, name := range []string{plannermemory.ToolListMemories, getWorkerToolUsageToolName} {
				t.Run(name, func(t *testing.T) {
					current := tools[name]
					processor := current.(interface {
						ProcessRequest(agent.Context, *model.LLMRequest) error
					})
					if err := processor.ProcessRequest(nil, nil); err == nil || err.Error() != "Planner model request is required" {
						t.Fatalf("nil request error = %v", err)
					}
					for _, shape := range []string{"empty", "non-function groups", "existing declarations", "empty declarations"} {
						t.Run(shape, func(t *testing.T) {
							existing := &genai.FunctionDeclaration{Name: "existing"}
							group := &genai.Tool{FunctionDeclarations: []*genai.FunctionDeclaration{existing}}
							request := &model.LLMRequest{}
							var originalGroups []*genai.Tool
							var originalDeclarations []*genai.FunctionDeclaration
							if shape != "empty" {
								originalGroups = []*genai.Tool{nil, {}}
								if shape == "empty declarations" {
									group.FunctionDeclarations = []*genai.FunctionDeclaration{}
								}
								if shape != "non-function groups" {
									originalGroups = append(originalGroups, group, &genai.Tool{FunctionDeclarations: []*genai.FunctionDeclaration{existing}})
									originalDeclarations = append(originalDeclarations, group.FunctionDeclarations...)
								}
								request.Config = &genai.GenerateContentConfig{Tools: append([]*genai.Tool(nil), originalGroups...)}
								request.Tools = map[string]any{"existing": existing}
							}
							if err := processor.ProcessRequest(nil, request); err != nil {
								t.Fatal(err)
							}
							if request.Tools[name] != current {
								t.Fatalf("registered %T instead of the original %T", request.Tools[name], current)
							}
							if shape != "empty" && request.Tools["existing"] != existing {
								t.Fatal("existing tool was replaced")
							}
							packed := request.Config.Tools
							if len(originalGroups) > 0 && !reflect.DeepEqual(packed[:len(originalGroups)], originalGroups) {
								t.Fatal("existing tool groups were replaced")
							}
							if shape == "empty" || shape == "non-function groups" {
								if len(packed) != len(originalGroups)+1 {
									t.Fatal("missing new function group")
								}
								group = packed[len(originalGroups)]
							} else if len(packed) != len(originalGroups) || len(packed[3].FunctionDeclarations) != 1 {
								t.Fatal("unrelated function group changed")
							}
							want := append(originalDeclarations, toolDeclaration(t, current))
							if !reflect.DeepEqual(group.FunctionDeclarations, want) {
								t.Fatalf("declarations = %+v, want %+v", group.FunctionDeclarations, want)
							}
							if err := processor.ProcessRequest(nil, request); err == nil || err.Error() != "duplicate tool \""+name+"\"" {
								t.Fatalf("duplicate error = %v", err)
							}
							if !reflect.DeepEqual(group.FunctionDeclarations, want) || request.Tools[name] != current {
								t.Fatal("duplicate mutated the request")
							}
						})
					}
					duplicate := &model.LLMRequest{Tools: map[string]any{name: "original"}}
					if err := processor.ProcessRequest(nil, duplicate); err == nil || duplicate.Config != nil || duplicate.Tools[name] != "original" {
						t.Fatalf("duplicate initialized config or replaced tool: %+v, %v", duplicate, err)
					}
				})
			}
		})
	}
}

func TestLocalToolsDispatchConcreteRunThroughADK(t *testing.T) {
	for _, profile := range []plannerProfile{streamlineProfile, routerProfile} {
		t.Run(profile.ref, func(t *testing.T) {
			args := map[string]any{}
			if profile.routesWorkers {
				args["worker_name"] = "builder"
			}
			modelValue := &scriptedModel{steps: []modelStep{
				functionStep(plannermemory.ToolListMemories, args),
				func(request *model.LLMRequest) (*model.LLMResponse, error) {
					if _, ok := request.Tools[plannermemory.ToolListMemories].(*plannerMemoryTool); !ok {
						t.Fatal("Memory dispatch lost its concrete instance")
					}
					response := requireFunctionResponse(t, request, plannermemory.ToolListMemories)
					if _, ok := response["result"]; !ok || response["error"] != nil {
						t.Fatalf("Memory Run response = %+v", response)
					}
					return functionStep(getWorkerToolUsageToolName, args)(request)
				},
				func(request *model.LLMRequest) (*model.LLMResponse, error) {
					if _, ok := request.Tools[getWorkerToolUsageToolName].(*plannerStateTool); !ok {
						t.Fatal("State dispatch lost its concrete instance")
					}
					response := requireFunctionResponse(t, request, getWorkerToolUsageToolName)
					failure, ok := response["error"].(map[string]any)
					if !ok || failure["code"] != "worker_state_unavailable" {
						t.Fatalf("State Run response = %+v", response)
					}
					return functionStep(finishToolName, map[string]any{
						"outcome": string(stageresult.StageFailed), "summary": "Worker State unavailable before dispatch",
						"artifacts": map[string]any{},
						"error":     map[string]any{"code": "state_unavailable", "message": "expected tool response", "retryable": false},
					})(request)
				},
			}}
			invocation := memoryInvocation(strings.TrimSuffix(profile.ref, "@1"), map[string][]string{
				"builder": {plannermemory.ToolListMemories},
			})
			instance := newMemoryPlanner(t, profile, newPlannerMemoryArtifactStore(), invocation, modelValue)
			result, err := instance.Run(t.Context())
			if err != nil || result.Outcome != stageresult.StageFailed {
				t.Fatalf("Run = (%+v, %v)", result, err)
			}
		})
	}
}

func TestLocalToolsResolveOnlyEligibleWorkerBindings(t *testing.T) {
	builder := workerBinding{logicalName: "builder"}
	reviewer := workerBinding{logicalName: "reviewer"}
	for _, test := range []struct {
		name     string
		profile  plannerProfile
		eligible []workerBinding
		worker   string
		want     string
	}{
		{"sole Streamline Worker", streamlineProfile, []workerBinding{builder}, "", "builder"},
		{"Streamline rejects explicit name", streamlineProfile, []workerBinding{builder}, "builder", ""},
		{"Streamline rejects multiple Workers", streamlineProfile, []workerBinding{builder, reviewer}, "", ""},
		{"empty selection", streamlineProfile, nil, "", ""},
		{"Router selected Worker", routerProfile, []workerBinding{builder, reviewer}, "reviewer", "reviewer"},
		{"Router operation-ineligible Worker", routerProfile, []workerBinding{builder}, "reviewer", ""},
		{"Router requires a name", routerProfile, []workerBinding{builder}, "", ""},
		{"Router exact name", routerProfile, []workerBinding{builder}, "Builder", ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			instance := &streamlinePlanner{profile: test.profile, workers: []workerBinding{builder, reviewer}}
			binding, ok := instance.resolveWorkerBinding(test.eligible, test.worker)
			if ok != (test.want != "") || binding.logicalName != test.want {
				t.Fatalf("Worker = (%+v, %v), want %q", binding, ok, test.want)
			}
		})
	}
}
