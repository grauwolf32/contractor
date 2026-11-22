package streamline

import (
	"encoding/json"
	"errors"
	"strconv"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"google.golang.org/adk/v2/model"
)

func TestGuidanceReachesBothPlannerModels(t *testing.T) {
	for _, profile := range []plannerProfile{streamlineProfile, routerProfile} {
		t.Run(profile.ref, func(t *testing.T) {
			log := &requestLog{}
			llm := &scriptedModel{steps: []modelStep{
				log.capture(addSubtaskStep("private task text", "private instruction")),
				log.capture(functionStep(finishToolName, map[string]any{"outcome": "failed", "summary": "done", "artifacts": map[string]any{}, "error": map[string]any{"code": "test", "message": "done", "retryable": false}})),
			}}
			sessions := newFakeSessions()
			factory, err := newFactory(profile, sessions, sessions, &fakeWorkerInvoker{}, &fakeInspector{}, unavailableWorkerStateReader{}, nil, testModelFactory(llm), Limits{})
			if err != nil {
				t.Fatal(err)
			}
			invocation := testInvocation("builder")
			invocation.Stage.Planner = resolved.PlannerRef{PlannerID: strings.TrimSuffix(profile.ref, "@1"), Version: "1"}
			instance, err := factory.Create(invocation)
			if err != nil {
				t.Fatal(err)
			}
			if _, err = instance.Run(t.Context()); err != nil {
				t.Fatal(err)
			}
			for i := range 2 {
				if strings.Count(log.at(i), "Current execution facts") != 1 {
					t.Fatalf("guidance missing/duplicated: %s", log.at(i))
				}
			}
			if !strings.Contains(log.at(1), `\"usedPlanSlots\":1`) {
				t.Fatal(log.at(1))
			}
		})
	}
}

func TestSharedRequestGuidanceMaximumAndReplacement(t *testing.T) {
	raw := contractstest.ReadFile(t, "api", "testdata", "request-guidance.json")
	var fixture struct {
		MaxBytes     int
		MaxPlanSlots contracts.PlannerSubtaskLimit
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.MaxBytes != maxGuidanceBytes {
		t.Fatal("shared ceiling drift")
	}
	invocation := testInvocation("builder")
	invocation.Stage.MaxSubtasks = fixture.MaxPlanSlots
	factory := mustFactory(t, newFakeSessions(), &fakeWorkerInvoker{}, &fakeInspector{}, &scriptedModel{}, Limits{})
	instance, err := factory.Create(invocation)
	if err != nil {
		t.Fatal(err)
	}
	p := instance.(*streamlinePlanner)
	for range fixture.MaxPlanSlots.Effective() {
		if _, err := p.plan.AddSubtask("private objective", "private instruction"); err != nil {
			t.Fatal(err)
		}
	}
	state := newExecutionState(p.limits)
	if err := state.beforeModel(); err != nil {
		t.Fatal(err)
	}
	request := &model.LLMRequest{}
	for range 2 {
		if err := p.prepareGuidance(state, request); err != nil {
			t.Fatal(err)
		}
	}
	if len(request.Contents) != 1 {
		t.Fatal("projection accumulated")
	}
	text := request.Contents[0].Parts[0].Text
	if len(text) > fixture.MaxBytes || strings.Contains(text, "private") || !strings.Contains(text, `"freePlanSlots":0`) {
		t.Fatal(text)
	}
}

func TestGuidanceShowsMixedThirtyOfThirtyTwoPlan(t *testing.T) {
	for _, profile := range []plannerProfile{streamlineProfile, routerProfile} {
		t.Run(profile.ref, func(t *testing.T) {
			workers := &fakeWorkerInvoker{results: map[string]stageresult.StageContentResult{
				"builder": stageResult("done", map[string]contracts.ArtifactRef{}),
			}}
			llm := &scriptedModel{}
			for range 30 {
				llm.steps = append(llm.steps, addSubtaskStep("Inspect input", "Read exact evidence"))
			}
			for i := range 28 {
				execute := functionStep(executeCurrentSubtaskToolName, executeArgs(strings.TrimSuffix(profile.ref, "@1"), strconv.Itoa(i)))
				llm.steps = append(llm.steps, func(request *model.LLMRequest) (*model.LLMResponse, error) {
					if i == 27 {
						workers.failures = map[string][]error{"builder": {errors.New("controlled failure")}}
					}
					return execute(request)
				})
			}
			llm.steps = append(llm.steps, func(request *model.LLMRequest) (*model.LLMResponse, error) {
				var facts struct {
					UsedPlanSlots, FreePlanSlots                            int
					SubtaskCounts                                           map[string]int
					CurrentSubtask                                          struct{ ID string }
					AdditionalModelCallsAfterThisCall, RemainingWorkerCalls int
				}
				text := request.Contents[len(request.Contents)-1].Parts[0].Text
				if err := json.Unmarshal([]byte(strings.TrimPrefix(text, guidancePreamble)), &facts); err != nil {
					t.Fatal(err)
				}
				if facts.UsedPlanSlots != 30 || facts.FreePlanSlots != 2 || facts.SubtaskCounts["succeeded"] != 27 || facts.SubtaskCounts["failed"] != 1 || facts.SubtaskCounts["pending"] != 2 || facts.CurrentSubtask.ID != "28" || facts.AdditionalModelCallsAfterThisCall != 5 || facts.RemainingWorkerCalls != 4 {
					t.Fatalf("mixed-plan facts: %s", text)
				}
				return functionStep(finishToolName, map[string]any{"outcome": "failed", "summary": "Stopped with work pending", "artifacts": map[string]any{}, "error": map[string]any{"code": "controlled", "message": "Stopped", "retryable": false}})(request)
			})
			sessions := newFakeSessions()
			factory, err := newFactory(profile, sessions, sessions, workers, &fakeInspector{}, unavailableWorkerStateReader{}, nil, testModelFactory(llm), Limits{})
			if err != nil {
				t.Fatal(err)
			}
			invocation := testInvocation("builder")
			invocation.Stage.Planner = resolved.PlannerRef{PlannerID: strings.TrimSuffix(profile.ref, "@1"), Version: "1"}
			invocation.Stage.Result.Artifacts = map[string]resolved.ArtifactSlot{}
			invocation.ModelAccess.ModelPolicy.MaxModelCalls = 64
			invocation.ModelAccess.ModelPolicy.MaxWorkerCalls = 32
			instance, err := factory.Create(invocation)
			if err != nil {
				t.Fatal(err)
			}
			if _, err = instance.Run(t.Context()); err != nil {
				t.Fatal(err)
			}
		})
	}
}
