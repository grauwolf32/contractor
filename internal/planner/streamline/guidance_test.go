package streamline

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
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

func TestGuidanceMaximumAndReplacement(t *testing.T) {
	raw, err := os.ReadFile("../../../api/testdata/request-guidance.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		MaxBytes     int
		MaxPlanSlots contracts.PlannerSubtaskLimit
	}
	if err = json.Unmarshal(raw, &fixture); err != nil {
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
