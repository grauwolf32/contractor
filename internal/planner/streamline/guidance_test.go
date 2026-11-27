package streamline

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

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
	factory := runtimeTestFactory(t, "streamline", newFakeSessions(), &fakeWorkerInvoker{})
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
	text, err := p.prepareGuidance(state)
	if err != nil {
		t.Fatal(err)
	}
	again, err := p.prepareGuidance(state)
	if err != nil || again != text {
		t.Fatal("guidance accumulated", err)
	}
	if len(text) > fixture.MaxBytes || strings.Contains(text, "private") || !strings.Contains(text, `"freePlanSlots":0`) {
		t.Fatal(text)
	}
}
