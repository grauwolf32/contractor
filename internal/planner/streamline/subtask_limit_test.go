package streamline

import (
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/planner/stageresult"
)

func TestModeledFactoriesEnforceWorkflowSubtaskLimitThroughTools(t *testing.T) {
	for _, profile := range []plannerProfile{streamlineProfile, routerProfile} {
		for _, limit := range []contracts.PlannerSubtaskLimit{0, 1, 64} {
			t.Run(fmt.Sprintf("%s/%d", profile.ref, limit), func(t *testing.T) {
				llm := &scriptedModel{}
				for range limit.Effective() + 1 {
					llm.steps = append(llm.steps, addSubtaskStep("Inspect", "Read the source"))
				}
				llm.steps = append(llm.steps, functionStep(finishToolName, map[string]any{
					"outcome": "failed", "summary": "Plan budget reached", "artifacts": map[string]any{},
					"error": map[string]any{"code": "plan_budget", "message": "Plan budget reached", "retryable": false},
				}))
				sessions := newFakeSessions()
				factory, err := newFactory(profile, sessions, sessions, &fakeWorkerInvoker{}, &fakeInspector{},
					unavailableWorkerStateReader{}, nil, testModelFactory(llm), Limits{})
				if err != nil {
					t.Fatal(err)
				}
				invocation := testInvocation("builder")
				invocation.Stage.Planner = resolved.PlannerRef{PlannerID: strings.TrimSuffix(profile.ref, "@1"), Version: "1"}
				invocation.Stage.MaxSubtasks = limit
				invocation.ModelAccess.ModelPolicy.MaxModelCalls = limit.Effective() + 2
				instance, err := factory.Create(invocation)
				if err != nil {
					t.Fatal(err)
				}
				result, err := instance.Run(t.Context())
				if err != nil || result.Outcome != stageresult.StageFailed {
					t.Fatalf("result = (%+v, %v)", result, err)
				}
				if plan := instance.(*streamlinePlanner).plan.Snapshot(); len(plan.Subtasks) != limit.Effective() {
					t.Fatalf("subtasks=%d, want %d", len(plan.Subtasks), limit.Effective())
				}
				report, ok := instance.(planner.ReportProvider).ExecutionReport()
				if !ok || len(report.ToolCalls) != limit.Effective()+2 {
					t.Fatalf("tool report count = %d", len(report.ToolCalls))
				}
				rejected := report.ToolCalls[limit.Effective()]
				if rejected.Error == nil || rejected.Error.Code != "planner_subtask_limit" {
					t.Fatalf("overflow tool call = %+v", rejected)
				}
			})
		}
	}
}
