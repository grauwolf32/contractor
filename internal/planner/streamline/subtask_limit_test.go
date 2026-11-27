package streamline

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"github.com/grauwolf32/contractor/internal/telemetry"
	collectortracev1 "go.opentelemetry.io/proto/otlp/collector/trace/v1"
	"google.golang.org/protobuf/proto"
)

func TestSharedPlannerTelemetrySubtaskIDs(t *testing.T) {
	var fixture struct {
		IDs []struct {
			Value string
			Valid bool
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "planner-subtask-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	for _, test := range fixture.IDs {
		want := "invalid"
		if test.Valid {
			want = test.Value
		}
		if got := safeSubtaskID(test.Value); got != want {
			t.Errorf("safeSubtaskID(%q) = %q, want %q", test.Value, got, want)
		}
	}
	if safeSubtaskID("") != "invalid" {
		t.Fatal("empty ID accepted")
	}
}

func TestModeledPlannerReportsThreeDigitSubtaskID(t *testing.T) {
	for _, profile := range []plannerProfile{streamlineProfile, routerProfile} {
		t.Run(profile.ref, func(t *testing.T) {
			llm := &operationScript{}
			for range 101 {
				llm.steps = append(llm.steps, addSubtaskStep("Inspect", "Read source"))
			}
			for index := range 101 {
				args := map[string]any{"subtask_id": fmt.Sprint(index)}
				if profile.routesWorkers {
					args["worker_name"] = "builder"
				}
				llm.steps = append(llm.steps, functionStep(executeCurrentSubtaskToolName, args))
			}
			llm.steps = append(llm.steps, functionStep(finishToolName, map[string]any{
				"outcome": "succeeded", "summary": "done", "artifacts": map[string]any{},
			}))
			sessions := newFakeSessions()
			workers := &fakeWorkerInvoker{results: map[string]stageresult.StageContentResult{
				"builder": stageResult("done", map[string]contracts.ArtifactRef{}),
			}}
			factory, err := runtimeScriptFactory(profile, sessions, workers, &fakeInspector{},
				unavailableWorkerStateReader{}, llm, Limits{})
			if err != nil {
				t.Fatal(err)
			}
			invocation := testInvocation("builder")
			invocation.Stage.Planner = resolved.PlannerRef{PlannerID: strings.TrimSuffix(profile.ref, "@1"), Version: "1"}
			invocation.Stage.MaxSubtasks = 101
			invocation.Stage.Result.Artifacts = map[string]resolved.ArtifactSlot{}
			invocation.ModelAccess.ModelPolicy.MaxModelCalls = len(llm.steps)
			invocation.ModelAccess.ModelPolicy.MaxWorkerCalls = 101
			adapter, payloads := streamlineTestTelemetry(t, invocation)
			invocation.Instrumentation = adapter.Instrumentation()
			instance, err := factory.Create(invocation)
			if err != nil {
				t.Fatal(err)
			}
			result, err := instance.Run(t.Context())
			if err != nil || result.Outcome != stageresult.StageSucceeded || len(workers.calls) != 101 {
				t.Fatalf("Run = (%+v, %v), Worker calls = %d", result, err, len(workers.calls))
			}
			report, _ := instance.(planner.ReportProvider).ExecutionReport()
			lastDispatch := report.ToolCalls[len(report.ToolCalls)-2]
			if lastDispatch.Tool != executeCurrentSubtaskToolName || lastDispatch.Arguments["subtaskId"] != "100" {
				t.Fatalf("last dispatch = %+v", lastDispatch)
			}
			if exported := adapter.Flush(t.Context()); !exported.Succeeded {
				t.Fatalf("export = %+v", exported)
			}
			var trace collectortracev1.ExportTraceServiceRequest
			if err := proto.Unmarshal(<-payloads, &trace); err != nil {
				t.Fatal(err)
			}
			for _, resource := range trace.ResourceSpans {
				for _, scope := range resource.ScopeSpans {
					for _, span := range scope.Spans {
						if span.Name != string(telemetry.PlannerSpanWorker) {
							continue
						}
						for _, attr := range span.Attributes {
							if attr.Key == "subtask.id" && attr.Value.GetStringValue() == "100" {
								return
							}
						}
					}
				}
			}
			t.Fatal("Worker span lost subtask ID 100")
		})
	}
}

func TestModeledFactoriesEnforceWorkflowSubtaskLimitThroughTools(t *testing.T) {
	for _, profile := range []plannerProfile{streamlineProfile, routerProfile} {
		for _, limit := range []contracts.PlannerSubtaskLimit{0, 1, 64} {
			t.Run(fmt.Sprintf("%s/%d", profile.ref, limit), func(t *testing.T) {
				llm := &operationScript{}
				for range limit.Effective() + 1 {
					llm.steps = append(llm.steps, addSubtaskStep("Inspect", "Read the source"))
				}
				llm.steps = append(llm.steps, functionStep(finishToolName, map[string]any{
					"outcome": "failed", "summary": "Plan budget reached", "artifacts": map[string]any{},
					"error": map[string]any{"code": "plan_budget", "message": "Plan budget reached", "retryable": false},
				}))
				sessions := newFakeSessions()
				factory, err := runtimeScriptFactory(profile, sessions, &fakeWorkerInvoker{}, &fakeInspector{},
					unavailableWorkerStateReader{}, llm, Limits{})
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
