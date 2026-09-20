package public

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

type failingAllocationReader struct {
	RunReader
	err error
}

func (r failingAllocationReader) ListStageAllocationsBatch(context.Context, []string) (map[string][]runstore.StageAllocation, error) {
	return nil, r.err
}

func TestRunDetailBatchFailuresPreserveOptionalDiagnostics(t *testing.T) {
	readErr := errors.New("private-batch-read-failure")
	for _, test := range []struct {
		name      string
		configure func(*Dependencies)
		stages    int
		status    int
		wantPlan  bool
	}{
		{
			name: "optional readers absent", stages: 2, status: http.StatusOK,
			configure: func(d *Dependencies) { d.Metrics, d.PlannerPlans = nil, nil },
		},
		{
			name: "metrics unavailable", stages: 2, status: http.StatusOK, wantPlan: true,
			configure: func(d *Dependencies) { d.Metrics = &fakeMetricsReader{err: readErr} },
		},
		{
			name: "plans unavailable", stages: 2, status: http.StatusInternalServerError,
			configure: func(d *Dependencies) { d.PlannerPlans = &fakePlannerPlanReader{err: readErr} },
		},
		{
			name: "allocations unavailable", stages: 2, status: http.StatusInternalServerError,
			configure: func(d *Dependencies) { d.Runs = failingAllocationReader{RunReader: d.Runs, err: readErr} },
		},
		{
			name: "empty run needs no related reads", status: http.StatusOK,
			configure: func(d *Dependencies) {
				d.Runs = failingAllocationReader{RunReader: d.Runs, err: readErr}
				d.Metrics = &fakeMetricsReader{err: readErr}
				d.PlannerPlans = &fakePlannerPlanReader{err: readErr}
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid",
				newTestAuthentication(t), mustTestOrigins(t), false, nil, test.configure)
			workflow, err := fixture.configs.Workflow("artifact-copy@1")
			if err != nil {
				t.Fatal(err)
			}
			stageJSON, err := json.Marshal(workflow.Stages[workflow.EntryStage])
			if err != nil {
				t.Fatal(err)
			}
			fixture.runs.runs["run-batch"] = queryRun("run-batch", "user-1", runstore.RunRunning, time.Now().UTC())
			for i := 0; i < test.stages; i++ {
				id := fmt.Sprintf("stage-%d", i)
				sessionID, invocationID := "session-"+id, "invocation-"+id
				fixture.runs.executions["run-batch"] = append(fixture.runs.executions["run-batch"], runstore.StageExecution{
					StageExecutionID: id, RunID: "run-batch", StageName: "copy", Attempt: i + 1,
					ExecutionConfigVariant: runstore.StageExecutionConfigBase,
					StageSpecSnapshot:      stageJSON, State: runstore.StageRunning,
					PlannerSessionID: &sessionID, PlannerInvocationID: &invocationID,
				})
				fixture.plans.plans[id] = planner.PlannerPlanProjection{
					Revision: 1, CurrentSubtaskID: "0",
					Subtasks: []planner.PlannerSubtask{{ID: "0", Objective: id, Instructions: "Inspect", Status: planner.PlannerSubtaskPending}},
				}
			}
			response := serveQuery(t, fixture.handler, "/v1/runs/run-batch")
			if response.Code != test.status || strings.Contains(response.Body.String(), readErr.Error()) {
				t.Fatalf("Run detail: status=%d body=%s", response.Code, response.Body.String())
			}
			if test.status != http.StatusOK {
				assertErrorCode(t, response, "internal_error")
				return
			}
			var detail runStatusResponse
			decodeQueryResponse(t, response, &detail)
			if len(detail.Attempts) != test.stages || detail.State != runstore.RunRunning {
				t.Fatalf("durable Run state lost: %+v", detail)
			}
			for _, attempt := range detail.Attempts {
				if attempt.Metrics != nil || (attempt.Plan != nil) != test.wantPlan || attempt.Resources == nil {
					t.Fatalf("optional data: %+v", attempt)
				}
				if test.wantPlan && attempt.Plan.Subtasks[0].Objective != attempt.StageExecutionID {
					t.Fatalf("plan belongs to another Stage: %+v", attempt)
				}
			}
		})
	}
}
