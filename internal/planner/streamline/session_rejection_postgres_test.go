package streamline

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	plannersession "github.com/grauwolf32/contractor/internal/planner/session"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"github.com/jackc/pgx/v5/pgxpool"
	"google.golang.org/adk/v2/model"
)

// Plan progress, a fact or a redacted ADK event that the real session store
// refuses under planner_session_state_size ends Streamline with the
// non-retryable planner_session_rejected, never the retryable
// planner_plan_persistence_failed or planner_execution_failed an outage
// returns (PLN-R106). The model step lowers the bound, with the production
// constraint and measure, below the state its write produces.
func TestPostgresStreamlineRejectedSessionWritesAreNotRetryable(t *testing.T) {
	failedFinish := functionStep(finishToolName, map[string]any{
		"outcome": string(stageresult.StageFailed), "summary": "analysis could not be completed",
		"artifacts": map[string]any{},
		"error": map[string]any{
			"code": "insufficient_evidence", "message": "required evidence was unavailable", "retryable": true,
		},
	})
	for _, test := range []struct {
		name string
		step modelStep
		// slack is how far the lowered bound lies above the stored state:
		// room for the model event's counters, not for a plan or a fact.
		slack int
	}{
		{name: "plan transition", step: addSubtaskStep("Analyze the input", "Produce the final report"), slack: 16},
		{name: "fact", step: failedFinish, slack: 16},
		{name: "ADK event", step: textStep("still planning"), slack: -1},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
			defer cancel()
			pool := isolatedStreamlinePool(t, ctx)
			store := runstore.NewPostgresStore(pool)
			createPreparingStreamlineStage(t, ctx, store)
			sessions, err := plannersession.New(store, plannersession.Options{})
			if err != nil {
				t.Fatal(err)
			}
			var lowerErr error
			lowered := func(request *model.LLMRequest) (*model.LLMResponse, error) {
				if lowerErr = lowerStreamlineStateBound(ctx, pool, test.slack); lowerErr != nil {
					return nil, lowerErr
				}
				return test.step(request)
			}
			llm := &scriptedModel{steps: []modelStep{lowered}, fallback: textStep("not finished")}
			invocation := testInvocation("builder")
			invocation.ModelAccess.ModelPolicy.MaxModelCalls = 4
			factory, err := NewConfiguredFactory(
				sessions, sessions, &fakeWorkerInvoker{}, &fakeInspector{}, unavailableWorkerStateReader{},
				testModelFactory(llm), Limits{MaxWallTime: time.Minute},
			)
			if err != nil {
				t.Fatal(err)
			}
			instance, err := factory.Create(invocation)
			if err != nil {
				t.Fatal(err)
			}
			_, err = instance.Run(ctx)
			if lowerErr != nil {
				t.Fatalf("lower the state bound: %v", lowerErr)
			}
			assertPlannerCode(t, err, "planner_session_rejected")
			if planner.FailureFrom(err).Retryable || !errors.Is(err, runstore.ErrInvalid) {
				t.Fatalf("error = %v, want non-retryable and wrapping runstore.ErrInvalid", err)
			}
			report, ok := instance.(planner.ReportProvider).ExecutionReport()
			if !ok || len(report.Errors) == 0 {
				t.Fatalf("report = (%+v, %t)", report, ok)
			}
			for _, reported := range report.Errors {
				if reported.Code != "planner_session_rejected" || reported.Retryable == nil || *reported.Retryable {
					t.Fatalf("report errors = %+v, want only non-retryable planner_session_rejected", report.Errors)
				}
			}
		})
	}
}

// createPreparingStreamlineStage creates the running Run and preparing
// StageExecution of testInvocation, whose Planner session the session
// service starts.
func createPreparingStreamlineStage(t *testing.T, ctx context.Context, runs *runstore.PostgresStore) {
	t.Helper()
	if _, err := runstoretest.CreateRun(ctx, runs, runstore.CreateRunParams{
		LaneClass: resolved.LaneClassModel,
		RunID:     "run-1", OwnerID: "owner", WorkflowName: "test", WorkflowVersion: "1",
		WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{}`),
		RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatal(err)
	}
	for _, step := range [][2]runstore.WorkflowRunState{
		{runstore.RunInitializing, runstore.RunPending}, {runstore.RunPending, runstore.RunRunning},
	} {
		if _, err := runs.TransitionRun(ctx, "run-1", step[0], step[1], runstore.Reason{Code: "test"}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := runs.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{
		LaneClass:        resolved.LaneClassModel,
		StageExecutionID: "stage-1", RunID: "run-1", StageName: "review", Attempt: 1,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: json.RawMessage(`{"objective":"build","session":"isolated"}`),
		StageContextSchemaVersion: contracts.APIVersion,
		StageContext:              runstore.StageContextSnapshot{Parameters: map[string]string{}, Artifacts: map[string]runstore.PinnedContextArtifact{}},
	}); err != nil {
		t.Fatal(err)
	}
}

// lowerStreamlineStateBound replaces planner_session_state_size with the same
// measure and a bound slack bytes above the stored state. NOT VALID keeps
// that state, so only later writes meet the lowered bound.
func lowerStreamlineStateBound(ctx context.Context, pool *pgxpool.Pool, slack int) error {
	var size int
	if err := pool.QueryRow(ctx, `SELECT octet_length(state::text) FROM planner_sessions`).Scan(&size); err != nil {
		return err
	}
	_, err := pool.Exec(ctx, fmt.Sprintf(`
ALTER TABLE planner_sessions DROP CONSTRAINT planner_session_state_size,
ADD CONSTRAINT planner_session_state_size CHECK (octet_length(state::text) <= %d) NOT VALID`, size+slack))
	return err
}
