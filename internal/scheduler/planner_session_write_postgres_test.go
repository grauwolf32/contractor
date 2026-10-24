package scheduler

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

// A Passthrough outcome that the real Planner session cannot record ends the
// StageExecution as an interrupted planner_session_unavailable of the
// outcome's retryability (PLN-R106). A summarized Worker result is final:
// on.interrupted takes its then branch instead of dispatching the Worker
// again, and no summary continuation starts without a durable candidate. A
// successful candidate stays retryable, so the same retry policy creates a
// fresh attempt.
func TestPostgresUnwrittenPlannerOutcomeFollowsItsRetryability(t *testing.T) {
	for _, test := range []struct {
		name       string
		summarized bool
		retryable  bool
	}{
		{name: "summarized Worker result", summarized: true, retryable: false},
		{name: "successful candidate", retryable: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
			defer cancel()
			h := newDeferredPlacementHarness(t, ctx, nil, func(workflow *resolved.ResolvedWorkflow) {
				configureSummaryWorkflow(t, workflow)
				stage := workflow.Stages[workflow.EntryStage]
				stage.On.Interrupted = resolved.TransitionAction{
					Kind: resolved.TransitionRetry,
					Retry: &resolved.RetryTransition{
						MaxAttempts: 2, Then: resolved.TransitionAction{Kind: resolved.TransitionFail},
					},
				}
				workflow.Stages[workflow.EntryStage] = stage
				if err := resolved.ResolveEscalationVariants(workflow); err != nil {
					t.Fatal(err)
				}
			})
			rejectPlannerCompletions(t, ctx, h.pool)
			if test.summarized {
				h.invoker.result = summarizedCandidate()
			}

			if worked, err := h.scheduler.RunOnce(ctx); !worked || err != nil {
				t.Fatalf("RunOnce = (%t, %v)", worked, err)
			}
			executions, err := h.store.ListStageExecutions(ctx, "run-1")
			if err != nil || len(executions) == 0 {
				t.Fatalf("StageExecutions = (%+v, %v)", executions, err)
			}
			source := executions[0]
			if source.State != runstore.StageInterrupted || source.Termination == nil ||
				source.Termination.Code != "planner_session_unavailable" ||
				source.Termination.Retryable != test.retryable || source.CandidateResult != nil ||
				source.AcceptedResult != nil || h.invoker.calls != 1 {
				t.Fatalf("source StageExecution %s: termination %+v, candidate %t, accepted %t, Worker calls %d",
					source.State, source.Termination, source.CandidateResult != nil, source.AcceptedResult != nil,
					h.invoker.calls)
			}
			assertPlannerSessionRunning(t, ctx, h.store, source)
			run, err := h.store.GetRun(ctx, "run-1")
			if err != nil {
				t.Fatal(err)
			}
			if test.retryable {
				if len(executions) != 2 || executions[1].Attempt != 2 ||
					executions[1].ExecutionConfigVariant != runstore.StageExecutionConfigBase ||
					executions[1].StageContext.Continuation != nil || run.State != runstore.RunRunning {
					t.Fatalf("retryable outcome: Run %s, StageExecution states %v", run.State, stageExecutionStates(executions))
				}
				return
			}
			if len(executions) != 1 || run.State != runstore.RunFailed ||
				run.StateReason.Code != "planner_session_unavailable" {
				t.Fatalf("final outcome: Run %s %+v, %d StageExecutions", run.State, run.StateReason, len(executions))
			}
			if escalated, err := h.store.HasSummaryEscalation(ctx, "run-1", h.workflow.EntryStage); err != nil || escalated {
				t.Fatalf("summary continuation started without a durable candidate: %t %v", escalated, err)
			}
		})
	}
}

// rejectPlannerCompletions makes the isolated schema refuse every Planner
// session completion with a non-transient PostgreSQL error, so the completion
// write fails after the Planner has its outcome and is not retried.
func rejectPlannerCompletions(t *testing.T, ctx context.Context, pool *pgxpool.Pool) {
	t.Helper()
	if _, err := pool.Exec(ctx, `
CREATE FUNCTION reject_planner_completion() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'test rejects the Planner session completion';
END
$$;
CREATE TRIGGER reject_planner_completion BEFORE UPDATE ON planner_sessions FOR EACH ROW
WHEN (NEW.state->>'status' = 'completed') EXECUTE FUNCTION reject_planner_completion();`); err != nil {
		t.Fatal(err)
	}
}

func assertPlannerSessionRunning(
	t *testing.T, ctx context.Context, store *runstore.PostgresStore, execution runstore.StageExecution,
) {
	t.Helper()
	if execution.PlannerSessionID == nil {
		t.Fatal("StageExecution has no Planner session")
	}
	session, err := store.GetPlannerSession(ctx, *execution.PlannerSessionID)
	if err != nil {
		t.Fatal(err)
	}
	var state struct {
		Status     string          `json:"status"`
		Completion json.RawMessage `json:"completion"`
	}
	if err := json.Unmarshal(session.State, &state); err != nil || state.Status != "running" || state.Completion != nil {
		t.Fatalf("Planner session state = %s (%v), want running without a completion", session.State, err)
	}
}
