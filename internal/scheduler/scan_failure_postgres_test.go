package scheduler

import (
	"context"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/stageresult"
)

// Planner/scan tests exercise the actual classification and completed-attempt
// reuse. This boundary test verifies that Scheduler persistence and transition
// selection preserve those classified failures, including the no-rescan scope.
func TestPostgresScanPlannerFailureRetryabilityControlsStageRetry(t *testing.T) {
	for _, test := range []struct {
		code      string
		retryable bool
	}{
		{"audit_scan_result_unavailable", true},
		{"audit_scan_result_invalid", false},
		{"scan_report_write_failed", false},
	} {
		t.Run(test.code, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
			defer cancel()
			h := newDeferredPlacementHarness(t, ctx, nil, func(workflow *resolved.ResolvedWorkflow) {
				stage := workflow.Stages[workflow.EntryStage]
				stage.On.Interrupted = resolved.TransitionAction{
					Kind:  resolved.TransitionRetry,
					Retry: &resolved.RetryTransition{MaxAttempts: 2, Then: resolved.TransitionAction{Kind: resolved.TransitionFail}},
				}
				workflow.Stages[workflow.EntryStage] = stage
			})
			h.scheduler.planners = scanFailureRegistry{
				sessions: h.scheduler.options.PlannerSessions,
				failure:  planner.NewError(test.code, "Scan planning or execution could not be completed", test.retryable, context.DeadlineExceeded),
			}
			if worked, err := h.scheduler.RunOnce(ctx); !worked || err != nil {
				t.Fatalf("RunOnce=(%t, %v)", worked, err)
			}
			executions, err := h.store.ListStageExecutions(ctx, "run-1")
			if err != nil || len(executions) == 0 {
				t.Fatalf("StageExecutions=(%+v, %v)", executions, err)
			}
			termination := executions[0].Termination
			if executions[0].State != runstore.StageInterrupted || termination == nil || termination.Code != test.code || termination.Retryable != test.retryable {
				t.Fatalf("persisted termination=%+v", termination)
			}
			wantAttempts := 1
			if test.retryable {
				wantAttempts = 2
			}
			if len(executions) != wantAttempts || h.invoker.calls != 0 {
				t.Fatalf("attempts=%d, Worker calls=%d; want %d queued attempts", len(executions), h.invoker.calls, wantAttempts)
			}
		})
	}
}

type scanFailureRegistry struct {
	sessions interface {
		Begin(context.Context, string) (planner.SessionStart, error)
	}
	failure error
}

func (r scanFailureRegistry) Create(_ string, invocation planner.Invocation) (planner.Planner, error) {
	return plannerFunc(func(ctx context.Context) (stageresult.StageContentResult, error) {
		if _, err := r.sessions.Begin(ctx, invocation.StageExecutionID); err != nil {
			return stageresult.StageContentResult{}, err
		}
		return stageresult.StageContentResult{}, r.failure
	}), nil
}
