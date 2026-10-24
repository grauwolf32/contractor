package router

import (
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/planner/streamline"
)

// Router records its outcome through the Streamline engine: a failed finish
// whose completion write fails ends with planner_session_unavailable of the
// candidate's retryability, and the report ends with that error (PLN-R106,
// PLN-R120).
func TestRouterUnwrittenFailedCandidateKeepsItsRetryability(t *testing.T) {
	for _, retryable := range []bool{false, true} {
		llm := &scriptedModel{steps: []modelStep{functionStep("finish", map[string]any{
			"outcome": "failed", "summary": "the API could not be described", "artifacts": map[string]any{},
			"error": map[string]any{
				"code": "insufficient_evidence", "message": "required evidence was unavailable", "retryable": retryable,
			},
		})}}
		sessions := newFakeSessions()
		sessions.completeErr = errors.New("database write failed")
		factory, err := NewConfiguredFactory(
			sessions, sessions, &fakeWorkerInvoker{}, fakeInspector{}, unavailableWorkerStateReader{},
			testModelFactory(llm), streamline.DefaultLimits(),
		)
		if err != nil {
			t.Fatal(err)
		}
		instance, err := factory.Create(testInvocation())
		if err != nil {
			t.Fatal(err)
		}
		result, err := instance.Run(t.Context())
		var plannerError *planner.Error
		if !errors.As(err, &plannerError) || plannerError.Code != "planner_session_unavailable" ||
			plannerError.Retryable != retryable || result.Outcome != "" || sessions.complete != nil {
			t.Fatalf("retryable=%t: Run = (%+v, %v), recorded %+v", retryable, result, err, sessions.complete)
		}
		report, _ := instance.(planner.ReportProvider).ExecutionReport()
		if len(report.Errors) != 1 || report.Errors[0].Code != "planner_session_unavailable" ||
			report.Errors[0].Retryable == nil || *report.Errors[0].Retryable != retryable {
			t.Fatalf("retryable=%t: report errors = %+v", retryable, report.Errors)
		}
	}
}
