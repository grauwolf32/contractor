package planner

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

// An outcome that the Passthrough Planner cannot record in its session ends
// the invocation with planner_session_unavailable, which keeps the outcome's
// retryability: retryable after a successful candidate or a retryable
// failure, non-retryable after a Planner failure or failed candidate whose
// error is non-retryable, a summarized Worker result included (PLN-R106).
// The report ends with the returned error and the a2a.invoke call keeps its
// verdict (PLN-R119).
func TestPassthroughUnwrittenOutcomeKeepsItsRetryability(t *testing.T) {
	revision := "result-r1"
	succeeded := workerCompletionFromCandidate(stageresult.StageContentResult{
		APIVersion: contracts.APIVersion, Outcome: stageresult.StageSucceeded, Summary: "report created",
		Artifacts: map[string]contracts.ArtifactRef{
			"report": {Namespace: "builder", Name: "report", Revision: &revision},
		},
	})
	workerFailure := func(code string, retryable bool) contracts.WorkerCompletion {
		return contracts.WorkerCompletion{
			APIVersion: contracts.APIVersion,
			Failure: &contracts.WorkerFailure{
				Code: code, Message: "Worker did not complete", Retryable: retryable,
			},
			InvocationID: "worker-invocation-failed", StateRevision: 3,
		}
	}
	summarized := workerCompletionFromCandidate(stageresult.StageContentResult{
		APIVersion: contracts.APIVersion, Summary: "Partial analysis; continue the report.",
		Artifacts: map[string]contracts.ArtifactRef{},
	})
	summarized.Result.Summarized = true
	for _, test := range []struct {
		name      string
		worker    *recordingWorker
		callCode  string
		recorded  string
		retryable bool
	}{
		{name: "successful candidate", worker: &recordingWorker{result: succeeded}, retryable: true},
		{
			name: "retryable Worker failure", worker: &recordingWorker{result: workerFailure("worker_budget_exhausted", true)},
			callCode: "worker_budget_exhausted", recorded: "worker_budget_exhausted", retryable: true,
		},
		{
			name: "non-retryable Worker failure", worker: &recordingWorker{result: workerFailure("tool_input_invalid", false)},
			callCode: "tool_input_invalid", recorded: "tool_input_invalid", retryable: false,
		},
		{
			name: "summarized Worker result", worker: &recordingWorker{result: summarized},
			recorded: stageresult.SummaryRequiredCode, retryable: false,
		},
		{
			name: "retryable Planner failure", worker: &recordingWorker{err: errors.New("A2A transport closed")},
			callCode: "worker_unavailable", recorded: "worker_unavailable", retryable: true,
		},
		{
			name: "non-retryable Planner failure",
			worker: &recordingWorker{err: NewError(
				"worker_input_required", "Worker requires input", false, nil,
			)},
			callCode: "worker_input_required", recorded: "worker_input_required", retryable: false,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			var attempted []Completion
			sessions := &memorySessions{completeHook: func(_ context.Context, completion Completion) error {
				attempted = append(attempted, completion)
				return errors.New("database write failed")
			}}
			factory, err := NewPassthroughFactory(sessions, test.worker, &fakeInspector{
				mediaTypes: map[string]string{"builder/report/result-r1": "application/json"},
			})
			if err != nil {
				t.Fatal(err)
			}
			instance, err := factory.Create(testInvocation())
			if err != nil {
				t.Fatal(err)
			}
			result, err := instance.Run(t.Context())
			assertUnwrittenOutcome(t, err, test.retryable)
			if result.Outcome != "" || test.worker.calls != 1 || len(attempted) != 1 || sessions.completion != nil {
				t.Fatalf("result = %+v, Worker calls = %d, writes = %+v, recorded = %+v",
					result, test.worker.calls, attempted, sessions.completion)
			}
			if code := attemptedCode(attempted[0]); code != test.recorded {
				t.Fatalf("attempted completion code = %q, want %q", code, test.recorded)
			}
			report, ok := instance.(ReportProvider).ExecutionReport()
			if !ok || len(report.ToolCalls) != 1 {
				t.Fatalf("report = (%+v, %t)", report, ok)
			}
			call := report.ToolCalls[0]
			if test.callCode == "" && (call.Outcome != reporting.ToolCallSucceeded || call.Error != nil) ||
				test.callCode != "" && (call.Outcome != reporting.ToolCallFailed || call.Error == nil ||
					call.Error.Code != test.callCode) {
				t.Fatalf("a2a.invoke call = %+v, want verdict %q", call, test.callCode)
			}
			assertReportEndsWith(t, report, "planner_session_unavailable", test.retryable)
			if len(report.Errors) != 1 {
				t.Fatalf("report errors = %+v, want only the returned error", report.Errors)
			}
		})
	}
}

// A session span carries the code of the error the Planner returns: a session
// that another invocation still runs is planner_invocation_in_progress,
// retryable (PLN-R107), and a write the store refuses as invalid is
// planner_session_rejected, non-retryable (PLN-R106), not
// planner_session_unavailable.
func TestSessionSpansCarryTheReturnedCode(t *testing.T) {
	for _, test := range []struct {
		name      string
		cause     error
		code      string
		outcome   string
		retryable bool
	}{
		{
			name: "invocation in progress", cause: errors.Join(errors.New("load"), ErrInvocationInProgress),
			code: "planner_invocation_in_progress", outcome: "rejected", retryable: true,
		},
		{
			name: "store failure", cause: errors.New("database write failed"),
			code: "planner_session_unavailable", outcome: "unavailable", retryable: true,
		},
		{
			name: "store rejection", cause: fmt.Errorf("start Planner: %w: state too large", runstore.ErrInvalid),
			code: "planner_session_rejected", outcome: "rejected", retryable: false,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			spans := &recordingInstrumentation{}
			_, err := StartSession(t.Context(), failingSessions{err: test.cause}, spans, "stage-execution-1")
			assertSessionError(t, err, test.code, test.retryable)
			err = RecordSessionRequest(
				t.Context(), failingSessions{err: test.cause}, spans, SessionIdentity{SessionID: "session-1"}, RequestFacts{},
			)
			assertSessionError(t, err, test.code, test.retryable)
			if len(spans.ended) != 2 {
				t.Fatalf("ended spans = %+v", spans.ended)
			}
			for _, span := range spans.ended {
				if span.outcome != test.outcome || span.attributes.ErrorCode != test.code {
					t.Fatalf("session span = %+v, want %s %s", span, test.outcome, test.code)
				}
			}
		})
	}
}

// A completion the store refuses as invalid, such as a state over its stored
// size bound, ends even a retryable outcome with the non-retryable
// planner_session_rejected: repeating the write is refused the same way
// (PLN-R106).
func TestPassthroughRejectedCompletionIsNotRetryable(t *testing.T) {
	sessions := &memorySessions{completeHook: func(context.Context, Completion) error {
		return fmt.Errorf("append Planner event: %w: planner_session_state_size", runstore.ErrInvalid)
	}}
	worker := &recordingWorker{err: errors.New("A2A transport closed")}
	factory, err := NewPassthroughFactory(sessions, worker, &fakeInspector{})
	if err != nil {
		t.Fatal(err)
	}
	instance, err := factory.Create(testInvocation())
	if err != nil {
		t.Fatal(err)
	}
	_, err = instance.Run(t.Context())
	assertSessionError(t, err, "planner_session_rejected", false)
	if !errors.Is(err, runstore.ErrInvalid) || sessions.completion != nil {
		t.Fatalf("error = %v, recorded = %+v, want the store rejection and no completion", err, sessions.completion)
	}
	report, ok := instance.(ReportProvider).ExecutionReport()
	if !ok {
		t.Fatal("no report")
	}
	assertReportEndsWith(t, report, "planner_session_rejected", false)
	assertZeroModelUsage(t, report)
}

func assertUnwrittenOutcome(t *testing.T, err error, retryable bool) {
	t.Helper()
	assertSessionError(t, err, "planner_session_unavailable", retryable)
}

func assertSessionError(t *testing.T, err error, code string, retryable bool) {
	t.Helper()
	var plannerError *Error
	if !errors.As(err, &plannerError) || plannerError.Code != code || plannerError.Retryable != retryable {
		t.Fatalf("error = %v, want %s retryable=%t", err, code, retryable)
	}
}

func assertReportEndsWith(t *testing.T, report reporting.ExecutionReport, code string, retryable bool) {
	t.Helper()
	if len(report.Errors) == 0 {
		t.Fatal("report has no errors")
	}
	last := report.Errors[len(report.Errors)-1]
	if last.Code != code || last.Retryable == nil || *last.Retryable != retryable {
		t.Fatalf("report errors = %+v, want last %s retryable=%t", report.Errors, code, retryable)
	}
}

func attemptedCode(completion Completion) string {
	switch {
	case completion.Failure != nil:
		return completion.Failure.Code
	case completion.Result != nil && completion.Result.Error != nil:
		return completion.Result.Error.Code
	default:
		return ""
	}
}

type failingSessions struct{ err error }

func (s failingSessions) Begin(context.Context, string) (SessionStart, error) {
	return SessionStart{}, s.err
}

func (s failingSessions) RecordRequest(context.Context, SessionIdentity, RequestFacts) error {
	return s.err
}

func (s failingSessions) Complete(context.Context, SessionIdentity, Completion) error { return s.err }

type endedSpan struct {
	outcome    string
	attributes telemetry.PlannerSpanAttributes
}

type recordingInstrumentation struct{ ended []endedSpan }

func (r *recordingInstrumentation) StartSpan(
	telemetry.PlannerSpanName, telemetry.PlannerSpanAttributes,
) telemetry.PlannerSpan {
	return recordingSpan{instrumentation: r}
}

type recordingSpan struct{ instrumentation *recordingInstrumentation }

func (s recordingSpan) End(outcome string, attributes telemetry.PlannerSpanAttributes) {
	s.instrumentation.ended = append(s.instrumentation.ended, endedSpan{outcome: outcome, attributes: attributes})
}
