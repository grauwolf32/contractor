package streamline

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// An outcome that Streamline cannot record in its session ends the invocation
// with planner_session_unavailable, which keeps the outcome's retryability
// (PLN-R106). The report ends with that returned error; a Planner failure
// that could not be recorded precedes it (PLN-R120).
func TestStreamlineUnwrittenOutcomeKeepsItsRetryability(t *testing.T) {
	const revision = "report-r1"
	failedFinish := func(retryable bool) modelStep {
		return functionStep(finishToolName, map[string]any{
			"outcome": string(stageresult.StageFailed), "summary": "analysis could not be completed",
			"artifacts": map[string]any{},
			"error": map[string]any{
				"code": "insufficient_evidence", "message": "required evidence was unavailable", "retryable": retryable,
			},
		})
	}
	rejected := func(*model.LLMRequest) (*model.LLMResponse, error) {
		return nil, &gatewayrecovery.FailureError{Failure: gatewayrecovery.Failure{Code: "gateway_access_denied"}}
	}
	for _, test := range []struct {
		name      string
		steps     []modelStep
		fallback  modelStep
		recorded  string
		failure   string
		retryable bool
	}{
		{
			name: "successful candidate",
			steps: []modelStep{
				addSubtaskStep("Analyze the input", "Produce the final report"),
				functionStep(executeCurrentSubtaskToolName, map[string]any{"subtask_id": "0"}),
				functionStep(finishToolName, map[string]any{
					"outcome": string(stageresult.StageSucceeded), "summary": "complete", "artifacts": map[string]any{
						"report": artifactArgs("review", "report", revision),
					},
				}),
			},
			retryable: true,
		},
		{name: "retryable failed candidate", steps: []modelStep{failedFinish(true)}, recorded: "insufficient_evidence", retryable: true},
		{name: "non-retryable failed candidate", steps: []modelStep{failedFinish(false)}, recorded: "insufficient_evidence", retryable: false},
		{
			name: "budget stop", fallback: textStep("not finished"),
			recorded: "planner_model_call_limit", failure: "planner_model_call_limit", retryable: true,
		},
		{
			name: "non-retryable Planner failure", steps: []modelStep{rejected},
			recorded: "planner_gateway_rejected", failure: "planner_gateway_rejected", retryable: false,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			sessions := newFakeSessions()
			var attempted []planner.Completion
			sessions.completeHook = func(_ context.Context, completion planner.Completion) error {
				attempted = append(attempted, completion)
				return errors.New("database write failed")
			}
			workers := &fakeWorkerInvoker{results: map[string]stageresult.StageContentResult{
				"builder": stageResult("report ready", map[string]contracts.ArtifactRef{
					"report": exactRef("review", "report", revision),
				}),
			}}
			inspector := &fakeInspector{mediaTypes: map[string]string{
				refKey(exactRef("inputs", "source", "source-r1")): "text/plain",
				refKey(exactRef("review", "report", revision)):    "application/json",
			}}
			llm := &scriptedModel{steps: test.steps, fallback: test.fallback}
			invocation := testInvocation("builder")
			invocation.ModelAccess.ModelPolicy.MaxModelCalls = 4
			instance, err := mustFactory(
				t, sessions, workers, inspector, llm, Limits{MaxWallTime: time.Minute},
			).Create(invocation)
			if err != nil {
				t.Fatal(err)
			}
			result, err := instance.Run(t.Context())
			assertUnwrittenOutcome(t, err, test.retryable)
			if result.Outcome != "" || len(attempted) != 1 || sessions.completion != nil {
				t.Fatalf("result = %+v, writes = %+v, recorded = %+v", result, attempted, sessions.completion)
			}
			if code := attemptedCode(attempted[0]); code != test.recorded {
				t.Fatalf("attempted completion code = %q, want %q", code, test.recorded)
			}
			report, ok := instance.(planner.ReportProvider).ExecutionReport()
			if !ok {
				t.Fatal("unwritten outcome left no Planner report")
			}
			want := []string{"planner_session_unavailable"}
			if test.failure != "" {
				want = []string{test.failure, "planner_session_unavailable"}
			}
			assertReportErrors(t, report, want, test.retryable)
			if test.failure == "" && !finishCalled(report) {
				t.Fatalf("tool calls = %+v, want the accepted finish call", report.ToolCalls)
			}
		})
	}
}

// Rejected tool selections never push the failure that ended the invocation
// or the returned error out of the report's 100 errors (LIM-OBS-5).
func TestStreamlineReportKeepsReturnedErrorAtItsErrorBound(t *testing.T) {
	state := newExecutionState(DefaultLimits())
	unknown := &model.LLMResponse{
		Content: genai.NewContentFromFunctionCall("worker_not_prepared", map[string]any{}, genai.RoleModel),
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount: 1, CandidatesTokenCount: 1, TotalTokenCount: 2,
		},
	}
	for range maxReportErrors + 20 {
		if _, err := state.afterModel(unknown, map[string]struct{}{}); err != nil {
			t.Fatal(err)
		}
	}
	state.setExternalFailure(planner.NewError("planner_model_call_limit", "Planner exhausted its model-call limit", true, nil))
	returned := planner.Failure{
		Code: "planner_session_unavailable", Message: "Planner durable session is unavailable", Retryable: true,
	}
	report := state.report("planner-session", time.Second, &returned)
	if len(report.Errors) != maxReportErrors {
		t.Fatalf("report has %d errors, want %d", len(report.Errors), maxReportErrors)
	}
	tail := report.Errors[maxReportErrors-3:]
	if tail[0].Code != "planner_tool_selection_invalid" || tail[1].Code != "planner_model_call_limit" ||
		tail[2].Code != "planner_session_unavailable" {
		t.Fatalf("report error tail = %+v", tail)
	}
	// The returned error is reported once when it is the recorded failure.
	recorded := state.report("planner-session", time.Second, &planner.Failure{
		Code: "planner_model_call_limit", Message: "Planner exhausted its model-call limit", Retryable: true,
	})
	if last := recorded.Errors[len(recorded.Errors)-1]; len(recorded.Errors) != maxReportErrors-1 ||
		last.Code != "planner_model_call_limit" {
		t.Fatalf("report errors = %d, last %+v", len(recorded.Errors), last)
	}
}

func assertUnwrittenOutcome(t *testing.T, err error, retryable bool) {
	t.Helper()
	var plannerError *planner.Error
	if !errors.As(err, &plannerError) || plannerError.Code != "planner_session_unavailable" ||
		plannerError.Retryable != retryable {
		t.Fatalf("error = %v, want planner_session_unavailable retryable=%t", err, retryable)
	}
}

// assertReportErrors checks the report's error codes in order and that the
// last one, the error the Planner returned, has the given retryability.
func assertReportErrors(t *testing.T, report reporting.ExecutionReport, codes []string, retryable bool) {
	t.Helper()
	if len(report.Errors) != len(codes) {
		t.Fatalf("report errors = %+v, want %v", report.Errors, codes)
	}
	for index, code := range codes {
		if report.Errors[index].Code != code {
			t.Fatalf("report errors = %+v, want %v", report.Errors, codes)
		}
	}
	last := report.Errors[len(report.Errors)-1]
	if last.Retryable == nil || *last.Retryable != retryable {
		t.Fatalf("returned report error = %+v, want retryable=%t", last, retryable)
	}
}

func attemptedCode(completion planner.Completion) string {
	switch {
	case completion.Failure != nil:
		return completion.Failure.Code
	case completion.Result != nil && completion.Result.Error != nil:
		return completion.Result.Error.Code
	default:
		return ""
	}
}

func finishCalled(report reporting.ExecutionReport) bool {
	for _, call := range report.ToolCalls {
		if call.Tool == finishToolName && call.Outcome == reporting.ToolCallSucceeded {
			return true
		}
	}
	return false
}

// Plan progress or a fact that the session store refuses as invalid, such as
// a state over its stored size bound, ends the invocation with the
// non-retryable planner_session_rejected, never the retryable
// planner_plan_persistence_failed that an outage returns (PLN-R106).
func TestStreamlineRejectedPlanWriteIsNotRetryable(t *testing.T) {
	for _, test := range []struct {
		name      string
		cause     error
		code      string
		retryable bool
	}{
		{name: "store outage", cause: errors.New("database write failed"),
			code: "planner_plan_persistence_failed", retryable: true},
		{name: "store rejection", cause: fmt.Errorf("append Planner event: %w: planner_session_state_size", runstore.ErrInvalid),
			code: "planner_session_rejected", retryable: false},
	} {
		t.Run(test.name, func(t *testing.T) {
			state := newExecutionState(DefaultLimits())
			failure := (&streamlinePlanner{}).planPersistenceFailure(state, test.cause)
			if failure.Code != test.code || failure.Retryable != test.retryable {
				t.Fatalf("failure = %+v, want %s retryable=%t", failure, test.code, test.retryable)
			}
			if state.failure == nil || state.failure.Code != test.code || !errors.Is(state.failure, test.cause) {
				t.Fatalf("state failure = %v, want %s wrapping the cause", state.failure, test.code)
			}
		})
	}
}
