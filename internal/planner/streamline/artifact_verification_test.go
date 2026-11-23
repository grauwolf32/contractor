package streamline

import (
	"context"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"github.com/jackc/pgx/v5/pgconn"
)

// A Worker request or result artifact that the store rejects is a
// non-retryable tool rejection, and one the store cannot verify is a
// retryable one. Either way only the dispatch fails: the Planner keeps its
// tool-rejection flow and still owns finish (PLN-R118).
func TestStreamlineClassifiesWorkerArtifactVerificationAndContinues(t *testing.T) {
	source := exactRef("inputs", "source", "source-r1")
	report := exactRef("review", "report", "report-r1")
	other := "report-r2"
	missing := func(contracts.ArtifactRef) (artifacts.Metadata, error) {
		return artifacts.Metadata{}, artifacts.ErrArtifactNotFound
	}
	transient := func(ref contracts.ArtifactRef) (artifacts.Metadata, error) {
		return artifacts.Metadata{}, fmt.Errorf("read artifact metadata %s/%s: %w", ref.Namespace, ref.Name,
			&pgconn.PgError{Code: persistencepostgres.SQLStateLockNotAvailable})
	}
	stored := func(ref contracts.ArtifactRef) (artifacts.Metadata, error) {
		return artifacts.Metadata{Ref: ref.Clone(), MediaType: "application/json"}, nil
	}
	for _, test := range []struct {
		name          string
		source        func(contracts.ArtifactRef) (artifacts.Metadata, error)
		report        func(contracts.ArtifactRef) (artifacts.Metadata, error)
		code          string
		retryable     bool
		workerInvoked bool
	}{
		{name: "missing result artifact", source: stored, report: missing,
			code: "invalid_worker_result", workerInvoked: true},
		{name: "result artifact of another revision", source: stored,
			report: func(ref contracts.ArtifactRef) (artifacts.Metadata, error) {
				return artifacts.Metadata{
					Ref:       contracts.ArtifactRef{Namespace: ref.Namespace, Name: ref.Name, Revision: &other},
					MediaType: "application/json",
				}, nil
			},
			code: "invalid_worker_result", workerInvoked: true},
		{name: "unverifiable result artifact", source: stored, report: transient,
			code: "result_artifact_unavailable", retryable: true, workerInvoked: true},
		{name: "missing request artifact", source: missing, report: stored, code: "worker_request_invalid"},
		{name: "unverifiable request artifact", source: transient, report: stored,
			code: "worker_artifact_unavailable", retryable: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			inspector, err := newStreamlineInspector("run-1", streamlineMetadataReader(
				func(_ context.Context, ref contracts.ArtifactRef) (artifacts.Metadata, error) {
					switch refKey(ref) {
					case refKey(source):
						return test.source(ref)
					case refKey(report):
						return test.report(ref)
					default:
						return artifacts.Metadata{}, artifacts.ErrArtifactNotFound
					}
				},
			))
			if err != nil {
				t.Fatal(err)
			}
			model := &scriptedModel{steps: []modelStep{
				addSubtaskStep("Analyze the input", "Produce the final report"),
				functionStep(executeCurrentSubtaskToolName, map[string]any{"subtask_id": "0"}),
				functionStep(finishToolName, map[string]any{
					"outcome": string(stageresult.StageFailed), "summary": "the report could not be verified",
					"artifacts": map[string]any{},
					"error": map[string]any{
						"code": "report_unverified", "message": "the Worker report was not usable", "retryable": false,
					},
				}),
			}}
			workers := &fakeWorkerInvoker{results: map[string]stageresult.StageContentResult{
				"builder": stageResult("report ready", map[string]contracts.ArtifactRef{"report": report}),
			}}
			sessions := newFakeSessions()
			factory, err := NewConfiguredFactory(
				sessions, sessions, workers, inspector, unavailableWorkerStateReader{}, testModelFactory(model), Limits{},
			)
			if err != nil {
				t.Fatal(err)
			}
			instance, err := factory.Create(testInvocation("builder"))
			if err != nil {
				t.Fatal(err)
			}
			result, err := instance.Run(t.Context())
			if err != nil || result.Outcome != stageresult.StageFailed || result.Error == nil ||
				result.Error.Code != "report_unverified" {
				t.Fatalf("Run = (%+v, %v), want the Planner's own failed finish", result, err)
			}
			if invoked := len(workers.calls) == 1; invoked != test.workerInvoked || len(workers.calls) > 1 {
				t.Fatalf("Worker calls = %d, want invoked=%t", len(workers.calls), test.workerInvoked)
			}
			if sessions.plan == nil || sessions.plan.Subtasks[0].Status != planner.PlannerSubtaskFailed {
				t.Fatalf("rejected dispatch plan = %+v", sessions.plan)
			}
			report, _ := instance.(planner.ReportProvider).ExecutionReport()
			found := false
			for _, call := range report.ToolCalls {
				if call.Tool != executeCurrentSubtaskToolName {
					continue
				}
				if found || call.Error == nil || call.Error.Code != test.code || call.Error.Retryable == nil ||
					*call.Error.Retryable != test.retryable {
					t.Fatalf("dispatch tool error = %+v (duplicate=%t), want %s retryable=%t",
						call.Error, found, test.code, test.retryable)
				}
				found = true
			}
			if !found {
				t.Fatalf("no dispatch tool call was recorded: %+v", report.ToolCalls)
			}
		})
	}
}

// A finish candidate that cannot be verified is a retryable finish
// rejection the model may repeat unchanged, while one whose artifact the
// RunScope rejects stays a non-retryable rejection; either way the Planner
// keeps reasoning until a valid finish (PLN-R72, PLN-R118).
func TestStreamlineFinishRejectionKeepsVerificationClass(t *testing.T) {
	report := exactRef("review", "report", "report-r1")
	for _, test := range []struct {
		name      string
		answers   []error
		retryable bool
		retry     map[string]any
	}{
		{
			name: "unverifiable artifact",
			answers: []error{fmt.Errorf("read artifact metadata: %w",
				&pgconn.PgError{Code: persistencepostgres.SQLStateLockNotAvailable}), nil},
			retryable: true,
			retry:     map[string]any{"report": materialArgs("review", "report")},
		},
		{
			name:    "missing artifact",
			answers: []error{artifacts.ErrArtifactNotFound},
			retry:   map[string]any{},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			lookups, unexpected := 0, ""
			inspector, err := newStreamlineInspector("run-1", streamlineMetadataReader(
				func(_ context.Context, ref contracts.ArtifactRef) (artifacts.Metadata, error) {
					if refKey(ref) != refKey(report) || lookups >= len(test.answers) {
						unexpected = refKey(ref)
						return artifacts.Metadata{}, artifacts.ErrArtifactNotFound
					}
					answer := test.answers[lookups]
					lookups++
					if answer != nil {
						return artifacts.Metadata{}, answer
					}
					return artifacts.Metadata{Ref: ref.Clone(), MediaType: "application/json"}, nil
				},
			))
			if err != nil {
				t.Fatal(err)
			}
			failedFinish := func(resultArtifacts map[string]any) modelStep {
				return functionStep(finishToolName, map[string]any{
					"outcome": string(stageresult.StageFailed), "summary": "partial report", "artifacts": resultArtifacts,
					"error": map[string]any{"code": "incomplete", "message": "the review is incomplete", "retryable": false},
				})
			}
			model := &scriptedModel{steps: []modelStep{
				failedFinish(map[string]any{"report": materialArgs("review", "report")}),
				failedFinish(test.retry),
			}}
			sessions := newFakeSessions()
			factory, err := NewConfiguredFactory(
				sessions, sessions, &fakeWorkerInvoker{}, inspector, unavailableWorkerStateReader{}, testModelFactory(model), Limits{},
			)
			if err != nil {
				t.Fatal(err)
			}
			invocation := testInvocation("builder")
			invocation.Context.Artifacts["selected_report"] = pointerRef(report)
			invocation.Stage.Context.Artifacts["selected_report"] = resolved.ContextArtifact{Namespace: report.Namespace, Name: report.Name, Required: true}
			instance, err := factory.Create(invocation)
			if err != nil {
				t.Fatal(err)
			}
			result, err := instance.Run(t.Context())
			if err != nil || result.Outcome != stageresult.StageFailed || len(result.Artifacts) != len(test.retry) ||
				unexpected != "" {
				t.Fatalf("Run = (%+v, %v), unexpected lookup %q", result, err, unexpected)
			}
			reported, _ := instance.(planner.ReportProvider).ExecutionReport()
			var finishes []reporting.ToolCallRecord
			for _, call := range reported.ToolCalls {
				if call.Tool == finishToolName {
					finishes = append(finishes, call)
				}
			}
			if len(finishes) != 2 || finishes[0].Error == nil || finishes[0].Error.Code != "finish_rejected" ||
				finishes[0].Error.Retryable == nil || *finishes[0].Error.Retryable != test.retryable ||
				finishes[1].Outcome != reporting.ToolCallSucceeded {
				t.Fatalf("finish calls = %+v, want one rejection retryable=%t then acceptance", finishes, test.retryable)
			}
		})
	}
}
