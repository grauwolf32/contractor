package planner

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"github.com/jackc/pgx/v5/pgconn"
)

// scriptedMetadataRepository answers every metadata lookup of the production
// ArtifactStore service with answer; any other Repository operation panics.
type scriptedMetadataRepository struct {
	artifacts.Repository
	answer  func(contracts.ArtifactRef) (artifacts.Metadata, error)
	queries int
}

func (r *scriptedMetadataRepository) Metadata(
	_ context.Context, _ artifacts.Scope, ref contracts.ArtifactRef,
) (artifacts.Metadata, error) {
	r.queries++
	return r.answer(ref)
}

func verificationInspector(t *testing.T, repository *scriptedMetadataRepository) ArtifactInspector {
	t.Helper()
	inspector, err := NewArtifactServiceInspector(artifacts.NewService(repository))
	if err != nil {
		t.Fatal(err)
	}
	return inspector
}

func verificationCandidate(refs map[string]contracts.ArtifactRef) stageresult.StageContentResult {
	return stageresult.StageContentResult{
		APIVersion: contracts.APIVersion, Outcome: stageresult.StageSucceeded, Summary: "done",
		Artifacts: refs,
	}
}

func verificationRef(name, revision string) contracts.ArtifactRef {
	return contracts.ArtifactRef{Namespace: "builder", Name: name, Revision: &revision}
}

func transientMetadataFailure(ref contracts.ArtifactRef) error {
	return fmt.Errorf("read artifact metadata %s/%s: %w", ref.Namespace, ref.Name,
		&pgconn.PgError{Code: persistencepostgres.SQLStateLockNotAvailable})
}

// A Stage candidate whose exact result reference the store rejects is a
// deterministic contract violation; a store that cannot answer leaves the
// candidate unverified and retryable (PLN-R118).
func TestValidateCandidateClassifiesResultArtifactVerification(t *testing.T) {
	contract := map[string]resolved.ArtifactSlot{
		"report": {Required: true, MediaTypes: []string{"application/json"}},
	}
	ref := verificationRef("report", "result-r1")
	other := "result-r2"
	for _, test := range []struct {
		name      string
		answer    func(contracts.ArtifactRef) (artifacts.Metadata, error)
		ctx       func(context.Context) context.Context
		code      string
		retryable bool
	}{
		{
			name: "missing exact revision",
			answer: func(contracts.ArtifactRef) (artifacts.Metadata, error) {
				return artifacts.Metadata{}, artifacts.ErrArtifactNotFound
			},
			code: "result_contract_violation",
		},
		{
			name: "store rejects the reference",
			answer: func(contracts.ArtifactRef) (artifacts.Metadata, error) {
				return artifacts.Metadata{}, artifacts.ErrInvalidName
			},
			code: "result_contract_violation",
		},
		{
			name: "store describes another revision",
			answer: func(ref contracts.ArtifactRef) (artifacts.Metadata, error) {
				return artifacts.Metadata{
					Ref:       contracts.ArtifactRef{Namespace: ref.Namespace, Name: ref.Name, Revision: &other},
					MediaType: "application/json",
				}, nil
			},
			code: "result_contract_violation",
		},
		{
			name: "store describes an inexact binding",
			answer: func(ref contracts.ArtifactRef) (artifacts.Metadata, error) {
				return artifacts.Metadata{
					Ref: contracts.ArtifactRef{Namespace: ref.Namespace, Name: ref.Name}, MediaType: "application/json",
				}, nil
			},
			code: "result_contract_violation",
		},
		{
			name: "transient database failure",
			answer: func(ref contracts.ArtifactRef) (artifacts.Metadata, error) {
				return artifacts.Metadata{}, transientMetadataFailure(ref)
			},
			code: "result_artifact_unavailable", retryable: true,
		},
		{
			name: "cancelled lookup",
			answer: func(contracts.ArtifactRef) (artifacts.Metadata, error) {
				return artifacts.Metadata{}, context.Canceled
			},
			code: "result_artifact_unavailable", retryable: true,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			repository := &scriptedMetadataRepository{answer: test.answer}
			failure := ValidateCandidate(
				t.Context(), "run-1", contract,
				verificationCandidate(map[string]contracts.ArtifactRef{"report": ref}),
				verificationInspector(t, repository),
			)
			if failure == nil || failure.Code != test.code || failure.Retryable != test.retryable {
				t.Fatalf("ValidateCandidate = %+v, want %s retryable=%t", failure, test.code, test.retryable)
			}
			if repository.queries != 1 {
				t.Fatalf("metadata queries = %d, want 1", repository.queries)
			}
		})
	}
}

// A malformed reference never reaches the store and is rejected with the
// same non-retryable class as a reference the store rejects.
func TestValidateCandidateRejectsMalformedReferenceBeforeStorage(t *testing.T) {
	contract := map[string]resolved.ArtifactSlot{"report": {Required: true, MediaTypes: []string{"*/*"}}}
	for name, ref := range map[string]contracts.ArtifactRef{
		"versionless":    {Namespace: "builder", Name: "report"},
		"invalid name":   verificationRef("bad name", "result-r1"),
		"empty revision": verificationRef("report", ""),
	} {
		t.Run(name, func(t *testing.T) {
			repository := &scriptedMetadataRepository{answer: func(contracts.ArtifactRef) (artifacts.Metadata, error) {
				t.Fatal("a malformed reference reached the ArtifactStore")
				return artifacts.Metadata{}, nil
			}}
			failure := ValidateCandidate(
				t.Context(), "run-1", contract,
				verificationCandidate(map[string]contracts.ArtifactRef{"report": ref}),
				verificationInspector(t, repository),
			)
			if failure == nil || failure.Retryable || repository.queries != 0 {
				t.Fatalf("ValidateCandidate = %+v after %d queries", failure, repository.queries)
			}
		})
	}
}

// Several failing artifacts are verified in result-name order, so the same
// candidate is always rejected for the same artifact and with the same class.
func TestValidateCandidateSelectsTheSameRejectionAcrossRuns(t *testing.T) {
	contract := map[string]resolved.ArtifactSlot{
		"alpha": {MediaTypes: []string{"*/*"}}, "beta": {MediaTypes: []string{"*/*"}},
		"gamma": {MediaTypes: []string{"*/*"}}, "delta": {MediaTypes: []string{"*/*"}},
	}
	for _, test := range []struct {
		name      string
		failing   map[string]func(contracts.ArtifactRef) error
		code      string
		retryable bool
		message   string
	}{
		{
			name: "rejected reference first",
			failing: map[string]func(contracts.ArtifactRef) error{
				"alpha": func(contracts.ArtifactRef) error { return artifacts.ErrArtifactNotFound },
				"beta":  transientMetadataFailure, "gamma": transientMetadataFailure,
				"delta": transientMetadataFailure,
			},
			code: "result_contract_violation", message: `"alpha"`,
		},
		{
			name: "unavailable reference first",
			failing: map[string]func(contracts.ArtifactRef) error{
				"alpha": transientMetadataFailure,
				"beta":  func(contracts.ArtifactRef) error { return artifacts.ErrArtifactNotFound },
				"gamma": func(contracts.ArtifactRef) error { return artifacts.ErrArtifactNotFound },
				"delta": func(contracts.ArtifactRef) error { return artifacts.ErrArtifactNotFound },
			},
			code: "result_artifact_unavailable", retryable: true, message: `"alpha"`,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			refs := map[string]contracts.ArtifactRef{}
			byName := map[string]func(contracts.ArtifactRef) error{}
			for name, fail := range test.failing {
				refs[name] = verificationRef(name, name+"-r1")
				byName[name] = fail
			}
			repository := &scriptedMetadataRepository{answer: func(ref contracts.ArtifactRef) (artifacts.Metadata, error) {
				return artifacts.Metadata{}, byName[ref.Name](ref)
			}}
			inspector := verificationInspector(t, repository)
			for attempt := range 64 {
				failure := ValidateCandidate(t.Context(), "run-1", contract, verificationCandidate(refs), inspector)
				if failure == nil || failure.Code != test.code || failure.Retryable != test.retryable ||
					!strings.Contains(failure.Message, test.message) {
					t.Fatalf("attempt %d: ValidateCandidate = %+v, want %s for %s", attempt, failure, test.code, test.message)
				}
			}
		})
	}
}

// A recorded result is revalidated with the same classification as a fresh
// candidate.
func TestRecoverCompletionClassifiesResultArtifactVerification(t *testing.T) {
	contract := map[string]resolved.ArtifactSlot{"report": {Required: true, MediaTypes: []string{"*/*"}}}
	recorded := verificationCandidate(map[string]contracts.ArtifactRef{"report": verificationRef("report", "result-r1")})
	for _, test := range []struct {
		err       func(contracts.ArtifactRef) error
		code      string
		retryable bool
	}{
		{func(contracts.ArtifactRef) error { return artifacts.ErrArtifactNotFound }, "result_contract_violation", false},
		{transientMetadataFailure, "result_artifact_unavailable", true},
	} {
		repository := &scriptedMetadataRepository{answer: func(ref contracts.ArtifactRef) (artifacts.Metadata, error) {
			return artifacts.Metadata{}, test.err(ref)
		}}
		_, err := RecoverCompletion(
			t.Context(), "run-1", contract, Completion{Result: &recorded}, verificationInspector(t, repository),
		)
		if failure := FailureFrom(err); failure.Code != test.code || failure.Retryable != test.retryable {
			t.Fatalf("recovered completion = %+v, want %s retryable=%t", failure, test.code, test.retryable)
		}
	}
}

// The Passthrough Worker invocation whose candidate is rejected is one failed
// a2a.invoke call, and its tool record, metrics and report-level error agree
// (PLN-R119).
func TestPassthroughRejectedCandidateIsOneFailedInvocation(t *testing.T) {
	for _, test := range []struct {
		name      string
		err       func(contracts.ArtifactRef) error
		code      string
		retryable bool
	}{
		{"missing result artifact", func(contracts.ArtifactRef) error { return artifacts.ErrArtifactNotFound },
			"result_contract_violation", false},
		{"unverifiable result artifact", transientMetadataFailure, "result_artifact_unavailable", true},
	} {
		t.Run(test.name, func(t *testing.T) {
			candidate := verificationCandidate(map[string]contracts.ArtifactRef{"report": verificationRef("report", "result-r1")})
			repository := &scriptedMetadataRepository{answer: func(ref contracts.ArtifactRef) (artifacts.Metadata, error) {
				return artifacts.Metadata{}, test.err(ref)
			}}
			sessions := &memorySessions{}
			worker := &recordingWorker{result: workerCompletionFromCandidate(candidate)}
			factory, err := NewPassthroughFactory(sessions, worker, verificationInspector(t, repository))
			if err != nil {
				t.Fatal(err)
			}
			instance, err := factory.Create(testInvocation())
			if err != nil {
				t.Fatal(err)
			}
			_, err = instance.Run(t.Context())
			var plannerError *Error
			if !errors.As(err, &plannerError) || plannerError.Code != test.code || plannerError.Retryable != test.retryable {
				t.Fatalf("Run error = %v, want %s retryable=%t", err, test.code, test.retryable)
			}
			if worker.calls != 1 || sessions.completion == nil || sessions.completion.Failure == nil ||
				sessions.completion.Failure.Code != test.code || sessions.completion.Failure.Retryable != test.retryable {
				t.Fatalf("Worker calls = %d, recorded completion = %+v", worker.calls, sessions.completion)
			}
			report, ok := instance.(ReportProvider).ExecutionReport()
			if !ok {
				t.Fatal("rejected candidate left no Planner execution report")
			}
			metrics := report.Metrics.Tools["a2a.invoke"]
			if metrics.Calls == nil || *metrics.Calls != 1 || metrics.Succeeded == nil || *metrics.Succeeded != 0 ||
				metrics.Failed == nil || *metrics.Failed != 1 {
				t.Fatalf("a2a.invoke metrics = %+v", metrics)
			}
			if len(report.ToolCalls) != 1 || report.ToolCalls[0].Tool != "a2a.invoke" ||
				report.ToolCalls[0].Outcome != reporting.ToolCallFailed || report.ToolCalls[0].Error == nil ||
				report.ToolCalls[0].Error.Code != test.code || report.ToolCalls[0].Error.Retryable == nil ||
				*report.ToolCalls[0].Error.Retryable != test.retryable {
				t.Fatalf("a2a.invoke tool calls = %+v", report.ToolCalls)
			}
			if len(report.Errors) != 1 || report.Errors[0].Code != test.code || report.Errors[0].Retryable == nil ||
				*report.Errors[0].Retryable != test.retryable {
				t.Fatalf("report errors = %+v", report.Errors)
			}
		})
	}
}

// A failed session write after the Planner judged the invocation changes only
// the report-level error: the a2a.invoke call keeps its verdict, and the
// Planner returns planner_session_unavailable instead of the candidate or the
// rejection, retryable after an accepted result and with the rejection's
// retryability otherwise (PLN-R106, PLN-R119).
func TestPassthroughSessionWriteFailureKeepsInvocationVerdict(t *testing.T) {
	for _, test := range []struct {
		name          string
		err           func(contracts.ArtifactRef) error
		callCode      string
		callRetryable bool
		retryable     bool
	}{
		{name: "accepted result", retryable: true},
		{
			name: "rejected candidate", err: func(contracts.ArtifactRef) error { return artifacts.ErrArtifactNotFound },
			callCode: "result_contract_violation", callRetryable: false, retryable: false,
		},
		{
			name: "unverifiable candidate", err: transientMetadataFailure,
			callCode: "result_artifact_unavailable", callRetryable: true, retryable: true,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			candidate := verificationCandidate(map[string]contracts.ArtifactRef{"report": verificationRef("report", "result-r1")})
			repository := &scriptedMetadataRepository{answer: func(ref contracts.ArtifactRef) (artifacts.Metadata, error) {
				if test.err != nil {
					return artifacts.Metadata{}, test.err(ref)
				}
				return artifacts.Metadata{Ref: ref.Clone(), MediaType: "application/json"}, nil
			}}
			var attempted []Completion
			sessions := &memorySessions{completeHook: func(_ context.Context, completion Completion) error {
				attempted = append(attempted, completion)
				return errors.New("database write failed")
			}}
			worker := &recordingWorker{result: workerCompletionFromCandidate(candidate)}
			factory, err := NewPassthroughFactory(sessions, worker, verificationInspector(t, repository))
			if err != nil {
				t.Fatal(err)
			}
			instance, err := factory.Create(testInvocation())
			if err != nil {
				t.Fatal(err)
			}
			result, err := instance.Run(t.Context())
			var plannerError *Error
			if !errors.As(err, &plannerError) || plannerError.Code != "planner_session_unavailable" ||
				plannerError.Retryable != test.retryable {
				t.Fatalf("Run error = %v, want planner_session_unavailable retryable=%t", err, test.retryable)
			}
			if result.Outcome != "" {
				t.Fatalf("Run returned candidate %+v with a failed session write", result)
			}
			if worker.calls != 1 || len(attempted) != 1 || sessions.completion != nil {
				t.Fatalf("Worker calls = %d, completion writes = %+v, recorded = %+v", worker.calls, attempted, sessions.completion)
			}
			if test.callCode == "" {
				if attempted[0].Result == nil || attempted[0].Failure != nil {
					t.Fatalf("attempted completion = %+v, want the accepted candidate", attempted[0])
				}
			} else if attempted[0].Failure == nil || attempted[0].Failure.Code != test.callCode {
				t.Fatalf("attempted completion = %+v, want failure %s", attempted[0], test.callCode)
			}

			report, ok := instance.(ReportProvider).ExecutionReport()
			if !ok {
				t.Fatal("failed session write left no Planner execution report")
			}
			succeeded, failed := int64(1), int64(0)
			if test.callCode != "" {
				succeeded, failed = 0, 1
			}
			metrics := report.Metrics.Tools["a2a.invoke"]
			if metrics.Calls == nil || *metrics.Calls != 1 || metrics.Succeeded == nil || *metrics.Succeeded != succeeded ||
				metrics.Failed == nil || *metrics.Failed != failed {
				t.Fatalf("a2a.invoke metrics = %+v", metrics)
			}
			if len(report.ToolCalls) != 1 || report.ToolCalls[0].Tool != "a2a.invoke" {
				t.Fatalf("tool calls = %+v", report.ToolCalls)
			}
			call := report.ToolCalls[0]
			if test.callCode == "" {
				if call.Outcome != reporting.ToolCallSucceeded || call.Error != nil {
					t.Fatalf("a2a.invoke call = %+v, want succeeded", call)
				}
			} else if call.Outcome != reporting.ToolCallFailed || call.Error == nil || call.Error.Code != test.callCode ||
				call.Error.Retryable == nil || *call.Error.Retryable != test.callRetryable {
				t.Fatalf("a2a.invoke call = %+v, want failed %s retryable=%t", call, test.callCode, test.callRetryable)
			}
			if len(report.Errors) != 1 || report.Errors[0].Code != "planner_session_unavailable" ||
				report.Errors[0].Retryable == nil || *report.Errors[0].Retryable != test.retryable {
				t.Fatalf("report errors = %+v", report.Errors)
			}
		})
	}
}
