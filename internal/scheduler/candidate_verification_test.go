package scheduler

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5/pgconn"
)

// verificationFailures are the outcomes of verifying one exact reference:
// the store rejects it, or it cannot answer.
var verificationFailures = []struct {
	name     string
	err      error
	rejected bool
}{
	{"missing revision", artifacts.ErrArtifactNotFound, true},
	{"rejected name", artifacts.ErrInvalidName, true},
	{"rejected scope", artifacts.ErrInvalidScope, true},
	{"versionless reference", artifacts.ErrExactRevisionRequired, true},
	{"malformed reference", contracts.Invalidf("artifact revision is required"), true},
	{"other revision", fmt.Errorf("%w: ArtifactStore resolved a different artifact revision", blob.ErrIntegrity), true},
	{"lock timeout", fmt.Errorf("read artifact metadata: %w",
		&pgconn.PgError{Code: persistencepostgres.SQLStateLockNotAvailable}), false},
	{"unclassified database failure", errors.New("temporary database failure"), false},
	{"transfer capacity", blob.ErrTransferCapacity, false},
	{"cancelled lookup", context.Canceled, false},
}

// The Planner and the Scheduler give every verification failure of a result
// artifact the same class: a rejected reference is a non-retryable contract
// violation in both, any other failure is retryable in both (PLN-R118,
// SCH-R177).
func TestResultArtifactVerificationVerdictMatchesPlanner(t *testing.T) {
	for _, test := range verificationFailures {
		t.Run(test.name, func(t *testing.T) {
			h := newSchedulerHarness(t)
			failing := resultResolverError{ArtifactResolver: h.artifacts, err: test.err}
			stage := h.workflow.Stages[h.workflow.EntryStage]
			plannerErr := planner.ValidateCandidate(
				t.Context(), h.store.run.RunID, stage.Result.Artifacts, h.planners.result.Clone(),
				resolverInspector{resolver: failing},
			)
			wantCode, wantRetryable := "result_verification_unavailable", true
			plannerCode := "result_artifact_unavailable"
			if test.rejected {
				wantCode, wantRetryable, plannerCode = "result_contract_violation", false, "result_contract_violation"
			}
			if plannerErr == nil || plannerErr.Code != plannerCode || plannerErr.Retryable != wantRetryable {
				t.Fatalf("Planner verdict = %+v, want %s retryable=%t", plannerErr, plannerCode, wantRetryable)
			}
			h.scheduler.artifacts = failing
			worked, err := h.scheduler.RunOnce(t.Context())
			if err != nil || !worked {
				t.Fatalf("RunOnce = (%v, %v)", worked, err)
			}
			execution := h.store.stages[0]
			if execution.State != runstore.StageInterrupted || execution.Termination == nil ||
				execution.Termination.Code != wantCode || execution.Termination.Retryable != wantRetryable {
				t.Fatalf("Scheduler termination = %+v, want %s retryable=%t", execution.Termination, wantCode, wantRetryable)
			}
		})
	}
}

// The saved summary of a summarized candidate is verified with the same
// classification as its result artifacts.
func TestSummaryArtifactVerificationKeepsItsClass(t *testing.T) {
	for _, test := range verificationFailures {
		t.Run(test.name, func(t *testing.T) {
			h := newSchedulerHarness(t)
			workflow, err := executableWorkflow{workflow: h.workflow}.selectStage(h.workflow.EntryStage)
			if err != nil {
				t.Fatal(err)
			}
			summary := exactRef("builder", "summary.stage-1", "summary-r1")
			candidate := summarizedCandidate()
			candidate.SummaryArtifact = &summary
			h.scheduler.artifacts = resultResolverError{ArtifactResolver: h.artifacts, err: test.err}
			err = h.scheduler.validateCandidate(t.Context(), h.store.run.RunID, workflow, candidate)
			var unavailable *candidateVerificationError
			if err == nil || errors.As(err, &unavailable) == test.rejected {
				t.Fatalf("summary verification = %v, want rejected=%t", err, test.rejected)
			}
		})
	}
}

type inexactMetadataRepository struct {
	artifacts.Repository
}

func (inexactMetadataRepository) Metadata(_ context.Context, _ artifacts.Scope, ref contracts.ArtifactRef) (artifacts.Metadata, error) {
	return artifacts.Metadata{Ref: contracts.ArtifactRef{Namespace: ref.Namespace, Name: ref.Name}, MediaType: "text/plain"}, nil
}

// A store answer without an exact revision breaks the store contract and no
// repeated lookup repairs it.
func TestArtifactResolverRejectsAnInexactStoreAnswer(t *testing.T) {
	resolver, err := NewArtifactServiceResolver(artifacts.NewService(inexactMetadataRepository{}))
	if err != nil {
		t.Fatal(err)
	}
	_, err = resolver.Resolve(t.Context(), "run-1", exactRef("builder", "copied", "result-r1"))
	if !errors.Is(err, blob.ErrIntegrity) || !artifacts.IsExactReferenceRejection(err) {
		t.Fatalf("inexact store answer = %v, want a rejected reference", err)
	}
}
