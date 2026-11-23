package findingintake

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore/querystore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

// intakeExecution carries one real authorization lane. An operation never
// impersonates a Worker, allocation, Runtime instance or logical agent.
type intakeExecution struct {
	RunID, StageID, ClaimID string
	Worker                  *controlplane.AllocationGrant
	Source                  *artifacts.ExecutionOrigin
	SourceVersion           string
	RequestDigest           string
	Fingerprint             *auditdomain.FindingFingerprint
	Candidate, Evidence     contracts.ArtifactRef
	Selection               *contracts.ArtifactRef
}

func (e intakeExecution) receiptID(input Submission) string {
	if e.Worker != nil {
		return deterministicID("receipt", e.Worker.AllocationID, input.InvocationID, input.SubmissionID)
	}
	return deterministicID("receipt", e.SourceVersion, input.Proposal.ClientKey)
}
func (e intakeExecution) proposalID(input Submission) string {
	if e.Worker != nil {
		return deterministicID("proposal", e.Worker.AllocationID, input.InvocationID, input.SubmissionID)
	}
	return deterministicID("proposal", e.SourceVersion, input.Proposal.ClientKey)
}
func (e intakeExecution) matchesOrigin(origin Origin) bool {
	if e.Worker != nil {
		return origin.RunID == e.RunID && origin.StageExecutionID == e.StageID && origin.LogicalAgentName == e.Worker.LogicalAgentName
	}
	return origin.Operation == contracts.PublishFindingsOperation && origin.Scan != nil && e.Source != nil && sameScanExecution(origin.Scan, e.Source) && origin.Scan.Artifact.SameExact(e.Source.Artifact)
}
func (s *Service) readExecutionReceipt(ctx context.Context, db querier, e intakeExecution, input Submission) (Receipt, error) {
	if e.Worker != nil {
		return readReceiptBySubmission(ctx, db, e.Worker.AllocationID, input.InvocationID, input.SubmissionID)
	}
	return scanReceipt(ctx, db, db.QueryRow(ctx, readExecutionReceiptSQL, e.receiptID(input)))
}
func (s *Service) findExecutionReplay(ctx context.Context, e intakeExecution, input Submission) (Receipt, bool, error) {
	if e.Worker != nil {
		return s.FindReplay(ctx, *e.Worker, input)
	}
	previous, err := s.readExecutionReceipt(ctx, s.pool, e, input)
	if errors.Is(err, pgx.ErrNoRows) {
		return Receipt{}, false, nil
	}
	if err != nil {
		return Receipt{}, false, err
	}
	if previous.RequestDigest != e.RequestDigest || !e.matchesOrigin(previous.Origin) {
		return Receipt{}, false, ErrConflict
	}
	return previous, true, nil
}
func (s *Service) verifyIntakeExecution(ctx context.Context, tx pgx.Tx, e intakeExecution, itemKey *string, incidental bool) (Origin, string, *string, error) {
	if e.Worker != nil {
		return verifyTrustedExecution(ctx, tx, *e.Worker, itemKey, incidental, s.storage)
	}
	execution, err := runstore.NewPostgresTxStore(tx, s.storage).LockStageOperation(ctx, e.RunID, e.StageID, e.ClaimID)
	if err != nil {
		return Origin{}, "", nil, err
	}
	if !pinnedCandidateSelection(execution, e) {
		return Origin{}, "", nil, ErrAccessDenied
	}
	run, err := runstore.NewPostgresStore(tx, s.storage).GetRun(ctx, e.RunID)
	if err != nil {
		return Origin{}, "", nil, err
	}
	origin := Origin{RunID: e.RunID, StageExecutionID: e.StageID, Operation: contracts.PublishFindingsOperation, Scan: e.Source,
		Workflow: WorkflowOrigin{Name: run.WorkflowName, Version: run.WorkflowVersion, SchemaVersion: run.WorkflowSchemaVersion, ClosureDigest: auditdomain.DigestBytes(run.WorkflowSnapshot)}}
	origin.Workflow.ConfigurationRef.Name = run.WorkflowName
	origin.Workflow.ConfigurationRef.Version = run.WorkflowVersion
	if run.AuditExecutionID != nil {
		key := ""
		if itemKey != nil {
			key = *itemKey
		}
		if incidental && itemKey == nil {
			auditID, err := querystore.New(tx, s.storage).ExecutionAuditOrigin(ctx, *run.AuditExecutionID, run.RunID)
			if err != nil {
				return Origin{}, "", nil, err
			}
			origin.Audit = &AuditOrigin{AuditID: auditID, ExecutionID: *run.AuditExecutionID}
		} else {
			task, err := querystore.New(tx, s.storage).ExecutionTaskOrigin(ctx, *run.AuditExecutionID, run.RunID, key)
			if err != nil {
				return Origin{}, "", nil, err
			}
			origin.Audit = &AuditOrigin{AuditID: task.AuditID, ExecutionID: *run.AuditExecutionID, TaskID: task.TaskID, Hypothesis: task.Hypothesis}
		}
	} else if itemKey != nil && (e.Source == nil || e.Source.RunID == e.RunID) {
		return Origin{}, "", nil, ErrInvalid
	}
	return origin, run.OwnerID, run.ProjectID, nil
}
func samePinnedInput(stage runstore.StageExecution, name string, ref contracts.ArtifactRef) bool {
	pinned, ok := stage.StageContext.Artifacts[name]
	return ok && pinned.Artifact != nil && pinned.Artifact.SameExact(ref)
}

// A selected set is fully validated once before mutation; immutable exact refs
// are then carried internally. The claim and selected set are fenced for each item.
func pinnedCandidateSelection(stage runstore.StageExecution, e intakeExecution) bool {
	if e.Selection != nil {
		return samePinnedInput(stage, "materials", *e.Selection)
	}
	return samePinnedInput(stage, "candidates", e.Candidate) && samePinnedInput(stage, "evidence", e.Evidence)
}
