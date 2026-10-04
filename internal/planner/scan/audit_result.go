package scan

import (
	"context"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditscan"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
)

func (p *execution) finishAudit(ctx context.Context, identity planner.ScanSessionIdentity, history []planner.ScanAttempt) (contracts.StageContentResult, error) {
	// Building the candidate has its own storage budget; recording it gets the
	// shared completion budget and transient retry.
	buildCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), journalIOTimeout)
	defer cancel()
	input := auditscan.ResultInput{RunID: p.invocation.RunID, CurrentStage: p.invocation.StageExecutionID,
		Task: p.audit.task, Manifest: p.audit.manifest, Attempts: history}
	assembled, err := auditscan.BuildResult(buildCtx, input,
		func(ctx context.Context, ref contracts.ArtifactRef, limit int) (artifacts.Payload, error) {
			return p.factory.artifacts.Read(ctx, p.invocation.RunID, ref, limit)
		})
	if err != nil {
		return contracts.StageContentResult{}, scanError("audit_scan_result_invalid", err)
	}
	slot := p.invocation.Stage.Result.Artifacts["report"].From
	target := contracts.ArtifactRef{Namespace: slot.Namespace, Name: slot.Name + "." + stableHash(p.invocation.StageExecutionID)}
	ref, err := p.factory.artifacts.Create(buildCtx, p.invocation.RunID, target, artifacts.Payload{MediaType: "application/zip", Data: assembled.Package})
	if err != nil {
		return contracts.StageContentResult{}, scanError("audit_scan_result_write_failed", err)
	}
	result := contracts.StageContentResult{APIVersion: contracts.APIVersion, Outcome: contracts.StageSucceeded, Summary: assembled.Summary, Artifacts: map[string]contracts.ArtifactRef{"report": ref}}
	if assembled.Retryable {
		result.Outcome = contracts.StageFailed
		result.Error = &contracts.TerminationError{Code: "scan_failed", Message: "The scanner failed with a known outcome", Retryable: true}
	}
	if err := planner.ValidateCandidate(buildCtx, p.invocation.RunID, p.invocation.Stage.Result.Artifacts, result, p.factory.inspector); err != nil {
		return contracts.StageContentResult{}, err
	}
	if err := planner.CompleteScanSession(ctx, p.factory.sessions, identity, planner.Completion{Result: &result}); err != nil {
		return contracts.StageContentResult{}, scanError("scan_completion_write_failed", err)
	}
	return result, nil
}
