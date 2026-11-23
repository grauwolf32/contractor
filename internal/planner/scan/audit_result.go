package scan

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditscan"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/stageresult"
)

func (p *execution) finishAudit(ctx context.Context, identity planner.ScanSessionIdentity, history []planner.ScanAttempt) (stageresult.StageContentResult, error) {
	// Building the candidate has its own storage budget; recording it gets the
	// shared completion budget and transient retry.
	buildCtx, cancel := context.WithTimeout(ctx, journalIOTimeout)
	defer cancel()
	input := auditscan.ResultInput{RunID: p.invocation.RunID, CurrentStage: p.invocation.StageExecutionID,
		Scanner: p.audit.task.Scanner, PreparationDigest: p.audit.task.PreparationDigest, Runnable: p.audit.task.Runnable, Gaps: p.audit.task.Gaps, Attempts: history}
	assembled, err := auditscan.BuildResult(buildCtx, input,
		func(ctx context.Context, ref contracts.ArtifactRef, limit int) (artifacts.Payload, error) {
			return p.factory.artifacts.Read(ctx, p.invocation.RunID, ref, limit)
		})
	if err != nil {
		if errors.Is(err, auditscan.ErrInvalid) {
			return stageresult.StageContentResult{}, p.scanError("audit_scan_result_invalid", err)
		}
		return stageresult.StageContentResult{}, p.scanError("audit_scan_result_unavailable", err)
	}
	slot := p.invocation.Stage.Result.Artifacts["report"].From
	target := contracts.ArtifactRef{Namespace: slot.Namespace, Name: slot.Name + "." + stableHash(p.invocation.StageExecutionID)}
	ref, err := p.factory.artifacts.Create(buildCtx, p.invocation.RunID, target, artifacts.Payload{MediaType: "application/json", Data: assembled.Data})
	if err != nil {
		return stageresult.StageContentResult{}, p.scanError("audit_scan_result_write_failed", err)
	}
	result := stageresult.StageContentResult{APIVersion: contracts.APIVersion, Outcome: stageresult.StageSucceeded, Summary: assembled.Summary, Artifacts: map[string]contracts.ArtifactRef{"report": ref}}
	if !assembled.ExecutionCompleted {
		result.Outcome = stageresult.StageFailed
		result.Error = &contracts.TerminationError{Code: "scan_failed", Message: "The scanner did not complete all selected work", Retryable: assembled.Retryable}
	}
	jobs := []planner.ScanJobRecord{}
	for _, attempt := range history {
		jobs = append(jobs, attempt.State.Jobs...)
	}
	if err := p.selectNormalizedMaterials(buildCtx, jobs, &result); err != nil {
		return stageresult.StageContentResult{}, err
	}
	if err := planner.ValidateCandidate(buildCtx, p.invocation.RunID, p.invocation.Stage.Result.Artifacts, result, p.factory.inspector); err != nil {
		return stageresult.StageContentResult{}, err
	}
	if err := planner.CompleteScanSession(ctx, p.factory.sessions, identity, planner.Completion{Result: &result}); err != nil {
		return stageresult.StageContentResult{}, p.scanError("scan_completion_write_failed", err)
	}
	return result, nil
}
