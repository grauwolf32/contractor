package auditscan

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/helpers/strictjson"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/scanplan"
)

// readRetained reads one exact retained scan artifact. Its content never
// changes, so a payload over the artifact's bound is invalid, not retryable.
func (b *resultBuilder) readRetained(ctx context.Context, ref contracts.ArtifactRef, limit int) (artifacts.Payload, error) {
	payload, err := b.read(ctx, ref, limit)
	if err != nil {
		return artifacts.Payload{}, err
	}
	if len(payload.Data) > limit {
		return artifacts.Payload{}, fmt.Errorf("%w: retained scan artifact exceeds its bound", ErrInvalid)
	}
	return payload, nil
}

func (b *resultBuilder) addEvidence(id, kind string, ref contracts.ArtifactRef, data []byte) bool {
	// Reserve 1 MiB for the bounded Run journal and exact descriptors. A final
	// canonical-byte check also covers JSON escaping and metadata overhead.
	if len(b.artifacts) >= 256 {
		b.gaps["scan_evidence_budget_exceeded"] = true
		return false
	}
	b.artifacts = append(b.artifacts, retainedArtifact{ID: id, Kind: kind, Ref: ref.Clone(), Digest: contentdigest.Bytes(data), SizeBytes: len(data)})
	return true
}

func (b *resultBuilder) addAttempt(ctx context.Context, attempt planner.ScanAttempt) error {
	b.needsRecovery = b.needsRecovery || planner.ScanAttemptNeedsRecovery(attempt)
	prefix := "attempt-" + hash([]byte(attempt.StageExecutionID))
	if err := b.addPlan(ctx, prefix, attempt.State); err != nil {
		return err
	}
	for index, job := range attempt.State.Jobs {
		b.recordOutcome(attempt.StageExecutionID, job)
		id := fmt.Sprintf("%s-report-%d", prefix, index)
		if err := b.addReport(ctx, id, job); err != nil {
			return err
		}
	}
	return nil
}

func (b *resultBuilder) addPlan(ctx context.Context, prefix string, state planner.ScanState) error {
	if state.Plan == nil {
		return nil
	}
	payload, err := b.readRetained(ctx, *state.Plan, scanplan.MaxPlanBytes)
	if errors.Is(err, artifacts.ErrArtifactNotFound) {
		b.gaps["scan_plan_unavailable"] = true
		return nil
	}
	if err != nil {
		return err
	}
	plan, err := scanplan.DecodePlan(payload.Data)
	if err != nil || payload.MediaType != scanplan.PlanMediaType || contentdigest.Bytes(payload.Data) != state.PlanDigest {
		return fmt.Errorf("%w: retained scan plan is invalid", ErrInvalid)
	}
	for _, candidate := range plan.Candidates {
		if candidate.Code != "" {
			b.gaps[candidate.Code] = true
		}
	}
	b.addEvidence(prefix+"-plan", "scan-plan", *state.Plan, payload.Data)
	return nil
}

func (b *resultBuilder) recordOutcome(stageID string, job planner.ScanJobRecord) {
	b.observed = true
	if job.Code != "" {
		b.gaps[job.Code] = true
	}
	if job.Status == planner.ScanJobStarted || job.Status == planner.ScanJobUnknown {
		b.gaps["scan_outcome_unknown"] = true
	}
	if stageID != b.input.CurrentStage {
		return
	}
	switch job.Status {
	case planner.ScanJobFailed, planner.ScanJobUnavailable:
		b.retryable = true
	case planner.ScanJobIncomplete:
		b.retryable = job.Code == "scan_deadline_exceeded"
	}
}

func (b *resultBuilder) addReport(ctx context.Context, id string, job planner.ScanJobRecord) error {
	if job.Report == nil {
		if job.Status == planner.ScanJobCompleted {
			b.gaps["scan_report_unavailable"] = true
		}
		return nil
	}
	payload, err := b.readRetained(ctx, *job.Report, MaxScannerReportBytes)
	if errors.Is(err, artifacts.ErrArtifactNotFound) {
		b.gaps["scan_report_unavailable"] = true
		return nil
	}
	if err != nil {
		return err
	}
	var header map[string]any
	if payload.MediaType != "application/json" || strictjson.Decode(payload.Data, &header) != nil || header["tool"] != "scan_"+b.input.Scanner {
		return fmt.Errorf("%w: retained scanner report is invalid", ErrInvalid)
	}
	if b.addEvidence(id, "scanner-report", *job.Report, payload.Data) && job.Status == planner.ScanJobCompleted {
		b.completed = true
	}
	return nil
}
