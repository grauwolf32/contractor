package auditscan

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/scanplan"
)

func (b *resultBuilder) addEvidence(id, kind string, data []byte) bool {
	// Reserve the domain's maximum manifest size for the package index and result
	// metadata. Check the final ZIP separately because compression adds overhead.
	payloadBudget := MaxResultBytes - auditdomain.MaximumManifestBytes
	if b.evidenceBytes+len(data) > payloadBudget || len(b.evidence) >= auditdomain.MaximumEvidencePerItem {
		b.gaps["scan_evidence_budget_exceeded"] = true
		return false
	}
	b.evidenceBytes += len(data)
	b.members = append(b.members, auditdomain.PackageInput{
		ID: id, Path: "evidence/" + id + ".json", MediaType: auditdomain.JSONMediaType, Data: data,
	})
	b.evidence = append(b.evidence, auditdomain.Evidence{
		ID: id, Kind: kind, Summary: "Retained " + kind + " from the assigned scan.", ContentMemberID: id,
	})
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
	payload, err := b.read(ctx, *state.Plan, scanplan.MaxPlanBytes)
	if errors.Is(err, artifacts.ErrArtifactNotFound) {
		b.gaps["scan_plan_unavailable"] = true
		return nil
	}
	if err != nil {
		return err
	}
	plan, err := scanplan.DecodePlan(payload.Data)
	if err != nil || payload.MediaType != scanplan.PlanMediaType || "sha256:"+hash(payload.Data) != state.PlanDigest {
		return fmt.Errorf("retained scan plan is invalid")
	}
	for _, candidate := range plan.Candidates {
		if candidate.Code != "" {
			b.gaps[candidate.Code] = true
		}
	}
	b.addEvidence(prefix+"-plan", "scan-plan", payload.Data)
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
	payload, err := b.read(ctx, *job.Report, MaxScannerReportBytes)
	if errors.Is(err, artifacts.ErrArtifactNotFound) {
		b.gaps["scan_report_unavailable"] = true
		return nil
	}
	if err != nil {
		return err
	}
	var header struct {
		Tool string `json:"tool"`
	}
	if payload.MediaType != auditdomain.JSONMediaType || json.Unmarshal(payload.Data, &header) != nil || header.Tool != "scan_"+b.input.Task.Scan.Scanner {
		return fmt.Errorf("retained scanner report is invalid")
	}
	if b.addEvidence(id, "scanner-report", payload.Data) && job.Status == planner.ScanJobCompleted {
		b.completed = true
	}
	return nil
}
