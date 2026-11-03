package auditimport

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditscan"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/planner"
	plannersession "github.com/grauwolf32/contractor/internal/planner/session"
)

// Existing durable scanner journals distinguish a known failure from an action
// that completed or may have executed. Recover missing ordinary report output
// from those exact journals; never re-execute traffic to repair publication.
func (i *Importer) recoverTaskScan(ctx context.Context, snapshot auditstore.ReconcileSnapshot, execution auditstore.Execution, pinned auditprofile.PinnedProgram) (*taskOutput, bool, error) {
	workflow := pinned.Workflows[execution.WorkflowRole]
	var stage *resolved.ResolvedStage
	for _, candidate := range workflow.Stages {
		if candidate.ScanPreparation != nil {
			copy := candidate
			stage = &copy
		}
	}
	if stage == nil {
		return nil, false, nil
	}
	store, ok := i.runs.(plannersession.ScanHistoryStore)
	if !ok {
		return nil, false, fmt.Errorf("scan history reader is unavailable")
	}
	history, err := plannersession.ReadAuditScanHistory(ctx, store, *execution.RunID)
	if err != nil {
		return nil, false, err
	}
	nonrepeatable := false
	for _, attempt := range history {
		if !attempt.Terminal {
			return nil, false, fmt.Errorf("terminal Run retains an active scan attempt")
		}
		nonrepeatable = nonrepeatable || planner.ScanAttemptNeedsRecovery(attempt)
	}
	binding := pinned.Program.Workflows[execution.WorkflowRole]
	logical, slot := "", ""
	for key, output := range binding.Outputs {
		if stage.WorkflowOutputs[output] == "report" {
			if logical != "" {
				return nil, false, fmt.Errorf("%w: duplicate scan report binding", ErrPermanent)
			}
			logical, slot = key, output
		}
	}
	if logical == "" {
		return nil, nonrepeatable, nil
	}
	source, frozen, err := i.artifacts.DescribeRunBinding(ctx, *execution.RunID, contracts.ArtifactRef{Namespace: "outputs", Name: slot})
	if err == nil && frozen && source.MediaType == "application/json" {
		return nil, nonrepeatable, nil
	}
	if err != nil && !errors.Is(err, artifacts.ErrArtifactNotFound) {
		return nil, false, err
	}
	assembled, err := auditscan.BuildResult(ctx, auditscan.ResultInput{RunID: *execution.RunID, Scanner: stage.ScanPreparation.Scanner, Runnable: true, Gaps: []string{"scan_report_publication_recovered"}, Attempts: history}, func(ctx context.Context, ref contracts.ArtifactRef, limit int) (artifacts.Payload, error) {
		exact, data, err := i.artifacts.ReadRunExact(ctx, *execution.RunID, ref)
		if err != nil {
			return artifacts.Payload{}, err
		}
		if len(data) > limit {
			return artifacts.Payload{}, fmt.Errorf("%w: retained scan report bound", auditscan.ErrInvalid)
		}
		return artifacts.Payload{MediaType: exact.MediaType, Data: data}, nil
	})
	if errors.Is(err, auditscan.ErrInvalid) {
		return nil, nonrepeatable, fmt.Errorf("%w: invalid frozen scan history", ErrPermanent)
	}
	if err != nil {
		return nil, nonrepeatable, err
	}
	if int64(len(assembled.Data)) > snapshot.Audit.Limits.MaxEvidenceBytes-snapshot.Audit.RetainedEvidenceBytes {
		return nil, nonrepeatable, auditstore.ErrEvidenceBudgetExhausted
	}
	digest := contentdigest.Bytes(assembled.Data)
	retained, err := i.artifacts.PutImmutableAudit(ctx, snapshot.Audit.ProjectID, contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(execution.AuditID), Name: "scan-recovery-" + strings.TrimPrefix(digest, "sha256:")}, artifacts.Payload{MediaType: "application/json", Data: assembled.Data})
	if err != nil {
		return nil, nonrepeatable, err
	}
	return &taskOutput{logical: logical, slot: slot, retained: retained}, nonrepeatable, nil
}
