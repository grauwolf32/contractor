package auditimport

import (
	"context"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditscan"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	plannersession "github.com/grauwolf32/contractor/internal/planner/session"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func isScanCheck(members []preparedMember) bool {
	return len(members) == 1 && members[0].task.Scan != nil
}

func scanHistoryNeedsRecovery(history []planner.ScanAttempt) bool {
	for _, attempt := range history {
		if planner.ScanAttemptNeedsRecovery(attempt) {
			return true
		}
	}
	return false
}

func (i *Importer) scanHistory(ctx context.Context, execution auditstore.Execution) (runstore.WorkflowRun, []planner.ScanAttempt, error) {
	store, ok := i.runs.(plannersession.ScanHistoryStore)
	if !ok || execution.RunID == nil {
		return runstore.WorkflowRun{}, nil, fmt.Errorf("Audit scan history is unavailable")
	}
	run, err := store.GetRun(ctx, *execution.RunID)
	if err != nil {
		return runstore.WorkflowRun{}, nil, err
	}
	if run.AuditExecutionID == nil || *run.AuditExecutionID != execution.ExecutionID {
		return runstore.WorkflowRun{}, nil, fmt.Errorf("Audit scan execution identity differs")
	}
	history, err := plannersession.ReadAuditScanHistory(ctx, store, run.RunID)
	if err != nil {
		return runstore.WorkflowRun{}, nil, err
	}
	for _, attempt := range history {
		if !attempt.Terminal {
			return runstore.WorkflowRun{}, nil, fmt.Errorf("Audit scan attempt is not terminal")
		}
	}
	return run, history, nil
}

// collectScanRecovery settles the Run's technical failure separately from the
// scanner's outcome. It retains canonical evidence even when final publication
// failed. Only known failures or attempts before invocation can create a new Run.
func (i *Importer) collectScanRecovery(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	snapshot auditstore.ReconcileSnapshot,
	execution auditstore.Execution,
	members []preparedMember,
	code string,
) (bool, error) {
	if !isScanCheck(members) {
		return false, fmt.Errorf("scan recovery requires one scan item")
	}
	run, history, err := i.scanHistory(ctx, execution)
	if err != nil {
		return false, err
	}
	if run.ProjectID == nil || *run.ProjectID != snapshot.Audit.ProjectID {
		return false, fmt.Errorf("Audit scan project identity differs")
	}
	manifestBytes, err := i.artifacts.ReadProjectExact(ctx, snapshot.Audit.ProjectID, execution.Manifest)
	if err != nil {
		return false, err
	}
	manifest, err := auditdomain.DecodeExecutionManifest(manifestBytes)
	if err != nil {
		return false, err
	}
	input := auditscan.ResultInput{RunID: run.RunID, Task: members[0].task, Manifest: manifest, Attempts: history}
	assembled, err := auditscan.BuildResult(ctx, input, i.scanRecoveryReader(run.RunID))
	if err != nil {
		return false, err
	}
	remainingBytes := snapshot.Audit.Limits.MaxEvidenceBytes - snapshot.Audit.RetainedEvidenceBytes
	if int64(len(assembled.Package)) > remainingBytes {
		return i.collectTechnical(ctx, claim, execution, members, auditstore.CollectionExecutionFailed,
			false, "evidence-budget-exhausted", auditstore.CoverageInconclusive)
	}
	return i.retainScanRecovery(ctx, claim, snapshot, execution, members[0], history, assembled.Package, code)
}

func (i *Importer) scanRecoveryReader(runID string) auditscan.ArtifactReader {
	return func(ctx context.Context, ref contracts.ArtifactRef, limit int) (artifacts.Payload, error) {
		metadata, data, err := i.artifacts.ReadRunExact(ctx, runID, ref)
		if err != nil {
			return artifacts.Payload{}, err
		}
		if len(data) > limit {
			return artifacts.Payload{}, fmt.Errorf("retained scan artifact exceeds bound")
		}
		return artifacts.Payload{MediaType: metadata.MediaType, Data: data}, nil
	}
}
