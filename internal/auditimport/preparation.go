package auditimport

import (
	"context"
	"encoding/json"
	"errors"
	"maps"
	"slices"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// Failed/partial preparation outputs remain attempt evidence. They never enter
// preparation_outputs, so neither dependencies nor inventory can consume them.
func (i *Importer) commitRoleFailure(ctx context.Context, claim auditstore.ControllerClaim, snapshot auditstore.ReconcileSnapshot, execution auditstore.Execution, disposition auditstore.CollectionDisposition, source *auditstore.ExactArtifact, code string) (bool, error) {
	if execution.Role != auditstore.ExecutionPrepare || execution.RunID == nil || code == "evidence-budget-exhausted" {
		return i.commitCollection(ctx, claim, execution, disposition, source, nil, &code, nil)
	}
	profile, err := config.DecodeResolvedAuditProfileSnapshot(snapshot.Audit.ProfileSnapshot)
	if err != nil {
		return false, err
	}
	binding := profile.Workflows[execution.WorkflowRole]
	remaining := snapshot.Audit.Limits.MaxEvidenceBytes - snapshot.Audit.RetainedEvidenceBytes
	links := []auditstore.ArtifactLink{}
	evidenceSource := source
	for _, name := range slices.Sorted(maps.Keys(binding.Outputs)) {
		slot := binding.Outputs[name]
		exact, _, frozen, err := i.artifacts.ReadRunBinding(ctx, *execution.RunID, contracts.ArtifactRef{Namespace: "outputs", Name: slot})
		if errors.Is(err, artifacts.ErrArtifactNotFound) || errors.Is(err, artifacts.ErrArtifactIntegrity) {
			continue
		}
		if err != nil {
			return false, err
		}
		if !frozen {
			continue
		}
		if evidenceSource == nil {
			value := exact
			evidenceSource = &value
		}
		if exact.SizeBytes > remaining {
			code = "evidence-budget-exhausted"
			return i.commitCollection(ctx, claim, execution, auditstore.CollectionInvalidResult, evidenceSource, nil, &code, nil)
		}
		remaining -= exact.SizeBytes
		retained, err := i.artifacts.RetainRunExact(ctx, *execution.RunID, exact, snapshot.Audit.ProjectID,
			contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(snapshot.Audit.AuditID), Name: auditdomain.DeterministicID("prepare-attempt-output", execution.ExecutionID, name)})
		if err != nil {
			return false, err
		}
		provenance, _ := json.Marshal(struct {
			Schema         string                   `json:"schema"`
			ExecutionID    string                   `json:"executionId"`
			RunID          string                   `json:"runId"`
			LogicalName    string                   `json:"logicalName"`
			WorkflowOutput string                   `json:"workflowOutput"`
			ErrorCode      string                   `json:"errorCode"`
			Source         auditstore.ExactArtifact `json:"source"`
		}{"contractor.audit.preparation-attempt.v1", execution.ExecutionID, *execution.RunID, name, slot, code, exact})
		links = append(links, auditstore.ArtifactLink{LogicalKey: "prepare-attempt:" + execution.ExecutionID + ":" + name, Artifact: retained, SourceProvenance: provenance})
	}
	changed, err := i.commitCollection(ctx, claim, execution, disposition, source, links, &code, nil)
	if errors.Is(err, auditstore.ErrEvidenceBudgetExhausted) {
		code = "evidence-budget-exhausted"
		return i.commitCollection(ctx, claim, execution, auditstore.CollectionInvalidResult, evidenceSource, nil, &code, nil)
	}
	return changed, err
}
