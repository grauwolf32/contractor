package auditimport

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// Failed/partial preparation outputs remain attempt evidence. They never enter
// preparation_outputs, so neither dependencies nor inventory can consume them.
func (i *Importer) commitRoleFailure(ctx context.Context, claim auditstore.ControllerClaim, snapshot auditstore.ReconcileSnapshot, execution auditstore.Execution, disposition auditstore.CollectionDisposition, source *auditstore.ExactArtifact, code string) (bool, error) {
	if execution.Role != auditstore.ExecutionPrepare || execution.RunID == nil || code == "evidence-budget-exhausted" {
		return i.commitCollection(ctx, claim, execution, disposition, source, nil, &code, nil)
	}
	profile, err := auditprofile.DecodeResolvedAuditProfileSnapshot(snapshot.Audit.ProfileSnapshot)
	if err != nil {
		return false, fmt.Errorf("%w: pinned AuditProfile is invalid", ErrPermanent)
	}
	binding := profile.Workflows[execution.WorkflowRole]
	remaining := snapshot.Audit.Limits.MaxEvidenceBytes - snapshot.Audit.RetainedEvidenceBytes
	type attemptOutput struct {
		name, slot string
		exact      auditstore.ExactArtifact
	}
	outputs := []attemptOutput{}
	evidenceSource := source
	for _, name := range slices.Sorted(maps.Keys(binding.Outputs)) {
		slot := binding.Outputs[name]
		exact, _, frozen, err := i.artifacts.ReadRunBinding(ctx, *execution.RunID, contracts.ArtifactRef{Namespace: "outputs", Name: slot})
		if errors.Is(err, artifacts.ErrArtifactNotFound) || errors.Is(err, blob.ErrIntegrity) {
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
		outputs = append(outputs, attemptOutput{name: name, slot: slot, exact: exact})
	}
	// Every output fits the budget before any is retained; the copies commit
	// in bounded artifact-write transactions (AUX-R98).
	retentions := make([]RunRetention, len(outputs))
	for index, output := range outputs {
		retentions[index] = RunRetention{
			Source: output.exact,
			Target: contracts.ArtifactRef{
				Namespace: auditdomain.ArtifactNamespace(snapshot.Audit.AuditID),
				Name:      auditdomain.DeterministicID("prepare-attempt-output", execution.ExecutionID, output.name),
			},
		}
	}
	retained, err := i.artifacts.RetainRunExact(ctx, *execution.RunID, snapshot.Audit.ProjectID, retentions)
	if err != nil {
		return false, err
	}
	if len(retained) != len(retentions) {
		return false, fmt.Errorf("retain Audit preparation attempt outputs: %d of %d revisions", len(retained), len(retentions))
	}
	links := make([]auditstore.ArtifactLink, 0, len(outputs))
	for index, output := range outputs {
		provenance, _ := json.Marshal(struct {
			Schema         string                   `json:"schema"`
			ExecutionID    string                   `json:"executionId"`
			RunID          string                   `json:"runId"`
			LogicalName    string                   `json:"logicalName"`
			WorkflowOutput string                   `json:"workflowOutput"`
			ErrorCode      string                   `json:"errorCode"`
			Source         auditstore.ExactArtifact `json:"source"`
		}{"contractor.audit.preparation-attempt.v1", execution.ExecutionID, *execution.RunID, output.name, output.slot, code, output.exact})
		links = append(links, auditstore.ArtifactLink{Scope: "project", LogicalKey: "prepare-attempt:" + execution.ExecutionID + ":" + output.name, Artifact: retained[index], SourceProvenance: provenance})
	}
	changed, err := i.commitCollection(ctx, claim, execution, disposition, source, links, &code, nil)
	if errors.Is(err, auditstore.ErrEvidenceBudgetExhausted) {
		code = "evidence-budget-exhausted"
		return i.commitCollection(ctx, claim, execution, auditstore.CollectionInvalidResult, evidenceSource, nil, &code, nil)
	}
	return changed, err
}
