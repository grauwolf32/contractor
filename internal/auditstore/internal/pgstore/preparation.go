package pgstore

import (
	"encoding/json"
	"errors"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
)

func decodePreparation(execution *auditstore.Execution, snapshot, outputs []byte) error {
	if snapshot != nil {
		var value auditstore.PreparationSnapshot
		if decodeStoredProvenance(snapshot, &value) != nil || validatePreparationSnapshot(&value) != nil || execution.Role != auditstore.ExecutionPrepare {
			return errors.New("stored Audit preparation snapshot is invalid")
		}
		execution.Preparation = &value
	} else if execution.Role == auditstore.ExecutionPrepare {
		return errors.New("stored Audit preparation snapshot is missing")
	}
	if decodeStoredProvenance(outputs, &execution.PreparationOutputs) != nil {
		return errors.New("stored Audit preparation outputs are invalid")
	}
	return nil
}

func validatePreparationSnapshot(snapshot *auditstore.PreparationSnapshot) error {
	if snapshot == nil || snapshot.Inputs == nil || snapshot.Parameters == nil ||
		len(snapshot.Inputs) > auditprofile.MaxAuditWorkflowMappings || len(snapshot.Parameters) > auditprofile.MaxAuditWorkflowMappings {
		return invalidf("preparation inputs and parameters are required and bounded")
	}
	for name, artifact := range snapshot.Inputs {
		if validateText("input name", name, 128, true) != nil || validateExactArtifact("preparation input", artifact, true) != nil {
			return invalidf("preparation input is invalid")
		}
	}
	for name, value := range snapshot.Parameters {
		if validateText("parameter name", name, 128, true) != nil || validateText("parameter value", value, auditprofile.MaxAuditLiteralParamBytes, false) != nil {
			return invalidf("preparation parameter is invalid")
		}
	}
	encoded, err := json.Marshal(snapshot)
	if err != nil || len(encoded) > auditstore.MaxExecutionInputsBytes {
		return invalidf("preparation snapshot exceeds bounds")
	}
	return nil
}

// Outputs become ordinary immutable Audit links in the same receipt write.
// The DB additionally proves their frozen Project binding and import lineage.
func withPreparationLinks(p auditstore.CollectParams) (auditstore.CollectParams, error) {
	if len(p.PreparationOutputs) > auditprofile.MaxAuditWorkflowMappings ||
		(len(p.PreparationOutputs) != 0 && (p.Disposition != auditstore.CollectionAccepted || len(p.Items) != 0)) {
		return p, invalidf("preparation output collection shape is invalid")
	}
	seen := map[string]bool{}
	p.Retained = append([]auditstore.ArtifactLink{}, p.Retained...)
	for _, output := range p.PreparationOutputs {
		if validateText("logical output name", output.LogicalName, 128, true) != nil ||
			validateText("Workflow output name", output.WorkflowOutput, 128, true) != nil || seen[output.LogicalName] ||
			validateExactArtifact("preparation source", output.Source, true) != nil ||
			validateExactArtifact("preparation retained output", output.Retained, true) != nil ||
			output.Source.Ref.Namespace != "outputs" || output.Source.Ref.Name != output.WorkflowOutput ||
			output.Retained.Ref.Namespace != auditdomain.ArtifactNamespace(p.Claim.AuditID) ||
			output.Source.Digest != output.Retained.Digest || output.Source.MediaType != output.Retained.MediaType || output.Source.SizeBytes != output.Retained.SizeBytes {
			return p, invalidf("preparation output identity or exact retention is invalid")
		}
		seen[output.LogicalName] = true
		provenance, _ := json.Marshal(struct {
			Schema      string                       `json:"schema"`
			ExecutionID string                       `json:"executionId"`
			Output      auditstore.PreparationOutput `json:"output"`
		}{"contractor.audit.preparation-output.v1", p.ExecutionID, output})
		p.Retained = append(p.Retained, auditstore.ArtifactLink{Scope: "audit", LogicalKey: "prepare:" + p.ExecutionID + ":" + output.LogicalName,
			Artifact: output.Retained, SourceProvenance: provenance,
		})
	}
	return p, nil
}
