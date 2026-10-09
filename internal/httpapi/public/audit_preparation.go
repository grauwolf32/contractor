package public

import (
	"net/http"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func (h *handler) auditDetailReadModel(r *http.Request, audit auditstore.Audit) (auditResponse, error) {
	response, err := auditReadModel(audit)
	if err != nil {
		return auditResponse{}, err
	}
	if response.Phase == auditdomain.AuditPhaseNotStarted {
		return response, nil
	}
	profile, err := config.DecodeResolvedAuditProfileSnapshot(audit.ProfileSnapshot)
	if err != nil || !profile.HasPreparation() {
		return response, nil
	}
	roles, err := h.dependencies.Audits.Preparation(r.Context(), principalUserID(r.Context()), audit.AuditID)
	if err != nil {
		return auditResponse{}, err
	}
	response.Preparation = &auditPreparationResponse{Roles: map[string]auditPreparationRoleResponse{}}
	for name, role := range roles {
		view := auditPreparationRoleResponse{Status: role.Status, Attempts: role.Attempts, MaxAttempts: role.MaxRunAttempts,
			ExecutionID: role.ExecutionID, RunID: role.RunID, Outputs: map[string]auditPreparationOutputResponse{}}
		for name, output := range role.Outputs {
			view.Outputs[name] = auditPreparationOutputResponse{
				ExecutionID: output.ExecutionID, RunID: output.RunID, WorkflowOutput: output.Output.WorkflowOutput,
				Artifact: auditPreparationArtifactResponse{Ref: output.Output.Retained.Ref, Digest: output.Output.Retained.Digest,
					MediaType: output.Output.Retained.MediaType, SizeBytes: output.Output.Retained.SizeBytes},
			}
		}
		response.Preparation.Roles[name] = view
	}
	return response, nil
}

// Prepared artifacts retain their producing execution even after source Run
// deletion. Artifact is the retained Project revision, never a current binding.
type auditPreparationOutputResponse struct {
	Artifact       auditPreparationArtifactResponse `json:"artifact"`
	ExecutionID    string                           `json:"executionId"`
	RunID          string                           `json:"runId"`
	WorkflowOutput string                           `json:"workflowOutput"`
}

type auditPreparationRoleResponse struct {
	Status      auditdomain.PreparationStatus             `json:"status"`
	Attempts    int                                       `json:"attempts"`
	MaxAttempts int                                       `json:"maxAttempts"`
	ExecutionID *string                                   `json:"executionId,omitempty"`
	RunID       *string                                   `json:"runId,omitempty"`
	Outputs     map[string]auditPreparationOutputResponse `json:"outputs"`
}

type auditPreparationResponse struct {
	Roles map[string]auditPreparationRoleResponse `json:"roles"`
}

// Metadata is mandatory for retained preparation output; a zero-byte artifact
// still has an explicit size rather than an absent field.
type auditPreparationArtifactResponse struct {
	Ref       contracts.ArtifactRef `json:"ref"`
	Digest    string                `json:"digest"`
	MediaType string                `json:"mediaType"`
	SizeBytes int64                 `json:"sizeBytes"`
}
