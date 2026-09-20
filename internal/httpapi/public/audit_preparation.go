package public

import (
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
)

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
