package public

import (
	"net/http"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// readAuditDetail answers the detail of audit, the state of an owner's Audit
// that a read or a committed mutation returned (ASC-R149). The service reads
// an Audit whose detail hydrates preparation again in one snapshot, so the
// response, its revision and its ETag come from one commit and a write that
// commits meanwhile never turns the answer into a conflict.
func (h *auditHandlers) readAuditDetail(r *http.Request, audit auditstore.Audit) (auditResponse, error) {
	detail, err := h.Audits.Detail(r.Context(), principalUserID(r.Context()), audit, auditDetailHydratesPreparation)
	if err != nil {
		return auditResponse{}, err
	}
	return auditDetailReadModel(detail)
}

// auditDetailReadModel is the public projection of one Audit's detail.
func auditDetailReadModel(detail auditservice.AuditDetail) (auditResponse, error) {
	response, err := auditReadModel(detail.Audit)
	if err != nil {
		return auditResponse{}, err
	}
	if detail.Preparation != nil {
		response.Preparation = auditPreparationReadModel(detail.Preparation)
		response.GeneratedInventory = detail.GeneratedInventory
	}
	return response, nil
}

// auditDetailHydratesPreparation reports whether the detail of an Audit
// carries preparation (ASC-R149): it has started and its pinned profile has
// a prepare role. Its generatedInventory exists only in the rounds phase
// (ASC-R146).
func auditDetailHydratesPreparation(audit auditstore.Audit) bool {
	return audit.Phase != auditdomain.AuditPhaseNotStarted && auditPinsPreparation(audit)
}

// auditListHydratesPreparation reports whether a list item carries
// preparation (ASC-R150): its phase precedes Rounds (preparing or
// inventory) and its pinned profile has a prepare role. Neither phase has
// generatedInventory, which exists only once the Audit reaches rounds
// (ASC-R146), so the list reads no inventory link or payload.
func auditListHydratesPreparation(audit auditstore.Audit) bool {
	return audit.Phase.PrecedesRounds() && auditPinsPreparation(audit)
}

// auditPinsPreparation reports whether the Audit's pinned profile has a
// prepare role. A snapshot that does not decode has none.
func auditPinsPreparation(audit auditstore.Audit) bool {
	profile, err := auditprofile.DecodeResolvedAuditProfileSnapshot(audit.ProfileSnapshot)
	return err == nil && profile.HasPreparation()
}

// auditPreparationReadModel is the public preparation of one Audit; the
// detail and both lists build it from the same role projection.
func auditPreparationReadModel(roles map[string]auditservice.PreparationRoleProjection) *auditPreparationResponse {
	result := &auditPreparationResponse{Roles: make(map[string]auditPreparationRoleResponse, len(roles))}
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
		result.Roles[name] = view
	}
	return result
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
