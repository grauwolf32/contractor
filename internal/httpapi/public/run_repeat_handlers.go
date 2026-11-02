package public

import (
	"context"
	"github.com/grauwolf32/contractor/internal/runservice"
	"net/http"
)

type RepeatDraftReader interface {
	Read(context.Context, string, string) (runservice.RepeatProjection, error)
}
type RunRepeatDependencies struct{ RepeatDrafts RepeatDraftReader }
type runRepeatHandlers struct{ RunRepeatDependencies }

func (h *runRepeatHandlers) getRunRepeatDraft(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	source, err := h.RepeatDrafts.Read(r.Context(), principalUserID(r.Context()), r.PathValue("runId"))
	if err != nil {
		return err
	}
	response := runRepeatDraftResponse{SourceRunID: source.SourceRunID, Authority: source.Authority, Workflow: source.Workflow, ProjectID: source.ProjectID, AuditID: source.AuditID, Notices: source.Notices}
	if source.Draft != nil {
		draft := source.Draft
		response.Draft = &runRepeatDraft{RequiredPlacementLabels: draft.RequiredPlacementLabels, Parameters: draft.Parameters, RuntimeLabels: draft.RuntimeLabels, Labels: draft.Labels, Inputs: make(map[string]runRepeatInputSelection, len(draft.Inputs))}
		for slot, input := range draft.Inputs {
			selection := runRepeatInputSelection{Status: input.Status, SourceScope: input.SourceScope, Artifact: input.Artifact, Code: input.Code, Message: input.Message}
			if input.Metadata != nil {
				selection.Metadata = &artifactMetadataResponse{Metadata: *input.Metadata, CanUploadNewVersion: input.CanReplace}
			}
			response.Draft.Inputs[slot] = selection
		}
	}
	writeJSON(w, http.StatusOK, response)
	return nil
}
