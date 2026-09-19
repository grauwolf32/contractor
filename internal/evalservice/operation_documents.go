package evalservice

import (
	"time"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

type inputCopyRequest struct {
	Inputs    map[string]evaldomain.Artifact `json:"inputs"`
	ProjectID string                         `json:"projectId"`
}
type projectCreationRequest struct {
	ProjectID string `json:"projectId"`
}
type executionCreated struct {
	ID string `json:"id"`
}
type cancellationRequest struct {
	ID          string    `json:"id"`
	Kind        string    `json:"kind"`
	RequestedAt time.Time `json:"requestedAt"`
}
type cancellationAccepted struct {
	ID        string `json:"id"`
	Requested bool   `json:"requested"`
}
type draftDeletionRequest struct {
	ID          string `json:"id"`
	Kind        string `json:"kind"`
	DeleteDraft bool   `json:"deleteDraft"`
}
type draftDeletionAccepted struct {
	ID          string `json:"id"`
	DeleteDraft bool   `json:"deleteDraft"`
}
type executionDeleted struct {
	ID      string `json:"id"`
	Deleted bool   `json:"deleted"`
}
type operationRejected struct {
	Reason string `json:"reason"`
}
type preparationRejected struct {
	RejectedBeforeCreation bool   `json:"rejectedBeforeCreation"`
	Code                   string `json:"code"`
}
type safeDiagnostic struct {
	Code     string `json:"code"`
	Field    string `json:"field"`
	Recovery string `json:"recovery"`
}
