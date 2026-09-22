package public

// Owner queue: queued item pages and the queue control body.

import (
	"time"

	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

type queueProjectResponse struct {
	ProjectID string            `json:"projectId"`
	Name      string            `json:"name"`
	Kind      projectstore.Kind `json:"kind"`
}

type queueItemResponse struct {
	RunID       string                     `json:"runId"`
	Project     *queueProjectResponse      `json:"project,omitempty"`
	Workflow    string                     `json:"workflow"`
	State       runstore.WorkflowRunState  `json:"state"`
	Labels      runstore.RunMetadataLabels `json:"labels"`
	EventCursor eventCursorResponse        `json:"eventCursor"`
	CreatedAt   time.Time                  `json:"createdAt"`
	UpdatedAt   time.Time                  `json:"updatedAt"`
}

type queuePageResponse struct {
	Items []queueItemResponse `json:"items"`
	Page  pageInfoResponse    `json:"page"`
}

type ownerQueueControlResponse struct {
	Paused    bool       `json:"paused"`
	Revision  string     `json:"revision"`
	UpdatedAt *time.Time `json:"updatedAt,omitempty"`
}

type updateOwnerQueueControlRequest struct {
	Paused *bool `json:"paused"`
}
