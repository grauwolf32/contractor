// Package scanstore owns the immutable sweep snapshot and bounded durable
// member progression. Scanner execution remains the ordinary Run machinery.
package scanstore

import (
	"encoding/json"
	"time"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

const (
	MaxProjects       = 1000
	MaxSnapshotBytes  = 8 << 20
	MaxChildBytes     = 16 << 10
	TickMembers       = 25
	TickBytes         = 4 << 20
	AdmissionDeadline = 24 * time.Hour
	Retention         = 30 * 24 * time.Hour
)

var (
	ErrInvalid     = apierror.New(apierror.InvalidRequest, "Scan sweep request is invalid")
	ErrNotFound    = apierror.New(apierror.NotFound, "Scan sweep not found")
	ErrConflict    = apierror.New(apierror.Conflict, "Scan sweep is not ready for this operation")
	ErrIdempotency = apierror.New(apierror.IdempotencyConflict, "Scan sweep key records another request")
)

type Counts struct {
	Total     int `json:"total"`
	Pending   int `json:"pending"`
	Admitted  int `json:"admitted"`
	Skipped   int `json:"skipped"`
	Failed    int `json:"failed"`
	Cancelled int `json:"cancelled"`
	Unsettled int `json:"unsettled"`
}

type Sweep struct {
	SweepID        string     `json:"sweepId"`
	Workflow       string     `json:"workflow"`
	SnapshotDigest string     `json:"snapshotDigest"`
	State          string     `json:"state"`
	Counts         Counts     `json:"counts"`
	CreatedAt      time.Time  `json:"createdAt"`
	SettledAt      *time.Time `json:"settledAt"`
}

type InputPin struct {
	Source   contracts.ExactArtifact `json:"source"`
	Retained contracts.ArtifactRef   `json:"retained"`
}

type Snapshot struct {
	SchemaVersion  int                       `json:"schemaVersion"`
	OwnerID        string                    `json:"ownerId"`
	Workflow       resolved.ResolvedWorkflow `json:"workflow"`
	WorkflowDigest string                    `json:"workflowDigest"`
	Runtime        runtimeconfig.RunSnapshot `json:"runtime"`
	Rulepack       *InputPin                 `json:"rulepack,omitempty"`
}

// Child is the captured public request before the trusted retained-input
// substitution. The create key is private and is never exposed to the owner.
type Child struct {
	Workflow   string                           `json:"workflow"`
	Parameters map[string]string                `json:"parameters"`
	Inputs     map[string]contracts.ArtifactRef `json:"artifacts"`
	Labels     runstore.RunMetadataLabels       `json:"labels"`
}

type Member struct {
	Ordinal   int                        `json:"ordinal"`
	ProjectID string                     `json:"projectId"`
	Target    string                     `json:"target"`
	State     string                     `json:"state"`
	Reason    string                     `json:"reason"`
	RunID     *string                    `json:"runId"`
	RunState  *runstore.WorkflowRunState `json:"runState"`
	Settled   bool                       `json:"settled"`
	Child     json.RawMessage            `json:"-"`
	Key       string                     `json:"-"`
	Digest    string                     `json:"-"`
	Attempts  int                        `json:"-"`
}

type CapturedProject struct {
	ProjectID string
	Target    string
	Reason    string
}
