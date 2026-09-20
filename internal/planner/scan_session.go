package planner

import (
	"context"

	"github.com/grauwolf32/contractor/internal/contracts"
)

const MaxScanJobs = 100

const (
	ScanJobPending     = "pending"
	ScanJobStarted     = "started"
	ScanJobCompleted   = "completed"
	ScanJobFailed      = "failed"
	ScanJobIncomplete  = "incomplete"
	ScanJobUnavailable = "unavailable"
	ScanJobUnknown     = "unknown"
)

// ScanSessionIdentity fences every scan mutation with the current durable
// Scheduler claim. A SessionIdentity alone cannot authorize scan dispatch.
type ScanSessionIdentity struct {
	SessionIdentity
	SchedulerClaimID string
}

// ScanJobRecord retains only exact artifact references and bounded audit facts.
// HTTP inputs, scanner arguments and credentials belong in the input artifacts.
type ScanJobRecord struct {
	ID             string                           `json:"id"`
	Worker         string                           `json:"worker"`
	Status         string                           `json:"status"`
	Code           string                           `json:"code,omitempty"`
	InputArtifacts map[string]contracts.ArtifactRef `json:"inputArtifacts"`
	Report         *contracts.ArtifactRef           `json:"report,omitempty"`
}

type ScanState struct {
	Plan       *contracts.ArtifactRef `json:"plan,omitempty"`
	PlanDigest string                 `json:"planDigest,omitempty"`
	Jobs       []ScanJobRecord        `json:"jobs"`
}

// BeginScan grants ownership only after persisting the Scheduler claim. On
// takeover, previously started jobs become unknown and pending jobs remain
// eligible. A completed session returns its durable completion without Invoke.
type ScanSessionStart struct {
	Identity   ScanSessionIdentity
	Invoke     bool
	State      ScanState
	Completion *Completion
}

type ScanSessionService interface {
	BeginScan(context.Context, string, string) (ScanSessionStart, error)
	InitializeScan(context.Context, ScanSessionIdentity, ScanState) error
	ClaimScanJob(context.Context, ScanSessionIdentity, string) (bool, error)
	FinishScanJob(context.Context, ScanSessionIdentity, ScanJobRecord) error
	CompleteScan(context.Context, ScanSessionIdentity, Completion) error
}
