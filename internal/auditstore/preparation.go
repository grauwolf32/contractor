package auditstore

import (
	"encoding/json"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
)

// PreparationSnapshot pins the complete resolved submission inputs and
// parameters before a Run exists. Recovery never resolves mutable bindings.
type PreparationSnapshot struct {
	Inputs     map[string]ExactArtifact `json:"inputs"`
	Parameters map[string]string        `json:"parameters"`
}

// PreparationOutput is stored on the immutable accepted execution receipt.
// Its containing execution supplies the Audit, role, attempt and source Run.
type PreparationOutput struct {
	LogicalName    string        `json:"logicalName"`
	WorkflowOutput string        `json:"workflowOutput"`
	Source         ExactArtifact `json:"source"`
	Retained       ExactArtifact `json:"retained"`
}

type AcceptedPreparationOutput struct {
	ExecutionID  string
	WorkflowRole string
	RoleAttempt  int
	RunID        string
	Output       PreparationOutput
}

type PreparationRole struct {
	WorkflowRole   string
	Status         auditdomain.PreparationStatus
	Attempts       int
	MaxRunAttempts int
	ExecutionID    *string
}

// PreparationAttempt is the latest attempt of a prepare role as the public
// preparation projection reads it (ASC-R41 to ASC-R43). Outputs are the
// retained output descriptors of an accepted attempt and empty otherwise.
type PreparationAttempt struct {
	ExecutionID string
	RoleAttempt int
	RunID       *string
	State       ExecutionState
	Outputs     []PreparationOutput
}

// ListedPreparationRole is one pinned prepare role of a listed Audit with its
// latest attempt, which is nil while the role has none.
type ListedPreparationRole struct {
	PreparationRole
	Latest *PreparationAttempt
}

// ListedAudit names one Audit of a list page at the revision the page read.
type ListedAudit struct {
	AuditID  string
	Revision uint64
}

type StartPreparationParams struct {
	OwnerID          string
	AuditID          string
	ExpectedRevision uint64
	BaselineSnapshot json.RawMessage
	DeadlineAt       time.Time
	InitialRetained  []ArtifactLink
	IdempotencyKey   string
	RequestDigest    string
}
