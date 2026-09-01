package controlplane

import (
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

type ArtifactReadPolicy string

const ReadCurrentRun ArtifactReadPolicy = "current_run"

type ArtifactWritePolicy string

const WriteInputsAndIntermediates ArtifactWritePolicy = "inputs_and_intermediates"

type BindingRequirement struct {
	LogicalAgentName string
	Namespace        string
	AgentTemplate    contracts.ResolvedAgentTemplate
	ExecutionConfig  AllocationExecutionConfig
}

type ReservationRequest struct {
	RunID            string
	StageExecutionID string
	Bindings         []BindingRequirement
}

type AllocationGrant struct {
	AllocationID      string
	RuntimeAgentID    string
	RuntimeInstanceID string
	RunID             string
	StageExecutionID  string
	LogicalAgentName  string
	Namespace         string
	ReadPolicy        ArtifactReadPolicy
	WritePolicy       ArtifactWritePolicy
	WriteFenced       bool
	Lost              bool
}

type AllocationLossReason string

const (
	LossControlLeaseExpired AllocationLossReason = "control_lease_expired"
	LossRuntimeMismatch     AllocationLossReason = "runtime_state_mismatch"
	LossRuntimeRestarted    AllocationLossReason = "runtime_restarted"
)

// AllocationLoss is an irreversible edge emitted once for the durable
// StageExecution owner. The Registry retains the fenced grant until normal
// bounded abort/release reconciliation removes it.
type AllocationLoss struct {
	AllocationID      string
	RuntimeAgentID    string
	RuntimeInstanceID string
	RunID             string
	StageExecutionID  string
	Reason            AllocationLossReason
}

type Reservation struct {
	Grant           AllocationGrant
	ControlURL      string
	A2AURL          string
	AgentTemplate   contracts.ResolvedAgentTemplate
	ExecutionConfig AllocationExecutionConfig
	LeaseExpiresAt  time.Time
}
