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
}

type ReservationRequest struct {
	RunID            string
	StageExecutionID string
	Bindings         []BindingRequirement
}

type AllocationGrant struct {
	AllocationID      string
	RuntimeInstanceID string
	RunID             string
	StageExecutionID  string
	LogicalAgentName  string
	Namespace         string
	ReadPolicy        ArtifactReadPolicy
	WritePolicy       ArtifactWritePolicy
	WriteFenced       bool
}

type Reservation struct {
	Grant          AllocationGrant
	ControlURL     string
	A2AURL         string
	AgentTemplate  contracts.ResolvedAgentTemplate
	LeaseExpiresAt time.Time
}
