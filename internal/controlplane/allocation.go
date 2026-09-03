package controlplane

import (
	"time"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

type ArtifactReadPolicy string

const ReadCurrentRun ArtifactReadPolicy = "current_run"

type ArtifactWritePolicy string

const WriteInputsAndIntermediates ArtifactWritePolicy = "inputs_and_intermediates"

type BindingRequirement struct {
	LogicalAgentName string
	Namespace        string
	AgentTemplate    contracts.ResolvedAgentTemplate
	ResolvedSkills   []contracts.ResolvedSkill
	ExecutionConfig  AllocationExecutionConfig
	Workspace        *contracts.AllocationWorkspaceSpecV2
	// RuntimeSelection is the immutable Workflow/Run/escalation selection.
	// Candidate Agent labels are deliberately resolved later by placement.
	RuntimeSelection *workflowconfig.ResolvedConsumerExecutionConfig
}

type ReservationRequest struct {
	RunID             string
	StageExecutionID  string
	RunMetadataLabels contracts.RunMetadataLabels
	Bindings          []BindingRequirement
	// RuntimeConfig is nil only for the legacy in-process Registry surface used
	// by focused capacity tests. Production placement always supplies the
	// immutable Run snapshot.
	RuntimeConfig *runtimeconfig.RunSnapshot
}

// CandidateEdge is one already-resolved, non-secret compatibility edge. The
// Registry still rechecks its frozen capability snapshot under the live-state
// mutex before installing any allocation IDs.
type CandidateEdge struct {
	LogicalAgentName          string
	RuntimeAgentID            string
	RuntimeAgentInstanceID    string
	RuntimeAgentLabelRevision uint64
	RequiredRuntimeAdapters   []contracts.RuntimeAdapterRef
}

// PinnedReservationConfig is attached only after durable allocation
// provenance commits. It never contains RuntimeSettings or secret material.
type PinnedReservationConfig struct {
	RuntimeAgentLabelRevision uint64
	Resolved                  runtimeconfig.ResolvedRuntimeConfig
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
	Grant                     AllocationGrant
	ControlURL                string
	A2AURL                    string
	AgentTemplate             contracts.ResolvedAgentTemplate
	ResolvedSkills            []contracts.ResolvedSkill
	ExecutionConfig           AllocationExecutionConfig
	Workspace                 *contracts.AllocationWorkspaceSpecV2
	RunMetadataLabels         contracts.RunMetadataLabels
	RuntimeAgentLabelRevision uint64
	ResolvedRuntimeConfig     *runtimeconfig.ResolvedRuntimeConfig
	LeaseExpiresAt            time.Time
}
