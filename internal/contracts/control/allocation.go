package control

// Allocation lifecycle on the private Control Plane/Runtime wire: the
// AllocationSpec a Runtime prepares, and the prepare, finalize, abort and
// release requests addressed to one allocation.
// Mirrors the Python runtime's contracts/allocation.py.

import (
	"encoding/json"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/contracts/runlabels"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
)

type PrepareAllocationResponse struct {
	APIVersion   string                 `json:"apiVersion"`
	WorkerHandle contracts.WorkerHandle `json:"workerHandle"`
}

func (r PrepareAllocationResponse) Validate() error {
	if err := contracts.ValidateAPIVersion(r.APIVersion); err != nil {
		return err
	}
	if err := contracts.ValidateOpaqueID("workerHandle.allocationId", r.WorkerHandle.AllocationID); err != nil {
		return err
	}
	if err := r.WorkerHandle.AgentTemplateRef.ValidateRef(); err != nil {
		return err
	}
	if err := r.WorkerHandle.WorkerRuntimeRef.ValidateRef(); err != nil {
		return err
	}
	if len(r.WorkerHandle.AgentCard) == 0 || r.WorkerHandle.LeaseExpiresAt.IsZero() {
		return contracts.Invalidf("workerHandle Agent Card and leaseExpiresAt are required")
	}
	return nil
}

type FinalizeAllocationRequest struct {
	APIVersion     string    `json:"apiVersion" strictjson:"required"`
	AllocationID   string    `json:"allocationId" strictjson:"required"`
	FinalizationID string    `json:"finalizationId" strictjson:"required"`
	Deadline       time.Time `json:"deadline" strictjson:"required"`
}

func (r FinalizeAllocationRequest) Validate() error {
	return validateLifecycleRequest(r.APIVersion, r.AllocationID, "finalizationId", r.FinalizationID, r.Deadline)
}

type AbortAllocationRequest struct {
	APIVersion   string                     `json:"apiVersion" strictjson:"required"`
	AllocationID string                     `json:"allocationId" strictjson:"required"`
	AbortID      string                     `json:"abortId" strictjson:"required"`
	Reason       contracts.TerminationError `json:"reason" strictjson:"required"`
	Deadline     time.Time                  `json:"deadline" strictjson:"required"`
}

func (r AbortAllocationRequest) Validate() error {
	if err := validateLifecycleRequest(r.APIVersion, r.AllocationID, "abortId", r.AbortID, r.Deadline); err != nil {
		return err
	}
	return r.Reason.Validate()
}

type ReleaseAllocationRequest struct {
	APIVersion   string `json:"apiVersion" strictjson:"required"`
	AllocationID string `json:"allocationId" strictjson:"required"`
}

func (r ReleaseAllocationRequest) Validate() error {
	if err := contracts.ValidateAPIVersion(r.APIVersion); err != nil {
		return err
	}
	return contracts.ValidateOpaqueID("allocationId", r.AllocationID)
}

func validateLifecycleRequest(apiVersion, allocationID, idField, idValue string, deadline time.Time) error {
	if err := contracts.ValidateAPIVersion(apiVersion); err != nil {
		return err
	}
	if err := contracts.ValidateOpaqueID("allocationId", allocationID); err != nil {
		return err
	}
	if err := contracts.ValidateOpaqueID(idField, idValue); err != nil {
		return err
	}
	if deadline.IsZero() {
		return contracts.Invalidf("deadline must not be zero")
	}
	return nil
}

type AllocationSpec struct {
	ProjectID                       *string                                         `json:"projectId,omitempty"`
	Engagement                      *engagementcontract.AllocationProjection        `json:"engagement,omitempty"`
	SummarizerModelPolicy           *contracts.ResolvedModelPolicy                  `json:"summarizerModelPolicy,omitempty"`
	CompactorModelPolicy            *contracts.ResolvedModelPolicy                  `json:"compactorModelPolicy,omitempty"`
	CompletionContract              *contracts.WorkerCompletionContract             `json:"completionContract,omitempty"`
	APIVersion                      string                                          `json:"apiVersion" strictjson:"required"`
	AllocationID                    string                                          `json:"allocationId" strictjson:"required"`
	RunID                           string                                          `json:"runId" strictjson:"required"`
	StageExecutionID                string                                          `json:"stageExecutionId" strictjson:"required"`
	LogicalAgentName                string                                          `json:"logicalAgentName" strictjson:"required"`
	Namespace                       string                                          `json:"namespace" strictjson:"required"`
	WorkerSessionMode               contracts.WorkerSessionMode                     `json:"workerSessionMode" strictjson:"required"`
	ContextManagement               contracts.WorkerContextManagement               `json:"contextManagement" strictjson:"required"`
	RunMetadataLabels               runlabels.RunMetadataLabels                     `json:"runMetadataLabels" strictjson:"required"`
	LeaseExpiresAt                  time.Time                                       `json:"leaseExpiresAt" strictjson:"required"`
	AgentTemplate                   contracts.ResolvedAgentTemplate                 `json:"agentTemplate" strictjson:"required"`
	ResolvedSkills                  []contracts.ResolvedSkill                       `json:"resolvedSkills" strictjson:"required"`
	ModelPolicy                     contracts.ResolvedModelPolicy                   `json:"modelPolicy,omitzero"`
	RuntimeSettings                 runtimesettings.RuntimeSettings                 `json:"runtimeSettings" strictjson:"required"`
	ResolvedRuntimeConfigProvenance runtimesettings.ResolvedRuntimeConfigProvenance `json:"resolvedRuntimeConfigProvenance" strictjson:"required"`
	Workspace                       *contracts.AllocationWorkspaceSpec              `json:"workspace,omitempty" strictjson:"nullable"`
	PerformanceMetrics              *reporting.PerformanceMetricsRequest            `json:"performanceMetrics,omitempty" strictjson:"nullable"`
}

func (s AllocationSpec) Validate() error {
	if err := s.validateEngagement(); err != nil {
		return err
	}
	if s.CompletionContract != nil {
		if err := s.CompletionContract.ValidateAllocation(s.Namespace); err != nil {
			return err
		}
	}
	if s.PerformanceMetrics != nil {
		if err := s.PerformanceMetrics.Validate(); err != nil {
			return err
		}
	}
	if err := s.validateIdentity(); err != nil {
		return err
	}
	if err := s.AgentTemplate.Validate(); err != nil {
		return err
	}
	if err := contracts.ValidateResolvedSkills(s.AgentTemplate, s.ResolvedSkills); err != nil {
		return err
	}
	if err := s.validateWorkerConfiguration(); err != nil {
		return err
	}
	if err := s.RuntimeSettings.Validate(); err != nil {
		return err
	}
	if s.Workspace != nil {
		if err := s.Workspace.Validate(); err != nil {
			return err
		}
	}
	return s.ResolvedRuntimeConfigProvenance.Validate()
}

type PrepareAllocationRequest struct {
	APIVersion string         `json:"apiVersion"`
	Spec       AllocationSpec `json:"spec"`
}

func (r PrepareAllocationRequest) Validate() error {
	if err := contracts.ValidateAPIVersion(r.APIVersion); err != nil {
		return err
	}
	return r.Spec.Validate()
}

func (s *AllocationSpec) UnmarshalJSON(data []byte) error {
	type wire AllocationSpec
	var value wire
	fields, err := contracts.DecodeStrictObject(data, &value)
	if err != nil {
		return err
	}
	if value.AgentTemplate.IsToolWorker() {
		if fields["modelPolicy"] != nil || fields["completionContract"] != nil {
			return contracts.Invalidf("tool@1 forbids modelPolicy and completionContract")
		}
		var settings map[string]json.RawMessage
		if err := json.Unmarshal(fields["runtimeSettings"], &settings); err != nil {
			return err
		}
		if settings["llmGatewayUrl"] != nil {
			return contracts.Invalidf("tool@1 forbids llmGatewayUrl")
		}
	}
	*s = AllocationSpec(value)
	return nil
}
