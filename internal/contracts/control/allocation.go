package control

// Allocation lifecycle on the private Control Plane/Runtime wire: the
// AllocationSpec a Runtime prepares, and the prepare, finalize, abort and
// release requests addressed to one allocation.
// Mirrors the Python runtime's contracts/allocation.py.

import (
	"encoding/json"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/runlabels"
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
	APIVersion     string    `json:"apiVersion"`
	AllocationID   string    `json:"allocationId"`
	FinalizationID string    `json:"finalizationId"`
	Deadline       time.Time `json:"deadline"`
}

func (r FinalizeAllocationRequest) Validate() error {
	return validateLifecycleRequest(r.APIVersion, r.AllocationID, "finalizationId", r.FinalizationID, r.Deadline)
}

type AbortAllocationRequest struct {
	APIVersion   string                     `json:"apiVersion"`
	AllocationID string                     `json:"allocationId"`
	AbortID      string                     `json:"abortId"`
	Reason       contracts.TerminationError `json:"reason"`
	Deadline     time.Time                  `json:"deadline"`
}

func (r AbortAllocationRequest) Validate() error {
	if err := validateLifecycleRequest(r.APIVersion, r.AllocationID, "abortId", r.AbortID, r.Deadline); err != nil {
		return err
	}
	return r.Reason.Validate()
}

type ReleaseAllocationRequest struct {
	APIVersion   string `json:"apiVersion"`
	AllocationID string `json:"allocationId"`
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
	CompletionContract              *contracts.WorkerCompletionContract       `json:"completionContract,omitempty"`
	APIVersion                      string                                    `json:"apiVersion"`
	AllocationID                    string                                    `json:"allocationId"`
	RunID                           string                                    `json:"runId"`
	StageExecutionID                string                                    `json:"stageExecutionId"`
	LogicalAgentName                string                                    `json:"logicalAgentName"`
	Namespace                       string                                    `json:"namespace"`
	WorkerSessionMode               contracts.WorkerSessionMode               `json:"workerSessionMode"`
	RunMetadataLabels               runlabels.RunMetadataLabels               `json:"runMetadataLabels"`
	LeaseExpiresAt                  time.Time                                 `json:"leaseExpiresAt"`
	AgentTemplate                   contracts.ResolvedAgentTemplate           `json:"agentTemplate"`
	ResolvedSkills                  []contracts.ResolvedSkill                 `json:"resolvedSkills"`
	ModelPolicy                     contracts.ResolvedModelPolicy             `json:"modelPolicy,omitzero"`
	RuntimeSettings                 contracts.RuntimeSettings                 `json:"runtimeSettings"`
	ResolvedRuntimeConfigProvenance contracts.ResolvedRuntimeConfigProvenance `json:"resolvedRuntimeConfigProvenance"`
	Workspace                       *contracts.AllocationWorkspaceSpec        `json:"workspace,omitempty"`
	PerformanceMetrics              *contracts.PerformanceMetricsRequest      `json:"performanceMetrics,omitempty"`
}

func (s AllocationSpec) Validate() error {
	if s.CompletionContract != nil {
		if err := s.CompletionContract.ValidateAllocation(s.Namespace, s.AgentTemplate); err != nil {
			return err
		}
	}
	if s.PerformanceMetrics != nil {
		if err := s.PerformanceMetrics.Validate(); err != nil {
			return err
		}
	}
	if err := contracts.ValidateAPIVersion(s.APIVersion); err != nil {
		return err
	}
	for field, value := range map[string]string{
		"allocationId": s.AllocationID, "runId": s.RunID,
		"stageExecutionId": s.StageExecutionID, "logicalAgentName": s.LogicalAgentName,
		"namespace": s.Namespace,
	} {
		if err := contracts.ValidateOpaqueID(field, value); err != nil {
			return err
		}
	}
	if contracts.ValidateArtifactName(s.Namespace) != nil || s.LeaseExpiresAt.IsZero() {
		return contracts.Invalidf("allocation namespace or lease is invalid")
	}
	if err := s.WorkerSessionMode.Validate(); err != nil {
		return err
	}
	if err := s.RunMetadataLabels.Validate(); err != nil {
		return err
	}
	if err := s.AgentTemplate.Validate(); err != nil {
		return err
	}
	if err := contracts.ValidateResolvedSkills(s.AgentTemplate, s.ResolvedSkills); err != nil {
		return err
	}
	if s.AgentTemplate.IsToolWorker() {
		if !s.ModelPolicy.IsZero() || s.RuntimeSettings.LLMGatewayURL != "" || s.RuntimeSettings.LLMGatewayToken != nil ||
			s.ResolvedRuntimeConfigProvenance.LLMGatewayConfig != nil || s.ResolvedRuntimeConfigProvenance.LLMCredential != nil || s.Workspace != nil || s.CompletionContract != nil {
			return contracts.Invalidf("tool@1 allocation forbids model access, project workspace and completion contracts")
		}
	} else {
		if s.RuntimeSettings.LLMGatewayURL == "" {
			return contracts.Invalidf("modeled allocation requires llmGatewayUrl")
		}
		if err := s.ModelPolicy.ValidateForWorker(len(s.AgentTemplate.Toolsets) > 0 || len(s.AgentTemplate.Skills) > 0); err != nil {
			return err
		}
	}
	if s.AgentTemplate.Summarizer != nil {
		if err := s.AgentTemplate.Summarizer.Validate(s.ModelPolicy); err != nil {
			return err
		}
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
