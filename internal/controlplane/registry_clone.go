package controlplane

// Deep copies for the values the registry hands out. Every snapshot,
// reservation and grant leaves the registry detached from the entry it was
// read from, so a caller can never mutate registry state through a returned
// value while the lock is not held.

import (
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func cloneRegistration(source contracts.AgentRegistration) contracts.AgentRegistration {
	result := source
	result.AllocationID = cloneString(source.AllocationID)
	result.InitialLabels = append([]string{}, source.InitialLabels...)
	result.SupportedRuntimes = append([]string{}, source.SupportedRuntimes...)
	result.SupportedSandboxProfiles = append([]string{}, source.SupportedSandboxProfiles...)
	result.SupportedRuntimeAdapters = append([]contracts.RuntimeAdapterRef{}, source.SupportedRuntimeAdapters...)
	result.SupportedPerformanceMetricsVersions = append(
		contracts.PerformanceMetricsVersions{}, source.SupportedPerformanceMetricsVersions...,
	)
	if source.WorkspaceCapabilities != nil {
		capabilities := *source.WorkspaceCapabilities
		capabilities.Modes = append([]contracts.WorkspaceMode{}, source.WorkspaceCapabilities.Modes...)
		result.WorkspaceCapabilities = &capabilities
	}
	result.SupportedToolsets = make([]contracts.ToolsetCapability, len(source.SupportedToolsets))
	for index, capability := range source.SupportedToolsets {
		result.SupportedToolsets[index] = capability
		result.SupportedToolsets[index].Tools = append([]string(nil), capability.Tools...)
	}
	return result
}

func clonePrincipal(source AuthenticatedPrincipal) AuthenticatedPrincipal {
	result := source
	result.Labels = append([]string{}, source.Labels...)
	return result
}

func cloneHeartbeatResponse(source contracts.HeartbeatResponse) contracts.HeartbeatResponse {
	result := source
	result.AllocationID = cloneString(source.AllocationID)
	return result
}

func cloneReservation(source Reservation) Reservation {
	result := source
	result.CompletionContract = contracts.CloneWorkerCompletionContract(source.CompletionContract)
	if source.CompletionCapabilities != nil {
		value := *source.CompletionCapabilities
		value.CompletionContracts = append([]string{}, value.CompletionContracts...)
		result.CompletionCapabilities = &value
	}
	result.AgentTemplate = cloneAgentTemplate(source.AgentTemplate)
	result.ResolvedSkills = contracts.CloneResolvedSkills(source.ResolvedSkills)
	result.ExecutionConfig = cloneAllocationExecutionConfig(source.ExecutionConfig)
	result.Workspace = contracts.CloneAllocationWorkspaceSpec(source.Workspace)
	result.RunMetadataLabels = source.RunMetadataLabels.Clone()
	if source.PerformanceMetrics != nil {
		request := *source.PerformanceMetrics
		result.PerformanceMetrics = &request
	}
	if source.ResolvedRuntimeConfig != nil {
		resolved := source.ResolvedRuntimeConfig.Clone()
		result.ResolvedRuntimeConfig = &resolved
	}
	return result
}

func cloneRuntimeSelection(
	source *workflowconfig.ResolvedConsumerExecutionConfig,
) *workflowconfig.ResolvedConsumerExecutionConfig {
	if source == nil {
		return nil
	}
	result := *source
	result.ModelPolicy = cloneModelPolicy(source.ModelPolicy)
	if source.LLMGateway != nil {
		gateway := *source.LLMGateway
		if source.LLMGateway.CredentialManager != nil {
			manager := *source.LLMGateway.CredentialManager
			gateway.CredentialManager = &manager
		}
		result.LLMGateway = &gateway
	}
	if source.Credential != nil {
		credential := *source.Credential
		result.Credential = &credential
	}
	return &result
}

func cloneRunSnapshot(source *runtimeconfig.RunSnapshot) *runtimeconfig.RunSnapshot {
	if source == nil {
		return nil
	}
	result := source.Clone()
	return &result
}

func cloneAgentTemplate(source contracts.ResolvedAgentTemplate) contracts.ResolvedAgentTemplate {
	result := source
	result.Execution = source.Execution.Clone()
	result.ModelPolicy = cloneModelPolicy(source.ModelPolicy)
	if source.Summarizer != nil {
		summarizer := *source.Summarizer
		if source.Summarizer.Instructions != nil {
			instructions := *source.Summarizer.Instructions
			summarizer.Instructions = &instructions
		}
		summarizer.ModelPolicy = cloneModelPolicy(source.Summarizer.ModelPolicy)
		summarizer.CumulativeBudget = cloneIntPointer(source.Summarizer.CumulativeBudget)
		result.Summarizer = &summarizer
	}
	result.Toolsets = make([]contracts.ToolsetSelection, len(source.Toolsets))
	for index, selection := range source.Toolsets {
		result.Toolsets[index] = selection
		result.Toolsets[index].Tools = append([]string(nil), selection.Tools...)
	}
	result.Skills = make([]contracts.ArtifactRef, len(source.Skills))
	for index, skill := range source.Skills {
		result.Skills[index] = skill
		if skill.Revision != nil {
			revision := *skill.Revision
			result.Skills[index].Revision = &revision
		}
	}
	return result
}

func cloneIntPointer(source *int) *int {
	if source == nil {
		return nil
	}
	value := *source
	return &value
}

func cloneModelPolicy(source contracts.ResolvedModelPolicy) contracts.ResolvedModelPolicy {
	result := source
	if source.Temperature != nil {
		temperature := *source.Temperature
		result.Temperature = &temperature
	}
	return result
}

func cloneString(source *string) *string {
	if source == nil {
		return nil
	}
	result := *source
	return &result
}

func cloneCredentialRef(source *contracts.LLMCredentialRef) *contracts.LLMCredentialRef {
	if source == nil {
		return nil
	}
	result := *source
	return &result
}
