package controlplane

// Deep copies for the values the registry hands out. Every snapshot,
// reservation and grant leaves the registry detached from the entry it was
// read from, so a caller can never mutate registry state through a returned
// value while the lock is not held.

import (
	"github.com/grauwolf32/contractor/internal/clone"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func cloneRegistration(source control.AgentRegistration) control.AgentRegistration {
	result := source
	result.AllocationID = clone.Pointer(source.AllocationID)
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
	result.SupportedToolsets = make([]control.ToolsetCapability, len(source.SupportedToolsets))
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

func cloneHeartbeatResponse(source control.HeartbeatResponse) control.HeartbeatResponse {
	result := source
	result.AllocationID = clone.Pointer(source.AllocationID)
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
	result.AgentTemplate = source.AgentTemplate.Clone()
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
	result.ModelPolicy = source.ModelPolicy.Clone()
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
