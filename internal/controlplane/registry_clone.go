package controlplane

// Deep copies for the values the registry hands out. Every snapshot,
// reservation and grant leaves the registry detached from the entry it was
// read from, so a caller can never mutate registry state through a returned
// value while the lock is not held.

import (
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/helpers/clone"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func cloneRegistration(source control.AgentRegistration) control.AgentRegistration {
	result := source
	result.AllocationID = clone.Pointer(source.AllocationID)
	result.PlacementLabels = append([]string{}, source.PlacementLabels...)
	result.SupportedRuntimes = append([]string{}, source.SupportedRuntimes...)
	result.SupportedSandboxProfiles = append([]string{}, source.SupportedSandboxProfiles...)
	result.SupportedRuntimeAdapters = append([]runtimesettings.RuntimeAdapterRef{}, source.SupportedRuntimeAdapters...)
	result.SupportedPerformanceMetricsVersions = append(
		reporting.PerformanceMetricsVersions{}, source.SupportedPerformanceMetricsVersions...,
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
	result.Grant = cloneAllocationGrant(source.Grant)
	result.PlacementLabels = append([]string{}, source.PlacementLabels...)
	result.CompletionContract = contracts.CloneWorkerCompletionContract(source.CompletionContract)
	if source.CompletionCapabilities != nil {
		value := *source.CompletionCapabilities
		value.CompletionContracts = append([]string{}, value.CompletionContracts...)
		result.CompletionCapabilities = &value
	}
	result.AgentTemplate = source.AgentTemplate.Clone()
	result.ContextManagement = source.ContextManagement.Clone()
	result.ResolvedSkills = contracts.CloneResolvedSkills(source.ResolvedSkills)
	result.ExecutionConfig = cloneAllocationExecutionConfig(source.ExecutionConfig)
	result.Workspace = contracts.CloneAllocationWorkspaceSpec(source.Workspace)
	result.RunMetadataLabels = source.RunMetadataLabels.Clone()
	if source.PerformanceMetrics != nil {
		request := *source.PerformanceMetrics
		result.PerformanceMetrics = &request
	}
	if source.ResolvedRuntimeConfig != nil {
		runtimeConfig := source.ResolvedRuntimeConfig.Clone()
		result.ResolvedRuntimeConfig = &runtimeConfig
	}
	return result
}

func cloneAllocationGrant(source AllocationGrant) AllocationGrant {
	result := source
	result.SitemapPolicy = source.SitemapPolicy.Clone()
	result.SitemapTools = append([]string(nil), source.SitemapTools...)
	result.ProjectArtifactReads = append([]ProjectArtifactReadTarget(nil), source.ProjectArtifactReads...)
	for i := range result.ProjectArtifactReads {
		result.ProjectArtifactReads[i].Operations = append([]string(nil), source.ProjectArtifactReads[i].Operations...)
	}
	return result
}

// reservationRuntimeSnapshot borrows populated slices for synchronous fingerprint
// encoding, preserving the empty-array representation without deep copying them.
func reservationRuntimeSnapshot(source *runtimeconfig.RunSnapshot) *runtimeconfig.RunSnapshot {
	if source == nil {
		return nil
	}
	result := *source
	if result.Labels == nil {
		result.Labels = []runtimeconfig.PinnedLabel{}
	}
	if result.LLMCredentialIDs == nil {
		result.LLMCredentialIDs = []string{}
	}
	if result.RuntimeCredentialIDs == nil {
		result.RuntimeCredentialIDs = []string{}
	}
	return &result
}
