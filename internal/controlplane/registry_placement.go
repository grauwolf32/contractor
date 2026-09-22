package controlplane

// Whether an agent may take a binding. Eligibility is about the agent's
// own state; compatibility is about what the request demands of it.
// Both are pure functions so placement can be reasoned about on its own.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func containsRuntimeAdapters(
	available []contracts.RuntimeAdapterRef,
	required []contracts.RuntimeAdapterRef,
) bool {
	availableIndex := 0
	for _, expected := range required {
		for availableIndex < len(available) && available[availableIndex] < expected {
			availableIndex++
		}
		if availableIndex == len(available) || available[availableIndex] != expected {
			return false
		}
	}
	return true
}

func isPlacementEligible(entry *agentEntry, monotonicNow time.Duration) bool {
	return entry.authoritativeAllocationID == nil && !entry.reconciliationRequired &&
		!entry.leaseExpired && !entry.superseded && entry.blockedByInstanceID == nil &&
		entry.registration.ObservedState == contracts.AgentIdle &&
		entry.confirmedLeaseDeadline > monotonicNow
}

func isBindingCompatible(registration contracts.AgentRegistration, binding BindingRequirement) bool {
	return contracts.ValidateWorkerCompletionSelection(binding.CompletionContract, binding.Namespace, binding.AgentTemplate) == nil &&
		contracts.SupportsWorkerCompletion(registration.Capabilities, binding.CompletionContract) &&
		isCompatible(registration, binding.AgentTemplate, binding.Workspace)
}

func isCompatible(
	registration contracts.AgentRegistration,
	template contracts.ResolvedAgentTemplate,
	workspace *contracts.AllocationWorkspaceSpec,
) bool {
	if !workflowconfig.SandboxWorkspaceCompatible(template, workspace, registration.WorkspaceCapabilities) {
		return false
	}
	runtime := template.Runtime.RuntimeID + "@" + template.Runtime.Version
	if !contains(registration.SupportedRuntimes, runtime) {
		return false
	}
	sandbox := template.SandboxProfile.SandboxProfileID + "@" + template.SandboxProfile.Version
	if !contains(registration.SupportedSandboxProfiles, sandbox) {
		return false
	}
	capabilities := make(map[string]map[string]struct{}, len(registration.SupportedToolsets))
	for _, capability := range registration.SupportedToolsets {
		tools := make(map[string]struct{}, len(capability.Tools))
		for _, tool := range capability.Tools {
			tools[tool] = struct{}{}
		}
		capabilities[capability.Ref] = tools
	}
	for _, selection := range template.Toolsets {
		ref := selection.Ref.ToolsetID + "@" + selection.Ref.Version
		tools, ok := capabilities[ref]
		if !ok {
			return false
		}
		for _, selected := range selection.Tools {
			if _, ok := tools[selected]; !ok {
				return false
			}
		}
	}
	if workspace != nil && !supportsWorkspaceMode(registration.WorkspaceCapabilities, workspace.Mode) {
		return false
	}
	return true
}

func supportsWorkspaceMode(
	capabilities *contracts.WorkspaceCapabilities,
	mode contracts.WorkspaceMode,
) bool {
	if capabilities == nil {
		return false
	}
	for _, supported := range capabilities.Modes {
		if supported == mode {
			return true
		}
	}
	return false
}

func normalizeReservationRequest(request ReservationRequest) (string, []BindingRequirement, error) {
	if strings.TrimSpace(request.RunID) == "" || strings.TrimSpace(request.StageExecutionID) == "" || len(request.Bindings) == 0 {
		return "", nil, fmt.Errorf("%w: run, StageExecution, and bindings are required", ErrInvalidRequest)
	}
	if request.RuntimeConfig != nil {
		if err := request.RuntimeConfig.Validate(); err != nil {
			return "", nil, fmt.Errorf("%w: Run RuntimeConfig snapshot is invalid", ErrInvalidRequest)
		}
	}
	metadataLabels, err := contracts.NormalizeRunMetadataLabels(request.RunMetadataLabels)
	if err != nil {
		return "", nil, fmt.Errorf("%w: Run metadata labels are invalid", ErrInvalidRequest)
	}
	bindings := make([]BindingRequirement, len(request.Bindings))
	seen := make(map[string]struct{}, len(request.Bindings))
	for index, binding := range request.Bindings {
		if strings.TrimSpace(binding.LogicalAgentName) == "" || contracts.ValidateArtifactName(binding.Namespace) != nil {
			return "", nil, fmt.Errorf("%w: binding name and valid Artifact namespace are required", ErrInvalidRequest)
		}
		if binding.Namespace == "inputs" || binding.Namespace == "outputs" {
			return "", nil, fmt.Errorf("%w: Agent binding cannot use a Run-reserved namespace", ErrInvalidRequest)
		}
		if err := contracts.ValidateWorkerCompletionSelection(binding.CompletionContract, binding.Namespace, binding.AgentTemplate); err != nil {
			return "", nil, fmt.Errorf("%w: %v", ErrInvalidRequest, err)
		}
		if err := binding.AgentTemplate.Validate(); err != nil {
			return "", nil, fmt.Errorf("%w: invalid AgentTemplate for %q: %v", ErrInvalidRequest, binding.LogicalAgentName, err)
		}
		if err := binding.WorkerSessionMode.Validate(); err != nil {
			return "", nil, fmt.Errorf("%w: invalid Worker session mode for %q", ErrInvalidRequest, binding.LogicalAgentName)
		}
		resolvedSkills := binding.ResolvedSkills
		if resolvedSkills == nil && len(binding.AgentTemplate.Skills) == 0 {
			resolvedSkills = []contracts.ResolvedSkill{}
		}
		if err := contracts.ValidateResolvedSkills(binding.AgentTemplate, resolvedSkills); err != nil {
			return "", nil, fmt.Errorf("%w: invalid resolved Skills for %q: %v", ErrInvalidRequest, binding.LogicalAgentName, err)
		}
		if err := binding.ExecutionConfig.Validate(); err != nil {
			return "", nil, fmt.Errorf("%w: invalid execution config for %q: %v", ErrInvalidRequest, binding.LogicalAgentName, err)
		}
		if binding.AgentTemplate.IsToolWorker() != (binding.ExecutionConfig == (AllocationExecutionConfig{})) {
			return "", nil, fmt.Errorf("%w: execution config does not match Worker runtime", ErrInvalidRequest)
		}
		if binding.Workspace != nil {
			if err := binding.Workspace.Validate(); err != nil {
				return "", nil, fmt.Errorf("%w: invalid workspace for %q: %v", ErrInvalidRequest, binding.LogicalAgentName, err)
			}
		}
		if (request.RuntimeConfig == nil) != (binding.RuntimeSelection == nil) {
			return "", nil, fmt.Errorf("%w: candidate Runtime selection is incomplete", ErrInvalidRequest)
		}
		if binding.RuntimeSelection != nil {
			if err := validateRuntimeSelection(*binding.RuntimeSelection, binding.AgentTemplate.IsToolWorker()); err != nil {
				return "", nil, fmt.Errorf("%w: invalid Runtime selection for %q", ErrInvalidRequest, binding.LogicalAgentName)
			}
		}
		if _, duplicate := seen[binding.LogicalAgentName]; duplicate {
			return "", nil, fmt.Errorf("%w: duplicate logical Agent name", ErrInvalidRequest)
		}
		seen[binding.LogicalAgentName] = struct{}{}
		bindings[index] = BindingRequirement{
			CompletionContract: contracts.CloneWorkerCompletionContract(binding.CompletionContract),
			LogicalAgentName:   binding.LogicalAgentName, Namespace: binding.Namespace,
			WorkerSessionMode: binding.WorkerSessionMode,
			AgentTemplate:     cloneAgentTemplate(binding.AgentTemplate),
			ResolvedSkills:    contracts.CloneResolvedSkills(resolvedSkills),
			ExecutionConfig:   cloneAllocationExecutionConfig(binding.ExecutionConfig),
			RuntimeSelection:  cloneRuntimeSelection(binding.RuntimeSelection),
			Workspace:         contracts.CloneAllocationWorkspaceSpec(binding.Workspace),
		}
	}
	sort.Slice(bindings, func(i, j int) bool { return bindings[i].LogicalAgentName < bindings[j].LogicalAgentName })
	encoded, err := json.Marshal(struct {
		RunID             string
		StageExecutionID  string
		RunMetadataLabels contracts.RunMetadataLabels
		Bindings          []BindingRequirement
		RuntimeConfig     *runtimeconfig.RunSnapshot
	}{
		RunID:             request.RunID,
		StageExecutionID:  request.StageExecutionID,
		RunMetadataLabels: metadataLabels,
		Bindings:          bindings,
		RuntimeConfig:     cloneRunSnapshot(request.RuntimeConfig),
	})
	if err != nil {
		return "", nil, fmt.Errorf("encode reservation request: %w", err)
	}
	digest := sha256.Sum256(encoded)
	return "sha256:" + hex.EncodeToString(digest[:]), bindings, nil
}

func validateRuntimeSelection(value workflowconfig.ResolvedConsumerExecutionConfig, toolWorker bool) error {
	if toolWorker {
		if !value.ModelPolicy.IsZero() || value.LLMGateway != nil || value.Credential != nil || value.Origins != (workflowconfig.ExecutionConfigOrigins{}) {
			return ErrInvalidRequest
		}
		return nil
	}
	if err := value.ModelPolicy.Validate(); err != nil || strings.TrimSpace(value.Origins.ModelPolicy) == "" {
		return ErrInvalidRequest
	}
	if value.LLMGateway != nil {
		if err := value.LLMGateway.Validate(); err != nil || strings.TrimSpace(value.Origins.LLMGateway) == "" {
			return ErrInvalidRequest
		}
	}
	if value.Credential != nil {
		if err := value.Credential.Validate(); err != nil || strings.TrimSpace(value.Origins.Credential) == "" {
			return ErrInvalidRequest
		}
	}
	return nil
}

func contains(values []string, expected string) bool {
	index := sort.SearchStrings(values, expected)
	return index < len(values) && values[index] == expected
}
