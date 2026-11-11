package controlplane

import (
	"slices"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/control"
)

// toolCapabilityIndex is detached from registration and never modified after
// construction. Registry entries keep it for their immutable process lifetime;
// optimistic placement builds one per candidate outside the Registry mutex.
type toolCapabilityIndex map[string]map[string]struct{}

func indexToolCapabilities(toolsets []control.ToolsetCapability) toolCapabilityIndex {
	result := make(toolCapabilityIndex, len(toolsets))
	for _, capability := range toolsets {
		tools := make(map[string]struct{}, len(capability.Tools))
		for _, tool := range capability.Tools {
			tools[tool] = struct{}{}
		}
		result[capability.Ref] = tools
	}
	return result
}

// completeCapabilityAssignmentWithEdges finds a complete injective
// binding-to-slot assignment over two already stable-ordered collections. Each
// augmentation uses an explicit queue and parent edges so request-controlled
// binding depth never becomes call-stack depth. It does not mutate Runtime
// Agent entries.
func completeCapabilityAssignmentWithEdges(
	available []*agentEntry,
	bindings []BindingRequirement,
	edges []CandidateEdge,
) ([]*agentEntry, bool) {
	if len(bindings) > len(available) {
		return nil, false
	}
	candidates, complete := buildCapabilityCandidates(available, bindings, edges)
	if !complete {
		return nil, false
	}
	bindingToAgent, complete := matchCapabilityCandidates(candidates, len(available))
	if !complete {
		return nil, false
	}
	selected := make([]*agentEntry, len(bindings))
	for bindingIndex, agentIndex := range bindingToAgent {
		if agentIndex < 0 {
			return nil, false
		}
		selected[bindingIndex] = available[agentIndex]
	}
	return selected, true
}

type candidateEdgeKey struct{ logicalAgentName, runtimeAgentInstanceID string }

func buildCapabilityCandidates(available []*agentEntry, bindings []BindingRequirement, edges []CandidateEdge) ([][]int, bool) {
	edgeSet := make(map[candidateEdgeKey]CandidateEdge, len(edges))
	for _, edge := range edges {
		edgeSet[candidateEdgeKey{edge.LogicalAgentName, edge.RuntimeAgentInstanceID}] = edge
	}
	candidates := make([][]int, len(bindings))
	for bindingIndex, binding := range bindings {
		for agentIndex, entry := range available {
			if !isBindingCompatibleWithTools(entry.registration, binding, entry.toolCapabilities) {
				continue
			}
			if edges != nil {
				edge, ok := edgeSet[candidateEdgeKey{binding.LogicalAgentName, entry.registration.InstanceID}]
				if !ok || edge.RuntimeAgentID != entry.principal.RuntimeAgentID ||
					edge.RuntimeAgentLabelRevision != entry.principal.LabelRevision ||
					!containsRuntimeAdapters(entry.registration.SupportedRuntimeAdapters, edge.RequiredRuntimeAdapters) {
					continue
				}
			}
			candidates[bindingIndex] = append(candidates[bindingIndex], agentIndex)
		}
		if len(candidates[bindingIndex]) == 0 {
			return nil, false
		}
	}
	return candidates, true
}

// matchCapabilityCandidates only reads the stable candidate graph. Registry
// reservation checks and mutations remain with the caller under its lock.
func matchCapabilityCandidates(candidates [][]int, agentCount int) ([]int, bool) {
	bindingToAgent := integersFilled(len(candidates), -1)
	agentToBinding := integersFilled(agentCount, -1)
	for startBinding := range candidates {
		freeAgent, parents := findAugmentingPath(candidates, agentToBinding, startBinding)
		if freeAgent == -1 {
			return nil, false
		}
		applyAugmentingPath(bindingToAgent, agentToBinding, parents, freeAgent)
	}
	return bindingToAgent, true
}

func findAugmentingPath(candidates [][]int, agentToBinding []int, startBinding int) (int, []int) {
	seenBindings := make([]bool, len(candidates))
	seenAgents := make([]bool, len(agentToBinding))
	parentBindingForAgent := integersFilled(len(agentToBinding), -1)
	queue := make([]int, 1, len(candidates))
	queue[0] = startBinding
	seenBindings[startBinding] = true
	for len(queue) > 0 {
		bindingIndex := queue[0]
		queue = queue[1:]
		for _, agentIndex := range candidates[bindingIndex] {
			if seenAgents[agentIndex] {
				continue
			}
			seenAgents[agentIndex] = true
			parentBindingForAgent[agentIndex] = bindingIndex
			occupiedBy := agentToBinding[agentIndex]
			if occupiedBy == -1 {
				return agentIndex, parentBindingForAgent
			}
			if !seenBindings[occupiedBy] {
				seenBindings[occupiedBy] = true
				queue = append(queue, occupiedBy)
			}
		}
	}
	return -1, parentBindingForAgent
}

func applyAugmentingPath(bindingToAgent, agentToBinding, parents []int, freeAgent int) {
	for currentAgent := freeAgent; currentAgent != -1; {
		currentBinding := parents[currentAgent]
		previousAgent := bindingToAgent[currentBinding]
		bindingToAgent[currentBinding] = currentAgent
		agentToBinding[currentAgent] = currentBinding
		currentAgent = previousAgent
	}
}

func integersFilled(length int, value int) []int {
	result := make([]int, length)
	for index := range result {
		result[index] = value
	}
	return result
}

func isBindingCompatible(registration control.AgentRegistration, binding BindingRequirement) bool {
	return isBindingCompatibleWithTools(registration, binding, indexToolCapabilities(registration.SupportedToolsets))
}

func isBindingCompatibleWithTools(registration control.AgentRegistration, binding BindingRequirement, tools toolCapabilityIndex) bool {
	return resolved.ValidateWorkerCompletionSelection(binding.CompletionContract, binding.Namespace, binding.AgentTemplate, binding.ContextManagement) == nil &&
		contracts.SupportsWorkerCompletion(registration.Capabilities, binding.CompletionContract) &&
		tools.supports(registration, binding.AgentTemplate, binding.Workspace)
}

func (capabilities toolCapabilityIndex) supports(
	registration control.AgentRegistration,
	template contracts.ResolvedAgentTemplate,
	workspace *contracts.AllocationWorkspaceSpec,
) bool {
	if !resolved.SandboxWorkspaceCompatible(template, workspace, registration.WorkspaceCapabilities) {
		return false
	}
	runtime := template.Runtime.RuntimeID + "@" + template.Runtime.Version
	if !slices.Contains(registration.SupportedRuntimes, runtime) {
		return false
	}
	sandbox := template.SandboxProfile.SandboxProfileID + "@" + template.SandboxProfile.Version
	if !slices.Contains(registration.SupportedSandboxProfiles, sandbox) {
		return false
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
