package config

import (
	"fmt"
	"sort"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func (l *loader) resolveAgentTemplate(
	selector Selector,
	spec *agentTemplateSpecSource,
) (contracts.ResolvedAgentTemplate, error) {
	if spec == nil {
		return contracts.ResolvedAgentTemplate{}, fmt.Errorf("spec is required")
	}
	if strings.TrimSpace(spec.Description) == "" {
		return contracts.ResolvedAgentTemplate{}, fmt.Errorf("spec.description must not be empty or whitespace-only")
	}

	runtime, err := ParseSelector(spec.Runtime)
	if err != nil {
		return contracts.ResolvedAgentTemplate{}, fmt.Errorf("spec.runtime: %w", err)
	}
	if _, ok := l.descriptors.WorkerRuntimes[runtime.String()]; !ok {
		return contracts.ResolvedAgentTemplate{}, fmt.Errorf("spec.runtime selects unknown WorkerRuntime %q", runtime)
	}

	instructions, err := l.resolveInstructions(spec.Instructions)
	if err != nil {
		return contracts.ResolvedAgentTemplate{}, fmt.Errorf("spec.instructions: %w", err)
	}

	policySelector, err := ParseSelector(spec.ModelPolicy)
	if err != nil {
		return contracts.ResolvedAgentTemplate{}, fmt.Errorf("spec.modelPolicy: %w", err)
	}
	policy, ok := l.policies[policySelector.String()]
	if !ok {
		return contracts.ResolvedAgentTemplate{}, fmt.Errorf("spec.modelPolicy selects unknown ModelPolicy %q", policySelector)
	}
	policy = cloneModelPolicy(policy)

	toolsets, err := l.resolveToolsets(spec.Toolsets)
	if err != nil {
		return contracts.ResolvedAgentTemplate{}, err
	}

	sandbox, err := ParseSelector(spec.SandboxProfile)
	if err != nil {
		return contracts.ResolvedAgentTemplate{}, fmt.Errorf("spec.sandboxProfile: %w", err)
	}
	if _, ok := l.descriptors.SandboxProfiles[sandbox.String()]; !ok {
		return contracts.ResolvedAgentTemplate{}, fmt.Errorf("spec.sandboxProfile selects unknown SandboxProfile %q", sandbox)
	}

	result := contracts.ResolvedAgentTemplate{
		Ref:         contracts.AgentTemplateRef{TemplateID: selector.ID, Version: selector.Version},
		Description: spec.Description,
		Runtime: contracts.WorkerRuntimeRef{
			RuntimeID: runtime.ID,
			Version:   runtime.Version,
		},
		Instructions: instructions,
		ModelPolicy:  policy,
		Toolsets:     toolsets,
		SandboxProfile: contracts.SandboxProfileRef{
			SandboxProfileID: sandbox.ID,
			Version:          sandbox.Version,
		},
	}
	digest, err := agentTemplateDigest(selector, result)
	if err != nil {
		return contracts.ResolvedAgentTemplate{}, fmt.Errorf("compute AgentTemplate digest: %w", err)
	}
	result.Ref.Digest = digest
	return result, nil
}

func (l *loader) resolveToolsets(source *[]toolsetSelectionSource) ([]contracts.ToolsetSelection, error) {
	if source == nil {
		return nil, fmt.Errorf("spec.toolsets is required (use [] to select none)")
	}
	result := make([]contracts.ToolsetSelection, 0, len(*source))
	seenRefs := make(map[string]struct{}, len(*source))
	visibleNames := make(map[string]string)
	for index, item := range *source {
		selector, err := ParseSelector(item.Ref)
		if err != nil {
			return nil, fmt.Errorf("spec.toolsets[%d].ref: %w", index, err)
		}
		if _, duplicate := seenRefs[selector.String()]; duplicate {
			return nil, fmt.Errorf("spec.toolsets contains duplicate ref %q", selector)
		}
		seenRefs[selector.String()] = struct{}{}
		descriptor, ok := l.descriptors.Toolsets[selector.String()]
		if !ok {
			return nil, fmt.Errorf("spec.toolsets[%d] selects unknown Toolset %q", index, selector)
		}
		if len(item.Tools) == 0 {
			return nil, fmt.Errorf("spec.toolsets[%d].tools must be non-empty", index)
		}

		exported := make(map[string]struct{}, len(descriptor.Tools))
		for _, tool := range descriptor.Tools {
			exported[tool] = struct{}{}
		}
		selected := make([]string, 0, len(item.Tools))
		seenSelected := make(map[string]struct{}, len(item.Tools))
		for _, tool := range item.Tools {
			if err := validateIdentifier("selected tool", tool); err != nil {
				return nil, fmt.Errorf("spec.toolsets[%d]: %w", index, err)
			}
			if _, duplicate := seenSelected[tool]; duplicate {
				return nil, fmt.Errorf("spec.toolsets[%d].tools contains duplicate %q", index, tool)
			}
			seenSelected[tool] = struct{}{}
			if _, exists := exported[tool]; !exists {
				return nil, fmt.Errorf("Toolset %q does not export selected tool %q", selector, tool)
			}
			if previous, collision := visibleNames[tool]; collision {
				return nil, fmt.Errorf("model-visible tool %q collides between Toolsets %s and %s", tool, previous, selector)
			}
			visibleNames[tool] = selector.String()
			selected = append(selected, tool)
		}
		sort.Strings(selected)
		result = append(result, contracts.ToolsetSelection{
			Ref:   contracts.ToolsetRef{ToolsetID: selector.ID, Version: selector.Version},
			Tools: selected,
		})
	}
	sort.Slice(result, func(i, j int) bool {
		left := result[i].Ref.ToolsetID + "@" + result[i].Ref.Version
		right := result[j].Ref.ToolsetID + "@" + result[j].Ref.Version
		return left < right
	})
	return result, nil
}
