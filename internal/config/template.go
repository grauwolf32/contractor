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
	summarizer, err := l.resolveWorkerSummarizer(spec.Summarizer, policy)
	if err != nil {
		return contracts.ResolvedAgentTemplate{}, fmt.Errorf("spec.summarizer: %w", err)
	}

	toolsets, err := l.resolveToolsets(spec.Toolsets)
	if err != nil {
		return contracts.ResolvedAgentTemplate{}, err
	}
	skills, err := resolveSkills(spec.Skills)
	if err != nil {
		return contracts.ResolvedAgentTemplate{}, err
	}
	if err := policy.ValidateForWorker(len(toolsets) > 0 || len(skills) > 0); err != nil {
		return contracts.ResolvedAgentTemplate{}, fmt.Errorf("spec.modelPolicy: %w", err)
	}
	if len(skills) > 0 {
		for _, toolset := range toolsets {
			for _, tool := range toolset.Tools {
				if contracts.IsNativeSkillToolName(tool) {
					return contracts.ResolvedAgentTemplate{}, fmt.Errorf("model-visible tool %q is reserved by Agent Skills", tool)
				}
			}
		}
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
		Summarizer:   summarizer,
		Toolsets:     toolsets,
		Skills:       skills,
		SandboxProfile: contracts.SandboxProfileRef{
			SandboxProfileID: sandbox.ID,
			Version:          sandbox.Version,
		},
	}
	if err := l.descriptors.ValidateSandboxToolCompatibility(result); err != nil {
		return contracts.ResolvedAgentTemplate{}, err
	}
	digest, err := agentTemplateDigest(selector, result)
	if err != nil {
		return contracts.ResolvedAgentTemplate{}, fmt.Errorf("compute AgentTemplate digest: %w", err)
	}
	result.Ref.Digest = digest
	return result, nil
}

func (l *loader) resolveWorkerSummarizer(
	source *workerSummarizerSource,
	workerPolicy contracts.ResolvedModelPolicy,
) (*contracts.WorkerSummarizerConfig, error) {
	if source == nil {
		return nil, nil
	}
	selector, err := ParseSelector(source.ModelPolicy)
	if err != nil {
		return nil, fmt.Errorf("modelPolicy: %w", err)
	}
	policy, ok := l.policies[selector.String()]
	if !ok {
		return nil, fmt.Errorf("modelPolicy selects unknown ModelPolicy %q", selector)
	}
	result := &contracts.WorkerSummarizerConfig{
		ModelPolicy:        cloneModelPolicy(policy),
		ContextWindowRatio: contracts.DefaultWorkerSummarizerContextWindowRatio,
		CumulativeBudget:   cloneInt(source.CumulativeBudget),
	}
	if source.ContextWindowRatio != nil {
		result.ContextWindowRatio = *source.ContextWindowRatio
	}
	if source.Instructions != nil {
		instructions, err := l.resolveInstructions(source.Instructions)
		if err != nil {
			return nil, fmt.Errorf("instructions: %w", err)
		}
		result.Instructions = &instructions
	}
	if err := result.Validate(workerPolicy); err != nil {
		return nil, err
	}
	return result, nil
}

func resolveSkills(source *[]artifactRefSource) ([]contracts.ArtifactRef, error) {
	if source == nil || len(*source) == 0 {
		return nil, nil
	}
	if len(*source) > contracts.MaxAgentTemplateSkills {
		return nil, fmt.Errorf("spec.skills may contain at most %d refs", contracts.MaxAgentTemplateSkills)
	}
	result := make([]contracts.ArtifactRef, 0, len(*source))
	seen := make(map[string]struct{}, len(*source))
	for index, item := range *source {
		ref := contracts.ArtifactRef{Namespace: item.Namespace, Name: item.Name, Revision: item.Revision}
		if err := ref.ValidateAgentSkillRef(); err != nil {
			return nil, fmt.Errorf("spec.skills[%d]: %w", index, err)
		}
		if _, duplicate := seen[item.Name]; duplicate {
			return nil, fmt.Errorf("spec.skills contains duplicate ref skills/%s", item.Name)
		}
		seen[item.Name] = struct{}{}
		result = append(result, contracts.ArtifactRef{Namespace: item.Namespace, Name: item.Name})
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Name < result[j].Name })
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
