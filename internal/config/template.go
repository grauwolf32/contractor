package config

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/clone"
	"go.yaml.in/yaml/v4"
)

func (l *loader) resolveAgentTemplate(
	selector resolved.Selector,
	spec *agentTemplateSpecSource,
) (contracts.ResolvedAgentTemplate, error) {
	if spec == nil {
		return contracts.ResolvedAgentTemplate{}, fmt.Errorf("spec is required")
	}
	if strings.TrimSpace(spec.Description) == "" {
		return contracts.ResolvedAgentTemplate{}, fmt.Errorf("spec.description must not be empty or whitespace-only")
	}

	runtime, err := resolved.ParseSelector(spec.Runtime)
	if err != nil {
		return contracts.ResolvedAgentTemplate{}, fmt.Errorf("spec.runtime: %w", err)
	}
	if !resolved.KnownWorkerRuntime(runtime.String()) {
		return contracts.ResolvedAgentTemplate{}, fmt.Errorf("spec.runtime selects unknown WorkerRuntime %q", runtime)
	}

	var instructions contracts.ResolvedInstructions
	var summarizer *contracts.WorkerSummarizerConfig
	if runtime.String() != "tool@1" {
		instructions, err = l.resolveInstructions(spec.Instructions)
		if err != nil {
			return contracts.ResolvedAgentTemplate{}, fmt.Errorf("spec.instructions: %w", err)
		}

		summarizer, err = l.resolveWorkerSummarizer(spec.Summarizer)
		if err != nil {
			return contracts.ResolvedAgentTemplate{}, fmt.Errorf("spec.summarizer: %w", err)
		}
	} else if spec.Instructions != nil || spec.Summarizer != nil || spec.Skills != nil {
		return contracts.ResolvedAgentTemplate{}, fmt.Errorf("tool@1 forbids instructions, summarizer and skills")
	}

	toolsets, err := l.resolveToolsets(spec.Toolsets)
	if err != nil {
		return contracts.ResolvedAgentTemplate{}, err
	}
	skills, err := resolveSkills(spec.Skills)
	if err != nil {
		return contracts.ResolvedAgentTemplate{}, err
	}

	sandbox, err := resolved.ParseSelector(spec.SandboxProfile)
	if err != nil {
		return contracts.ResolvedAgentTemplate{}, fmt.Errorf("spec.sandboxProfile: %w", err)
	}
	if !resolved.KnownSandboxProfile(sandbox.String()) {
		return contracts.ResolvedAgentTemplate{}, fmt.Errorf("spec.sandboxProfile selects unknown SandboxProfile %q", sandbox)
	}

	responseRecovery, err := optionalStringFromYAML(spec.ResponseRecovery, false, "spec.responseRecovery")
	if err != nil {
		return contracts.ResolvedAgentTemplate{}, err
	}
	if responseRecovery.Present && responseRecovery.Value != "document@1" {
		return contracts.ResolvedAgentTemplate{}, fmt.Errorf("spec.responseRecovery must select document@1")
	}
	result := contracts.ResolvedAgentTemplate{
		Ref:         contracts.AgentTemplateRef{TemplateID: selector.ID, Version: selector.Version},
		Description: spec.Description,
		Runtime: contracts.WorkerRuntimeRef{
			RuntimeID: runtime.ID,
			Version:   runtime.Version,
		},
		Instructions:     instructions,
		Summarizer:       summarizer,
		ResponseRecovery: responseRecovery.Value,
		Execution:        spec.Execution.Clone(),
		Toolsets:         toolsets,
		Skills:           skills,
		SandboxProfile: contracts.SandboxProfileRef{
			SandboxProfileID: sandbox.ID,
			Version:          sandbox.Version,
		},
	}
	if err := result.ValidateResponseRecovery(); err != nil {
		return contracts.ResolvedAgentTemplate{}, err
	}
	if err := result.ValidateToolExecution(); err != nil {
		return contracts.ResolvedAgentTemplate{}, err
	}
	if err := resolved.ValidateSandboxToolCompatibility(result); err != nil {
		return contracts.ResolvedAgentTemplate{}, err
	}
	if spec.Sitemap.Kind != 0 {
		var policy contracts.ProjectSitemapPolicy
		if err := spec.Sitemap.Decode(&policy); err != nil {
			return contracts.ResolvedAgentTemplate{}, fmt.Errorf("spec.sitemap: %w", err)
		}
		if err := policy.Validate(); err != nil {
			return contracts.ResolvedAgentTemplate{}, fmt.Errorf("spec.sitemap: %w", err)
		}
		result.Sitemap = &policy
	}
	for _, selection := range toolsets {
		if selection.Ref.ToolsetID == "sitemap" && result.Sitemap == nil {
			return contracts.ResolvedAgentTemplate{}, fmt.Errorf("sitemap@1 selection requires spec.sitemap")
		}
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
) (*contracts.WorkerSummarizerConfig, error) {
	if source == nil {
		return nil, nil
	}
	result := &contracts.WorkerSummarizerConfig{
		ContextWindowRatio: contracts.DefaultWorkerSummarizerContextWindowRatio,
		CumulativeBudget:   clone.Pointer(source.CumulativeBudget),
	}
	if source.ContextWindowRatio != nil {
		result.ContextWindowRatio = *source.ContextWindowRatio
	}
	includeReasoning, err := resolveIncludeReasoning(source.IncludeReasoning)
	if err != nil {
		return nil, err
	}
	result.IncludeReasoning = includeReasoning
	if source.Instructions != nil {
		instructions, err := l.resolveInstructions(source.Instructions)
		if err != nil {
			return nil, fmt.Errorf("instructions: %w", err)
		}
		result.Instructions = &instructions
	}
	if err := result.Validate(); err != nil {
		return nil, err
	}
	return result, nil
}

// resolveIncludeReasoning resolves the authoring choice of TPL-R47: omission is
// false, and only a YAML boolean is accepted; null, strings and other scalars
// are rejected rather than read as omission.
func resolveIncludeReasoning(node yaml.Node) (bool, error) {
	if node.IsZero() {
		return false, nil
	}
	if node.Kind != yaml.ScalarNode || node.ShortTag() != "!!bool" {
		return false, fmt.Errorf("includeReasoning must be a boolean")
	}
	value, err := strconv.ParseBool(node.Value)
	if err != nil {
		return false, fmt.Errorf("includeReasoning must be a boolean")
	}
	return value, nil
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
		selector, err := resolved.ParseSelector(item.Ref)
		if err != nil {
			return nil, fmt.Errorf("spec.toolsets[%d].ref: %w", index, err)
		}
		if _, duplicate := seenRefs[selector.String()]; duplicate {
			return nil, fmt.Errorf("spec.toolsets contains duplicate ref %q", selector)
		}
		seenRefs[selector.String()] = struct{}{}
		exportedTools, ok := resolved.ToolsetTools(selector.String())
		if !ok {
			return nil, fmt.Errorf("spec.toolsets[%d] selects unknown Toolset %q", index, selector)
		}
		if len(item.Tools) == 0 {
			return nil, fmt.Errorf("spec.toolsets[%d].tools must be non-empty", index)
		}

		exported := make(map[string]struct{}, len(exportedTools))
		for _, tool := range exportedTools {
			exported[tool] = struct{}{}
		}
		selected := make([]string, 0, len(item.Tools))
		seenSelected := make(map[string]struct{}, len(item.Tools))
		for _, tool := range item.Tools {
			if err := resolved.ValidateIdentifier("selected tool", tool); err != nil {
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
