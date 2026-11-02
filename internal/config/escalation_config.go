package config

import (
	"fmt"
	"maps"
	"slices"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"go.yaml.in/yaml/v4"
)

func (s *escalationExecutionConfigSource) UnmarshalYAML(node *yaml.Node) error {
	type plain escalationExecutionConfigSource
	if err := node.Load((*plain)(s), yaml.WithKnownFields(), yaml.WithUniqueKeys()); err != nil {
		return err
	}
	s.plannerPresent = yamlMappingHasKey(node, "planner")
	s.agentsPresent = yamlMappingHasKey(node, "agents")
	return nil
}

// yamlAliasTarget exposes the standard YAML alias target to validators that
// inspect raw nodes instead of decoding them into a typed value.
func yamlAliasTarget(node *yaml.Node) *yaml.Node {
	if node != nil && node.Kind == yaml.AliasNode {
		return node.Alias
	}
	return node
}

func yamlMappingHasKey(node *yaml.Node, key string) bool {
	node = yamlAliasTarget(node)
	if node == nil || node.Kind != yaml.MappingNode {
		return false
	}
	// Let the YAML library resolve aliases and native merge keys, including
	// the presence of explicit null members inherited through a merge.
	var fields map[string]yaml.Node
	if err := node.Decode(&fields); err != nil {
		return false
	}
	_, present := fields[key]
	return present
}

func stageExecutionConfigPatchFromEscalationSource(
	source *escalationExecutionConfigSource,
	field string,
) (stageExecutionConfigPatch, error) {
	if source == nil {
		return stageExecutionConfigPatch{}, fmt.Errorf("%s is required", field)
	}
	if !source.plannerPresent && !source.agentsPresent {
		return stageExecutionConfigPatch{}, fmt.Errorf("%s must contain planner and/or agents", field)
	}
	result := stageExecutionConfigPatch{}
	if source.plannerPresent {
		if source.Planner == nil {
			return stageExecutionConfigPatch{}, fmt.Errorf("%s.planner must not be null", field)
		}
		selection, err := executionSelectionFromYAML(*source.Planner, true, field+".planner")
		if err != nil {
			return stageExecutionConfigPatch{}, err
		}
		result.Planner = &selection
	}
	if source.agentsPresent {
		if source.Agents == nil {
			return stageExecutionConfigPatch{}, fmt.Errorf("%s.agents must not be null", field)
		}
		if len(*source.Agents) == 0 {
			return stageExecutionConfigPatch{}, fmt.Errorf("%s.agents must be a non-empty mapping", field)
		}
		result.Agents = make(map[string]executionSelectionPatch, len(*source.Agents))
		for logicalName, sourceSelection := range *source.Agents {
			if err := resolved.ValidateWorkflowName(field+" logical Agent name", logicalName); err != nil {
				return stageExecutionConfigPatch{}, err
			}
			selection, err := executionSelectionFromYAML(
				sourceSelection, true, field+".agents."+logicalName,
			)
			if err != nil {
				return stageExecutionConfigPatch{}, err
			}
			result.Agents[logicalName] = selection
		}
	}
	return result, nil
}

func (l *loader) resolveStageExecutionConfigOverride(
	patch stageExecutionConfigPatch,
) (resolved.ResolvedStageExecutionConfigOverride, error) {
	result := resolved.ResolvedStageExecutionConfigOverride{}
	if patch.Planner != nil {
		selection, err := l.resolveExecutionSelectionOverride(*patch.Planner)
		if err != nil {
			return resolved.ResolvedStageExecutionConfigOverride{}, fmt.Errorf("planner: %w", err)
		}
		result.Planner = &selection
	}
	if len(patch.Agents) > 0 {
		result.Agents = make(map[string]resolved.ResolvedExecutionSelectionOverride, len(patch.Agents))
		for _, logicalName := range slices.Sorted(maps.Keys(patch.Agents)) {
			selection, err := l.resolveExecutionSelectionOverride(patch.Agents[logicalName])
			if err != nil {
				return resolved.ResolvedStageExecutionConfigOverride{}, fmt.Errorf("agents.%s: %w", logicalName, err)
			}
			result.Agents[logicalName] = selection
		}
	}
	return result, nil
}

func (l *loader) resolveExecutionSelectionOverride(
	patch executionSelectionPatch,
) (resolved.ResolvedExecutionSelectionOverride, error) {
	result := resolved.ResolvedExecutionSelectionOverride{}
	if patch.ModelPolicy.Present {
		selector, err := resolved.ParseSelector(patch.ModelPolicy.Value)
		if err != nil {
			return resolved.ResolvedExecutionSelectionOverride{}, fmt.Errorf("modelPolicy: %w", err)
		}
		policy, ok := l.policies[selector.String()]
		if !ok {
			return resolved.ResolvedExecutionSelectionOverride{}, fmt.Errorf(
				"modelPolicy selects unknown ModelPolicy %q", selector,
			)
		}
		pinned := policy.Clone()
		result.ModelPolicy = &pinned
	}
	if patch.SummarizerModelPolicy.Present {
		selector, err := resolved.ParseSelector(patch.SummarizerModelPolicy.Value)
		if err != nil {
			return result, fmt.Errorf("summarizerModelPolicy: %w", err)
		}
		policy, ok := l.policies[selector.String()]
		if !ok {
			return result, fmt.Errorf("summarizerModelPolicy selects unknown ModelPolicy %q", selector)
		}
		if err := policy.ValidateForWorkerSummarizer(); err != nil {
			return result, err
		}
		pinned := policy.Clone()
		result.SummarizerModelPolicy = &pinned
	}
	if patch.CompactorModelPolicy.Present {
		policy, err := l.compactorPolicy(patch.CompactorModelPolicy.Value)
		if err != nil {
			return result, err
		}
		result.CompactorModelPolicy = &policy
	}
	if patch.LLMGateway.Present {
		selector, err := resolved.ParseSelector(patch.LLMGateway.Value)
		if err != nil {
			return resolved.ResolvedExecutionSelectionOverride{}, fmt.Errorf("llmGateway: %w", err)
		}
		gateway, ok := l.gateways[selector.String()]
		if !ok {
			return resolved.ResolvedExecutionSelectionOverride{}, fmt.Errorf(
				"llmGateway selects unknown LLMGatewayConfig %q", selector,
			)
		}
		pinned := gateway.Clone()
		result.LLMGateway = &pinned
	}
	if patch.Credential.Present {
		credential := &resolved.ResolvedCredentialOverride{Clear: patch.Credential.Null}
		if !patch.Credential.Null {
			ref := llmgateway.LLMCredentialRef{CredentialID: patch.Credential.Value}
			if err := ref.Validate(); err != nil {
				return resolved.ResolvedExecutionSelectionOverride{}, fmt.Errorf("credential is invalid: %w", err)
			}
			credential.Ref = &ref
		}
		result.Credential = credential
	}
	return result, nil
}
