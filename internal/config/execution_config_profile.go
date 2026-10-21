package config

import (
	"fmt"
	"maps"
	"slices"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"go.yaml.in/yaml/v4"
)

func (s *executionConfigProfileSpecSource) UnmarshalYAML(node *yaml.Node) error {
	type plain executionConfigProfileSpecSource
	if err := node.Load((*plain)(s), yaml.WithKnownFields(), yaml.WithUniqueKeys()); err != nil {
		return err
	}
	s.plannerPresent = yamlMappingHasKey(node, "planner")
	s.agentsPresent = yamlMappingHasKey(node, "agents")
	return nil
}

func (s *escalationExecutionConfigSource) UnmarshalYAML(node *yaml.Node) error {
	type plain escalationExecutionConfigSource
	if err := node.Load((*plain)(s), yaml.WithKnownFields(), yaml.WithUniqueKeys()); err != nil {
		return err
	}
	s.refPresent = yamlMappingHasKey(node, "ref")
	s.plannerPresent = yamlMappingHasKey(node, "planner")
	s.agentsPresent = yamlMappingHasKey(node, "agents")
	return nil
}

func yamlMappingHasKey(node *yaml.Node, key string) bool {
	if node.Kind != yaml.MappingNode {
		return false
	}
	for index := 0; index+1 < len(node.Content); index += 2 {
		if node.Content[index].Value == key {
			return true
		}
	}
	return false
}

func (l *loader) loadExecutionConfigs() error {
	files, err := l.discover("execution-configs")
	if err != nil {
		return err
	}
	for _, file := range files {
		document, decodeErr := decodeOne[executionConfigDocument](file)
		if decodeErr != nil {
			return decodeErr
		}
		selector, resolveErr := validateEnvelope(
			document.APIVersion, document.Kind, executionConfigKind, document.Metadata,
		)
		if resolveErr != nil {
			return fmt.Errorf("%s: %w", file.relative, resolveErr)
		}
		if _, exists := l.executionConfigs[selector.String()]; exists {
			return fmt.Errorf("%s: duplicate ExecutionConfig identity %s", file.relative, selector)
		}
		patch, resolveErr := stageExecutionConfigPatchFromProfileSource(document.Spec, "spec")
		if resolveErr != nil {
			return fmt.Errorf("%s: %w", file.relative, resolveErr)
		}
		override, resolveErr := l.resolveStageExecutionConfigOverride(patch)
		if resolveErr != nil {
			return fmt.Errorf("%s: %w", file.relative, resolveErr)
		}
		digest, digestErr := executionConfigDigest(selector, patch)
		if digestErr != nil {
			return fmt.Errorf("%s: compute ExecutionConfig digest: %w", file.relative, digestErr)
		}
		l.executionConfigs[selector.String()] = resolved.ResolvedExecutionConfigProfile{
			Ref: resolved.ExecutionConfigRef{
				ConfigID: selector.ID, Version: selector.Version, Digest: digest,
			},
			Override: override,
		}
		l.sources[configurationSourceKey(ConfigurationExecutionConfigs, selector.String())] = file.source
	}
	return nil
}

func stageExecutionConfigPatchFromProfileSource(
	source *executionConfigProfileSpecSource,
	field string,
) (resolved.StageExecutionConfigPatch, error) {
	if source == nil {
		return resolved.StageExecutionConfigPatch{}, fmt.Errorf("%s is required", field)
	}
	return stageExecutionConfigPatchFromParts(
		source.Planner, source.Agents, source.plannerPresent, source.agentsPresent, field,
	)
}

func stageExecutionConfigPatchFromEscalationSource(
	source *escalationExecutionConfigSource,
	field string,
) (resolved.StageExecutionConfigPatch, error) {
	if source == nil {
		return resolved.StageExecutionConfigPatch{}, fmt.Errorf("%s is required", field)
	}
	return stageExecutionConfigPatchFromParts(
		source.Planner, source.Agents, source.plannerPresent, source.agentsPresent, field,
	)
}

func stageExecutionConfigPatchFromParts(
	planner *executionSelectionSource,
	agents *map[string]executionSelectionSource,
	plannerPresent bool,
	agentsPresent bool,
	field string,
) (resolved.StageExecutionConfigPatch, error) {
	if !plannerPresent && !agentsPresent {
		return resolved.StageExecutionConfigPatch{}, fmt.Errorf("%s must contain planner and/or agents", field)
	}
	result := resolved.StageExecutionConfigPatch{}
	if plannerPresent {
		if planner == nil {
			return resolved.StageExecutionConfigPatch{}, fmt.Errorf("%s.planner must not be null", field)
		}
		selection, err := executionSelectionFromYAML(*planner, true, field+".planner")
		if err != nil {
			return resolved.StageExecutionConfigPatch{}, err
		}
		result.Planner = &selection
	}
	if agentsPresent {
		if agents == nil {
			return resolved.StageExecutionConfigPatch{}, fmt.Errorf("%s.agents must not be null", field)
		}
		if len(*agents) == 0 {
			return resolved.StageExecutionConfigPatch{}, fmt.Errorf("%s.agents must be a non-empty mapping", field)
		}
		result.Agents = make(map[string]resolved.ExecutionSelectionPatch, len(*agents))
		for logicalName, sourceSelection := range *agents {
			if err := resolved.ValidateWorkflowName(field+" logical Agent name", logicalName); err != nil {
				return resolved.StageExecutionConfigPatch{}, err
			}
			selection, err := executionSelectionFromYAML(
				sourceSelection, true, field+".agents."+logicalName,
			)
			if err != nil {
				return resolved.StageExecutionConfigPatch{}, err
			}
			result.Agents[logicalName] = selection
		}
	}
	return result, nil
}

func (l *loader) resolveStageExecutionConfigOverride(
	patch resolved.StageExecutionConfigPatch,
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
	patch resolved.ExecutionSelectionPatch,
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
