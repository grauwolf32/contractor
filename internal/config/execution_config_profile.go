package config

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
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
		l.executionConfigs[selector.String()] = ResolvedExecutionConfigProfile{
			Ref: ExecutionConfigRef{
				ConfigID: selector.ID, Version: selector.Version, Digest: digest,
			},
			Override: override,
		}
	}
	return nil
}

func stageExecutionConfigPatchFromProfileSource(
	source *executionConfigProfileSpecSource,
	field string,
) (StageExecutionConfigPatch, error) {
	if source == nil {
		return StageExecutionConfigPatch{}, fmt.Errorf("%s is required", field)
	}
	return stageExecutionConfigPatchFromParts(
		source.Planner, source.Agents, source.plannerPresent, source.agentsPresent, field,
	)
}

func stageExecutionConfigPatchFromEscalationSource(
	source *escalationExecutionConfigSource,
	field string,
) (StageExecutionConfigPatch, error) {
	if source == nil {
		return StageExecutionConfigPatch{}, fmt.Errorf("%s is required", field)
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
) (StageExecutionConfigPatch, error) {
	if !plannerPresent && !agentsPresent {
		return StageExecutionConfigPatch{}, fmt.Errorf("%s must contain planner and/or agents", field)
	}
	result := StageExecutionConfigPatch{}
	if plannerPresent {
		if planner == nil {
			return StageExecutionConfigPatch{}, fmt.Errorf("%s.planner must not be null", field)
		}
		selection, err := executionSelectionFromYAML(*planner, true, field+".planner")
		if err != nil {
			return StageExecutionConfigPatch{}, err
		}
		result.Planner = &selection
	}
	if agentsPresent {
		if agents == nil {
			return StageExecutionConfigPatch{}, fmt.Errorf("%s.agents must not be null", field)
		}
		if len(*agents) == 0 {
			return StageExecutionConfigPatch{}, fmt.Errorf("%s.agents must be a non-empty mapping", field)
		}
		result.Agents = make(map[string]ExecutionSelectionPatch, len(*agents))
		for logicalName, sourceSelection := range *agents {
			if err := validateMapKey(field+" logical Agent name", logicalName); err != nil {
				return StageExecutionConfigPatch{}, err
			}
			selection, err := executionSelectionFromYAML(
				sourceSelection, true, field+".agents."+logicalName,
			)
			if err != nil {
				return StageExecutionConfigPatch{}, err
			}
			result.Agents[logicalName] = selection
		}
	}
	return result, nil
}

func (l *loader) resolveStageExecutionConfigOverride(
	patch StageExecutionConfigPatch,
) (ResolvedStageExecutionConfigOverride, error) {
	result := ResolvedStageExecutionConfigOverride{}
	if patch.Planner != nil {
		selection, err := l.resolveExecutionSelectionOverride(*patch.Planner)
		if err != nil {
			return ResolvedStageExecutionConfigOverride{}, fmt.Errorf("planner: %w", err)
		}
		result.Planner = &selection
	}
	if len(patch.Agents) > 0 {
		result.Agents = make(map[string]ResolvedExecutionSelectionOverride, len(patch.Agents))
		for _, logicalName := range sortedPatchKeys(patch.Agents) {
			selection, err := l.resolveExecutionSelectionOverride(patch.Agents[logicalName])
			if err != nil {
				return ResolvedStageExecutionConfigOverride{}, fmt.Errorf("agents.%s: %w", logicalName, err)
			}
			result.Agents[logicalName] = selection
		}
	}
	return result, nil
}

func (l *loader) resolveExecutionSelectionOverride(
	patch ExecutionSelectionPatch,
) (ResolvedExecutionSelectionOverride, error) {
	result := ResolvedExecutionSelectionOverride{}
	if patch.modelPolicy.present {
		selector, err := ParseSelector(patch.modelPolicy.value)
		if err != nil {
			return ResolvedExecutionSelectionOverride{}, fmt.Errorf("modelPolicy: %w", err)
		}
		policy, ok := l.policies[selector.String()]
		if !ok {
			return ResolvedExecutionSelectionOverride{}, fmt.Errorf(
				"modelPolicy selects unknown ModelPolicy %q", selector,
			)
		}
		resolved := cloneModelPolicy(policy)
		result.ModelPolicy = &resolved
	}
	if patch.llmGateway.present {
		selector, err := ParseSelector(patch.llmGateway.value)
		if err != nil {
			return ResolvedExecutionSelectionOverride{}, fmt.Errorf("llmGateway: %w", err)
		}
		gateway, ok := l.gateways[selector.String()]
		if !ok {
			return ResolvedExecutionSelectionOverride{}, fmt.Errorf(
				"llmGateway selects unknown LLMGatewayConfig %q", selector,
			)
		}
		resolved := cloneLLMGatewayConfig(gateway)
		result.LLMGateway = &resolved
	}
	if patch.credential.present {
		credential := &ResolvedCredentialOverride{Clear: patch.credential.null}
		if !patch.credential.null {
			ref := contracts.LLMCredentialRef{CredentialID: patch.credential.value}
			if err := ref.Validate(); err != nil {
				return ResolvedExecutionSelectionOverride{}, fmt.Errorf("credential is invalid: %w", err)
			}
			credential.Ref = &ref
		}
		result.Credential = credential
	}
	return result, nil
}
