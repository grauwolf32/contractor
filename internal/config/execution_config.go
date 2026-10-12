package config

import (
	"context"
	"fmt"
	"maps"
	"slices"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"go.yaml.in/yaml/v4"
)

const (
	originAgentTemplate  = "agentTemplate.modelPolicy"
	originWorkflowPrefix = "workflow.executionConfig"
	originRunPrefix      = "run.executionConfig"
)

func (l *loader) resolveWorkflowExecutionConfig(
	workflow *resolved.ResolvedWorkflow,
	source *workflowExecutionConfigSource,
) error {
	patch, err := executionConfigPatchFromYAML(source)
	if err != nil {
		return err
	}
	for stageName, stage := range workflow.Stages {
		base := resolved.ResolvedStageExecutionConfig{
			Agents: make(map[string]resolved.ResolvedConsumerExecutionConfig, len(stage.Agents)),
		}
		for logicalName, binding := range stage.Agents {
			if binding.Template.IsToolWorker() {
				base.Agents[logicalName] = resolved.ResolvedConsumerExecutionConfig{}
				continue
			}
			base.Agents[logicalName] = resolved.ResolvedConsumerExecutionConfig{
				ModelPolicy: binding.Template.ModelPolicy.Clone(),
				Origins:     resolved.ExecutionConfigOrigins{ModelPolicy: originAgentTemplate},
			}
		}
		stage.ExecutionConfig = base
		workflow.Stages[stageName] = stage
	}
	if err := l.applyExecutionConfigPatch(workflow, patch, originWorkflowPrefix); err != nil {
		return fmt.Errorf("spec.executionConfig: %w", err)
	}
	if err := resolved.ValidateWorkflowExecutionConfigs(*workflow); err != nil {
		return err
	}
	return resolved.ResolveEscalationVariants(workflow)
}

// ResolveRunWorkflow applies the two Run override layers to a caller-owned
// clone and rechecks every selected credential against current active metadata.
// A credential lookup that cannot confirm presence or absence returns an error
// classified as ErrCredentialLookup; every other error is a validation failure.
func (s *Snapshot) ResolveRunWorkflow(
	ctx context.Context,
	raw string,
	patch resolved.ExecutionConfigPatch,
	credentials resolved.CredentialLookup,
) (resolved.ResolvedWorkflow, error) {
	workflow, err := s.Workflow(raw)
	if err != nil {
		return resolved.ResolvedWorkflow{}, err
	}
	resolver := loader{policies: s.policies, gateways: s.gateways}
	if err := resolver.applyExecutionConfigPatch(&workflow, patch, originRunPrefix); err != nil {
		return resolved.ResolvedWorkflow{}, err
	}
	if err := resolved.ValidateWorkflowExecutionConfigs(workflow); err != nil {
		return resolved.ResolvedWorkflow{}, err
	}
	if err := resolved.ResolveEscalationVariants(&workflow); err != nil {
		return resolved.ResolvedWorkflow{}, err
	}
	if err := resolved.ValidateWorkflowGraph(workflow); err != nil {
		return resolved.ResolvedWorkflow{}, err
	}
	// Run overrides can enlarge the catalog's snapshot past its bound; the
	// check precedes every credential lookup (CAT-R76).
	if _, err := resolved.EncodeResolvedWorkflowSnapshot(workflow); err != nil {
		return resolved.ResolvedWorkflow{}, err
	}
	if err := resolved.ValidateResolvedWorkflowCredentials(ctx, workflow, credentials); err != nil {
		return resolved.ResolvedWorkflow{}, err
	}
	return workflow, nil
}

func (l *loader) applyExecutionConfigPatch(
	workflow *resolved.ResolvedWorkflow,
	patch resolved.ExecutionConfigPatch,
	originPrefix string,
) error {
	plannerApplied := false
	for _, stageName := range slices.Sorted(maps.Keys(workflow.Stages)) {
		stage := workflow.Stages[stageName]
		if patch.Planner != nil && !resolved.IsModelFreePlanner(stage.Planner) {
			if err := l.applyPlannerSelection(
				&stage, *patch.Planner, originPrefix+".planner",
			); err != nil {
				return fmt.Errorf("Stage %q: %w", stageName, err)
			}
			plannerApplied = true
		}
		if patch.Workers != nil {
			for _, logicalName := range slices.Sorted(maps.Keys(stage.ExecutionConfig.Agents)) {
				if stage.Agents[logicalName].Template.IsToolWorker() {
					return fmt.Errorf("tool@1 does not accept Worker model defaults")
				}
				selection := stage.ExecutionConfig.Agents[logicalName]
				if err := l.applySelection(
					&selection, *patch.Workers, originPrefix+".workers",
				); err != nil {
					return fmt.Errorf("Stage %q Agent %q: %w", stageName, logicalName, err)
				}
				stage.ExecutionConfig.Agents[logicalName] = selection
			}
		}
		workflow.Stages[stageName] = stage
	}
	if patch.Planner != nil && !plannerApplied {
		return fmt.Errorf("planner defaults do not apply to any modeled Stage")
	}

	for _, stageName := range slices.Sorted(maps.Keys(patch.Stages)) {
		stagePatch := patch.Stages[stageName]
		stage, ok := workflow.Stages[stageName]
		if !ok {
			return fmt.Errorf("stages names unknown Stage %q", stageName)
		}
		stageOrigin := originPrefix + ".stages." + stageName
		if stagePatch.Planner != nil {
			if err := l.applyPlannerSelection(
				&stage, *stagePatch.Planner, stageOrigin+".planner",
			); err != nil {
				return fmt.Errorf("Stage %q: %w", stageName, err)
			}
		}
		for _, logicalName := range slices.Sorted(maps.Keys(stagePatch.Agents)) {
			if stage.Agents[logicalName].Template.IsToolWorker() {
				return fmt.Errorf("tool@1 does not accept model execution overrides")
			}
			selection, ok := stage.ExecutionConfig.Agents[logicalName]
			if !ok {
				return fmt.Errorf("stages.%s.agents names unknown logical Agent %q", stageName, logicalName)
			}
			if err := l.applySelection(
				&selection, stagePatch.Agents[logicalName], stageOrigin+".agents."+logicalName,
			); err != nil {
				return fmt.Errorf("Stage %q Agent %q: %w", stageName, logicalName, err)
			}
			stage.ExecutionConfig.Agents[logicalName] = selection
		}
		workflow.Stages[stageName] = stage
	}
	return nil
}

func (l *loader) applyPlannerSelection(
	stage *resolved.ResolvedStage,
	patch resolved.ExecutionSelectionPatch,
	origin string,
) error {
	ref := stage.Planner.PlannerID + "@" + stage.Planner.Version
	if resolved.IsModelFreePlanner(stage.Planner) {
		return fmt.Errorf("%s does not accept Planner model configuration", ref)
	}
	if stage.ExecutionConfig.Planner == nil {
		stage.ExecutionConfig.Planner = &resolved.ResolvedConsumerExecutionConfig{}
	}
	return l.applySelection(stage.ExecutionConfig.Planner, patch, origin)
}

func (l *loader) applySelection(
	selection *resolved.ResolvedConsumerExecutionConfig,
	patch resolved.ExecutionSelectionPatch,
	origin string,
) error {
	if patch.ModelPolicy.Present {
		selector, err := resolved.ParseSelector(patch.ModelPolicy.Value)
		if err != nil {
			return fmt.Errorf("modelPolicy: %w", err)
		}
		policy, ok := l.policies[selector.String()]
		if !ok {
			return fmt.Errorf("modelPolicy selects unknown ModelPolicy %q", selector)
		}
		selection.ModelPolicy = policy.Clone()
		selection.Origins.ModelPolicy = origin
	}
	if patch.LLMGateway.Present {
		selector, err := resolved.ParseSelector(patch.LLMGateway.Value)
		if err != nil {
			return fmt.Errorf("llmGateway: %w", err)
		}
		gateway, ok := l.gateways[selector.String()]
		if !ok {
			return fmt.Errorf("llmGateway selects unknown LLMGatewayConfig %q", selector)
		}
		pinned := gateway.Clone()
		selection.LLMGateway = &pinned
		selection.Origins.LLMGateway = origin
	}
	if patch.Credential.Present {
		selection.Origins.Credential = origin
		selection.Credential = nil
		if !patch.Credential.Null {
			ref := llmgateway.LLMCredentialRef{CredentialID: patch.Credential.Value}
			if err := ref.Validate(); err != nil {
				return fmt.Errorf("credential is invalid: %w", err)
			}
			selection.Credential = &ref
		}
	}
	return nil
}

func executionConfigPatchFromYAML(source *workflowExecutionConfigSource) (resolved.ExecutionConfigPatch, error) {
	if source == nil {
		return resolved.ExecutionConfigPatch{}, nil
	}
	result := resolved.ExecutionConfigPatch{Stages: make(map[string]resolved.StageExecutionConfigPatch, len(source.Stages))}
	var err error
	if source.Planner != nil {
		value, parseErr := executionSelectionFromYAML(*source.Planner, false, "planner")
		if parseErr != nil {
			return resolved.ExecutionConfigPatch{}, parseErr
		}
		result.Planner = &value
	}
	if source.Workers != nil {
		value, parseErr := executionSelectionFromYAML(*source.Workers, false, "workers")
		if parseErr != nil {
			return resolved.ExecutionConfigPatch{}, parseErr
		}
		result.Workers = &value
	}
	for stageName, sourceStage := range source.Stages {
		if err = resolved.ValidateWorkflowName("executionConfig Stage name", stageName); err != nil {
			return resolved.ExecutionConfigPatch{}, err
		}
		stage := resolved.StageExecutionConfigPatch{Agents: make(map[string]resolved.ExecutionSelectionPatch, len(sourceStage.Agents))}
		if sourceStage.Planner != nil {
			value, parseErr := executionSelectionFromYAML(
				*sourceStage.Planner, false, "stages."+stageName+".planner",
			)
			if parseErr != nil {
				return resolved.ExecutionConfigPatch{}, parseErr
			}
			stage.Planner = &value
		}
		for logicalName, sourceSelection := range sourceStage.Agents {
			if err = resolved.ValidateWorkflowName("executionConfig logical Agent name", logicalName); err != nil {
				return resolved.ExecutionConfigPatch{}, err
			}
			value, parseErr := executionSelectionFromYAML(
				sourceSelection, false, "stages."+stageName+".agents."+logicalName,
			)
			if parseErr != nil {
				return resolved.ExecutionConfigPatch{}, parseErr
			}
			stage.Agents[logicalName] = value
		}
		result.Stages[stageName] = stage
	}
	return result, nil
}

func executionSelectionFromYAML(
	source executionSelectionSource,
	allowCredentialNull bool,
	field string,
) (resolved.ExecutionSelectionPatch, error) {
	modelPolicy, err := optionalStringFromYAML(source.ModelPolicy, false, field+".modelPolicy")
	if err != nil {
		return resolved.ExecutionSelectionPatch{}, err
	}
	llmGateway, err := optionalStringFromYAML(source.LLMGateway, false, field+".llmGateway")
	if err != nil {
		return resolved.ExecutionSelectionPatch{}, err
	}
	credential, err := optionalStringFromYAML(source.Credential, allowCredentialNull, field+".credential")
	if err != nil {
		return resolved.ExecutionSelectionPatch{}, err
	}
	result := resolved.ExecutionSelectionPatch{
		ModelPolicy: modelPolicy, LLMGateway: llmGateway, Credential: credential,
	}
	if !result.HasAny() {
		return resolved.ExecutionSelectionPatch{}, fmt.Errorf("%s must select at least one field", field)
	}
	return result, nil
}

func optionalStringFromYAML(
	node yaml.Node,
	allowNull bool,
	field string,
) (resolved.OptionalString, error) {
	if node.IsZero() {
		return resolved.OptionalString{}, nil
	}
	if node.Kind != yaml.ScalarNode {
		return resolved.OptionalString{}, fmt.Errorf("%s must be a string", field)
	}
	if node.ShortTag() == "!!null" {
		if !allowNull {
			return resolved.OptionalString{}, fmt.Errorf("%s must not be null", field)
		}
		return resolved.OptionalString{Present: true, Null: true}, nil
	}
	if node.ShortTag() != "!!str" {
		return resolved.OptionalString{}, fmt.Errorf("%s must be a string", field)
	}
	return resolved.OptionalString{Present: true, Value: node.Value}, nil
}
