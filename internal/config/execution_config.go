package config

import (
	"context"
	"fmt"
	"reflect"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
	"go.yaml.in/yaml/v4"
)

const (
	originAgentTemplate  = "agentTemplate.modelPolicy"
	originWorkflowPrefix = "workflow.executionConfig"
	originRunPrefix      = "run.executionConfig"
)

// CredentialMetadata is the safe lookup result used while a Run is resolved.
// It deliberately contains no token or provider response.
type CredentialMetadata struct {
	Ref        contracts.LLMCredentialRef
	LLMGateway contracts.LLMGatewayConfigRef
}

type CredentialLookup interface {
	LookupLLMCredential(context.Context, string) (CredentialMetadata, error)
}

func (l *loader) resolveWorkflowExecutionConfig(
	workflow *ResolvedWorkflow,
	source *workflowExecutionConfigSource,
) error {
	patch, err := executionConfigPatchFromYAML(source)
	if err != nil {
		return err
	}
	for stageName, stage := range workflow.Stages {
		resolved := ResolvedStageExecutionConfig{
			Agents: make(map[string]ResolvedConsumerExecutionConfig, len(stage.Agents)),
		}
		for logicalName, binding := range stage.Agents {
			resolved.Agents[logicalName] = ResolvedConsumerExecutionConfig{
				ModelPolicy: cloneModelPolicy(binding.Template.ModelPolicy),
				Origins:     ExecutionConfigOrigins{ModelPolicy: originAgentTemplate},
			}
		}
		stage.ExecutionConfig = resolved
		workflow.Stages[stageName] = stage
	}
	if err := l.applyExecutionConfigPatch(workflow, patch, originWorkflowPrefix); err != nil {
		return fmt.Errorf("spec.executionConfig: %w", err)
	}
	if err := validateWorkflowExecutionConfigs(*workflow); err != nil {
		return err
	}
	return resolveWorkflowEscalationVariants(workflow)
}

// ResolveRunWorkflow applies the two Run override layers to a caller-owned
// clone and rechecks every selected credential against current active metadata.
func (s *Snapshot) ResolveRunWorkflow(
	ctx context.Context,
	raw string,
	patch ExecutionConfigPatch,
	credentials CredentialLookup,
) (ResolvedWorkflow, error) {
	workflow, err := s.Workflow(raw)
	if err != nil {
		return ResolvedWorkflow{}, err
	}
	resolver := loader{policies: s.policies, gateways: s.gateways}
	if err := resolver.applyExecutionConfigPatch(&workflow, patch, originRunPrefix); err != nil {
		return ResolvedWorkflow{}, err
	}
	if err := validateWorkflowExecutionConfigs(workflow); err != nil {
		return ResolvedWorkflow{}, err
	}
	if err := resolveWorkflowEscalationVariants(&workflow); err != nil {
		return ResolvedWorkflow{}, err
	}
	if err := ValidateWorkflowGraph(workflow); err != nil {
		return ResolvedWorkflow{}, err
	}
	if err := validateRunCredentials(ctx, workflow, credentials); err != nil {
		return ResolvedWorkflow{}, err
	}
	return workflow, nil
}

func resolveWorkflowEscalationVariants(workflow *ResolvedWorkflow) error {
	for _, stageName := range sortedPatchKeys(workflow.Stages) {
		stage := workflow.Stages[stageName]
		actions := []struct {
			outcome string
			action  *TransitionAction
		}{
			{outcome: "failed", action: &stage.On.Failed},
			{outcome: "interrupted", action: &stage.On.Interrupted},
		}
		for _, item := range actions {
			if item.action.Kind != TransitionEscalate || item.action.Escalate == nil {
				continue
			}
			variantStage := stage
			variantStage.ExecutionConfig = cloneStageExecutionConfig(stage.ExecutionConfig)
			origin := escalationExecutionConfigOrigin(
				stageName, item.outcome, item.action.Escalate.ExecutionConfig,
			)
			if err := applyResolvedStageExecutionConfigOverride(
				&variantStage, item.action.Escalate.ExecutionConfig.Override, origin,
			); err != nil {
				return fmt.Errorf("Stage %q on.%s.escalate.executionConfig: %w", stageName, item.outcome, err)
			}
			if err := validateStageExecutionConfig(stageName, variantStage); err != nil {
				return fmt.Errorf("Stage %q on.%s.escalate.executionConfig: %w", stageName, item.outcome, err)
			}
			item.action.Escalate.ExecutionConfig.Effective = cloneStageExecutionConfig(
				variantStage.ExecutionConfig,
			)
		}
		workflow.Stages[stageName] = stage
	}
	return nil
}

func validateStageEscalationVariant(
	stageName string,
	outcome string,
	stage ResolvedStage,
	action TransitionAction,
) error {
	if action.Kind != TransitionEscalate || action.Escalate == nil {
		return nil
	}
	variant := action.Escalate.ExecutionConfig
	if variant.Ref != nil {
		if err := validateExecutionConfigRef(*variant.Ref); err != nil {
			return fmt.Errorf("Stage %q on.%s.escalate.executionConfig.ref: %w", stageName, outcome, err)
		}
	}
	variantStage := stage
	variantStage.ExecutionConfig = cloneStageExecutionConfig(stage.ExecutionConfig)
	if err := applyResolvedStageExecutionConfigOverride(
		&variantStage,
		variant.Override,
		escalationExecutionConfigOrigin(stageName, outcome, variant),
	); err != nil {
		return fmt.Errorf("Stage %q on.%s.escalate.executionConfig: %w", stageName, outcome, err)
	}
	if err := validateStageExecutionConfig(stageName, variantStage); err != nil {
		return fmt.Errorf("Stage %q on.%s.escalate.executionConfig: %w", stageName, outcome, err)
	}
	if !reflect.DeepEqual(variantStage.ExecutionConfig, variant.Effective) {
		return fmt.Errorf(
			"Stage %q on.%s.escalate.executionConfig effective value does not match its pinned override",
			stageName, outcome,
		)
	}
	return nil
}

func applyResolvedStageExecutionConfigOverride(
	stage *ResolvedStage,
	override ResolvedStageExecutionConfigOverride,
	origin string,
) error {
	if override.Planner == nil && len(override.Agents) == 0 {
		return fmt.Errorf("override must contain planner and/or agents")
	}
	if override.Planner != nil {
		if !resolvedExecutionSelectionOverrideHasAny(*override.Planner) {
			return fmt.Errorf("planner must select at least one field")
		}
		if stage.Planner.PlannerID+"@"+stage.Planner.Version == "passthrough@1" {
			return fmt.Errorf("passthrough@1 does not accept Planner model configuration")
		}
		if stage.ExecutionConfig.Planner == nil {
			return fmt.Errorf("modeled Planner has no base executionConfig")
		}
		if err := applyResolvedExecutionSelectionOverride(
			stage.ExecutionConfig.Planner, *override.Planner, origin+".planner",
		); err != nil {
			return fmt.Errorf("planner: %w", err)
		}
	}
	if override.Agents != nil && len(override.Agents) == 0 {
		return fmt.Errorf("agents must be a non-empty mapping when present")
	}
	for _, logicalName := range sortedPatchKeys(override.Agents) {
		patch := override.Agents[logicalName]
		if !resolvedExecutionSelectionOverrideHasAny(patch) {
			return fmt.Errorf("agents.%s must select at least one field", logicalName)
		}
		selection, ok := stage.ExecutionConfig.Agents[logicalName]
		if !ok {
			return fmt.Errorf("agents names unknown logical Agent %q", logicalName)
		}
		if err := applyResolvedExecutionSelectionOverride(
			&selection, patch, origin+".agents."+logicalName,
		); err != nil {
			return fmt.Errorf("agents.%s: %w", logicalName, err)
		}
		stage.ExecutionConfig.Agents[logicalName] = selection
	}
	return nil
}

func applyResolvedExecutionSelectionOverride(
	selection *ResolvedConsumerExecutionConfig,
	override ResolvedExecutionSelectionOverride,
	origin string,
) error {
	if override.ModelPolicy != nil {
		if err := override.ModelPolicy.Validate(); err != nil {
			return fmt.Errorf("modelPolicy is invalid: %w", err)
		}
		selection.ModelPolicy = cloneModelPolicy(*override.ModelPolicy)
		selection.Origins.ModelPolicy = origin
	}
	if override.LLMGateway != nil {
		if err := override.LLMGateway.Validate(); err != nil {
			return fmt.Errorf("llmGateway is invalid: %w", err)
		}
		selection.LLMGateway = cloneLLMGatewayConfig(*override.LLMGateway)
		selection.Origins.LLMGateway = origin
	}
	if override.Credential != nil {
		if override.Credential.Clear == (override.Credential.Ref != nil) {
			return fmt.Errorf("credential override must select exactly clear or ref")
		}
		selection.Credential = nil
		selection.Origins.Credential = origin
		if override.Credential.Ref != nil {
			if err := override.Credential.Ref.Validate(); err != nil {
				return fmt.Errorf("credential is invalid: %w", err)
			}
			ref := *override.Credential.Ref
			selection.Credential = &ref
		}
	}
	return nil
}

func resolvedExecutionSelectionOverrideHasAny(value ResolvedExecutionSelectionOverride) bool {
	return value.ModelPolicy != nil || value.LLMGateway != nil || value.Credential != nil
}

func escalationExecutionConfigOrigin(
	stageName string,
	outcome string,
	variant ResolvedEscalationExecutionConfig,
) string {
	if variant.Ref != nil {
		return "executionConfig." + variant.Ref.ConfigID + "@" + variant.Ref.Version
	}
	return "workflow.stages." + stageName + ".on." + outcome + ".escalate.executionConfig"
}

func (l *loader) applyExecutionConfigPatch(
	workflow *ResolvedWorkflow,
	patch ExecutionConfigPatch,
	originPrefix string,
) error {
	plannerApplied := false
	for _, stageName := range sortedPatchKeys(workflow.Stages) {
		stage := workflow.Stages[stageName]
		if patch.Planner != nil && stage.Planner.PlannerID+"@"+stage.Planner.Version != "passthrough@1" {
			if err := l.applyPlannerSelection(
				&stage, *patch.Planner, originPrefix+".planner",
			); err != nil {
				return fmt.Errorf("Stage %q: %w", stageName, err)
			}
			plannerApplied = true
		}
		if patch.Workers != nil {
			for _, logicalName := range sortedPatchKeys(stage.ExecutionConfig.Agents) {
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

	for _, stageName := range sortedPatchKeys(patch.Stages) {
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
		for _, logicalName := range sortedPatchKeys(stagePatch.Agents) {
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
	stage *ResolvedStage,
	patch ExecutionSelectionPatch,
	origin string,
) error {
	ref := stage.Planner.PlannerID + "@" + stage.Planner.Version
	if ref == "passthrough@1" {
		return fmt.Errorf("passthrough@1 does not accept Planner model configuration")
	}
	if stage.ExecutionConfig.Planner == nil {
		stage.ExecutionConfig.Planner = &ResolvedConsumerExecutionConfig{}
	}
	return l.applySelection(stage.ExecutionConfig.Planner, patch, origin)
}

func (l *loader) applySelection(
	selection *ResolvedConsumerExecutionConfig,
	patch ExecutionSelectionPatch,
	origin string,
) error {
	if patch.modelPolicy.present {
		selector, err := ParseSelector(patch.modelPolicy.value)
		if err != nil {
			return fmt.Errorf("modelPolicy: %w", err)
		}
		policy, ok := l.policies[selector.String()]
		if !ok {
			return fmt.Errorf("modelPolicy selects unknown ModelPolicy %q", selector)
		}
		selection.ModelPolicy = cloneModelPolicy(policy)
		selection.Origins.ModelPolicy = origin
	}
	if patch.llmGateway.present {
		selector, err := ParseSelector(patch.llmGateway.value)
		if err != nil {
			return fmt.Errorf("llmGateway: %w", err)
		}
		gateway, ok := l.gateways[selector.String()]
		if !ok {
			return fmt.Errorf("llmGateway selects unknown LLMGatewayConfig %q", selector)
		}
		selection.LLMGateway = cloneLLMGatewayConfig(gateway)
		selection.Origins.LLMGateway = origin
	}
	if patch.credential.present {
		selection.Origins.Credential = origin
		selection.Credential = nil
		if !patch.credential.null {
			ref := contracts.LLMCredentialRef{CredentialID: patch.credential.value}
			if err := ref.Validate(); err != nil {
				return fmt.Errorf("credential is invalid: %w", err)
			}
			selection.Credential = &ref
		}
	}
	return nil
}

func executionConfigPatchFromYAML(source *workflowExecutionConfigSource) (ExecutionConfigPatch, error) {
	if source == nil {
		return ExecutionConfigPatch{}, nil
	}
	result := ExecutionConfigPatch{Stages: make(map[string]StageExecutionConfigPatch, len(source.Stages))}
	var err error
	if source.Planner != nil {
		value, parseErr := executionSelectionFromYAML(*source.Planner, false, "planner")
		if parseErr != nil {
			return ExecutionConfigPatch{}, parseErr
		}
		result.Planner = &value
	}
	if source.Workers != nil {
		value, parseErr := executionSelectionFromYAML(*source.Workers, false, "workers")
		if parseErr != nil {
			return ExecutionConfigPatch{}, parseErr
		}
		result.Workers = &value
	}
	for stageName, sourceStage := range source.Stages {
		if err = validateMapKey("executionConfig Stage name", stageName); err != nil {
			return ExecutionConfigPatch{}, err
		}
		stage := StageExecutionConfigPatch{Agents: make(map[string]ExecutionSelectionPatch, len(sourceStage.Agents))}
		if sourceStage.Planner != nil {
			value, parseErr := executionSelectionFromYAML(
				*sourceStage.Planner, false, "stages."+stageName+".planner",
			)
			if parseErr != nil {
				return ExecutionConfigPatch{}, parseErr
			}
			stage.Planner = &value
		}
		for logicalName, sourceSelection := range sourceStage.Agents {
			if err = validateMapKey("executionConfig logical Agent name", logicalName); err != nil {
				return ExecutionConfigPatch{}, err
			}
			value, parseErr := executionSelectionFromYAML(
				sourceSelection, false, "stages."+stageName+".agents."+logicalName,
			)
			if parseErr != nil {
				return ExecutionConfigPatch{}, parseErr
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
) (ExecutionSelectionPatch, error) {
	modelPolicy, err := optionalStringFromYAML(source.ModelPolicy, false, field+".modelPolicy")
	if err != nil {
		return ExecutionSelectionPatch{}, err
	}
	llmGateway, err := optionalStringFromYAML(source.LLMGateway, false, field+".llmGateway")
	if err != nil {
		return ExecutionSelectionPatch{}, err
	}
	credential, err := optionalStringFromYAML(source.Credential, allowCredentialNull, field+".credential")
	if err != nil {
		return ExecutionSelectionPatch{}, err
	}
	result := ExecutionSelectionPatch{
		modelPolicy: modelPolicy, llmGateway: llmGateway, credential: credential,
	}
	if !result.hasAny() {
		return ExecutionSelectionPatch{}, fmt.Errorf("%s must select at least one field", field)
	}
	return result, nil
}

func optionalStringFromYAML(
	node yaml.Node,
	allowNull bool,
	field string,
) (optionalString, error) {
	if node.IsZero() {
		return optionalString{}, nil
	}
	if node.Kind != yaml.ScalarNode {
		return optionalString{}, fmt.Errorf("%s must be a string", field)
	}
	if node.ShortTag() == "!!null" {
		if !allowNull {
			return optionalString{}, fmt.Errorf("%s must not be null", field)
		}
		return optionalString{present: true, null: true}, nil
	}
	if node.ShortTag() != "!!str" {
		return optionalString{}, fmt.Errorf("%s must be a string", field)
	}
	return optionalString{present: true, value: node.Value}, nil
}

func validateWorkflowExecutionConfigs(workflow ResolvedWorkflow) error {
	for stageName, stage := range workflow.Stages {
		if err := validateStageExecutionConfig(stageName, stage); err != nil {
			return err
		}
	}
	return nil
}

func validateStageExecutionConfig(stageName string, stage ResolvedStage) error {
	plannerRef := stage.Planner.PlannerID + "@" + stage.Planner.Version
	if plannerRef == "passthrough@1" {
		if stage.ExecutionConfig.Planner != nil {
			return fmt.Errorf("Stage %q passthrough@1 must not have Planner executionConfig", stageName)
		}
	} else {
		if stage.ExecutionConfig.Planner == nil {
			return fmt.Errorf("Stage %q modeled Planner has no executionConfig", stageName)
		}
		if err := validateConsumerExecutionConfig(*stage.ExecutionConfig.Planner, true, false); err != nil {
			return fmt.Errorf("Stage %q Planner executionConfig: %w", stageName, err)
		}
	}
	if len(stage.ExecutionConfig.Agents) != len(stage.Agents) {
		return fmt.Errorf("Stage %q executionConfig Agent set is incomplete", stageName)
	}
	for logicalName, binding := range stage.Agents {
		selection, ok := stage.ExecutionConfig.Agents[logicalName]
		if !ok {
			return fmt.Errorf("Stage %q Agent %q has no executionConfig", stageName, logicalName)
		}
		hasTools := len(binding.Template.Toolsets) > 0
		if err := validateConsumerExecutionConfig(selection, false, hasTools); err != nil {
			return fmt.Errorf("Stage %q Agent %q executionConfig: %w", stageName, logicalName, err)
		}
	}
	return nil
}

func validateConsumerExecutionConfig(
	selection ResolvedConsumerExecutionConfig,
	planner bool,
	hasTools bool,
) error {
	var err error
	if planner {
		err = selection.ModelPolicy.ValidateForPlanner()
	} else {
		err = selection.ModelPolicy.ValidateForWorker(hasTools)
	}
	if err != nil {
		return err
	}
	if err := selection.LLMGateway.Validate(); err != nil {
		return err
	}
	if strings.TrimSpace(selection.Origins.ModelPolicy) == "" ||
		strings.TrimSpace(selection.Origins.LLMGateway) == "" {
		return fmt.Errorf("modelPolicy and llmGateway origins are required")
	}
	if selection.Credential != nil {
		if err := selection.Credential.Validate(); err != nil {
			return err
		}
		if strings.TrimSpace(selection.Origins.Credential) == "" {
			return fmt.Errorf("credential origin is required")
		}
	}
	return nil
}

func validateRunCredentials(
	ctx context.Context,
	workflow ResolvedWorkflow,
	lookup CredentialLookup,
) error {
	for stageName, stage := range workflow.Stages {
		configs := map[string]ResolvedStageExecutionConfig{"base": stage.ExecutionConfig}
		for outcome, action := range map[string]TransitionAction{
			"failed escalation":      stage.On.Failed,
			"interrupted escalation": stage.On.Interrupted,
		} {
			if action.Kind == TransitionEscalate && action.Escalate != nil {
				configs[outcome] = action.Escalate.ExecutionConfig.Effective
			}
		}
		for _, variant := range sortedPatchKeys(configs) {
			if err := validateStageExecutionConfigCredentials(
				ctx, stageName, variant, configs[variant], lookup,
			); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateStageExecutionConfigCredentials(
	ctx context.Context,
	stageName string,
	variant string,
	config ResolvedStageExecutionConfig,
	lookup CredentialLookup,
) error {
	selections := make(map[string]ResolvedConsumerExecutionConfig, len(config.Agents)+1)
	if config.Planner != nil {
		selections["planner"] = *config.Planner
	}
	for logicalName, selection := range config.Agents {
		selections["agent "+logicalName] = selection
	}
	for _, consumer := range sortedPatchKeys(selections) {
		selection := selections[consumer]
		if selection.Credential == nil {
			continue
		}
		if lookup == nil {
			return fmt.Errorf(
				"Stage %q %s %s selects unavailable credential %q",
				stageName, variant, consumer, selection.Credential.CredentialID,
			)
		}
		metadata, err := lookup.LookupLLMCredential(ctx, selection.Credential.CredentialID)
		if err != nil {
			return fmt.Errorf("Stage %q %s %s credential is unavailable", stageName, variant, consumer)
		}
		if metadata.Ref != *selection.Credential || metadata.LLMGateway != selection.LLMGateway.Ref {
			return fmt.Errorf(
				"Stage %q %s %s credential is bound to another LLMGatewayConfig",
				stageName, variant, consumer,
			)
		}
	}
	return nil
}
