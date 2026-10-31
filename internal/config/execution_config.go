package config

import (
	"context"
	"fmt"
	"maps"
	"slices"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"go.yaml.in/yaml/v4"
)

const (
	originWorkflowPrefix = "workflow.executionConfig"
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
		// resolveStage pins only a compact@1 Stage's compactor policy here.
		base := resolved.ResolvedStageExecutionConfig{
			Planner: stage.ExecutionConfig.Planner,
			Agents:  make(map[string]resolved.ResolvedConsumerExecutionConfig, len(stage.Agents)),
		}
		for logicalName := range stage.Agents {
			base.Agents[logicalName] = resolved.ResolvedConsumerExecutionConfig{}
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

// ResolveRunWorkflow captures deployment choices in a caller-owned clone and
// checks every selected credential against current active metadata.
// A credential lookup that cannot confirm presence or absence returns an error
// classified as ErrCredentialLookup; every other error is a validation failure.
func (s *Snapshot) ResolveRunWorkflow(
	ctx context.Context,
	raw string,
	credentials resolved.CredentialLookup,
) (resolved.ResolvedWorkflow, error) {
	return s.resolveRunWorkflow(ctx, raw, credentials, nil)
}

func (s *Snapshot) resolveRunWorkflow(ctx context.Context, raw string, credentials resolved.CredentialLookup, defaults map[string]contracts.ResolvedModelPolicy) (resolved.ResolvedWorkflow, error) {
	workflow, err := s.Workflow(raw)
	if err != nil {
		return resolved.ResolvedWorkflow{}, err
	}
	applyModelDefaults(&workflow, defaults)
	if err := resolved.ValidateWorkflowExecutionConfigs(workflow); err != nil {
		return resolved.ResolvedWorkflow{}, err
	}
	if err := resolved.ResolveEscalationVariants(&workflow); err != nil {
		return resolved.ResolvedWorkflow{}, err
	}
	if err := resolved.ValidateWorkflowGraph(workflow); err != nil {
		return resolved.ResolvedWorkflow{}, err
	}
	// Deployment model defaults can enlarge the catalog's snapshot; the
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
	patch executionConfigPatch,
	originPrefix string,
) error {
	plannerApplied, compactorApplied := false, false
	for _, stageName := range slices.Sorted(maps.Keys(workflow.Stages)) {
		stage := workflow.Stages[stageName]
		if patch.Planner != nil && !resolved.IsModelFreeStage(stage) {
			// A Workflow-wide compactor selection applies to every
			// compact@1 Stage and is no selection for the others (PLN-R125).
			selection := *patch.Planner
			if !resolved.PlannerCompactsContext(stage) {
				selection.CompactorModelPolicy = optionalSelectionString{}
			}
			if selection.HasAny() {
				if err := l.applyPlannerSelection(
					&stage, selection, originPrefix+".planner",
				); err != nil {
					return fmt.Errorf("Stage %q: %w", stageName, err)
				}
			}
			plannerApplied = true
			compactorApplied = compactorApplied || selection.CompactorModelPolicy.Present
		}
		if patch.Workers != nil {
			for _, logicalName := range slices.Sorted(maps.Keys(stage.ExecutionConfig.Agents)) {
				if stage.Agents[logicalName].Template.IsToolWorker() {
					return fmt.Errorf("tool@1 does not accept Worker model defaults")
				}
				selection := stage.ExecutionConfig.Agents[logicalName]
				if err := l.applySelection(
					&selection, *patch.Workers, originPrefix+".workers", false,
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
	if patch.Planner != nil && patch.Planner.CompactorModelPolicy.Present && !compactorApplied {
		return fmt.Errorf("planner compactorModelPolicy does not apply to any compact@1 Stage")
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
				&selection, stagePatch.Agents[logicalName], stageOrigin+".agents."+logicalName, false,
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
	patch executionSelectionPatch,
	origin string,
) error {
	ref := stage.Planner.Selector()
	if resolved.IsModelFreeStage(*stage) {
		return fmt.Errorf("%s does not accept Planner model configuration", ref)
	}
	if patch.CompactorModelPolicy.Present && !resolved.PlannerCompactsContext(*stage) {
		return fmt.Errorf("compactorModelPolicy requires plannerContextManagement %s", resolved.PlannerContextCompact)
	}
	if stage.ExecutionConfig.Planner == nil {
		stage.ExecutionConfig.Planner = &resolved.ResolvedConsumerExecutionConfig{}
	}
	return l.applySelection(stage.ExecutionConfig.Planner, patch, origin, true)
}

func (l *loader) applySelection(
	selection *resolved.ResolvedConsumerExecutionConfig,
	patch executionSelectionPatch,
	origin string,
	planner bool,
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
	if patch.SummarizerModelPolicy.Present {
		selector, err := resolved.ParseSelector(patch.SummarizerModelPolicy.Value)
		if err != nil {
			return fmt.Errorf("summarizerModelPolicy: %w", err)
		}
		policy, ok := l.policies[selector.String()]
		if !ok {
			return fmt.Errorf("summarizerModelPolicy selects unknown ModelPolicy %q", selector)
		}
		if err := policy.ValidateForWorkerSummarizer(); err != nil {
			return err
		}
		pinned := policy.Clone()
		selection.SummarizerModelPolicy = &pinned
		selection.Origins.SummarizerModelPolicy = origin
	}
	if patch.CompactorModelPolicy.Present {
		if !planner {
			return fmt.Errorf("compactorModelPolicy applies only to a Planner")
		}
		policy, err := l.compactorPolicy(patch.CompactorModelPolicy.Value)
		if err != nil {
			return err
		}
		selection.CompactorModelPolicy = &policy
		selection.Origins.CompactorModelPolicy = origin
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

func executionConfigPatchFromYAML(source *workflowExecutionConfigSource) (executionConfigPatch, error) {
	if source == nil {
		return executionConfigPatch{}, nil
	}
	result := executionConfigPatch{Stages: make(map[string]stageExecutionConfigPatch, len(source.Stages))}
	var err error
	if source.Planner != nil {
		value, parseErr := executionSelectionFromYAML(*source.Planner, false, "planner")
		if parseErr != nil {
			return executionConfigPatch{}, parseErr
		}
		result.Planner = &value
	}
	if source.Workers != nil {
		value, parseErr := executionSelectionFromYAML(*source.Workers, false, "workers")
		if parseErr != nil {
			return executionConfigPatch{}, parseErr
		}
		result.Workers = &value
	}
	for stageName, sourceStage := range source.Stages {
		if err = resolved.ValidateWorkflowName("executionConfig Stage name", stageName); err != nil {
			return executionConfigPatch{}, err
		}
		stage := stageExecutionConfigPatch{Agents: make(map[string]executionSelectionPatch, len(sourceStage.Agents))}
		if sourceStage.Planner != nil {
			value, parseErr := executionSelectionFromYAML(
				*sourceStage.Planner, false, "stages."+stageName+".planner",
			)
			if parseErr != nil {
				return executionConfigPatch{}, parseErr
			}
			stage.Planner = &value
		}
		for logicalName, sourceSelection := range sourceStage.Agents {
			if err = resolved.ValidateWorkflowName("executionConfig logical Agent name", logicalName); err != nil {
				return executionConfigPatch{}, err
			}
			value, parseErr := executionSelectionFromYAML(
				sourceSelection, false, "stages."+stageName+".agents."+logicalName,
			)
			if parseErr != nil {
				return executionConfigPatch{}, parseErr
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
) (executionSelectionPatch, error) {
	modelPolicy, err := optionalStringFromYAML(source.ModelPolicy, false, field+".modelPolicy")
	if err != nil {
		return executionSelectionPatch{}, err
	}
	summaryPolicy, err := optionalStringFromYAML(source.SummarizerModelPolicy, false, field+".summarizerModelPolicy")
	if err != nil {
		return executionSelectionPatch{}, err
	}
	compactorPolicy, err := optionalStringFromYAML(source.CompactorModelPolicy, false, field+".compactorModelPolicy")
	if err != nil {
		return executionSelectionPatch{}, err
	}
	llmGateway, err := optionalStringFromYAML(source.LLMGateway, false, field+".llmGateway")
	if err != nil {
		return executionSelectionPatch{}, err
	}
	credential, err := optionalStringFromYAML(source.Credential, allowCredentialNull, field+".credential")
	if err != nil {
		return executionSelectionPatch{}, err
	}
	result := executionSelectionPatch{
		ModelPolicy: modelPolicy, SummarizerModelPolicy: summaryPolicy, CompactorModelPolicy: compactorPolicy,
		LLMGateway: llmGateway, Credential: credential,
	}
	if !result.HasAny() {
		return executionSelectionPatch{}, fmt.Errorf("%s must select at least one field", field)
	}
	return result, nil
}

func optionalStringFromYAML(
	node yaml.Node,
	allowNull bool,
	field string,
) (optionalSelectionString, error) {
	if node.IsZero() {
		return optionalSelectionString{}, nil
	}
	if node.Kind != yaml.ScalarNode {
		return optionalSelectionString{}, fmt.Errorf("%s must be a string", field)
	}
	if node.ShortTag() == "!!null" {
		if !allowNull {
			return optionalSelectionString{}, fmt.Errorf("%s must not be null", field)
		}
		return optionalSelectionString{Present: true, Null: true}, nil
	}
	if node.ShortTag() != "!!str" {
		return optionalSelectionString{}, fmt.Errorf("%s must be a string", field)
	}
	return optionalSelectionString{Present: true, Value: node.Value}, nil
}

// compactorPolicy resolves the exact compactor ModelPolicy a compact@1 Stage
// or a Planner execution selection names (TPL-R117).
func (l *loader) compactorPolicy(raw string) (contracts.ResolvedModelPolicy, error) {
	selector, err := resolved.ParseSelector(raw)
	if err != nil {
		return contracts.ResolvedModelPolicy{}, fmt.Errorf("compactorModelPolicy: %w", err)
	}
	policy, ok := l.policies[selector.String()]
	if !ok {
		return contracts.ResolvedModelPolicy{}, fmt.Errorf("compactorModelPolicy selects unknown ModelPolicy %q", selector)
	}
	if err := policy.ValidateForCompactor(); err != nil {
		return contracts.ResolvedModelPolicy{}, err
	}
	return policy.Clone(), nil
}
