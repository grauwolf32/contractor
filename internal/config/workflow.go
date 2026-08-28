package config

import (
	"fmt"
	"strings"
)

func (l *loader) resolveWorkflow(selector Selector, spec *workflowSpecSource) (ResolvedWorkflow, error) {
	if spec == nil {
		return ResolvedWorkflow{}, fmt.Errorf("spec is required")
	}
	parameters, err := resolveParameters(spec.Parameters)
	if err != nil {
		return ResolvedWorkflow{}, err
	}
	inputs, err := resolveArtifactSlots("spec.inputs", spec.Inputs)
	if err != nil {
		return ResolvedWorkflow{}, err
	}
	outputs, err := resolveArtifactSlots("spec.outputs", spec.Outputs)
	if err != nil {
		return ResolvedWorkflow{}, err
	}
	if spec.Stages == nil || len(*spec.Stages) == 0 {
		return ResolvedWorkflow{}, fmt.Errorf("spec.stages must be a non-empty mapping")
	}
	if err := validateMapKey("spec.entryStage", spec.EntryStage); err != nil {
		return ResolvedWorkflow{}, err
	}
	if _, ok := (*spec.Stages)[spec.EntryStage]; !ok {
		return ResolvedWorkflow{}, fmt.Errorf("spec.entryStage %q does not name a Stage", spec.EntryStage)
	}

	stages := make(map[string]ResolvedStage, len(*spec.Stages))
	for name, source := range *spec.Stages {
		if err := validateMapKey("Stage name", name); err != nil {
			return ResolvedWorkflow{}, err
		}
		stage, stageErr := l.resolveStage(name, source, outputs, *spec.Stages)
		if stageErr != nil {
			return ResolvedWorkflow{}, fmt.Errorf("spec.stages.%s: %w", name, stageErr)
		}
		stages[name] = stage
	}

	return ResolvedWorkflow{
		Ref:        WorkflowRef{Name: selector.ID, Version: selector.Version},
		Parameters: parameters,
		Inputs:     inputs,
		Outputs:    outputs,
		EntryStage: spec.EntryStage,
		Stages:     stages,
	}, nil
}

func resolveParameters(source *map[string]parameterSlotSource) (map[string]ParameterSlot, error) {
	if source == nil {
		return nil, fmt.Errorf("spec.parameters is required (use {} for none)")
	}
	result := make(map[string]ParameterSlot, len(*source))
	for name, slot := range *source {
		if err := validateMapKey("parameter name", name); err != nil {
			return nil, err
		}
		if slot.Required == nil {
			return nil, fmt.Errorf("spec.parameters.%s.required is required", name)
		}
		result[name] = ParameterSlot{Required: *slot.Required}
	}
	return result, nil
}

func resolveArtifactSlots(field string, source *map[string]artifactSlotSource) (map[string]ArtifactSlot, error) {
	if source == nil {
		return nil, fmt.Errorf("%s is required (use {} for none)", field)
	}
	return resolveArtifactSlotMap(field, *source)
}

func resolveArtifactSlotMap(field string, source map[string]artifactSlotSource) (map[string]ArtifactSlot, error) {
	result := make(map[string]ArtifactSlot, len(source))
	for name, slot := range source {
		if err := validateMapKey(field+" slot name", name); err != nil {
			return nil, err
		}
		if slot.Required == nil {
			return nil, fmt.Errorf("%s.%s.required is required", field, name)
		}
		mediaTypes, err := validateMediaTypes(field+"."+name, slot.MediaTypes)
		if err != nil {
			return nil, err
		}
		result[name] = ArtifactSlot{Required: *slot.Required, MediaTypes: mediaTypes}
	}
	return result, nil
}

func (l *loader) resolveStage(
	stageName string,
	source stageSource,
	workflowOutputs map[string]ArtifactSlot,
	allStages map[string]stageSource,
) (ResolvedStage, error) {
	if strings.TrimSpace(source.Objective) == "" {
		return ResolvedStage{}, fmt.Errorf("objective must not be empty or whitespace-only")
	}
	instructions, err := l.resolveInstructions(source.Instructions)
	if err != nil {
		return ResolvedStage{}, fmt.Errorf("instructions: %w", err)
	}
	planner, err := ParseSelector(source.Planner)
	if err != nil {
		return ResolvedStage{}, fmt.Errorf("planner: %w", err)
	}
	if _, ok := l.descriptors.PlannerFactories[planner.String()]; !ok {
		return ResolvedStage{}, fmt.Errorf("planner selects unknown PlannerFactory %q", planner)
	}
	agents, err := l.resolveAgentBindings(source.Agents)
	if err != nil {
		return ResolvedStage{}, err
	}
	context, err := resolveStageContext(source.Context)
	if err != nil {
		return ResolvedStage{}, err
	}
	result, err := resolveStageResult(source.Result)
	if err != nil {
		return ResolvedStage{}, err
	}
	mappings, err := resolveWorkflowOutputMappings(stageName, source.WorkflowOutputs, workflowOutputs, result.Artifacts)
	if err != nil {
		return ResolvedStage{}, err
	}
	transitions, err := resolveTransitions(source.On, allStages)
	if err != nil {
		return ResolvedStage{}, err
	}

	return ResolvedStage{
		Objective:       source.Objective,
		Instructions:    instructions,
		Planner:         PlannerRef{PlannerID: planner.ID, Version: planner.Version},
		Agents:          agents,
		Context:         context,
		Result:          result,
		WorkflowOutputs: mappings,
		On:              transitions,
	}, nil
}

func (l *loader) resolveAgentBindings(source map[string]agentBindingSource) (map[string]ResolvedAgentBinding, error) {
	if len(source) == 0 {
		return nil, fmt.Errorf("agents must be a non-empty mapping")
	}
	result := make(map[string]ResolvedAgentBinding, len(source))
	for logicalName, binding := range source {
		if err := validateMapKey("logical Agent name", logicalName); err != nil {
			return nil, err
		}
		selector, err := ParseSelector(binding.Template)
		if err != nil {
			return nil, fmt.Errorf("agents.%s.template: %w", logicalName, err)
		}
		template, ok := l.templates[selector.String()]
		if !ok {
			return nil, fmt.Errorf("agents.%s.template selects unknown AgentTemplate %q", logicalName, selector)
		}
		namespace := logicalName
		if binding.Namespace != nil {
			namespace = *binding.Namespace
		}
		if err := validateArtifactComponent("agents."+logicalName+".namespace", namespace); err != nil {
			return nil, err
		}
		if namespace == "inputs" || namespace == "outputs" {
			return nil, fmt.Errorf("agents.%s.namespace %q is reserved", logicalName, namespace)
		}
		result[logicalName] = ResolvedAgentBinding{
			Template:  cloneAgentTemplate(template),
			Namespace: namespace,
		}
	}
	return result, nil
}

func resolveStageContext(source *stageContextSource) (StageContext, error) {
	result := StageContext{Artifacts: make(map[string]ContextArtifact)}
	if source == nil {
		return result, nil
	}
	if source.Artifacts == nil {
		return StageContext{}, fmt.Errorf("context.artifacts is required when context is present")
	}
	for localName, artifact := range *source.Artifacts {
		if err := validateMapKey("context artifact name", localName); err != nil {
			return StageContext{}, err
		}
		if err := validateArtifactComponent("context.artifacts."+localName+".namespace", artifact.Namespace); err != nil {
			return StageContext{}, err
		}
		if err := validateArtifactComponent("context.artifacts."+localName+".name", artifact.Name); err != nil {
			return StageContext{}, err
		}
		if artifact.Required == nil {
			return StageContext{}, fmt.Errorf("context.artifacts.%s.required is required", localName)
		}
		result.Artifacts[localName] = ContextArtifact{
			Namespace: artifact.Namespace,
			Name:      artifact.Name,
			Required:  *artifact.Required,
		}
	}
	return result, nil
}

func resolveStageResult(source *stageResultSource) (StageResultContract, error) {
	result := StageResultContract{Artifacts: make(map[string]ArtifactSlot)}
	if source == nil {
		return result, nil
	}
	if source.Artifacts == nil {
		return StageResultContract{}, fmt.Errorf("result.artifacts is required when result is present")
	}
	artifacts, err := resolveArtifactSlotMap("result.artifacts", *source.Artifacts)
	if err != nil {
		return StageResultContract{}, err
	}
	result.Artifacts = artifacts
	return result, nil
}

func resolveWorkflowOutputMappings(
	stageName string,
	source map[string]string,
	workflowOutputs map[string]ArtifactSlot,
	stageResults map[string]ArtifactSlot,
) (map[string]string, error) {
	result := make(map[string]string, len(source))
	for outputName, resultName := range source {
		outputSlot, ok := workflowOutputs[outputName]
		if !ok {
			return nil, fmt.Errorf("workflowOutputs key %q is not a declared Workflow output", outputName)
		}
		resultSlot, ok := stageResults[resultName]
		if !ok {
			return nil, fmt.Errorf("workflowOutputs.%s value %q is not a result artifact of Stage %q", outputName, resultName, stageName)
		}
		if !mediaTypesIntersect(outputSlot.MediaTypes, resultSlot.MediaTypes) {
			return nil, fmt.Errorf("workflowOutputs.%s maps incompatible result and output media types", outputName)
		}
		result[outputName] = resultName
	}
	return result, nil
}

func resolveTransitions(source *stageTransitionsSource, stages map[string]stageSource) (StageTransitions, error) {
	if source == nil || source.Succeeded == nil || source.Failed == nil || source.Interrupted == nil {
		return StageTransitions{}, fmt.Errorf("on must contain succeeded, failed, and interrupted")
	}
	succeeded, err := resolveTransitionAction("on.succeeded", source.Succeeded, stages, map[TransitionKind]bool{
		TransitionNext: true, TransitionSucceed: true,
	})
	if err != nil {
		return StageTransitions{}, err
	}
	failed, err := resolveTransitionAction("on.failed", source.Failed, stages, map[TransitionKind]bool{
		TransitionNext: true, TransitionRetry: true, TransitionFail: true,
	})
	if err != nil {
		return StageTransitions{}, err
	}
	interrupted, err := resolveTransitionAction("on.interrupted", source.Interrupted, stages, map[TransitionKind]bool{
		TransitionNext: true, TransitionRetry: true, TransitionFail: true,
	})
	if err != nil {
		return StageTransitions{}, err
	}
	return StageTransitions{Succeeded: succeeded, Failed: failed, Interrupted: interrupted}, nil
}

func resolveTransitionAction(
	field string,
	source *transitionActionSource,
	stages map[string]stageSource,
	allowed map[TransitionKind]bool,
) (TransitionAction, error) {
	if source == nil {
		return TransitionAction{}, fmt.Errorf("%s is required", field)
	}
	count := 0
	if source.Next != nil {
		count++
	}
	if source.Retry != nil {
		count++
	}
	if source.Succeed != nil {
		count++
	}
	if source.Fail != nil {
		count++
	}
	if count != 1 {
		return TransitionAction{}, fmt.Errorf("%s must select exactly one of next, retry, succeed, or fail", field)
	}

	var action TransitionAction
	switch {
	case source.Next != nil:
		action = TransitionAction{Kind: TransitionNext, NextStage: *source.Next}
		if err := validateMapKey(field+".next", action.NextStage); err != nil {
			return TransitionAction{}, err
		}
		if _, ok := stages[action.NextStage]; !ok {
			return TransitionAction{}, fmt.Errorf("%s.next names unknown Stage %q", field, action.NextStage)
		}
	case source.Retry != nil:
		if source.Retry.MaxAttempts < 2 {
			return TransitionAction{}, fmt.Errorf("%s.retry.maxAttempts must be at least 2", field)
		}
		then, err := resolveTransitionAction(field+".retry.then", source.Retry.Then, stages, map[TransitionKind]bool{
			TransitionNext: true, TransitionFail: true,
		})
		if err != nil {
			return TransitionAction{}, err
		}
		action = TransitionAction{Kind: TransitionRetry, Retry: &RetryTransition{MaxAttempts: source.Retry.MaxAttempts, Then: then}}
	case source.Succeed != nil:
		action = TransitionAction{Kind: TransitionSucceed}
	case source.Fail != nil:
		action = TransitionAction{Kind: TransitionFail}
	}
	if !allowed[action.Kind] {
		return TransitionAction{}, fmt.Errorf("%s action %q is not allowed", field, action.Kind)
	}
	return action, nil
}

func validateMapKey(field, value string) error {
	if err := validateIdentifier(field, value); err != nil {
		return err
	}
	if strings.Contains(value, "/") {
		return fmt.Errorf("%s %q must not contain slash", field, value)
	}
	return nil
}

func validateArtifactComponent(field, value string) error {
	return validateMapKey(field, value)
}
