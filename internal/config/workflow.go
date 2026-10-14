package config

import (
	"errors"
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"go.yaml.in/yaml/v4"
)

func (l *loader) resolveWorkflow(selector resolved.Selector, spec *workflowSpecSource) (resolved.ResolvedWorkflow, error) {
	if spec == nil {
		return resolved.ResolvedWorkflow{}, fmt.Errorf("spec is required")
	}
	presentation, err := resolveWorkflowPresentation(spec.Presentation)
	if err != nil {
		return resolved.ResolvedWorkflow{}, err
	}
	parameters, err := resolveParameters(spec.Parameters)
	if err != nil {
		return resolved.ResolvedWorkflow{}, err
	}
	inputs, err := resolveArtifactSlots("spec.inputs", spec.Inputs, false)
	if err != nil {
		return resolved.ResolvedWorkflow{}, err
	}
	outputs, err := resolveArtifactSlots("spec.outputs", spec.Outputs, true)
	if err != nil {
		return resolved.ResolvedWorkflow{}, err
	}
	if spec.Stages == nil || len(*spec.Stages) == 0 {
		return resolved.ResolvedWorkflow{}, fmt.Errorf("spec.stages must be a non-empty mapping")
	}
	if err := resolved.ValidateWorkflowName("spec.entryStage", spec.EntryStage); err != nil {
		return resolved.ResolvedWorkflow{}, err
	}
	if _, ok := (*spec.Stages)[spec.EntryStage]; !ok {
		return resolved.ResolvedWorkflow{}, fmt.Errorf("spec.entryStage %q does not name a Stage", spec.EntryStage)
	}

	stages := make(map[string]resolved.ResolvedStage, len(*spec.Stages))
	for name, source := range *spec.Stages {
		if err := resolved.ValidateWorkflowName("Stage name", name); err != nil {
			return resolved.ResolvedWorkflow{}, err
		}
		stage, stageErr := l.resolveStage(name, source, outputs, *spec.Stages)
		if stageErr != nil {
			return resolved.ResolvedWorkflow{}, fmt.Errorf("spec.stages.%s: %w", name, stageErr)
		}
		stages[name] = stage
	}
	// As for Stages, the decoded auditTask belongs to this load alone.
	workflow := resolved.ResolvedWorkflow{
		Ref:          resolved.WorkflowRef{Name: selector.ID, Version: selector.Version},
		AuditTask:    spec.AuditTask,
		Presentation: presentation,
		Parameters:   parameters,
		Inputs:       inputs,
		Outputs:      outputs,
		EntryStage:   spec.EntryStage,
		Stages:       stages,
	}
	if err := l.resolveWorkflowExecutionConfig(&workflow, spec.ExecutionConfig); err != nil {
		return resolved.ResolvedWorkflow{}, err
	}
	if err := resolved.ValidateWorkflowGraph(workflow); err != nil {
		return resolved.ResolvedWorkflow{}, err
	}
	if _, err := resolved.WorkflowSkillRefs(workflow); err != nil {
		return resolved.ResolvedWorkflow{}, err
	}
	// A Run without execution overrides stores exactly this snapshot, so it
	// must fit its bound (CAT-R76); ResolveRunWorkflow measures overrides.
	if _, err := resolved.EncodeResolvedWorkflowSnapshot(workflow); err != nil {
		return resolved.ResolvedWorkflow{}, err
	}
	return workflow, nil
}

func resolveWorkflowPresentation(source *workflowPresentationSource) (*resolved.WorkflowPresentation, error) {
	if source == nil {
		return nil, nil
	}
	for _, field := range []struct {
		name    string
		value   string
		maximum int
	}{
		{name: "displayName", value: source.DisplayName, maximum: 160},
		{name: "description", value: source.Description, maximum: 2000},
	} {
		if !utf8.ValidString(field.value) || strings.TrimSpace(field.value) == "" {
			return nil, fmt.Errorf("spec.presentation.%s must be non-empty UTF-8", field.name)
		}
		if utf8.RuneCountInString(field.value) > field.maximum {
			return nil, fmt.Errorf(
				"spec.presentation.%s must contain at most %d Unicode characters",
				field.name, field.maximum,
			)
		}
	}
	return &resolved.WorkflowPresentation{
		DisplayName: source.DisplayName,
		Description: source.Description,
	}, nil
}

func resolveParameters(source *map[string]parameterSlotSource) (map[string]resolved.ParameterSlot, error) {
	if source == nil {
		return nil, fmt.Errorf("spec.parameters is required (use {} for none)")
	}
	result := make(map[string]resolved.ParameterSlot, len(*source))
	for name, slot := range *source {
		if err := resolved.ValidateMapKey("parameter name", name); err != nil {
			return nil, err
		}
		if slot.Required == nil {
			return nil, fmt.Errorf("spec.parameters.%s.required is required", name)
		}
		result[name] = resolved.ParameterSlot{Required: *slot.Required}
	}
	return result, nil
}

func resolveArtifactSlots(
	field string,
	source *map[string]artifactSlotSource,
	allowPrimary bool,
) (map[string]resolved.ArtifactSlot, error) {
	if source == nil {
		return nil, fmt.Errorf("%s is required (use {} for none)", field)
	}
	return resolveArtifactSlotMap(field, *source, false, allowPrimary)
}

func resolveArtifactSlotMap(
	field string,
	source map[string]artifactSlotSource,
	allowFrom bool,
	allowPrimary bool,
) (map[string]resolved.ArtifactSlot, error) {
	result := make(map[string]resolved.ArtifactSlot, len(source))
	for name, slot := range source {
		if err := resolved.ValidateArtifactComponent(field+" slot name", name); err != nil {
			return nil, err
		}
		if slot.Required == nil {
			return nil, fmt.Errorf("%s.%s.required is required", field, name)
		}
		mediaTypes, err := resolved.ValidateMediaTypes(field+"."+name, slot.MediaTypes)
		if err != nil {
			return nil, err
		}
		if slot.From != nil && !allowFrom {
			return nil, fmt.Errorf("%s.%s.from is allowed only for Stage result artifacts", field, name)
		}
		if slot.Primary != nil && !allowPrimary {
			return nil, fmt.Errorf("%s.%s.primary is allowed only for Workflow outputs", field, name)
		}
		var from *resolved.ArtifactBinding
		if slot.From != nil {
			if err := resolved.ValidateArtifactComponent(field+"."+name+".from.namespace", slot.From.Namespace); err != nil {
				return nil, err
			}
			if err := resolved.ValidateArtifactComponent(field+"."+name+".from.name", slot.From.Name); err != nil {
				return nil, err
			}
			if artifactpolicy.IsPurposeReservedNamespace(slot.From.Namespace) ||
				artifactpolicy.IsReservedMemoryBinding(slot.From.Namespace, slot.From.Name) {
				return nil, fmt.Errorf("%s.%s.from identifies a Runtime-reserved binding", field, name)
			}
			from = &resolved.ArtifactBinding{Namespace: slot.From.Namespace, Name: slot.From.Name}
		}
		primary := false
		if slot.Primary != nil {
			primary = *slot.Primary
		}
		result[name] = resolved.ArtifactSlot{
			Required: *slot.Required, MediaTypes: mediaTypes, From: from, Primary: primary,
		}
	}
	return result, nil
}

func (l *loader) resolveStage(
	stageName string,
	source stageSource,
	workflowOutputs map[string]resolved.ArtifactSlot,
	allStages map[string]stageSource,
) (resolved.ResolvedStage, error) {
	if strings.TrimSpace(source.Objective) == "" {
		return resolved.ResolvedStage{}, fmt.Errorf("objective must not be empty or whitespace-only")
	}
	session, err := resolveWorkerSessionMode(source.Session)
	if err != nil {
		return resolved.ResolvedStage{}, fmt.Errorf("session: %w", err)
	}
	instructions, err := l.resolveInstructions(source.Instructions)
	if err != nil {
		return resolved.ResolvedStage{}, fmt.Errorf("instructions: %w", err)
	}
	planner, err := resolved.ParseSelector(source.Planner)
	if err != nil {
		return resolved.ResolvedStage{}, fmt.Errorf("planner: %w", err)
	}
	plannerRef := resolved.PlannerRef{PlannerID: planner.ID, Version: planner.Version}
	if !resolved.KnownPlanner(plannerRef) {
		return resolved.ResolvedStage{}, fmt.Errorf("planner selects unknown PlannerFactory %q", planner)
	}
	agents, err := l.resolveAgentBindings(source.Agents)
	if err != nil {
		return resolved.ResolvedStage{}, err
	}
	if err := resolved.ValidatePlannerAgents(plannerRef, len(agents)); err != nil {
		return resolved.ResolvedStage{}, err
	}
	context, err := resolveStageContext(source.Context)
	if err != nil {
		return resolved.ResolvedStage{}, err
	}
	result, err := resolveStageResult(source.Result)
	if err != nil {
		return resolved.ResolvedStage{}, err
	}
	if err := resolved.ValidateStageWorkspace(context, result, agents); err != nil {
		return resolved.ResolvedStage{}, err
	}
	if plannerRef != (resolved.PlannerRef{PlannerID: "scan-plan", Version: "1"}) {
		if err := validateStageResultBindings(result, context.Workspace, agents); err != nil {
			return resolved.ResolvedStage{}, err
		}
	}
	mappings, err := resolveWorkflowOutputMappings(stageName, source.WorkflowOutputs, workflowOutputs, result.Artifacts)
	if err != nil {
		return resolved.ResolvedStage{}, err
	}
	transitions, err := l.resolveTransitions(source.On, allStages)
	if err != nil {
		return resolved.ResolvedStage{}, err
	}

	// The decoded source belongs to this load alone and the Snapshot keeps
	// deep copies, so the Stage takes the scan policies without copying them.
	stage := resolved.ResolvedStage{
		Objective:       source.Objective,
		Instructions:    instructions,
		Planner:         plannerRef,
		ScanPlan:        source.ScanPlan,
		AuditScan:       source.AuditScan,
		Session:         session,
		Agents:          agents,
		Context:         context,
		Result:          result,
		WorkflowOutputs: mappings,
		On:              transitions,
	}
	if err := resolved.ValidateScanPlanStage(stage); err != nil {
		return resolved.ResolvedStage{}, err
	}
	return stage, nil
}

func resolveWorkerSessionMode(source yaml.Node) (contracts.WorkerSessionMode, error) {
	if source.Kind == 0 {
		return contracts.WorkerSessionIsolated, nil
	}
	if source.Kind != yaml.ScalarNode || source.ShortTag() != "!!str" {
		return "", fmt.Errorf("must be isolated or shared")
	}
	mode := contracts.WorkerSessionMode(source.Value)
	if err := mode.Validate(); err != nil {
		return "", fmt.Errorf("must be isolated or shared")
	}
	return mode, nil
}

func (l *loader) resolveAgentBindings(source map[string]agentBindingSource) (map[string]resolved.ResolvedAgentBinding, error) {
	if len(source) == 0 {
		return nil, fmt.Errorf("agents must be a non-empty mapping")
	}
	result := make(map[string]resolved.ResolvedAgentBinding, len(source))
	for logicalName, binding := range source {
		if err := resolved.ValidateWorkflowName("logical Agent name", logicalName); err != nil {
			return nil, err
		}
		selector, err := resolved.ParseSelector(binding.Template)
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
		if err := resolved.ValidateArtifactComponent("agents."+logicalName+".namespace", namespace); err != nil {
			return nil, err
		}
		if artifactpolicy.IsPurposeReservedNamespace(namespace) {
			return nil, fmt.Errorf("agents.%s.namespace %q is reserved", logicalName, namespace)
		}
		result[logicalName] = resolved.ResolvedAgentBinding{
			Template:  template.Clone(),
			Namespace: namespace,
		}
	}
	return result, nil
}

func resolveStageContext(source *stageContextSource) (resolved.StageContext, error) {
	result := resolved.StageContext{Artifacts: make(map[string]resolved.ContextArtifact)}
	if source == nil {
		return result, nil
	}
	if source.Artifacts == nil {
		return resolved.StageContext{}, fmt.Errorf("context.artifacts is required when context is present")
	}
	for localName, artifact := range *source.Artifacts {
		if err := resolved.ValidateMapKey("context artifact name", localName); err != nil {
			return resolved.StageContext{}, err
		}
		if err := resolved.ValidateArtifactComponent("context.artifacts."+localName+".namespace", artifact.Namespace); err != nil {
			return resolved.StageContext{}, err
		}
		if err := resolved.ValidateArtifactComponent("context.artifacts."+localName+".name", artifact.Name); err != nil {
			return resolved.StageContext{}, err
		}
		if artifactpolicy.IsReservedMemoryBinding(artifact.Namespace, artifact.Name) {
			return resolved.StageContext{}, fmt.Errorf(
				"context.artifacts.%s identifies a reserved Memory binding", localName,
			)
		}
		if artifact.Required == nil {
			return resolved.StageContext{}, fmt.Errorf("context.artifacts.%s.required is required", localName)
		}
		result.Artifacts[localName] = resolved.ContextArtifact{
			Namespace: artifact.Namespace,
			Name:      artifact.Name,
			Required:  *artifact.Required,
		}
	}
	workspace, err := resolveWorkspaceContext(source.Workspace, result.Artifacts)
	if err != nil {
		return resolved.StageContext{}, err
	}
	result.Workspace = workspace
	return result, nil
}

func resolveWorkspaceContext(
	source *workspaceContextSource,
	artifacts map[string]resolved.ContextArtifact,
) (*resolved.WorkspaceContext, error) {
	if source == nil {
		return nil, nil
	}
	mode := contracts.WorkspaceMode(source.Mode)
	if err := mode.Validate(); err != nil {
		return nil, fmt.Errorf("context.workspace.mode must be direct or overlay")
	}
	if len(source.Sources) == 0 || len(source.Sources) > 32 {
		return nil, fmt.Errorf("context.workspace.sources must contain from 1 through 32 entries")
	}
	result := &resolved.WorkspaceContext{Mode: mode, Sources: make([]resolved.WorkspaceSource, len(source.Sources))}
	for index, candidate := range source.Sources {
		if err := resolved.ValidateMapKey("context.workspace.sources artifact alias", candidate.Artifact); err != nil {
			return nil, err
		}
		artifact, exists := artifacts[candidate.Artifact]
		if !exists {
			return nil, fmt.Errorf("context.workspace.sources[%d].artifact names unknown context artifact %q", index, candidate.Artifact)
		}
		if !artifact.Required {
			return nil, fmt.Errorf("context.workspace.sources[%d].artifact must name a required context artifact", index)
		}
		if err := validateWorkspaceTarget(candidate.Target); err != nil {
			return nil, fmt.Errorf("context.workspace.sources[%d].target: %w", index, err)
		}
		result.Sources[index] = resolved.WorkspaceSource{Artifact: candidate.Artifact, Target: candidate.Target}
	}
	for index, candidate := range result.Sources {
		for otherIndex, other := range result.Sources {
			if index == otherIndex {
				continue
			}
			if candidate.Target == other.Target || candidate.Target == "" || strings.HasPrefix(other.Target, candidate.Target+"/") {
				return nil, fmt.Errorf("context.workspace source targets must be unique and non-overlapping")
			}
		}
	}
	if source.State != nil {
		if err := resolved.ValidateMapKey("context.workspace.state.artifact", source.State.Artifact); err != nil {
			return nil, err
		}
		if _, exists := artifacts[source.State.Artifact]; !exists {
			return nil, fmt.Errorf("context.workspace.state.artifact names unknown context artifact %q", source.State.Artifact)
		}
		result.State = &resolved.WorkspaceStateInput{Artifact: source.State.Artifact}
	}
	if source.Export != nil {
		if mode != contracts.WorkspaceModeOverlay {
			return nil, fmt.Errorf("context.workspace.export requires overlay mode")
		}
		if err := resolved.ValidateMapKey("context.workspace.export.state", source.Export.State); err != nil {
			return nil, err
		}
		if err := resolved.ValidateMapKey("context.workspace.export.diff", source.Export.Diff); err != nil {
			return nil, err
		}
		if source.Export.State == source.Export.Diff {
			return nil, fmt.Errorf("context.workspace export slots must be distinct")
		}
		result.Export = &resolved.WorkspaceExport{State: source.Export.State, Diff: source.Export.Diff}
	}
	return result, nil
}

// validateWorkspaceTarget accepts "" (the workspace root) or a relative
// directory of the workspace target grammar (WSP-R13) that allocation
// validation applies again.
func validateWorkspaceTarget(value string) error {
	if value == "" {
		return nil
	}
	if problem := contracts.RelativePathProblem(
		value, contracts.MaxWorkspaceTargetBytes, contracts.MaxWorkspaceTargetComponents,
	); problem != "" {
		return errors.New(problem)
	}
	return nil
}

func resolveStageResult(source *stageResultSource) (resolved.StageResultContract, error) {
	result := resolved.StageResultContract{Artifacts: make(map[string]resolved.ArtifactSlot)}
	if source == nil {
		return result, nil
	}
	if source.Artifacts == nil {
		return resolved.StageResultContract{}, fmt.Errorf("result.artifacts is required when result is present")
	}
	artifacts, err := resolveArtifactSlotMap("result.artifacts", *source.Artifacts, true, false)
	if err != nil {
		return resolved.StageResultContract{}, err
	}
	result.Artifacts = artifacts
	return result, nil
}

func validateStageResultBindings(
	result resolved.StageResultContract,
	workspace *resolved.WorkspaceContext,
	agents map[string]resolved.ResolvedAgentBinding,
) error {
	exports := map[string]struct{}{}
	if workspace != nil && workspace.Export != nil {
		exports[workspace.Export.State] = struct{}{}
		exports[workspace.Export.Diff] = struct{}{}
	}
	namespaces := make(map[string]struct{}, len(agents))
	for _, agent := range agents {
		namespaces[agent.Namespace] = struct{}{}
	}
	for name, slot := range result.Artifacts {
		if _, runtimeOwned := exports[name]; runtimeOwned {
			if slot.From != nil {
				return fmt.Errorf("result.artifacts.%s.from must be omitted for Runtime-owned workspace export", name)
			}
			continue
		}
		if slot.From == nil {
			return fmt.Errorf("result.artifacts.%s.from is required", name)
		}
		if _, assigned := namespaces[slot.From.Namespace]; !assigned {
			return fmt.Errorf(
				"result.artifacts.%s.from.namespace %q is not assigned to a Stage Agent",
				name,
				slot.From.Namespace,
			)
		}
	}
	return nil
}

func resolveWorkflowOutputMappings(
	stageName string,
	source map[string]string,
	workflowOutputs map[string]resolved.ArtifactSlot,
	stageResults map[string]resolved.ArtifactSlot,
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
		if !resolved.MediaTypesIntersect(outputSlot.MediaTypes, resultSlot.MediaTypes) {
			return nil, fmt.Errorf("workflowOutputs.%s maps incompatible result and output media types", outputName)
		}
		result[outputName] = resultName
	}
	return result, nil
}

func (l *loader) resolveTransitions(source *stageTransitionsSource, stages map[string]stageSource) (resolved.StageTransitions, error) {
	if source == nil || source.Succeeded == nil || source.Failed == nil || source.Interrupted == nil {
		return resolved.StageTransitions{}, fmt.Errorf("on must contain succeeded, failed, and interrupted")
	}
	succeeded, err := l.resolveTransitionAction("on.succeeded", source.Succeeded, stages, map[resolved.TransitionKind]bool{
		resolved.TransitionNext: true, resolved.TransitionSucceed: true,
	})
	if err != nil {
		return resolved.StageTransitions{}, err
	}
	failed, err := l.resolveTransitionAction("on.failed", source.Failed, stages, map[resolved.TransitionKind]bool{
		resolved.TransitionNext: true, resolved.TransitionRetry: true, resolved.TransitionEscalate: true, resolved.TransitionFail: true,
	})
	if err != nil {
		return resolved.StageTransitions{}, err
	}
	interrupted, err := l.resolveTransitionAction("on.interrupted", source.Interrupted, stages, map[resolved.TransitionKind]bool{
		resolved.TransitionNext: true, resolved.TransitionRetry: true, resolved.TransitionEscalate: true, resolved.TransitionFail: true,
	})
	if err != nil {
		return resolved.StageTransitions{}, err
	}
	result := resolved.StageTransitions{Succeeded: succeeded, Failed: failed, Interrupted: interrupted}
	if source.summarizedPresent && source.Summarized == nil {
		return resolved.StageTransitions{}, fmt.Errorf("on.summarized must not be null")
	}
	if source.Summarized != nil {
		action, err := l.resolveTransitionAction("on.summarized", source.Summarized, stages, map[resolved.TransitionKind]bool{
			resolved.TransitionEscalate: true,
		})
		if err != nil {
			return resolved.StageTransitions{}, err
		}
		result.Summarized = &action
	}
	return result, nil
}

func (l *loader) resolveTransitionAction(
	field string,
	source *transitionActionSource,
	stages map[string]stageSource,
	allowed map[resolved.TransitionKind]bool,
) (resolved.TransitionAction, error) {
	if source == nil {
		return resolved.TransitionAction{}, fmt.Errorf("%s is required", field)
	}
	count := 0
	if source.Next != nil {
		count++
	}
	if source.Retry != nil {
		count++
	}
	if source.Escalate != nil {
		count++
	}
	if source.Succeed != nil {
		count++
	}
	if source.Fail != nil {
		count++
	}
	if count != 1 {
		return resolved.TransitionAction{}, fmt.Errorf("%s must select exactly one of next, retry, escalate, succeed, or fail", field)
	}

	var action resolved.TransitionAction
	switch {
	case source.Next != nil:
		action = resolved.TransitionAction{Kind: resolved.TransitionNext, NextStage: *source.Next}
		if err := resolved.ValidateWorkflowName(field+".next", action.NextStage); err != nil {
			return resolved.TransitionAction{}, err
		}
		if _, ok := stages[action.NextStage]; !ok {
			return resolved.TransitionAction{}, fmt.Errorf("%s.next names unknown Stage %q", field, action.NextStage)
		}
	case source.Retry != nil:
		if source.Retry.MaxAttempts < 2 {
			return resolved.TransitionAction{}, fmt.Errorf("%s.retry.maxAttempts must be at least 2", field)
		}
		then, err := l.resolveTransitionAction(field+".retry.then", source.Retry.Then, stages, map[resolved.TransitionKind]bool{
			resolved.TransitionNext: true, resolved.TransitionFail: true,
		})
		if err != nil {
			return resolved.TransitionAction{}, err
		}
		action = resolved.TransitionAction{Kind: resolved.TransitionRetry, Retry: &resolved.RetryTransition{MaxAttempts: source.Retry.MaxAttempts, Then: then}}
	case source.Escalate != nil:
		if source.Escalate.MaxAttempts < 1 {
			return resolved.TransitionAction{}, fmt.Errorf("%s.escalate.maxAttempts must be at least 1", field)
		}
		plannerRef, deadlineSeconds, err := resolveEscalationPlanner(field, source.Escalate)
		if err != nil {
			return resolved.TransitionAction{}, err
		}
		executionConfig, err := l.resolveEscalationExecutionConfig(
			field+".escalate.executionConfig", source.Escalate.ExecutionConfig,
		)
		if err != nil {
			return resolved.TransitionAction{}, err
		}
		then, err := l.resolveTransitionAction(
			field+".escalate.then", source.Escalate.Then, stages,
			map[resolved.TransitionKind]bool{resolved.TransitionNext: true, resolved.TransitionFail: true},
		)
		if err != nil {
			return resolved.TransitionAction{}, err
		}
		action = resolved.TransitionAction{
			Kind: resolved.TransitionEscalate,
			Escalate: &resolved.EscalateTransition{
				MaxAttempts:     source.Escalate.MaxAttempts,
				Planner:         plannerRef,
				DeadlineSeconds: deadlineSeconds,
				ExecutionConfig: executionConfig,
				Then:            then,
			},
		}
	case source.Succeed != nil:
		action = resolved.TransitionAction{Kind: resolved.TransitionSucceed}
	case source.Fail != nil:
		action = resolved.TransitionAction{Kind: resolved.TransitionFail}
	}
	if !allowed[action.Kind] {
		return resolved.TransitionAction{}, fmt.Errorf("%s action %q is not allowed", field, action.Kind)
	}
	return action, nil
}

func (l *loader) resolveEscalationExecutionConfig(
	field string,
	source *escalationExecutionConfigSource,
) (resolved.ResolvedEscalationExecutionConfig, error) {
	if source == nil {
		return resolved.ResolvedEscalationExecutionConfig{}, fmt.Errorf("%s is required", field)
	}
	refPresent := source.refPresent
	inlinePresent := source.plannerPresent || source.agentsPresent
	if refPresent == inlinePresent {
		return resolved.ResolvedEscalationExecutionConfig{}, fmt.Errorf(
			"%s must contain exactly ref or inline planner/agents", field,
		)
	}
	if refPresent {
		value, err := optionalStringFromYAML(source.Ref, false, field+".ref")
		if err != nil {
			return resolved.ResolvedEscalationExecutionConfig{}, err
		}
		selector, err := resolved.ParseSelector(value.Value)
		if err != nil {
			return resolved.ResolvedEscalationExecutionConfig{}, fmt.Errorf("%s.ref: %w", field, err)
		}
		profile, ok := l.executionConfigs[selector.String()]
		if !ok {
			return resolved.ResolvedEscalationExecutionConfig{}, fmt.Errorf(
				"%s.ref selects unknown ExecutionConfig %q", field, selector,
			)
		}
		ref := profile.Ref
		return resolved.ResolvedEscalationExecutionConfig{
			Ref: &ref, Override: profile.Override.Clone(),
		}, nil
	}

	patch, err := stageExecutionConfigPatchFromEscalationSource(source, field)
	if err != nil {
		return resolved.ResolvedEscalationExecutionConfig{}, err
	}
	override, err := l.resolveStageExecutionConfigOverride(patch)
	if err != nil {
		return resolved.ResolvedEscalationExecutionConfig{}, fmt.Errorf("%s: %w", field, err)
	}
	return resolved.ResolvedEscalationExecutionConfig{Override: override}, nil
}
