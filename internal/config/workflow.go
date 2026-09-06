package config

import (
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/contracts"
	"go.yaml.in/yaml/v4"
	"golang.org/x/text/unicode/norm"
)

func (l *loader) resolveWorkflow(selector Selector, spec *workflowSpecSource) (ResolvedWorkflow, error) {
	if spec == nil {
		return ResolvedWorkflow{}, fmt.Errorf("spec is required")
	}
	presentation, err := resolveWorkflowPresentation(spec.Presentation)
	if err != nil {
		return ResolvedWorkflow{}, err
	}
	parameters, err := resolveParameters(spec.Parameters)
	if err != nil {
		return ResolvedWorkflow{}, err
	}
	inputs, err := resolveArtifactSlots("spec.inputs", spec.Inputs, false)
	if err != nil {
		return ResolvedWorkflow{}, err
	}
	outputs, err := resolveArtifactSlots("spec.outputs", spec.Outputs, true)
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
	workflow := ResolvedWorkflow{
		Ref:          WorkflowRef{Name: selector.ID, Version: selector.Version},
		Presentation: presentation,
		Parameters:   parameters,
		Inputs:       inputs,
		Outputs:      outputs,
		EntryStage:   spec.EntryStage,
		Stages:       stages,
	}
	if err := l.resolveWorkflowExecutionConfig(&workflow, spec.ExecutionConfig); err != nil {
		return ResolvedWorkflow{}, err
	}
	if err := ValidateWorkflowGraph(workflow); err != nil {
		return ResolvedWorkflow{}, err
	}
	if _, err := WorkflowSkillRefs(workflow); err != nil {
		return ResolvedWorkflow{}, err
	}
	return workflow, nil
}

func resolveWorkflowPresentation(source *workflowPresentationSource) (*WorkflowPresentation, error) {
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
	return &WorkflowPresentation{
		DisplayName: source.DisplayName,
		Description: source.Description,
	}, nil
}

// WorkflowSkillRefs returns the sorted logical union retained by a Run
// snapshot, including templates used only by later Stages.
func WorkflowSkillRefs(workflow ResolvedWorkflow) ([]contracts.ArtifactRef, error) {
	union := make(map[string]contracts.ArtifactRef)
	for _, stage := range workflow.Stages {
		for _, binding := range stage.Agents {
			if err := binding.Template.Validate(); err != nil {
				return nil, fmt.Errorf("invalid AgentTemplate %s@%s: %w", binding.Template.Ref.TemplateID, binding.Template.Ref.Version, err)
			}
			for _, skill := range binding.Template.Skills {
				union[skill.Name] = skill
			}
		}
	}
	if len(union) > contracts.MaxWorkflowRunSkills {
		return nil, fmt.Errorf("WorkflowRun may retain at most %d distinct skills", contracts.MaxWorkflowRunSkills)
	}
	names := make([]string, 0, len(union))
	for name := range union {
		names = append(names, name)
	}
	sort.Strings(names)
	result := make([]contracts.ArtifactRef, 0, len(names))
	for _, name := range names {
		result = append(result, union[name])
	}
	return result, nil
}

// WorkflowSkillSets returns one sorted name set per distinct retained
// AgentTemplate. It is used only for per-template package byte limits.
func WorkflowSkillSets(workflow ResolvedWorkflow) [][]string {
	templates := make(map[string][]string)
	for _, stage := range workflow.Stages {
		for _, binding := range stage.Agents {
			if len(binding.Template.Skills) == 0 {
				continue
			}
			key := binding.Template.Ref.TemplateID + "@" + binding.Template.Ref.Version + ":" + binding.Template.Ref.Digest
			if _, exists := templates[key]; exists {
				continue
			}
			names := make([]string, len(binding.Template.Skills))
			for index, skill := range binding.Template.Skills {
				names[index] = skill.Name
			}
			templates[key] = names
		}
	}
	keys := make([]string, 0, len(templates))
	for key := range templates {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	result := make([][]string, 0, len(keys))
	for _, key := range keys {
		result = append(result, append([]string(nil), templates[key]...))
	}
	return result
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

func resolveArtifactSlots(
	field string,
	source *map[string]artifactSlotSource,
	allowPrimary bool,
) (map[string]ArtifactSlot, error) {
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
) (map[string]ArtifactSlot, error) {
	result := make(map[string]ArtifactSlot, len(source))
	for name, slot := range source {
		if err := validateArtifactComponent(field+" slot name", name); err != nil {
			return nil, err
		}
		if slot.Required == nil {
			return nil, fmt.Errorf("%s.%s.required is required", field, name)
		}
		mediaTypes, err := validateMediaTypes(field+"."+name, slot.MediaTypes)
		if err != nil {
			return nil, err
		}
		if slot.From != nil && !allowFrom {
			return nil, fmt.Errorf("%s.%s.from is allowed only for Stage result artifacts", field, name)
		}
		if slot.Primary != nil && !allowPrimary {
			return nil, fmt.Errorf("%s.%s.primary is allowed only for Workflow outputs", field, name)
		}
		var from *ArtifactBinding
		if slot.From != nil {
			if err := validateArtifactComponent(field+"."+name+".from.namespace", slot.From.Namespace); err != nil {
				return nil, err
			}
			if err := validateArtifactComponent(field+"."+name+".from.name", slot.From.Name); err != nil {
				return nil, err
			}
			if artifactpolicy.IsPurposeReservedNamespace(slot.From.Namespace) ||
				artifactpolicy.IsReservedMemoryBinding(slot.From.Namespace, slot.From.Name) {
				return nil, fmt.Errorf("%s.%s.from identifies a Runtime-reserved binding", field, name)
			}
			from = &ArtifactBinding{Namespace: slot.From.Namespace, Name: slot.From.Name}
		}
		primary := false
		if slot.Primary != nil {
			primary = *slot.Primary
		}
		result[name] = ArtifactSlot{
			Required: *slot.Required, MediaTypes: mediaTypes, From: from, Primary: primary,
		}
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
	session, err := resolveWorkerSessionMode(source.Session)
	if err != nil {
		return ResolvedStage{}, fmt.Errorf("session: %w", err)
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
	plannerRef := PlannerRef{PlannerID: planner.ID, Version: planner.Version}
	if err := validatePlannerAgentCardinality(plannerRef, len(agents)); err != nil {
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
	if err := validateStageWorkspace(context, result, agents, l.descriptors); err != nil {
		return ResolvedStage{}, err
	}
	if err := validateStageResultBindings(result, context.Workspace, agents); err != nil {
		return ResolvedStage{}, err
	}
	mappings, err := resolveWorkflowOutputMappings(stageName, source.WorkflowOutputs, workflowOutputs, result.Artifacts)
	if err != nil {
		return ResolvedStage{}, err
	}
	transitions, err := l.resolveTransitions(source.On, allStages)
	if err != nil {
		return ResolvedStage{}, err
	}

	return ResolvedStage{
		Objective:       source.Objective,
		Instructions:    instructions,
		Planner:         plannerRef,
		Session:         session,
		Agents:          agents,
		Context:         context,
		Result:          result,
		WorkflowOutputs: mappings,
		On:              transitions,
	}, nil
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
		if artifactpolicy.IsPurposeReservedNamespace(namespace) {
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
		if artifactpolicy.IsReservedMemoryBinding(artifact.Namespace, artifact.Name) {
			return StageContext{}, fmt.Errorf(
				"context.artifacts.%s identifies a reserved Memory binding", localName,
			)
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
	workspace, err := resolveWorkspaceContext(source.Workspace, result.Artifacts)
	if err != nil {
		return StageContext{}, err
	}
	result.Workspace = workspace
	return result, nil
}

func resolveWorkspaceContext(
	source *workspaceContextSource,
	artifacts map[string]ContextArtifact,
) (*WorkspaceContext, error) {
	if source == nil {
		return nil, nil
	}
	mode := contracts.WorkspaceModeV2(source.Mode)
	if err := mode.Validate(); err != nil {
		return nil, fmt.Errorf("context.workspace.mode must be direct or overlay")
	}
	if len(source.Sources) == 0 || len(source.Sources) > 32 {
		return nil, fmt.Errorf("context.workspace.sources must contain from 1 through 32 entries")
	}
	result := &WorkspaceContext{Mode: mode, Sources: make([]WorkspaceSource, len(source.Sources))}
	for index, candidate := range source.Sources {
		if err := validateMapKey("context.workspace.sources artifact alias", candidate.Artifact); err != nil {
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
		result.Sources[index] = WorkspaceSource{Artifact: candidate.Artifact, Target: candidate.Target}
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
		if err := validateMapKey("context.workspace.state.artifact", source.State.Artifact); err != nil {
			return nil, err
		}
		if _, exists := artifacts[source.State.Artifact]; !exists {
			return nil, fmt.Errorf("context.workspace.state.artifact names unknown context artifact %q", source.State.Artifact)
		}
		result.State = &WorkspaceStateInput{Artifact: source.State.Artifact}
	}
	if source.Export != nil {
		if mode != contracts.WorkspaceModeOverlay {
			return nil, fmt.Errorf("context.workspace.export requires overlay mode")
		}
		if err := validateMapKey("context.workspace.export.state", source.Export.State); err != nil {
			return nil, err
		}
		if err := validateMapKey("context.workspace.export.diff", source.Export.Diff); err != nil {
			return nil, err
		}
		if source.Export.State == source.Export.Diff {
			return nil, fmt.Errorf("context.workspace export slots must be distinct")
		}
		result.Export = &WorkspaceExport{State: source.Export.State, Diff: source.Export.Diff}
	}
	return result, nil
}

func validateWorkspaceTarget(value string) error {
	if value == "" {
		return nil
	}
	if value != norm.NFC.String(value) || len([]byte(value)) > 1024 || strings.HasPrefix(value, "/") ||
		strings.ContainsAny(value, "\\\x00") || strings.Contains(value, "://") {
		return fmt.Errorf("must be a normalized relative POSIX directory")
	}
	parts := strings.Split(value, "/")
	if len(parts) > 32 {
		return fmt.Errorf("must contain at most 32 components")
	}
	for _, part := range parts {
		if part == "" || part == "." || part == ".." {
			return fmt.Errorf("contains an invalid component")
		}
		for _, character := range part {
			if character < 0x20 || character == 0x7f {
				return fmt.Errorf("contains a control character")
			}
		}
	}
	first := parts[0]
	if len(first) >= 2 && ((first[0] >= 'A' && first[0] <= 'Z') || (first[0] >= 'a' && first[0] <= 'z')) && first[1] == ':' {
		return fmt.Errorf("must not use Windows drive syntax")
	}
	return nil
}

func validateStageWorkspace(
	context StageContext,
	result StageResultContract,
	agents map[string]ResolvedAgentBinding,
	registered ...Descriptors,
) error {
	descriptors := MVPDescriptors()
	if len(registered) > 0 {
		descriptors = registered[0]
	}
	workspaceToolsets := false
	changesToolset := false
	for _, agent := range agents {
		if err := descriptors.ValidateSandboxToolCompatibility(agent.Template); err != nil {
			return err
		}
		profile := agent.Template.SandboxProfile.SandboxProfileID + "@" + agent.Template.SandboxProfile.Version
		required := descriptors.SandboxProfiles[profile]
		if required.WorkspaceMode != "" && (context.Workspace == nil || context.Workspace.Mode != required.WorkspaceMode) {
			return fmt.Errorf("SandboxProfile %s requires context.workspace.mode %s", profile, required.WorkspaceMode)
		}
		for _, selection := range agent.Template.Toolsets {
			ref := selection.Ref.ToolsetID + "@" + selection.Ref.Version
			switch ref {
			case "filesystem@1", "edit-files@1", "code-analysis@1", "taint-annotations@1":
				workspaceToolsets = true
			case "workspace-changes@1":
				workspaceToolsets = true
				changesToolset = true
			}
		}
	}
	if workspaceToolsets && context.Workspace == nil {
		return fmt.Errorf("workspace-dependent Toolsets require context.workspace")
	}
	if changesToolset && context.Workspace != nil && context.Workspace.Mode != contracts.WorkspaceModeOverlay {
		return fmt.Errorf("workspace-changes@1 requires context.workspace.mode overlay")
	}
	if context.Workspace == nil || context.Workspace.Export == nil {
		return nil
	}
	export := context.Workspace.Export
	state, stateExists := result.Artifacts[export.State]
	diff, diffExists := result.Artifacts[export.Diff]
	if !stateExists || len(state.MediaTypes) != 1 || state.MediaTypes[0] != "application/vnd.contractor.workspace-overlay+json" {
		return fmt.Errorf("context.workspace.export.state must name a result slot with exact workspace overlay media type")
	}
	if !diffExists || len(diff.MediaTypes) != 1 || diff.MediaTypes[0] != "text/x-diff" {
		return fmt.Errorf("context.workspace.export.diff must name a result slot with exact text/x-diff media type")
	}
	return nil
}

func resolveStageResult(source *stageResultSource) (StageResultContract, error) {
	result := StageResultContract{Artifacts: make(map[string]ArtifactSlot)}
	if source == nil {
		return result, nil
	}
	if source.Artifacts == nil {
		return StageResultContract{}, fmt.Errorf("result.artifacts is required when result is present")
	}
	artifacts, err := resolveArtifactSlotMap("result.artifacts", *source.Artifacts, true, false)
	if err != nil {
		return StageResultContract{}, err
	}
	result.Artifacts = artifacts
	return result, nil
}

func validateStageResultBindings(
	result StageResultContract,
	workspace *WorkspaceContext,
	agents map[string]ResolvedAgentBinding,
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

func (l *loader) resolveTransitions(source *stageTransitionsSource, stages map[string]stageSource) (StageTransitions, error) {
	if source == nil || source.Succeeded == nil || source.Failed == nil || source.Interrupted == nil {
		return StageTransitions{}, fmt.Errorf("on must contain succeeded, failed, and interrupted")
	}
	succeeded, err := l.resolveTransitionAction("on.succeeded", source.Succeeded, stages, map[TransitionKind]bool{
		TransitionNext: true, TransitionSucceed: true,
	})
	if err != nil {
		return StageTransitions{}, err
	}
	failed, err := l.resolveTransitionAction("on.failed", source.Failed, stages, map[TransitionKind]bool{
		TransitionNext: true, TransitionRetry: true, TransitionEscalate: true, TransitionFail: true,
	})
	if err != nil {
		return StageTransitions{}, err
	}
	interrupted, err := l.resolveTransitionAction("on.interrupted", source.Interrupted, stages, map[TransitionKind]bool{
		TransitionNext: true, TransitionRetry: true, TransitionEscalate: true, TransitionFail: true,
	})
	if err != nil {
		return StageTransitions{}, err
	}
	return StageTransitions{Succeeded: succeeded, Failed: failed, Interrupted: interrupted}, nil
}

func (l *loader) resolveTransitionAction(
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
		return TransitionAction{}, fmt.Errorf("%s must select exactly one of next, retry, escalate, succeed, or fail", field)
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
		then, err := l.resolveTransitionAction(field+".retry.then", source.Retry.Then, stages, map[TransitionKind]bool{
			TransitionNext: true, TransitionFail: true,
		})
		if err != nil {
			return TransitionAction{}, err
		}
		action = TransitionAction{Kind: TransitionRetry, Retry: &RetryTransition{MaxAttempts: source.Retry.MaxAttempts, Then: then}}
	case source.Escalate != nil:
		if source.Escalate.MaxAttempts < 1 {
			return TransitionAction{}, fmt.Errorf("%s.escalate.maxAttempts must be at least 1", field)
		}
		executionConfig, err := l.resolveEscalationExecutionConfig(
			field+".escalate.executionConfig", source.Escalate.ExecutionConfig,
		)
		if err != nil {
			return TransitionAction{}, err
		}
		then, err := l.resolveTransitionAction(
			field+".escalate.then", source.Escalate.Then, stages,
			map[TransitionKind]bool{TransitionNext: true, TransitionFail: true},
		)
		if err != nil {
			return TransitionAction{}, err
		}
		action = TransitionAction{
			Kind: TransitionEscalate,
			Escalate: &EscalateTransition{
				MaxAttempts:     source.Escalate.MaxAttempts,
				ExecutionConfig: executionConfig,
				Then:            then,
			},
		}
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

func (l *loader) resolveEscalationExecutionConfig(
	field string,
	source *escalationExecutionConfigSource,
) (ResolvedEscalationExecutionConfig, error) {
	if source == nil {
		return ResolvedEscalationExecutionConfig{}, fmt.Errorf("%s is required", field)
	}
	refPresent := source.refPresent
	inlinePresent := source.plannerPresent || source.agentsPresent
	if refPresent == inlinePresent {
		return ResolvedEscalationExecutionConfig{}, fmt.Errorf(
			"%s must contain exactly ref or inline planner/agents", field,
		)
	}
	if refPresent {
		value, err := optionalStringFromYAML(source.Ref, false, field+".ref")
		if err != nil {
			return ResolvedEscalationExecutionConfig{}, err
		}
		selector, err := ParseSelector(value.value)
		if err != nil {
			return ResolvedEscalationExecutionConfig{}, fmt.Errorf("%s.ref: %w", field, err)
		}
		profile, ok := l.executionConfigs[selector.String()]
		if !ok {
			return ResolvedEscalationExecutionConfig{}, fmt.Errorf(
				"%s.ref selects unknown ExecutionConfig %q", field, selector,
			)
		}
		ref := profile.Ref
		return ResolvedEscalationExecutionConfig{
			Ref: &ref, Override: cloneStageExecutionConfigOverride(profile.Override),
		}, nil
	}

	patch, err := stageExecutionConfigPatchFromEscalationSource(source, field)
	if err != nil {
		return ResolvedEscalationExecutionConfig{}, err
	}
	override, err := l.resolveStageExecutionConfigOverride(patch)
	if err != nil {
		return ResolvedEscalationExecutionConfig{}, fmt.Errorf("%s: %w", field, err)
	}
	return ResolvedEscalationExecutionConfig{Override: override}, nil
}

// ValidateWorkflowGraph verifies the resolved serial control-flow contract.
// It is exported so the Scheduler can distrust and revalidate a durable JSON
// snapshot without consulting the mutable configuration directory.
func ValidateWorkflowGraph(workflow ResolvedWorkflow) error {
	if len(workflow.Stages) == 0 {
		return fmt.Errorf("Workflow Graph must contain at least one Stage")
	}
	if _, ok := workflow.Stages[workflow.EntryStage]; !ok {
		return fmt.Errorf("Workflow Graph entry Stage %q does not exist", workflow.EntryStage)
	}
	names := make([]string, 0, len(workflow.Stages))
	adjacency := make(map[string]map[string]struct{}, len(workflow.Stages))
	for name := range workflow.Stages {
		names = append(names, name)
		adjacency[name] = make(map[string]struct{})
	}
	sort.Strings(names)
	for _, name := range names {
		stage := workflow.Stages[name]
		if err := stage.Session.Validate(); err != nil {
			return fmt.Errorf("Stage %q: %w", name, err)
		}
		if err := validatePlannerAgentCardinality(stage.Planner, len(stage.Agents)); err != nil {
			return fmt.Errorf("Stage %q: %w", name, err)
		}
		if err := validateResolvedStageMappings(name, stage, workflow.Outputs); err != nil {
			return err
		}
		checks := []struct {
			field   string
			action  TransitionAction
			allowed map[TransitionKind]bool
		}{
			{"succeeded", stage.On.Succeeded, map[TransitionKind]bool{TransitionNext: true, TransitionSucceed: true}},
			{"failed", stage.On.Failed, map[TransitionKind]bool{TransitionNext: true, TransitionRetry: true, TransitionEscalate: true, TransitionFail: true}},
			{"interrupted", stage.On.Interrupted, map[TransitionKind]bool{TransitionNext: true, TransitionRetry: true, TransitionEscalate: true, TransitionFail: true}},
		}
		for _, check := range checks {
			field := "Stage " + name + " on." + check.field
			if err := validateResolvedTransition(field, check.action, workflow.Stages, check.allowed); err != nil {
				return err
			}
			if err := validateStageEscalationVariant(name, check.field, stage, check.action); err != nil {
				return err
			}
			continuation := transitionContinuation(check.action)
			if continuation.Kind == TransitionNext {
				adjacency[name][continuation.NextStage] = struct{}{}
			}
		}
	}
	if err := validateWorkflowExecutionConfigs(workflow); err != nil {
		return err
	}
	if err := validateFindingsReaderInput(workflow); err != nil {
		return err
	}

	reachable := make(map[string]bool, len(workflow.Stages))
	var visit func(string)
	visit = func(name string) {
		if reachable[name] {
			return
		}
		reachable[name] = true
		targets := sortedSet(adjacency[name])
		for _, target := range targets {
			visit(target)
		}
	}
	visit(workflow.EntryStage)
	for _, name := range names {
		if !reachable[name] {
			return fmt.Errorf("Workflow Graph Stage %q is unreachable from entry Stage %q", name, workflow.EntryStage)
		}
	}

	order, err := topologicalStageOrder(names, adjacency)
	if err != nil {
		return err
	}
	flows := make(map[string]workflowOutputFlow, len(workflow.Stages))
	flows[workflow.EntryStage] = newWorkflowOutputFlow()
	for _, name := range order {
		incoming := flows[name]
		if !incoming.initialized {
			return fmt.Errorf("Workflow Graph Stage %q has no executable predecessor", name)
		}
		stage := workflow.Stages[name]
		for outputName := range stage.WorkflowOutputs {
			if _, alreadyBound := incoming.may[outputName]; alreadyBound {
				return fmt.Errorf("Workflow output %q may be mapped more than once on one Graph path", outputName)
			}
		}
		succeeded := incoming.clone()
		for outputName, resultName := range stage.WorkflowOutputs {
			if stage.Result.Artifacts[resultName].Required {
				succeeded.must[outputName] = struct{}{}
			}
			succeeded.may[outputName] = struct{}{}
		}
		if err := applyWorkflowTransition(
			name, stage.On.Succeeded, succeeded, workflow.Outputs, flows,
		); err != nil {
			return err
		}
		for _, action := range []TransitionAction{stage.On.Failed, stage.On.Interrupted} {
			if err := applyWorkflowTransition(
				name, action, incoming, workflow.Outputs, flows,
			); err != nil {
				return err
			}
		}
	}
	return nil
}

func validatePlannerAgentCardinality(planner PlannerRef, agents int) error {
	reference := planner.PlannerID + "@" + planner.Version
	switch reference {
	case "passthrough@1", "streamline@1":
		if agents != 1 {
			return fmt.Errorf("%s requires exactly one logical Agent binding; use router@1 for multiple bindings", reference)
		}
	case "router@1":
		if agents < 1 {
			return fmt.Errorf("router@1 requires at least one logical Agent binding")
		}
	default:
		if agents < 1 {
			return fmt.Errorf("%s requires at least one logical Agent binding", reference)
		}
	}
	return nil
}

func validateResolvedStageMappings(
	stageName string,
	stage ResolvedStage,
	workflowOutputs map[string]ArtifactSlot,
) error {
	for outputName, resultName := range stage.WorkflowOutputs {
		output, ok := workflowOutputs[outputName]
		if !ok {
			return fmt.Errorf("Stage %q maps undeclared Workflow output %q", stageName, outputName)
		}
		result, ok := stage.Result.Artifacts[resultName]
		if !ok {
			return fmt.Errorf("Stage %q maps output %q from undeclared result %q", stageName, outputName, resultName)
		}
		if !mediaTypesIntersect(output.MediaTypes, result.MediaTypes) {
			return fmt.Errorf("Stage %q maps output %q with incompatible media types", stageName, outputName)
		}
	}
	return nil
}

func validateResolvedTransition(
	field string,
	action TransitionAction,
	stages map[string]ResolvedStage,
	allowed map[TransitionKind]bool,
) error {
	if !allowed[action.Kind] {
		return fmt.Errorf("%s Transition action %q is not allowed", field, action.Kind)
	}
	switch action.Kind {
	case TransitionNext:
		if action.Retry != nil || action.Escalate != nil || action.NextStage == "" {
			return fmt.Errorf("%s next Transition has an invalid payload", field)
		}
		if _, ok := stages[action.NextStage]; !ok {
			return fmt.Errorf("%s next Transition names unknown Stage %q", field, action.NextStage)
		}
	case TransitionRetry:
		if action.NextStage != "" || action.Escalate != nil || action.Retry == nil || action.Retry.MaxAttempts < 2 {
			return fmt.Errorf("%s retry Transition requires bounded maxAttempts >= 2", field)
		}
		if err := validateResolvedTransition(
			field+".retry.then",
			action.Retry.Then,
			stages,
			map[TransitionKind]bool{TransitionNext: true, TransitionFail: true},
		); err != nil {
			return err
		}
	case TransitionEscalate:
		if action.NextStage != "" || action.Retry != nil || action.Escalate == nil || action.Escalate.MaxAttempts < 1 {
			return fmt.Errorf("%s escalate Transition requires bounded maxAttempts >= 1", field)
		}
		if err := validateResolvedTransition(
			field+".escalate.then",
			action.Escalate.Then,
			stages,
			map[TransitionKind]bool{TransitionNext: true, TransitionFail: true},
		); err != nil {
			return err
		}
	case TransitionSucceed, TransitionFail:
		if action.NextStage != "" || action.Retry != nil || action.Escalate != nil {
			return fmt.Errorf("%s terminal Transition has an invalid payload", field)
		}
	default:
		return fmt.Errorf("%s has unknown Transition action %q", field, action.Kind)
	}
	return nil
}

func transitionContinuation(action TransitionAction) TransitionAction {
	if action.Kind == TransitionRetry && action.Retry != nil {
		return action.Retry.Then
	}
	if action.Kind == TransitionEscalate && action.Escalate != nil {
		return action.Escalate.Then
	}
	return action
}

func topologicalStageOrder(
	names []string,
	adjacency map[string]map[string]struct{},
) ([]string, error) {
	indegree := make(map[string]int, len(names))
	for _, name := range names {
		for target := range adjacency[name] {
			indegree[target]++
		}
	}
	ready := make([]string, 0, len(names))
	for _, name := range names {
		if indegree[name] == 0 {
			ready = append(ready, name)
		}
	}
	order := make([]string, 0, len(names))
	for len(ready) > 0 {
		sort.Strings(ready)
		name := ready[0]
		ready = ready[1:]
		order = append(order, name)
		for _, target := range sortedSet(adjacency[name]) {
			indegree[target]--
			if indegree[target] == 0 {
				ready = append(ready, target)
			}
		}
	}
	if len(order) != len(names) {
		return nil, fmt.Errorf("Workflow Graph contains a next Transition Cycle")
	}
	return order, nil
}

type workflowOutputFlow struct {
	initialized bool
	must        map[string]struct{}
	may         map[string]struct{}
}

func newWorkflowOutputFlow() workflowOutputFlow {
	return workflowOutputFlow{
		initialized: true,
		must:        make(map[string]struct{}),
		may:         make(map[string]struct{}),
	}
}

func (f workflowOutputFlow) clone() workflowOutputFlow {
	result := newWorkflowOutputFlow()
	for name := range f.must {
		result.must[name] = struct{}{}
	}
	for name := range f.may {
		result.may[name] = struct{}{}
	}
	return result
}

func mergeWorkflowOutputFlow(current, incoming workflowOutputFlow) workflowOutputFlow {
	if !current.initialized {
		return incoming.clone()
	}
	result := current.clone()
	for name := range result.must {
		if _, present := incoming.must[name]; !present {
			delete(result.must, name)
		}
	}
	for name := range incoming.may {
		result.may[name] = struct{}{}
	}
	return result
}

func applyWorkflowTransition(
	stageName string,
	action TransitionAction,
	flow workflowOutputFlow,
	outputs map[string]ArtifactSlot,
	flows map[string]workflowOutputFlow,
) error {
	action = transitionContinuation(action)
	switch action.Kind {
	case TransitionNext:
		flows[action.NextStage] = mergeWorkflowOutputFlow(flows[action.NextStage], flow)
	case TransitionSucceed:
		for outputName, slot := range outputs {
			if _, bound := flow.must[outputName]; slot.Required && !bound {
				return fmt.Errorf(
					"Workflow Graph may succeed at Stage %q without required output %q",
					stageName,
					outputName,
				)
			}
		}
	case TransitionFail:
		return nil
	default:
		return fmt.Errorf("Workflow Graph has unresolved Transition action %q", action.Kind)
	}
	return nil
}

func sortedSet(values map[string]struct{}) []string {
	result := make([]string, 0, len(values))
	for value := range values {
		result = append(result, value)
	}
	sort.Strings(result)
	return result
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
	if err := contracts.ValidateArtifactName(value); err != nil {
		return fmt.Errorf("%s: %w", field, err)
	}
	return nil
}
