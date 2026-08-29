package config

import (
	"fmt"
	"sort"
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
	workflow := ResolvedWorkflow{
		Ref:        WorkflowRef{Name: selector.ID, Version: selector.Version},
		Parameters: parameters,
		Inputs:     inputs,
		Outputs:    outputs,
		EntryStage: spec.EntryStage,
		Stages:     stages,
	}
	if err := ValidateWorkflowGraph(workflow); err != nil {
		return ResolvedWorkflow{}, err
	}
	return workflow, nil
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
		if err := validateResolvedStageMappings(name, stage, workflow.Outputs); err != nil {
			return err
		}
		checks := []struct {
			field   string
			action  TransitionAction
			allowed map[TransitionKind]bool
		}{
			{"succeeded", stage.On.Succeeded, map[TransitionKind]bool{TransitionNext: true, TransitionSucceed: true}},
			{"failed", stage.On.Failed, map[TransitionKind]bool{TransitionNext: true, TransitionRetry: true, TransitionFail: true}},
			{"interrupted", stage.On.Interrupted, map[TransitionKind]bool{TransitionNext: true, TransitionRetry: true, TransitionFail: true}},
		}
		for _, check := range checks {
			field := "Stage " + name + " on." + check.field
			if err := validateResolvedTransition(field, check.action, workflow.Stages, check.allowed); err != nil {
				return err
			}
			continuation := transitionContinuation(check.action)
			if continuation.Kind == TransitionNext {
				adjacency[name][continuation.NextStage] = struct{}{}
			}
		}
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
		if action.Retry != nil || action.NextStage == "" {
			return fmt.Errorf("%s next Transition has an invalid payload", field)
		}
		if _, ok := stages[action.NextStage]; !ok {
			return fmt.Errorf("%s next Transition names unknown Stage %q", field, action.NextStage)
		}
	case TransitionRetry:
		if action.NextStage != "" || action.Retry == nil || action.Retry.MaxAttempts < 2 {
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
	case TransitionSucceed, TransitionFail:
		if action.NextStage != "" || action.Retry != nil {
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
	return validateMapKey(field, value)
}
