package resolved

import (
	"fmt"
	"maps"
	"slices"
	"sort"

	"github.com/grauwolf32/contractor/internal/contracts"
)

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

// ValidateStageWorkspace checks a Stage's workspace against the registered
// descriptors of its Agents' templates: the SandboxProfile a Toolset requires,
// the workspace mode a SandboxProfile needs, the workspace that
// workspace-dependent Toolsets need, and the exact media types of the
// workspace export result slots.
func ValidateStageWorkspace(
	context StageContext,
	result StageResultContract,
	agents map[string]ResolvedAgentBinding,
) error {
	registered := registeredDescriptors()
	workspaceToolsets := false
	changesToolset := false
	for _, agent := range agents {
		if err := sandboxToolCompatibility(registered, agent.Template); err != nil {
			return err
		}
		profile := agent.Template.SandboxProfile.SandboxProfileID + "@" + agent.Template.SandboxProfile.Version
		required := registered.SandboxProfiles[profile]
		if required.WorkspaceMode != "" && (context.Workspace == nil || context.Workspace.Mode != required.WorkspaceMode) {
			return fmt.Errorf("SandboxProfile %s requires context.workspace.mode %s", profile, required.WorkspaceMode)
		}
		for _, selection := range agent.Template.Toolsets {
			ref := selection.Ref.ToolsetID + "@" + selection.Ref.Version
			if registered.Toolsets[ref].RequiresWorkspace {
				workspaceToolsets = true
			}
			if ref == "workspace-changes@1" {
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

// ValidateWorkflowGraph verifies the resolved serial control-flow contract.
// It is exported so the Scheduler can distrust and revalidate a durable JSON
// snapshot without consulting the mutable configuration directory.
func ValidateWorkflowGraph(workflow ResolvedWorkflow) error {
	if err := ValidateAuditTaskWorkflow(workflow); err != nil {
		return err
	}
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
		if err := ValidateWorkflowName("Stage name", name); err != nil {
			return err
		}
		if err := validateStageAgentNames(fmt.Sprintf("Stage %q", name), stage.Agents); err != nil {
			return err
		}
		if err := stage.Session.Validate(); err != nil {
			return fmt.Errorf("Stage %q: %w", name, err)
		}
		if err := ValidatePlannerAgents(stage.Planner, len(stage.Agents)); err != nil {
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
			if err := ValidateStageEscalationVariant(name, check.field, stage, check.action); err != nil {
				return err
			}
			continuation := transitionContinuation(check.action)
			if continuation.Kind == TransitionNext {
				adjacency[name][continuation.NextStage] = struct{}{}
			}
		}
	}
	if err := ValidateWorkflowExecutionConfigs(workflow); err != nil {
		return err
	}
	if err := validateToolWorkflowInputs(workflow, registeredDescriptors().Toolsets); err != nil {
		return err
	}

	reachable := make(map[string]bool, len(workflow.Stages))
	var visit func(string)
	visit = func(name string) {
		if reachable[name] {
			return
		}
		reachable[name] = true
		targets := slices.Sorted(maps.Keys(adjacency[name]))
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
		if !MediaTypesIntersect(output.MediaTypes, result.MediaTypes) {
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
		for _, target := range slices.Sorted(maps.Keys(adjacency[name])) {
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
