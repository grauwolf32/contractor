package scheduler

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strings"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runstore"
)

type executableWorkflow struct {
	workflow  resolved.ResolvedWorkflow
	stageName string
	stage     resolved.ResolvedStage
}

func decodeExecutableWorkflow(run runstore.WorkflowRun) (executableWorkflow, error) {
	if run.WorkflowSchemaVersion != contracts.APIVersion || len(run.WorkflowSnapshot) == 0 {
		return executableWorkflow{}, fmt.Errorf("%w: Workflow snapshot version is invalid or the snapshot is empty", ErrUnsupportedWorkflow)
	}
	workflow, err := resolved.DecodeResolvedWorkflowSnapshot(run.WorkflowSnapshot)
	if err != nil {
		return executableWorkflow{}, fmt.Errorf("%w: decode Workflow snapshot: %v", ErrUnsupportedWorkflow, err)
	}
	// PostgreSQL returns jsonb text with its own spacing and number format,
	// so the bound is measured on the snapshot's encoding, exactly as Run
	// creation measured it (CAT-R76).
	if _, err := resolved.EncodeResolvedWorkflowSnapshot(workflow); err != nil {
		return executableWorkflow{}, fmt.Errorf("%w: %v", ErrUnsupportedWorkflow, err)
	}
	if workflow.Ref.Name != run.WorkflowName || workflow.Ref.Version != run.WorkflowVersion ||
		strings.TrimSpace(workflow.EntryStage) == "" {
		return executableWorkflow{}, fmt.Errorf("%w: Workflow identity or entry Stage is invalid", ErrUnsupportedWorkflow)
	}
	// The decoder has checked every Stage's Planner and logical Agent count
	// against the registered PlannerFactory descriptors.
	for stageName, stage := range workflow.Stages {
		if strings.TrimSpace(stage.Objective) == "" || strings.TrimSpace(stage.Instructions.Text) == "" {
			return executableWorkflow{}, fmt.Errorf("%w: Stage %q semantic input is incomplete", ErrUnsupportedWorkflow, stageName)
		}
		for logicalName, binding := range stage.Agents {
			if strings.TrimSpace(logicalName) == "" || strings.TrimSpace(binding.Namespace) == "" ||
				artifactpolicy.IsPurposeReservedNamespace(binding.Namespace) ||
				strings.Contains(binding.Namespace, "/") {
				return executableWorkflow{}, fmt.Errorf("%w: Stage %q Agent binding is invalid", ErrUnsupportedWorkflow, stageName)
			}
			if err := binding.Template.Validate(); err != nil {
				return executableWorkflow{}, fmt.Errorf("%w: resolved AgentTemplate is invalid: %v", ErrUnsupportedWorkflow, err)
			}
		}
	}
	result := executableWorkflow{workflow: workflow}
	return result.selectStage(workflow.EntryStage)
}

func (w executableWorkflow) selectStage(name string) (executableWorkflow, error) {
	stage, ok := w.workflow.Stages[name]
	if !ok {
		return executableWorkflow{}, fmt.Errorf("%w: Workflow Stage %q does not exist", ErrUnsupportedWorkflow, name)
	}
	w.stageName = name
	w.stage = stage
	return w, nil
}

func (w executableWorkflow) selectExecution(
	execution runstore.StageExecution,
) (executableWorkflow, error) {
	selected, err := w.selectStage(execution.StageName)
	if err != nil {
		return executableWorkflow{}, err
	}
	stage := selected.stage
	switch execution.ExecutionConfigVariant {
	case runstore.StageExecutionConfigBase:
		if execution.EscalationOrdinal != nil {
			return executableWorkflow{}, fmt.Errorf("base StageExecution has an escalation ordinal")
		}
	case runstore.StageExecutionConfigFailedEscalation:
		if execution.EscalationOrdinal == nil || *execution.EscalationOrdinal <= 0 ||
			stage.On.Failed.Kind != resolved.TransitionEscalate || stage.On.Failed.Escalate == nil ||
			*execution.EscalationOrdinal > stage.On.Failed.Escalate.MaxAttempts {
			return executableWorkflow{}, fmt.Errorf("failed-escalation StageExecution identity is invalid")
		}
		stage.ExecutionConfig = stage.On.Failed.Escalate.ExecutionConfig.Effective
	case runstore.StageExecutionConfigInterruptedEscalation:
		if execution.EscalationOrdinal == nil || *execution.EscalationOrdinal <= 0 ||
			stage.On.Interrupted.Kind != resolved.TransitionEscalate || stage.On.Interrupted.Escalate == nil ||
			*execution.EscalationOrdinal > stage.On.Interrupted.Escalate.MaxAttempts {
			return executableWorkflow{}, fmt.Errorf("interrupted-escalation StageExecution identity is invalid")
		}
		stage.ExecutionConfig = stage.On.Interrupted.Escalate.ExecutionConfig.Effective
	case runstore.StageExecutionConfigSummaryEscalation:
		if stage.On.Summarized == nil || stage.On.Summarized.Kind != resolved.TransitionEscalate ||
			stage.On.Summarized.Escalate == nil || stage.On.Summarized.Escalate.Planner == nil ||
			execution.EscalationOrdinal == nil || *execution.EscalationOrdinal != 1 || execution.StageContext.Continuation == nil {
			return executableWorkflow{}, fmt.Errorf("summary-escalation StageExecution identity is invalid")
		}
		stage.Planner = *stage.On.Summarized.Escalate.Planner
		stage.ExecutionConfig = stage.On.Summarized.Escalate.ExecutionConfig.Effective
	default:
		return executableWorkflow{}, fmt.Errorf(
			"StageExecution has unknown executionConfig variant %q", execution.ExecutionConfigVariant,
		)
	}
	selected.stage = stage
	return selected, nil
}

func validatePersistedExecution(
	execution runstore.StageExecution,
	run runstore.WorkflowRun,
	workflow executableWorkflow,
) error {
	if execution.CreatedAt.IsZero() {
		return fmt.Errorf("persisted StageExecution has no creation time")
	}
	if execution.RunID != run.RunID || execution.StageName != workflow.stageName || execution.Attempt <= 0 ||
		(execution.Attempt == 1) != (execution.PreviousExecutionID == nil) ||
		execution.StageSpecSchemaVersion != contracts.APIVersion ||
		execution.StageContextSchemaVersion != contracts.APIVersion {
		return fmt.Errorf("persisted StageExecution identity differs from the Workflow snapshot")
	}
	persisted, err := resolved.DecodeResolvedStageSnapshot(execution.StageSpecSnapshot)
	if err != nil {
		return fmt.Errorf("decode persisted Stage spec: %w", err)
	}
	if !reflect.DeepEqual(persisted, workflow.stage) {
		return fmt.Errorf("persisted Stage spec differs from the immutable Workflow snapshot")
	}
	if err := execution.StageContext.Validate(); err != nil {
		return err
	}
	if (execution.ExecutionConfigVariant == runstore.StageExecutionConfigSummaryEscalation) != (execution.StageContext.Continuation != nil) {
		return fmt.Errorf("StageExecution continuation differs from its configuration variant")
	}
	if continuation := execution.StageContext.Continuation; continuation != nil {
		if continuation.SourceStageExecutionID == execution.StageExecutionID {
			return fmt.Errorf("summary continuation refers to itself")
		}
		for _, binding := range workflow.stage.Agents {
			if continuation.SummaryArtifact.Namespace != binding.Namespace {
				return fmt.Errorf("summary continuation changed the Agent Namespace")
			}
		}
	}
	if !reflect.DeepEqual(execution.StageContext.Parameters, run.Parameters) ||
		len(execution.StageContext.Artifacts) != len(workflow.stage.Context.Artifacts) {
		return fmt.Errorf("persisted StageContext differs from the immutable Run input")
	}
	for name, declaration := range workflow.stage.Context.Artifacts {
		pinned, ok := execution.StageContext.Artifacts[name]
		if !ok || pinned.Required != declaration.Required {
			return fmt.Errorf("persisted StageContext artifact %q differs from its declaration", name)
		}
		if pinned.Artifact != nil &&
			(pinned.Artifact.Namespace != declaration.Namespace || pinned.Artifact.Name != declaration.Name) {
			return fmt.Errorf("persisted StageContext artifact %q identifies another binding", name)
		}
	}
	return nil
}

func stageSnapshot(stage resolved.ResolvedStage) (json.RawMessage, error) {
	encoded, err := json.Marshal(stage)
	if err != nil {
		return nil, fmt.Errorf("encode resolved Stage snapshot: %w", err)
	}
	return encoded, nil
}

func bindingRequirements(
	stage resolved.ResolvedStage,
	snapshot []agentskills.RunSkillSnapshot,
	context runstore.StageContextSnapshot,
) ([]controlplane.BindingRequirement, error) {
	workspace, err := projectAllocationWorkspace(stage.Context.Workspace, context)
	if err != nil {
		return nil, err
	}
	resolvedByName := make(map[string]contracts.ResolvedSkill, len(snapshot))
	for _, skill := range snapshot {
		if err := skill.Validate(); err != nil || !skill.Initialized() {
			return nil, fmt.Errorf("Run Skill snapshot is not initialized")
		}
		if _, duplicate := resolvedByName[skill.Name]; duplicate {
			return nil, fmt.Errorf("Run Skill snapshot contains duplicate names")
		}
		artifact := *skill.Artifact
		revision := *artifact.Revision
		artifact.Revision = &revision
		resolvedByName[skill.Name] = contracts.ResolvedSkill{
			Name: skill.Name, Artifact: artifact, PackageDigest: skill.PackageDigest,
		}
	}
	names := make([]string, 0, len(stage.Agents))
	for name := range stage.Agents {
		names = append(names, name)
	}
	sort.Strings(names)
	result := make([]controlplane.BindingRequirement, 0, len(names))
	for _, name := range names {
		binding := stage.Agents[name]
		selection := stage.ExecutionConfig.Agents[name]
		resolvedSkills := make([]contracts.ResolvedSkill, 0, len(binding.Template.Skills))
		for _, logical := range binding.Template.Skills {
			resolved, ok := resolvedByName[logical.Name]
			if !ok {
				return nil, fmt.Errorf("Run Skill %q is unavailable for Agent %q", logical.Name, name)
			}
			resolvedSkills = append(resolvedSkills, resolved)
		}
		if err := contracts.ValidateResolvedSkills(binding.Template, resolvedSkills); err != nil {
			return nil, fmt.Errorf("resolve Skills for Agent %q: %w", name, err)
		}
		result = append(result, controlplane.BindingRequirement{
			LogicalAgentName:  name,
			Namespace:         binding.Namespace,
			WorkerSessionMode: stage.Session,
			AgentTemplate:     binding.Template,
			ResolvedSkills:    resolvedSkills,
			ExecutionConfig:   allocationExecutionConfig(stage, name),
			RuntimeSelection:  &selection,
			Workspace:         contracts.CloneAllocationWorkspaceSpec(workspace),
		})
	}
	return result, nil
}

func projectAllocationWorkspace(
	workspace *resolved.WorkspaceContext,
	context runstore.StageContextSnapshot,
) (*contracts.AllocationWorkspaceSpec, error) {
	if workspace == nil {
		return nil, nil
	}
	result := &contracts.AllocationWorkspaceSpec{
		Mode:    workspace.Mode,
		Sources: make([]contracts.AllocationWorkspaceSource, 0, len(workspace.Sources)),
	}
	for _, source := range workspace.Sources {
		pinned, exists := context.Artifacts[source.Artifact]
		if !exists || pinned.Artifact == nil {
			return nil, fmt.Errorf("workspace source %q has no exact StageContext pin", source.Artifact)
		}
		result.Sources = append(result.Sources, contracts.AllocationWorkspaceSource{
			Artifact: pinned.Artifact.Clone(), Target: source.Target,
		})
	}
	if workspace.State != nil {
		pinned, exists := context.Artifacts[workspace.State.Artifact]
		if !exists {
			return nil, fmt.Errorf("workspace state %q is absent from StageContext", workspace.State.Artifact)
		}
		if pinned.Artifact != nil {
			result.State = &contracts.AllocationWorkspaceState{
				Artifact: pinned.Artifact.Clone(),
			}
		}
	}
	if workspace.Export != nil {
		result.Export = &contracts.AllocationWorkspaceExport{
			State: workspace.Export.State, Diff: workspace.Export.Diff,
		}
	}
	if err := result.Validate(); err != nil {
		return nil, fmt.Errorf("project allocation workspace: %w", err)
	}
	return result, nil
}

func allocationExecutionConfig(
	stage resolved.ResolvedStage,
	logicalAgentName string,
) controlplane.AllocationExecutionConfig {
	selection := stage.ExecutionConfig.Agents[logicalAgentName]
	result := controlplane.AllocationExecutionConfig{
		ModelPolicy: selection.ModelPolicy.Ref,
	}
	if selection.LLMGateway != nil {
		result.LLMGateway = selection.LLMGateway.Ref
	}
	if selection.Credential != nil {
		credential := *selection.Credential
		result.Credential = &credential
	}
	return result
}

func sameAllocationExecutionConfig(
	left controlplane.AllocationExecutionConfig,
	right controlplane.AllocationExecutionConfig,
) bool {
	if left.ModelPolicy != right.ModelPolicy || left.LLMGateway != right.LLMGateway {
		return false
	}
	return left.Credential == nil && right.Credential == nil ||
		left.Credential != nil && right.Credential != nil && *left.Credential == *right.Credential
}
