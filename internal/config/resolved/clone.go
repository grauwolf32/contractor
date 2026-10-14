package resolved

// The configuration Snapshot hands out, and every holder of a decoded
// snapshot keeps, values that share no mutable state with another holder.
// These are the deep copies they use.

// Clone returns a deep copy of workflow that shares no mutable state with it,
// so a decoded snapshot can be reused without a holder seeing another's edits.
func (workflow ResolvedWorkflow) Clone() ResolvedWorkflow { return cloneWorkflow(workflow) }

// Clone returns a deep copy of execution.
func (execution ResolvedStageExecutionConfig) Clone() ResolvedStageExecutionConfig {
	return cloneStageExecutionConfig(execution)
}

// Clone returns a deep copy of override.
func (override ResolvedStageExecutionConfigOverride) Clone() ResolvedStageExecutionConfigOverride {
	return cloneStageExecutionConfigOverride(override)
}

// Clone returns a deep copy of profile.
func (profile ResolvedExecutionConfigProfile) Clone() ResolvedExecutionConfigProfile {
	return cloneExecutionConfigProfile(profile)
}

func cloneWorkflow(source ResolvedWorkflow) ResolvedWorkflow {
	result := source
	result.AuditTask = cloneAuditTask(source.AuditTask)
	if source.Presentation != nil {
		presentation := *source.Presentation
		result.Presentation = &presentation
	}
	result.Parameters = make(map[string]ParameterSlot, len(source.Parameters))
	for name, slot := range source.Parameters {
		result.Parameters[name] = slot
	}
	result.Inputs = CloneArtifactSlots(source.Inputs)
	result.Outputs = CloneArtifactSlots(source.Outputs)
	result.Stages = make(map[string]ResolvedStage, len(source.Stages))
	for name, stage := range source.Stages {
		result.Stages[name] = cloneStage(stage)
	}
	return result
}

// CloneArtifactSlots returns a never-nil copy whose slots share no media types
// or source references with source.
func CloneArtifactSlots(source map[string]ArtifactSlot) map[string]ArtifactSlot {
	result := make(map[string]ArtifactSlot, len(source))
	for name, slot := range source {
		slot.MediaTypes = append([]string(nil), slot.MediaTypes...)
		if slot.From != nil {
			from := *slot.From
			slot.From = &from
		}
		result[name] = slot
	}
	return result
}

func cloneStage(source ResolvedStage) ResolvedStage {
	result := source
	result.ScanPlan = cloneScanPlanPolicy(source.ScanPlan)
	result.AuditScan = cloneAuditScan(source.AuditScan)
	result.Agents = make(map[string]ResolvedAgentBinding, len(source.Agents))
	for name, binding := range source.Agents {
		binding.Template = binding.Template.Clone()
		result.Agents[name] = binding
	}
	result.ExecutionConfig = cloneStageExecutionConfig(source.ExecutionConfig)
	result.Context.Artifacts = make(map[string]ContextArtifact, len(source.Context.Artifacts))
	for name, artifact := range source.Context.Artifacts {
		result.Context.Artifacts[name] = artifact
	}
	if source.Context.Workspace != nil {
		workspace := *source.Context.Workspace
		workspace.Sources = append([]WorkspaceSource(nil), source.Context.Workspace.Sources...)
		if source.Context.Workspace.State != nil {
			state := *source.Context.Workspace.State
			workspace.State = &state
		}
		if source.Context.Workspace.Export != nil {
			export := *source.Context.Workspace.Export
			workspace.Export = &export
		}
		result.Context.Workspace = &workspace
	}
	result.Result.Artifacts = CloneArtifactSlots(source.Result.Artifacts)
	result.WorkflowOutputs = make(map[string]string, len(source.WorkflowOutputs))
	for output, artifact := range source.WorkflowOutputs {
		result.WorkflowOutputs[output] = artifact
	}
	result.On = StageTransitions{
		Succeeded:   cloneTransition(source.On.Succeeded),
		Failed:      cloneTransition(source.On.Failed),
		Interrupted: cloneTransition(source.On.Interrupted),
	}
	if source.On.Summarized != nil {
		action := cloneTransition(*source.On.Summarized)
		result.On.Summarized = &action
	}
	return result
}

func cloneStageExecutionConfig(source ResolvedStageExecutionConfig) ResolvedStageExecutionConfig {
	result := ResolvedStageExecutionConfig{
		Agents: make(map[string]ResolvedConsumerExecutionConfig, len(source.Agents)),
	}
	if source.Planner != nil {
		planner := cloneConsumerExecutionConfig(*source.Planner)
		result.Planner = &planner
	}
	for name, selection := range source.Agents {
		result.Agents[name] = cloneConsumerExecutionConfig(selection)
	}
	return result
}

func cloneConsumerExecutionConfig(source ResolvedConsumerExecutionConfig) ResolvedConsumerExecutionConfig {
	result := source
	result.ModelPolicy = source.ModelPolicy.Clone()
	if source.LLMGateway != nil {
		gateway := source.LLMGateway.Clone()
		result.LLMGateway = &gateway
	}
	if source.Credential != nil {
		credential := *source.Credential
		result.Credential = &credential
	}
	return result
}

func cloneExecutionConfigProfile(source ResolvedExecutionConfigProfile) ResolvedExecutionConfigProfile {
	result := source
	result.Override = cloneStageExecutionConfigOverride(source.Override)
	return result
}

func cloneStageExecutionConfigOverride(
	source ResolvedStageExecutionConfigOverride,
) ResolvedStageExecutionConfigOverride {
	result := ResolvedStageExecutionConfigOverride{}
	if source.Planner != nil {
		planner := cloneExecutionSelectionOverride(*source.Planner)
		result.Planner = &planner
	}
	if source.Agents != nil {
		result.Agents = make(map[string]ResolvedExecutionSelectionOverride, len(source.Agents))
		for name, selection := range source.Agents {
			result.Agents[name] = cloneExecutionSelectionOverride(selection)
		}
	}
	return result
}

func cloneExecutionSelectionOverride(
	source ResolvedExecutionSelectionOverride,
) ResolvedExecutionSelectionOverride {
	result := source
	if source.ModelPolicy != nil {
		policy := source.ModelPolicy.Clone()
		result.ModelPolicy = &policy
	}
	if source.LLMGateway != nil {
		gateway := source.LLMGateway.Clone()
		result.LLMGateway = &gateway
	}
	if source.Credential != nil {
		credential := *source.Credential
		if source.Credential.Ref != nil {
			ref := *source.Credential.Ref
			credential.Ref = &ref
		}
		result.Credential = &credential
	}
	return result
}

func cloneTransition(source TransitionAction) TransitionAction {
	result := source
	if source.Retry != nil {
		result.Retry = &RetryTransition{
			MaxAttempts: source.Retry.MaxAttempts,
			Then:        cloneTransition(source.Retry.Then),
		}
	}
	if source.Escalate != nil {
		result.Escalate = &EscalateTransition{
			MaxAttempts:     source.Escalate.MaxAttempts,
			DeadlineSeconds: source.Escalate.DeadlineSeconds,
			ExecutionConfig: cloneEscalationExecutionConfig(source.Escalate.ExecutionConfig),
			Then:            cloneTransition(source.Escalate.Then),
		}
		if source.Escalate.Planner != nil {
			ref := *source.Escalate.Planner
			result.Escalate.Planner = &ref
		}
	}
	return result
}

func cloneEscalationExecutionConfig(
	source ResolvedEscalationExecutionConfig,
) ResolvedEscalationExecutionConfig {
	result := source
	if source.Ref != nil {
		ref := *source.Ref
		result.Ref = &ref
	}
	result.Override = cloneStageExecutionConfigOverride(source.Override)
	result.Effective = cloneStageExecutionConfig(source.Effective)
	return result
}
