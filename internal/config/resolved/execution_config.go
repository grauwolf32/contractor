package resolved

import (
	"fmt"
	"maps"
	"reflect"
	"slices"
	"strings"
)

// ResolveEscalationVariants pins, for every escalate transition of workflow,
// the effective Stage execution configuration its override produces on top of
// the Stage's base configuration, and validates it. The catalog load and Run
// Workflow resolution call it after they change a base configuration.
func ResolveEscalationVariants(workflow *ResolvedWorkflow) error {
	for _, stageName := range slices.Sorted(maps.Keys(workflow.Stages)) {
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
			if err := ValidateStageExecutionConfig(stageName, variantStage); err != nil {
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

// ValidateStageEscalationVariant checks that the effective configuration an
// escalate action pins is exactly what its override produces for stage.
// Other actions pass.
func ValidateStageEscalationVariant(
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
	if err := ValidateStageExecutionConfig(stageName, variantStage); err != nil {
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
		if IsModelFreePlanner(stage.Planner) {
			return fmt.Errorf("%s@%s does not accept Planner model configuration", stage.Planner.PlannerID, stage.Planner.Version)
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
	for _, logicalName := range slices.Sorted(maps.Keys(override.Agents)) {
		patch := override.Agents[logicalName]
		if !resolvedExecutionSelectionOverrideHasAny(patch) {
			return fmt.Errorf("agents.%s must select at least one field", logicalName)
		}
		selection, ok := stage.ExecutionConfig.Agents[logicalName]
		if !ok {
			return fmt.Errorf("agents names unknown logical Agent %q", logicalName)
		}
		if stage.Agents[logicalName].Template.IsToolWorker() {
			return fmt.Errorf("tool@1 does not accept model execution overrides")
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
		selection.ModelPolicy = override.ModelPolicy.Clone()
		selection.Origins.ModelPolicy = origin
	}
	if override.LLMGateway != nil {
		if err := override.LLMGateway.Validate(); err != nil {
			return fmt.Errorf("llmGateway is invalid: %w", err)
		}
		gateway := override.LLMGateway.Clone()
		selection.LLMGateway = &gateway
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

// ValidateWorkflowExecutionConfigs checks every Stage's scan plan, tool
// Worker bindings and base execution configuration against its Planner and
// Agents.
func ValidateWorkflowExecutionConfigs(workflow ResolvedWorkflow) error {
	for stageName, stage := range workflow.Stages {
		if err := ValidateScanPlanStage(stage); err != nil {
			return fmt.Errorf("Stage %q: %w", stageName, err)
		}
		if stage.ScanPlan != nil {
			if err := validateScanPlanInputMedia(workflow, stage); err != nil {
				return fmt.Errorf("Stage %q: %w", stageName, err)
			}
		}
		for _, binding := range stage.Agents {
			if !binding.Template.IsToolWorker() {
				continue
			}
			if stage.ScanPlan != nil {
				// The validated scan policy supplies per-job inputs and reports.
				continue
			}
			execution := binding.Template.Execution
			if execution == nil {
				return fmt.Errorf("tool@1 requires execution")
			}
			for _, source := range execution.Arguments {
				switch source.Source {
				case "parameter":
					if _, ok := workflow.Parameters[source.Name]; !ok {
						return fmt.Errorf("tool@1 argument names undeclared parameter")
					}
				case "artifact":
					if _, ok := stage.Context.Artifacts[source.Name]; !ok {
						return fmt.Errorf("tool@1 argument names undeclared context artifact")
					}
				}
			}
			output, ok := stage.Result.Artifacts[execution.ResultArtifact]
			if !ok || !output.Required || output.From == nil || output.From.Namespace != binding.Namespace || strings.HasPrefix(output.From.Name, "tool-invocation.") {
				return fmt.Errorf("tool@1 requires an owned report result binding")
			}
			if !slices.Contains(output.MediaTypes, "application/json") {
				return fmt.Errorf("tool@1 report must accept application/json")
			}
		}
		if err := ValidateStageExecutionConfig(stageName, stage); err != nil {
			return err
		}
	}
	return nil
}

// ValidateStageExecutionConfig checks that stage's execution configuration
// covers exactly its Planner and logical Agents and that each selection fits
// its consumer: none for a model-free Planner or a tool Worker, a complete
// model selection for every other consumer.
func ValidateStageExecutionConfig(stageName string, stage ResolvedStage) error {
	plannerRef := stage.Planner.PlannerID + "@" + stage.Planner.Version
	if IsModelFreePlanner(stage.Planner) {
		if stage.ExecutionConfig.Planner != nil {
			return fmt.Errorf("Stage %q %s must not have Planner executionConfig", stageName, plannerRef)
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
		hasTools := len(binding.Template.Toolsets) > 0 || len(binding.Template.Skills) > 0
		if binding.Template.IsToolWorker() {
			if !selection.ModelPolicy.IsZero() || selection.LLMGateway != nil || selection.Credential != nil || selection.Origins != (ExecutionConfigOrigins{}) {
				return fmt.Errorf("tool@1 does not accept model execution configuration")
			}
			if stage.Context.Workspace != nil {
				return fmt.Errorf("tool@1 does not support project workspace")
			}
			continue
		}
		if err := validateConsumerExecutionConfig(selection, false, hasTools); err != nil {
			return fmt.Errorf("Stage %q Agent %q executionConfig: %w", stageName, logicalName, err)
		}
		if binding.Template.Summarizer != nil {
			if err := binding.Template.Summarizer.Validate(selection.ModelPolicy); err != nil {
				return fmt.Errorf("Stage %q Agent %q summarizer: %w", stageName, logicalName, err)
			}
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
		err = ValidatePlannerModelPolicy(selection.ModelPolicy)
	} else {
		err = selection.ModelPolicy.ValidateForWorker(hasTools)
	}
	if err != nil {
		return err
	}
	if selection.LLMGateway == nil {
		if planner {
			return fmt.Errorf("Planner llmGateway is required")
		}
		if strings.TrimSpace(selection.Origins.LLMGateway) != "" {
			return fmt.Errorf("absent Worker llmGateway must not have an origin")
		}
	} else {
		if err := selection.LLMGateway.Validate(); err != nil {
			return err
		}
		if strings.TrimSpace(selection.Origins.LLMGateway) == "" {
			return fmt.Errorf("llmGateway origin is required")
		}
	}
	if strings.TrimSpace(selection.Origins.ModelPolicy) == "" {
		return fmt.Errorf("modelPolicy origin is required")
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
