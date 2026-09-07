package config

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// AuditWorkerCompletion is an immutable opt-in binding, never a public Run override.
type AuditWorkerCompletion struct {
	Kind  string `json:"kind" yaml:"kind"`
	Stage string `json:"stage" yaml:"stage"`
	Agent string `json:"agent" yaml:"agent"`
}

func ValidateAuditWorkerCompletion(binding ResolvedAuditWorkflowBinding) error {
	c := binding.WorkerCompletion
	if c == nil {
		return nil
	}
	if c.Kind != contracts.AuditCheckResultsV1 || binding.Kind != AuditWorkflowCheck {
		return fmt.Errorf("workerCompletion requires a check binding and audit-check-results@1")
	}
	if err := contracts.ValidateArtifactName(c.Stage); err != nil {
		return err
	}
	if err := contracts.ValidateArtifactName(c.Agent); err != nil {
		return err
	}
	stage, ok := binding.Workflow.Stages[c.Stage]
	if !ok {
		return fmt.Errorf("workerCompletion names an absent Stage")
	}
	if err := validateAuditCompletionStage(binding, c.Stage, c.Agent, stage); err != nil {
		return err
	}
	// Escalation currently replaces effective model access, while the stage's
	// template/result ownership stays fixed. Validate the fully effective stage
	// for every bounded retry/escalation branch, including future template checks.
	var visit func(TransitionAction, int, string) error
	visit = func(action TransitionAction, depth int, outcome string) error {
		if depth > 32 {
			return fmt.Errorf("workerCompletion transition depth exceeded")
		}
		if action.Escalate != nil {
			variant := stage
			variant.ExecutionConfig = cloneStageExecutionConfig(action.Escalate.ExecutionConfig.Effective)
			if err := validateStageEscalationVariant(c.Stage, outcome, stage, action); err != nil {
				return err
			}
			if err := validateAuditCompletionStage(binding, c.Stage, c.Agent, variant); err != nil {
				return err
			}
			return visit(action.Escalate.Then, depth+1, outcome)
		}
		if action.Retry != nil {
			return visit(action.Retry.Then, depth+1, outcome)
		}
		return nil
	}
	for outcome, action := range map[string]TransitionAction{"succeeded": stage.On.Succeeded, "failed": stage.On.Failed, "interrupted": stage.On.Interrupted} {
		if err := visit(action, 0, outcome); err != nil {
			return err
		}
	}
	return nil
}

func validateAuditCompletionStage(binding ResolvedAuditWorkflowBinding, stageName, agentName string, stage ResolvedStage) error {
	if stage.Planner.PlannerID != "passthrough" || stage.Planner.Version != "1" || len(stage.Agents) != 1 {
		return fmt.Errorf("workerCompletion requires a single-Worker passthrough@1 Stage")
	}
	agent, ok := stage.Agents[agentName]
	if !ok {
		return fmt.Errorf("workerCompletion names an absent Worker")
	}
	if err := contracts.ValidateAuditCompletionTemplate(agent.Template); err != nil {
		return err
	}
	if err := validateStageExecutionConfig(stageName, stage); err != nil {
		return err
	}
	output, ok := binding.Outputs["result"]
	if !ok {
		return fmt.Errorf("workerCompletion requires the canonical logical result output")
	}
	resultName, ok := stage.WorkflowOutputs[output]
	if !ok {
		return fmt.Errorf("workerCompletion Stage does not supply the canonical result output")
	}
	result, ok := stage.Result.Artifacts[resultName]
	if !ok || !result.Required || result.From == nil || result.From.Namespace != agent.Namespace ||
		!mediaTypesIntersect(result.MediaTypes, []string{"application/zip"}) {
		return fmt.Errorf("workerCompletion result must be a required ZIP bound to the selected Worker namespace")
	}
	for otherName, other := range binding.Workflow.Stages {
		if otherName != stageName {
			if _, exists := other.WorkflowOutputs[output]; exists {
				return fmt.Errorf("workerCompletion canonical result has another Stage producer")
			}
		}
	}
	// Input names are mappings, not fixed aliases; each trusted source must be
	// required by the Workflow and passed as a required input into this Stage.
	for _, source := range []AuditWorkflowInputSource{AuditInputFromItemPackage, AuditInputFromExecutionManifest} {
		count, mappedCount := 0, 0
		for slot, mapping := range binding.Inputs {
			if mapping.Source != source {
				continue
			}
			mappedCount++
			if !binding.Workflow.Inputs[slot].Required {
				return fmt.Errorf("workerCompletion input must be required")
			}
			for _, context := range stage.Context.Artifacts {
				if context.Namespace == "inputs" && context.Name == slot && context.Required {
					count++
				}
			}
		}
		if mappedCount != 1 || count != 1 {
			return fmt.Errorf("workerCompletion requires one mapped task and execution manifest")
		}
	}
	return nil
}
