package auditprofile

import (
	"fmt"
	"maps"
	"slices"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
)

type WorkerCompletionTarget struct {
	Stage       string
	Agent       string
	ContextSlot string
	Result      contracts.ArtifactRef
}

// ResolveCompletionTarget binds the existing Runtime assembler to declared
// ordinary Workflow inputs/outputs. Workflows without its toolset need no
// completion strategy. No role, archive or execution manifest is inferred.
func ResolveCompletionTarget(workflow resolved.ResolvedWorkflow, binding auditcontract.ProgramWorkflow) (*WorkerCompletionTarget, error) {
	var target *WorkerCompletionTarget
	for _, name := range slices.Sorted(maps.Keys(workflow.Stages)) {
		stage := workflow.Stages[name]
		for _, agentName := range slices.Sorted(maps.Keys(stage.Agents)) {
			agent := stage.Agents[agentName]
			if !agent.Template.SelectsAuditCompletion() {
				continue
			}
			if target != nil || len(stage.Agents) != 1 || stage.Planner.Selector() != resolved.PassthroughPlanner || resolved.ValidateAuditCompletionTemplate(agent.Template) != nil {
				return nil, fmt.Errorf("common result assembly requires one passthrough Worker target")
			}
			output, found := stage.Result.Artifacts[stage.WorkflowOutputs[binding.Outputs["result"]]]
			if !found || !output.Required || output.From == nil || output.From.Namespace != agent.Namespace || !contracts.AcceptsMediaType(output.MediaTypes, "application/json") {
				return nil, fmt.Errorf("common result assembly requires the declared JSON result")
			}
			contextSlot := ""
			for _, slot := range slices.Sorted(maps.Keys(binding.Inputs)) {
				if binding.Inputs[slot].Source != "task-context" {
					continue
				}
				for _, input := range stage.Context.Artifacts {
					if input.Required && input.Namespace == "inputs" && input.Name == slot {
						contextSlot = slot
						break
					}
				}
				if contextSlot != "" {
					break
				}
			}
			if contextSlot == "" {
				return nil, fmt.Errorf("common result assembly requires its task context in Stage inputs")
			}
			target = &WorkerCompletionTarget{Stage: name, Agent: agentName, ContextSlot: contextSlot, Result: contracts.ArtifactRef{Namespace: output.From.Namespace, Name: output.From.Name}}
		}
	}
	return target, nil
}
