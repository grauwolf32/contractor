package auditprofile

import (
	"fmt"
	"maps"
	"slices"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
)

// ResolveProposalTarget resolves roles from the pinned ordinary program and
// Workflow graph. No input, output, Workflow or Agent name is reserved for it.
func ResolveProposalTarget(workflow resolved.ResolvedWorkflow, binding auditcontract.ProgramWorkflow) (*WorkerCompletionTarget, error) {
	output := binding.Outputs["proposals"]
	selects := false
	for _, stage := range workflow.Stages {
		for _, agent := range stage.Agents {
			selects = selects || agent.Template.SelectsHypothesisProposals()
		}
	}
	if output == "" {
		for _, mapping := range binding.Inputs {
			if mapping.Source == "prior-results" {
				return nil, fmt.Errorf("proposal context requires its declared output role")
			}
		}
		if selects {
			return nil, fmt.Errorf("proposal tool requires its declared output role")
		}
		return nil, nil
	}
	context := ""
	for slot, mapping := range binding.Inputs {
		if mapping.Source == "prior-results" {
			if context != "" {
				return nil, fmt.Errorf("proposal context role is ambiguous")
			}
			context = slot
		}
	}
	input, found := workflow.Inputs[context]
	if context == "" || !found || !input.Required || input.MaterialSet || !input.AcceptsInputMediaType("application/json") {
		return nil, fmt.Errorf("proposal context requires one declared JSON input")
	}
	var target *WorkerCompletionTarget
	for _, name := range slices.Sorted(maps.Keys(workflow.Stages)) {
		stage := workflow.Stages[name]
		result, found := stage.Result.Artifacts[stage.WorkflowOutputs[output]]
		if !found {
			continue
		}
		if target != nil || !result.Required || result.From == nil || !contracts.AcceptsMediaType(result.MediaTypes, "application/json") {
			return nil, fmt.Errorf("proposal output requires one authoritative JSON target")
		}
		agentName := ""
		for name, agent := range stage.Agents {
			if agent.Namespace == result.From.Namespace {
				if agentName != "" {
					return nil, fmt.Errorf("proposal output namespace is ambiguous")
				}
				agentName = name
			}
		}
		if agentName == "" {
			return nil, fmt.Errorf("proposal output lacks its ordinary Worker")
		}
		visible := false
		for _, input := range stage.Context.Artifacts {
			visible = visible || input.Required && input.Namespace == "inputs" && input.Name == context
		}
		if !visible {
			return nil, fmt.Errorf("proposal context is absent from Stage inputs")
		}
		target = &WorkerCompletionTarget{Stage: name, Agent: agentName, ContextSlot: context, Result: contracts.ArtifactRef{Namespace: result.From.Namespace, Name: result.From.Name}}
	}
	if target == nil {
		return nil, fmt.Errorf("proposal output is not published by the Workflow")
	}
	for name, stage := range workflow.Stages {
		for agentName, agent := range stage.Agents {
			if agent.Template.SelectsHypothesisProposals() && (name != target.Stage || agentName != target.Agent) {
				return nil, fmt.Errorf("proposal tool does not own its configured output target")
			}
		}
	}
	return target, nil
}
