package runservice

import (
	"context"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func workflowHasMaterialSets(workflow resolved.ResolvedWorkflow) bool {
	for _, slot := range workflow.Inputs {
		if slot.MaterialSet {
			return true
		}
	}
	return false
}

// Only this adapter interprets Workflow input declarations and system exclusions.
func checkWorkflowInputs(ctx context.Context, s *artifacts.Service, workflow resolved.ResolvedWorkflow, inputs map[string]contracts.ArtifactRef, authority artifacts.MaterialAuthority, source func(string) (artifacts.Scope, error), excluded map[string]bool, expected map[string]contracts.ExactArtifact, budget *contracts.MaterialBudget) (*artifacts.CheckedInputs, error) {
	selected := make([]artifacts.MaterialInput, 0, len(inputs))
	for _, slot := range sortedInputSlots(inputs) {
		scope, err := source(slot)
		if err != nil {
			return nil, err
		}
		contract := workflow.Inputs[slot]
		input := artifacts.MaterialInput{Name: slot, Source: scope, Ref: inputs[slot], MediaTypes: contract.MediaTypes, Set: contract.MaterialSet, Required: contract.Required, Research: !excluded[slot]}
		if exact, found := expected[slot]; found {
			input.Expected = &exact
		}
		selected = append(selected, input)
	}
	return s.CheckInputMaterials(ctx, authority, selected, budget)
}
