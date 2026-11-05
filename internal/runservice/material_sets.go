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
func artifactServiceMetadata(ctx context.Context, s *artifacts.Service, scope artifacts.Scope, ref contracts.ArtifactRef) (artifacts.Metadata, error) {
	store, e := s.Scoped(scope)
	if e != nil {
		return artifacts.Metadata{}, e
	}
	return store.Metadata(ctx, ref)
}

func preflightMaterialSets(ctx context.Context, s *artifacts.Service, workflow resolved.ResolvedWorkflow, inputs map[string]contracts.ArtifactRef, authority artifacts.MaterialAuthority, source func(string) (artifacts.Scope, error), excluded map[string]bool, budget *contracts.MaterialBudget) error {
	if !workflowHasMaterialSets(workflow) {
		return nil
	}
	for _, slot := range sortedInputSlots(inputs) {
		scope, err := source(slot)
		if err != nil {
			return err
		}
		contract := workflow.Inputs[slot]
		if contract.MaterialSet {
			if err := s.ValidateInputSet(ctx, artifacts.MaterialSetInput{Authority: authority, Source: scope, Ref: inputs[slot], MediaTypes: contract.MediaTypes, Required: contract.Required, Budget: budget}); err != nil {
				return err
			}
			continue
		}
		if excluded[slot] {
			continue
		}
		m, err := artifactServiceMetadata(ctx, s, scope, inputs[slot])
		if err != nil {
			return err
		}
		if !contract.AcceptsInputMediaType(m.MediaType) {
			return ErrInvalid
		}
		if err := budget.Add(contracts.MaterialDescriptor{ScopeKind: string(scope.Kind()), ScopeID: scope.ID(), Ref: m.Ref, Digest: m.Digest, MediaType: m.MediaType, SizeBytes: m.Size}); err != nil {
			return err
		}
	}
	return nil
}
