package artifacts

import (
	"context"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// InputMapping is reconstructed from persisted input bytes and fork lineage.
// Source describes the accepted source; Local describes the actual Run revision.
type InputMapping struct {
	Local  contracts.ExactArtifact
	Source contracts.MaterialDescriptor
}

// DescribeInputMaterials verifies ordinary copies and declared flat-set views.
// It needs no preparation object or history scan. Callers supply declarations
// and retain responsibility for Task/execution membership and owner authority.
func (s *Service) DescribeInputMaterials(ctx context.Context, runID string, authority MaterialAuthority, inputs []MaterialInput) (map[string]InputMapping, error) {
	runScope, err := RunScope(runID)
	if err != nil {
		return nil, err
	}
	run, err := s.Scoped(runScope)
	if err != nil {
		return nil, err
	}
	refs := make([]contracts.ArtifactRef, 0, len(inputs))
	names := make([]string, 0, len(inputs))
	for _, input := range inputs {
		if input.Ref.Namespace != "inputs" || input.Name != input.Ref.Name || input.Ref.ValidateExact() != nil {
			return nil, ErrInvalidName
		}
		refs = append(refs, input.Ref)
		names = append(names, input.Name)
	}
	actual, err := run.DescribeExact(ctx, refs)
	if err != nil {
		return nil, err
	}
	edges, err := s.inputForks(ctx, runScope, "inputs", names)
	if err != nil {
		return nil, err
	}
	originals := make([]contracts.ArtifactRef, 0, len(edges))
	for _, edge := range edges {
		originals = append(originals, edge.Target)
	}
	initial, err := run.DescribeExact(ctx, originals)
	if err != nil {
		return nil, err
	}
	byName := map[string]int{}
	for i, input := range inputs {
		byName[input.Name] = i
	}
	result := map[string]InputMapping{}
	for i, edge := range edges {
		index := byName[edge.Target.Name]
		spec := inputs[index]
		local := actual[index]
		if spec.Expected != nil {
			if err := spec.Expected.Verify(local); err != nil {
				return nil, err
			}
		}
		if !spec.Set {
			if len(spec.MediaTypes) > 0 && !contracts.AcceptsMediaType(spec.MediaTypes, local.MediaType) || !local.Ref.SameExact(edge.Target) || local.Digest != initial[i].Digest || local.MediaType != initial[i].MediaType || local.SizeBytes != initial[i].SizeBytes {
				return nil, fmt.Errorf("%w: ordinary input revision changed", blob.ErrIntegrity)
			}
		} else if err := s.verifyInputSetView(ctx, runScope, authority, spec, initial[i], local); err != nil {
			return nil, err
		}
		source := contracts.MaterialDescriptor{ScopeKind: string(edge.SourceScope), ScopeID: edge.SourceScopeID, Ref: edge.Source, Digest: initial[i].Digest, MediaType: initial[i].MediaType, SizeBytes: initial[i].SizeBytes}
		result[spec.Name] = InputMapping{Local: local, Source: source}
	}
	return result, nil
}

// inputForks follows only the two ordinary input edge kinds for selected names.
// Native, unlinked bindings are not inputs; ambiguous origins are never chosen.
func (s *Service) inputForks(ctx context.Context, scope Scope, namespace string, names []string) ([]LineageEdge, error) {
	result := make([]LineageEdge, 0, len(names))
	found := map[string]bool{}
	for start := 0; start < len(names); start += MaxExactReadBatchSize {
		for _, kind := range []string{LineageInputFork, LineageAuditInput} {
			edges, err := s.repository.DescribeForkSources(ctx, scope, namespace, names[start:min(start+MaxExactReadBatchSize, len(names))], kind, true)
			if err != nil {
				return nil, err
			}
			for _, edge := range edges {
				if found[edge.Target.Name] {
					return nil, blob.ErrIntegrity
				}
				found[edge.Target.Name] = true
				result = append(result, edge)
			}
		}
	}
	if len(result) != len(names) {
		return nil, fmt.Errorf("%w: input fork is missing", blob.ErrIntegrity)
	}
	return result, nil
}

func (s *Service) verifyInputSetView(ctx context.Context, runScope Scope, authority MaterialAuthority, spec MaterialInput, initial, local contracts.ExactArtifact) error {
	if initial.MediaType != "application/json" || local.MediaType != "application/json" || initial.SizeBytes > contracts.MaxMaterialDescriptorBytes || local.SizeBytes > contracts.MaxMaterialDescriptorBytes {
		return blob.ErrIntegrity
	}
	run, err := s.Scoped(runScope)
	if err != nil {
		return err
	}
	old, err := run.ReadExact(ctx, initial)
	if err != nil {
		return err
	}
	source, err := contracts.DecodeMaterialSet(old.Payload.Data)
	if err != nil {
		return err
	}
	current, err := run.ReadExact(ctx, local)
	if err != nil {
		return err
	}
	view, err := contracts.DecodeMaterialSet(current.Payload.Data)
	if err != nil {
		return err
	}
	originals, err := s.originalMaterials(ctx, source.Materials)
	if err != nil {
		return err
	}
	expected := map[string]contracts.MaterialDescriptor{}
	for _, m := range originals {
		if !(m.ScopeKind == "project" && m.ScopeID == authority.ProjectID || m.ScopeKind == "audit" && m.ScopeID == authority.AuditID || m.ScopeKind == "user" && m.ScopeID == authority.OwnerID && authority.ProjectID == "") || !contracts.AcceptsMediaType(spec.MediaTypes, m.MediaType) {
			return ErrInvalidScope
		}
		expected[contracts.MaterialHandle(m)] = m
	}
	if len(expected) != len(view.Materials) || spec.Required && len(expected) == 0 {
		return blob.ErrIntegrity
	}
	refs := make([]contracts.ArtifactRef, 0, len(view.Materials))
	names := make([]string, 0, len(view.Materials))
	views := map[string]contracts.MaterialDescriptor{}
	for _, m := range view.Materials {
		original, found := expected[m.Ref.Name]
		if !found || m.ScopeKind != "run" || m.ScopeID != runScope.ID() || m.Ref.Namespace != "material-inputs" || m.Digest != original.Digest || m.MediaType != original.MediaType || m.SizeBytes != original.SizeBytes {
			return blob.ErrIntegrity
		}
		refs = append(refs, m.Ref)
		names = append(names, m.Ref.Name)
		views[m.Ref.Name] = m
	}
	actual, err := run.DescribeExact(ctx, refs)
	if err != nil {
		return err
	}
	for _, m := range actual {
		if err := views[m.Ref.Name].Exact().Verify(m); err != nil {
			return err
		}
	}
	edges, err := s.inputForks(ctx, runScope, "material-inputs", names)
	if err != nil {
		return err
	}
	physical := make([]contracts.MaterialDescriptor, 0, len(edges))
	for _, edge := range edges {
		m := views[edge.Target.Name]
		if !m.Ref.SameExact(edge.Target) {
			return blob.ErrIntegrity
		}
		m.ScopeKind, m.ScopeID, m.Ref = string(edge.SourceScope), edge.SourceScopeID, edge.Source
		physical = append(physical, m)
	}
	canonical, err := s.originalMaterials(ctx, physical)
	if err != nil {
		return err
	}
	for i, m := range canonical {
		if contracts.MaterialKey(m) != contracts.MaterialKey(expected[edges[i].Target.Name]) {
			return blob.ErrIntegrity
		}
	}
	return nil
}
