package artifacts

import (
	"context"
	"fmt"
	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// MaterialAuthority is supplied by the authenticated owner, never the JSON.
type MaterialAuthority struct{ OwnerID, ProjectID, AuditID string }

type materialSetSelection struct {
	Authority  MaterialAuthority
	Source     Scope
	Ref        contracts.ArtifactRef
	MediaTypes []string
	Required   bool
	Budget     *contracts.MaterialBudget
}

func (s *Service) checkMaterialSet(ctx context.Context, p materialSetSelection, meta contracts.ExactArtifact) (contracts.MaterialSet, []contracts.MaterialDescriptor, error) {
	if p.Budget == nil || p.Authority.OwnerID == "" || p.Ref.ValidateExact() != nil {
		return contracts.MaterialSet{}, nil, ErrInvalidName
	}
	if !(p.Source.Kind() == ScopeUser && p.Source.ID() == p.Authority.OwnerID && p.Authority.ProjectID == "" || p.Source.Kind() == ScopeProject && p.Source.ID() == p.Authority.ProjectID || p.Source.Kind() == ScopeAudit && p.Source.ID() == p.Authority.AuditID) {
		return contracts.MaterialSet{}, nil, ErrInvalidScope
	}
	source, err := s.Scoped(p.Source)
	if err != nil {
		return contracts.MaterialSet{}, nil, err
	}

	if meta.SizeBytes > contracts.MaxMaterialDescriptorBytes {
		return contracts.MaterialSet{}, nil, contracts.Invalidf("material descriptor bytes exceeded")
	}
	if meta.MediaType != "application/json" {
		return contracts.MaterialSet{}, nil, fmt.Errorf("%w: material set descriptor", ErrInvalidMediaType)
	}
	if err := p.Budget.AddSetDocument(contracts.MaterialDescriptor{ScopeKind: string(p.Source.Kind()), ScopeID: p.Source.ID(), Ref: meta.Ref, Digest: meta.Digest, MediaType: meta.MediaType, SizeBytes: meta.SizeBytes}, meta.SizeBytes); err != nil {
		return contracts.MaterialSet{}, nil, err
	}
	raw, err := source.ReadExact(ctx, meta)
	if err != nil {
		return contracts.MaterialSet{}, nil, err
	}
	set, err := contracts.DecodeMaterialSet(raw.Payload.Data)
	if err != nil {
		return contracts.MaterialSet{}, nil, err
	}
	if p.Required && len(set.Materials) == 0 {
		return contracts.MaterialSet{}, nil, fmt.Errorf("%w: required material set is empty", ErrInvalidName)
	}
	scopes := make([]Scope, len(set.Materials))
	groups := map[Scope][]contracts.ArtifactRef{}
	groupOrder := []Scope{}
	seen := map[Scope]map[string]bool{}
	for i, m := range set.Materials {
		if artifactpolicy.IsModelHiddenBinding(m.Ref.Namespace, m.Ref.Name) || !contracts.AcceptsMediaType(p.MediaTypes, m.MediaType) {
			return contracts.MaterialSet{}, nil, ErrInvalidMediaType
		}
		switch {
		case m.ScopeKind == "project" && m.ScopeID == p.Authority.ProjectID:
			scopes[i], err = ProjectScope(m.ScopeID)
		case m.ScopeKind == "user" && m.ScopeID == p.Authority.OwnerID && p.Authority.ProjectID == "":
			scopes[i], err = UserScope(m.ScopeID)
		case m.ScopeKind == "audit" && m.ScopeID == p.Authority.AuditID:
			scopes[i], err = AuditScope(m.ScopeID)
		default:
			return contracts.MaterialSet{}, nil, ErrInvalidScope
		}
		if err != nil {
			return contracts.MaterialSet{}, nil, err
		}
		if seen[scopes[i]] == nil {
			seen[scopes[i]] = map[string]bool{}
			groupOrder = append(groupOrder, scopes[i])
		}
		if !seen[scopes[i]][m.Ref.Key()] {
			groups[scopes[i]] = append(groups[scopes[i]], m.Ref)
			seen[scopes[i]][m.Ref.Key()] = true
		}

	}
	actual := map[Scope]map[string]contracts.ExactArtifact{}
	for _, scope := range groupOrder {
		store, e := s.Scoped(scope)
		if e != nil {
			return contracts.MaterialSet{}, nil, e
		}
		found, e := store.DescribeExact(ctx, groups[scope])
		if e != nil {
			return contracts.MaterialSet{}, nil, e
		}
		actual[scope] = map[string]contracts.ExactArtifact{}
		for _, m := range found {
			actual[scope][m.Ref.Key()] = m
		}
	}
	for i, m := range set.Materials {
		if e := m.Exact().Verify(actual[scopes[i]][m.Ref.Key()]); e != nil {
			return contracts.MaterialSet{}, nil, e
		}
	}
	originals, err := s.originalMaterials(ctx, set.Materials)
	if err != nil {
		return contracts.MaterialSet{}, nil, err
	}
	for _, m := range originals {
		if err := p.Budget.Add(m); err != nil {
			return contracts.MaterialSet{}, nil, err
		}
	}
	return set, originals, nil
}
