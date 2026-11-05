package artifacts

import (
	"context"
	"slices"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// OriginalMaterials resolves one ordinary Project/User-to-Audit retention hop.
// Callers validate the selected exact metadata and authority before using it.
// Native Audit materials keep their own identity; content digests never merge
// independent sources. At most one material budget is read, with bounded
// indexed fork lookups, without scanning lineage or interpreting payloads.
func (s *Service) OriginalMaterials(ctx context.Context, materials []contracts.MaterialDescriptor) ([]contracts.MaterialDescriptor, error) {
	if len(materials) > contracts.MaxSelectedMaterials {
		return nil, ErrInvalidName
	}
	result := slices.Clone(materials)
	type group struct {
		scope     Scope
		namespace string
	}
	groups := map[group][]string{}
	indices := map[group]map[string][]int{}
	for i, m := range materials {
		if err := m.Validate(); err != nil {
			return nil, err
		}
		if m.ScopeKind != "audit" {
			continue
		}
		scope, err := AuditScope(m.ScopeID)
		if err != nil {
			return nil, err
		}
		key := group{scope, m.Ref.Namespace}
		if indices[key] == nil {
			indices[key] = map[string][]int{}
		}
		if len(indices[key][m.Ref.Name]) == 0 {
			groups[key] = append(groups[key], m.Ref.Name)
		}
		indices[key][m.Ref.Name] = append(indices[key][m.Ref.Name], i)
	}
	for key, names := range groups {
		edges, err := s.repository.DescribeForkSources(ctx, key.scope, key.namespace, names, LineageAuditInput, true)
		if err != nil {
			return nil, err
		}
		for _, edge := range edges {
			for _, i := range indices[key][edge.Target.Name] {
				if materials[i].Ref.SameExact(edge.Target) {
					result[i].ScopeKind, result[i].ScopeID, result[i].Ref = string(edge.SourceScope), edge.SourceScopeID, edge.Source
				}
			}
		}
	}
	return result, nil
}
