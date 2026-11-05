package artifacts

import (
	"context"
	"sort"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

// MaterialInput describes a caller-authorized input. The caller interprets
// Workflow roles and exclusions; only Set asks to expand an explicit flat set.
type MaterialInput struct {
	Name                    string
	Source                  Scope
	Ref                     contracts.ArtifactRef
	Expected                *contracts.ExactArtifact
	Original                *contracts.MaterialDescriptor // trusted exact input correspondence, never model JSON
	MediaTypes              []string
	Set, Required, Research bool
}

type checkedMaterial struct{ physical, original contracts.MaterialDescriptor }
type checkedInput struct {
	spec     MaterialInput
	exact    contracts.ExactArtifact
	members  []checkedMaterial
	original contracts.MaterialDescriptor
}

// CheckedInputs is a short-lived checked selection, never a durable registry.
// It can be retained at admission or materialized in the Run transaction.
type CheckedInputs struct{ inputs []checkedInput }

// PreparedMaterial pairs readable local bytes with their original identity.
type PreparedMaterial struct{ Local, Source contracts.MaterialDescriptor }

// PreparedInputs is the complete local result consumed by context builders.
type PreparedInputs struct {
	Inputs    map[string]contracts.ExactArtifact
	Materials map[string]PreparedMaterial
}

// Refs returns the ordinary local Workflow bindings, including added contexts.
func (p PreparedInputs) Refs() map[string]contracts.ArtifactRef {
	refs := make(map[string]contracts.ArtifactRef, len(p.Inputs))
	for name, m := range p.Inputs {
		refs[name] = m.Ref
	}
	return refs
}

// SetMembers returns the exact physical members which admission must retain.
// Descriptor inputs themselves remain ordinary caller-owned input retentions.
func (p *CheckedInputs) SetMembers() []contracts.MaterialDescriptor {
	unique := map[string]contracts.MaterialDescriptor{}
	for _, input := range p.inputs {
		for _, m := range input.members {
			unique[contracts.MaterialKey(m.physical)] = m.physical
		}
	}
	keys := make([]string, 0, len(unique))
	for k := range unique {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	result := make([]contracts.MaterialDescriptor, 0, len(keys))
	for _, k := range keys {
		result = append(result, unique[k])
	}
	return result
}

// CheckInputMaterials owns exact checking, flat expansion, source identity and
// shared budget accounting. A nil budget preserves ordinary non-research input
// semantics. Every source Scope is authorized by the caller; refs inside sets
// are independently checked against authority by the Artifact owner.
func (s *Service) CheckInputMaterials(ctx context.Context, authority MaterialAuthority, inputs []MaterialInput, budget *contracts.MaterialBudget) (*CheckedInputs, error) {
	result := &CheckedInputs{}
	inputs = append([]MaterialInput(nil), inputs...)
	sort.Slice(inputs, func(i, j int) bool { return inputs[i].Name < inputs[j].Name })
	refs := map[Scope][]contracts.ArtifactRef{}
	for i, input := range inputs {
		if contracts.ValidateArtifactName(input.Name) != nil || ValidateRef(input.Ref) != nil || input.Set && input.Ref.Revision == nil || i > 0 && inputs[i-1].Name == input.Name {
			return nil, ErrInvalidName
		}
		if input.Ref.Revision != nil {
			refs[input.Source] = append(refs[input.Source], input.Ref)
		}
	}
	metadata := map[Scope]map[string]contracts.ExactArtifact{}
	for scope, selection := range refs {
		store, err := s.Scoped(scope)
		if err != nil {
			return nil, err
		}
		values, err := store.DescribeExact(ctx, selection)
		if err != nil {
			return nil, err
		}
		metadata[scope] = map[string]contracts.ExactArtifact{}
		for _, m := range values {
			metadata[scope][m.Ref.Key()] = m
		}
	}
	for _, input := range inputs {
		exact := metadata[input.Source][input.Ref.Key()]
		// Ordinary input selectors may name current; freeze that choice once.
		// Explicit material sets always require an exact descriptor revision.
		if input.Ref.Revision == nil {
			store, err := s.Scoped(input.Source)
			if err != nil {
				return nil, err
			}
			m, err := store.Metadata(ctx, input.Ref)
			if err != nil {
				return nil, err
			}
			exact = m.Exact()
		}
		if input.Expected != nil {
			if err := input.Expected.Verify(exact); err != nil {
				return nil, err
			}
		}
		checked := checkedInput{spec: input, exact: exact}
		if input.Set {
			if budget == nil {
				return nil, ErrInvalidName
			}
			set, originals, err := s.checkMaterialSet(ctx, materialSetSelection{Authority: authority, Source: input.Source, Ref: input.Ref, MediaTypes: input.MediaTypes, Required: input.Required, Budget: budget}, exact)
			if err != nil {
				return nil, err
			}
			for i, m := range set.Materials {
				checked.members = append(checked.members, checkedMaterial{m, originals[i]})
			}
		} else {
			if len(input.MediaTypes) > 0 && !contracts.AcceptsMediaType(input.MediaTypes, exact.MediaType) {
				return nil, ErrInvalidMediaType
			}
			checked.original = contracts.MaterialDescriptor{ScopeKind: string(input.Source.Kind()), ScopeID: input.Source.ID(), Ref: exact.Ref, Digest: exact.Digest, MediaType: exact.MediaType, SizeBytes: exact.SizeBytes}
		}
		if input.Original != nil {
			if input.Set || input.Expected == nil || input.Original.Validate() != nil || input.Original.Digest != exact.Digest || input.Original.MediaType != exact.MediaType || input.Original.SizeBytes != exact.SizeBytes {
				return nil, ErrInvalidName
			}
			checked.original = *input.Original
		}
		result.inputs = append(result.inputs, checked)
	}
	singletonIndices := []int{}
	singletons := []contracts.MaterialDescriptor{}
	for i, input := range result.inputs {
		if !input.spec.Set && input.spec.Research {
			singletonIndices = append(singletonIndices, i)
			singletons = append(singletons, input.original)
		}
	}
	for start := 0; start < len(singletons); start += contracts.MaxSelectedMaterials {
		originals, err := s.originalMaterials(ctx, singletons[start:min(start+contracts.MaxSelectedMaterials, len(singletons))])
		if err != nil {
			return nil, err
		}
		for i, m := range originals {
			result.inputs[singletonIndices[start+i]].original = m
			if budget != nil {
				if err := budget.Add(m); err != nil {
					return nil, err
				}
			}
		}
	}
	// Validate canonical retained sources once, before any fork. This also
	// prevents a mismatched/corrupt edge from turning metadata into authority.
	canonical := map[string]contracts.MaterialDescriptor{}
	for _, input := range result.inputs {
		if !input.spec.Set {
			physical := contracts.MaterialDescriptor{ScopeKind: string(input.spec.Source.Kind()), ScopeID: input.spec.Source.ID(), Ref: input.exact.Ref, Digest: input.exact.Digest, MediaType: input.exact.MediaType, SizeBytes: input.exact.SizeBytes}
			if input.spec.Research && contracts.MaterialKey(physical) != contracts.MaterialKey(input.original) {
				canonical[contracts.MaterialKey(input.original)] = input.original
			}
		}
		for _, m := range input.members {
			if contracts.MaterialKey(m.physical) != contracts.MaterialKey(m.original) {
				canonical[contracts.MaterialKey(m.original)] = m.original
			}
		}
	}
	groups := map[Scope][]contracts.MaterialDescriptor{}
	for _, m := range canonical {
		if !(m.ScopeKind == "audit" && m.ScopeID == authority.AuditID || m.ScopeKind == "project" && m.ScopeID == authority.ProjectID || m.ScopeKind == "user" && m.ScopeID == authority.OwnerID && authority.ProjectID == "") {
			return nil, ErrInvalidScope
		}
		scope, err := materialDescriptorScope(m)
		if err != nil {
			return nil, err
		}
		groups[scope] = append(groups[scope], m)
	}
	for scope, values := range groups {
		refs := make([]contracts.ArtifactRef, 0, len(values))
		for _, m := range values {
			refs = append(refs, m.Ref)
		}
		store, err := s.Scoped(scope)
		if err != nil {
			return nil, err
		}
		actual, err := store.DescribeExact(ctx, refs)
		if err != nil {
			return nil, err
		}
		for i, m := range values {
			if err := m.Exact().Verify(actual[i]); err != nil {
				return nil, err
			}
		}
	}

	return result, nil
}

// MaterialMappings exposes already checked leaf correspondence for new exact
// selections. Set expansion stays inside the common material operation.
func (p *CheckedInputs) MaterialMappings() map[string]PreparedMaterial {
	result := map[string]PreparedMaterial{}
	for _, input := range p.inputs {
		if input.spec.Set || !input.spec.Research {
			continue
		}
		m := input.exact
		result[input.spec.Name] = PreparedMaterial{Source: input.original, Local: contracts.MaterialDescriptor{ScopeKind: string(input.spec.Source.Kind()), ScopeID: input.spec.Source.ID(), Ref: m.Ref, Digest: m.Digest, MediaType: m.MediaType, SizeBytes: m.SizeBytes}}
	}
	return result
}

// MaterializeInputs copies one checked selection without repeating its reads or
// budget calculation. The caller owns the transaction and Run dispatch fence.
func (s *Service) MaterializeInputs(ctx context.Context, runID string, checked *CheckedInputs) (PreparedInputs, error) {
	run, err := RunScope(runID)
	if err != nil {
		return PreparedInputs{}, err
	}
	result := PreparedInputs{Inputs: map[string]contracts.ExactArtifact{}, Materials: map[string]PreparedMaterial{}}
	members := map[string]contracts.MaterialDescriptor{}
	for _, input := range checked.inputs {
		local := contracts.MaterialSet{Schema: contracts.MaterialSetSchema, Materials: []contracts.MaterialDescriptor{}}
		for _, member := range input.members {
			key := contracts.MaterialHandle(member.original)
			m, found := members[key]
			if !found {
				scope, err := materialDescriptorScope(member.original)
				if err != nil {
					return PreparedInputs{}, err
				}
				kind := LineageInputFork
				if scope.Kind() == ScopeAudit {
					kind = LineageAuditInput
				}
				fork, err := s.Fork(ctx, Fork{Kind: kind, Source: scope, SourceRef: member.original.Ref, Target: run, TargetRef: contracts.ArtifactRef{Namespace: "material-inputs", Name: key}, Frozen: true, Replay: true})
				if err != nil {
					return PreparedInputs{}, err
				}
				m = member.original
				m.ScopeKind, m.ScopeID, m.Ref = "run", runID, fork.TargetRef
				members[key] = m
			}
			local.Materials = append(local.Materials, m)
			if _, found := result.Materials[key]; !found {
				result.Materials[key] = PreparedMaterial{Local: m, Source: member.original}
			}
		}
		kind := LineageInputFork
		if input.spec.Source.Kind() == ScopeAudit {
			kind = LineageAuditInput
		}
		fork, err := s.Fork(ctx, Fork{Kind: kind, Source: input.spec.Source, SourceRef: input.exact.Ref, Target: run, TargetRef: contracts.ArtifactRef{Namespace: "inputs", Name: input.spec.Name}})
		if err != nil {
			return PreparedInputs{}, err
		}
		exact := input.exact
		exact.Ref = fork.TargetRef
		if input.spec.Set {
			data, err := strictjson.Canonical(local)
			if err != nil {
				return PreparedInputs{}, err
			}
			written, err := s.Replace(ctx, run, contracts.ArtifactRef{Namespace: "inputs", Name: input.spec.Name}, Payload{MediaType: "application/json", Data: data}, *fork.TargetRef.Revision)
			if err != nil {
				return PreparedInputs{}, err
			}
			exact = contracts.ExactContent(written.Ref, written.MediaType, data)
		} else if input.spec.Research {
			key := contracts.MaterialHandle(input.original)
			if _, found := result.Materials[key]; !found {
				result.Materials[key] = PreparedMaterial{Source: input.original, Local: contracts.MaterialDescriptor{ScopeKind: "run", ScopeID: runID, Ref: exact.Ref, Digest: exact.Digest, MediaType: exact.MediaType, SizeBytes: exact.SizeBytes}}
			}
		}
		result.Inputs[input.spec.Name] = exact
	}
	return result, nil
}

func materialDescriptorScope(m contracts.MaterialDescriptor) (Scope, error) {
	switch m.ScopeKind {
	case "project":
		return ProjectScope(m.ScopeID)
	case "audit":
		return AuditScope(m.ScopeID)
	case "user":
		return UserScope(m.ScopeID)
	case "run":
		return RunScope(m.ScopeID)
	}
	return Scope{}, ErrInvalidScope
}
