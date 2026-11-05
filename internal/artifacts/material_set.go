package artifacts

import (
	"context"
	"fmt"

	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

// MaterialAuthority is supplied by the authenticated/locked Run owner, never
// read from a selection document. It permits no cross-Project or cross-Audit read.
type MaterialAuthority struct{ OwnerID, ProjectID, AuditID string }

// MaterialSetInput prepares one declared ordinary Workflow input in the Run
// creation transaction. MediaTypes constrains members, not the JSON descriptor.
type MaterialSetInput struct {
	Authority   MaterialAuthority
	Source      Scope
	Ref         contracts.ArtifactRef
	RunID, Slot string
	MediaTypes  []string
	Required    bool
	Budget      *contracts.MaterialBudget
}

// MaterializeInputSet validates the whole explicit set, creates ordinary exact
// member forks, then replaces the input working copy with a local descriptor.
// The initial input_fork remains its repeat provenance. The enclosing Run
// transaction owns rollback and the dispatch fence; payload blobs are shared.
func (s *Service) MaterializeInputSet(ctx context.Context, p MaterialSetInput) (contracts.ExactArtifact, error) {
	if p.Budget == nil || p.Authority.OwnerID == "" || p.Ref.ValidateExact() != nil || contracts.ValidateArtifactName(p.Slot) != nil {
		return contracts.ExactArtifact{}, ErrInvalidName
	}
	set, scopes, err := s.prepareInputSet(ctx, p)
	if err != nil {
		return contracts.ExactArtifact{}, err
	}
	run, err := RunScope(p.RunID)
	if err != nil {
		return contracts.ExactArtifact{}, err
	}
	local := contracts.MaterialSet{Schema: contracts.MaterialSetSchema, Materials: make([]contracts.MaterialDescriptor, 0, len(set.Materials))}
	for i, m := range set.Materials {
		kind := LineageInputFork
		if scopes[i].Kind() == ScopeAudit {
			kind = LineageAuditInput
		}
		fork, e := s.Fork(ctx, Fork{Kind: kind, Source: scopes[i], SourceRef: m.Ref, Target: run, TargetRef: contracts.ArtifactRef{Namespace: "material-inputs", Name: contracts.MaterialHandle(m)}, Frozen: true, Replay: true})
		if e != nil {
			return contracts.ExactArtifact{}, e
		}
		m.ScopeKind = "run"
		m.ScopeID = p.RunID
		m.Ref = fork.TargetRef
		local.Materials = append(local.Materials, m)
	}
	kind := LineageInputFork
	if p.Source.Kind() == ScopeAudit {
		kind = LineageAuditInput
	}
	input, err := s.Fork(ctx, Fork{Kind: kind, Source: p.Source, SourceRef: p.Ref, Target: run, TargetRef: contracts.ArtifactRef{Namespace: "inputs", Name: p.Slot}})
	if err != nil {
		return contracts.ExactArtifact{}, err
	}
	data, err := strictjson.Canonical(local)
	if err != nil {
		return contracts.ExactArtifact{}, err
	}
	written, err := s.Replace(ctx, run, contracts.ArtifactRef{Namespace: "inputs", Name: p.Slot}, Payload{MediaType: "application/json", Data: data}, *input.TargetRef.Revision)
	if err != nil {
		return contracts.ExactArtifact{}, err
	}
	return contracts.ExactContent(written.Ref, written.MediaType, data), nil
}

// ValidateInputSet checks a selected set without creating bindings. Run creation
// preflights every set and singleton against one cumulative budget before writes.
func (s *Service) ValidateInputSet(ctx context.Context, p MaterialSetInput) error {
	_, _, err := s.prepareInputSet(ctx, p)
	return err
}
func (s *Service) prepareInputSet(ctx context.Context, p MaterialSetInput) (contracts.MaterialSet, []Scope, error) {
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
	meta, err := source.Metadata(ctx, p.Ref)
	if err != nil {
		return contracts.MaterialSet{}, nil, err
	}
	if meta.Size > contracts.MaxMaterialDescriptorBytes {
		return contracts.MaterialSet{}, nil, contracts.Invalidf("material descriptor bytes exceeded")
	}
	if meta.MediaType != "application/json" {
		return contracts.MaterialSet{}, nil, fmt.Errorf("%w: material set descriptor", ErrInvalidMediaType)
	}
	if err := p.Budget.AddSetDocument(contracts.MaterialDescriptor{ScopeKind: string(p.Source.Kind()), ScopeID: p.Source.ID(), Ref: meta.Ref, Digest: meta.Digest, MediaType: meta.MediaType, SizeBytes: meta.Size}, meta.Size); err != nil {
		return contracts.MaterialSet{}, nil, err
	}
	raw, err := source.Read(ctx, p.Ref)
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
		if e := p.Budget.Add(m); e != nil {
			return contracts.MaterialSet{}, nil, e
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
	return set, scopes, nil
}
