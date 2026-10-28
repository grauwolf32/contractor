package projectsitemap

import (
	"context"
	"encoding/json"
	"errors"
	"slices"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/strictjson"
	"github.com/jackc/pgx/v5"
)

const SnapshotMediaType = "application/vnd.contractor.project-sitemap-context+json"

// SnapshotSelection selects only from the exact view the caller reviewed.
type SnapshotSelection struct {
	Revision                string   `json:"revision" strictjson:"required"`
	ExpectedPolicyDigest    string   `json:"expectedPolicyDigest" strictjson:"required"`
	RequireReferencedInputs bool     `json:"requireReferencedInputs,omitempty"`
	Groups                  []string `json:"groups,omitempty"`
	Members                 []string `json:"members,omitempty"`
}

func (s *SnapshotSelection) UnmarshalJSON(data []byte) error {
	type wire SnapshotSelection
	var value wire
	if err := strictjson.DecodeNonNull(data, &value); err != nil {
		return ErrInvalid
	}
	*s = SnapshotSelection(value)
	return s.Validate()
}

type SnapshotResult struct {
	Artifact contracts.ExactArtifact `json:"artifact"`
	Replayed bool                    `json:"replayed"`
}

type snapshotDocument struct {
	Schema       string                         `json:"schema"`
	Source       contracts.ArtifactRef          `json:"source"`
	PolicyDigest string                         `json:"policyDigest"`
	Policy       contracts.ProjectSitemapPolicy `json:"policy"`
	Inputs       []contracts.ExactArtifact      `json:"inputs"`
	Selection    SnapshotSelection              `json:"selection"`
	Document     Document                       `json:"document"`
}

func (s SnapshotSelection) Validate() error {
	if !contracts.ValidResourceID(s.Revision) || s.Revision == "absent" || !contentdigest.Valid(s.ExpectedPolicyDigest) || len(s.Groups)+len(s.Members) == 0 || len(s.Groups)+len(s.Members) > MaxBatch {
		return ErrInvalid
	}
	seen := map[string]bool{}
	for _, id := range append(slices.Clone(s.Groups), s.Members...) {
		if seen[id] || !hashPattern.MatchString(id) {
			return ErrInvalid
		}
		seen[id] = true
	}
	return nil
}

// SnapshotAllocation uses ordinary frozen materialization and lineage in the
// caller's Run namespace. Exact source selection and protection are atomic;
// the caller's transient view supplies no bytes or retention authority.
func (s *Service) SnapshotAllocation(ctx context.Context, access AllocationAccess, keyValue string, selection SnapshotSelection) (SnapshotResult, error) {
	if !contracts.ValidIdempotencyKey(keyValue) || selection.Validate() != nil {
		return SnapshotResult{}, ErrInvalid
	}
	if selection.ExpectedPolicyDigest != access.PolicyDigest {
		return SnapshotResult{}, ErrConflict
	}
	selection.Groups = slices.Sorted(slices.Values(selection.Groups))
	selection.Members = slices.Sorted(slices.Values(selection.Members))
	ctx, cancel := context.WithTimeout(ctx, OperationTimeout)
	defer cancel()
	var result SnapshotResult
	var err error
	for range MaxAttempts {
		err = postgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			if err := checkProject(ctx, tx, access.OwnerID, access.ProjectID, true); err != nil {
				return err
			}
			repository := artifacts.NewPostgresTxRepository(tx)
			store, err := artifacts.NewService(repository).Project(access.ProjectID)
			if err != nil {
				return err
			}
			ref := mapRef()
			ref.Revision = &selection.Revision
			read, err := store.Read(ctx, ref)
			if err != nil {
				return err
			}
			if read.Payload.MediaType != MediaType {
				return ErrInvalid
			}
			document, err := Decode(read.Payload.Data)
			if err != nil {
				return err
			}
			view, err := SafeProjection(document, access.Policy, access.RunID, access.Inputs)
			if err != nil {
				return err
			}
			selected, inputs, err := selectSnapshot(view, selection, access.RunID, access.Inputs)
			if err != nil {
				return err
			}
			if selection.RequireReferencedInputs && slices.ContainsFunc(selected.Gaps, func(gap Gap) bool { return gap.Reason == "source-unavailable" }) {
				return artifacts.ErrArtifactNotFound
			}
			name := "sitemap-context-" + key("snapshot", keyValue)
			selected, inputs, err = retainSnapshotInputs(ctx, repository, access, name, selected, inputs)
			if err != nil {
				return err
			}
			encoded, err := json.Marshal(snapshotDocument{
				Schema: "project-sitemap-context@1", Source: ref, PolicyDigest: access.PolicyDigest,
				Policy: access.Policy, Inputs: inputs, Selection: selection, Document: selected,
			})
			if err != nil || len(encoded) > MaxDocumentBytes {
				return ErrInvalid
			}
			source, _ := artifacts.ProjectScope(access.ProjectID)
			target, err := artifacts.RunScope(access.RunID)
			if err != nil {
				return err
			}
			targetRef := contracts.ArtifactRef{Namespace: access.Namespace, Name: name}
			written, replayed, err := repository.CreateContextProjection(ctx, source, ref, target, targetRef, artifacts.Payload{MediaType: SnapshotMediaType, Data: encoded})
			if err != nil {
				return err
			}
			consumer, err := artifacts.NewService(repository).Run(access.RunID)
			if err != nil {
				return err
			}
			metadata, err := consumer.Metadata(ctx, written.Ref)
			if err != nil {
				return err
			}
			result = SnapshotResult{Artifact: metadata.Exact(), Replayed: replayed}
			return nil
		})
		if err == nil || !postgres.IsTransactionConflict(err) && !errors.Is(err, artifacts.ErrArtifactConflict) {
			return result, err
		}
		if ctx.Err() != nil {
			return SnapshotResult{}, ctx.Err()
		}
	}
	return SnapshotResult{}, err
}

func selectSnapshot(view Document, selection SnapshotSelection, runID string, inputs []contracts.ExactArtifact) (Document, []contracts.ExactArtifact, error) {
	groups, members := map[string]bool{}, map[string]bool{}
	for _, id := range selection.Groups {
		groups[id] = true
	}
	for _, id := range selection.Members {
		members[id] = true
	}
	kept := []Group{}
	keptMembers := map[string]bool{}
	for _, group := range view.Groups {
		whole := groups[group.ID]
		delete(groups, group.ID)
		selected := []Member{}
		for _, member := range group.Members {
			if whole || members[member.ID] {
				selected = append(selected, member)
				keptMembers[member.ID] = true
			}
			delete(members, member.ID)
		}
		if whole || len(selected) > 0 {
			group.Members = selected
			kept = append(kept, group)
		}
	}
	if len(groups) > 0 || len(members) > 0 {
		return Document{}, nil, artifacts.ErrArtifactNotFound
	}
	view.Rules = slices.DeleteFunc(view.Rules, func(rule Rule) bool {
		return !slices.ContainsFunc(kept, func(group Group) bool {
			return group.Origin == rule.Origin && group.Method == rule.Method && group.Route == rule.Path
		})
	})
	// Grouping stores a manual declaration at its concrete path, separately
	// from the template group holding its member. Preserve only declarations
	// for the exact selected identities; a companion group adds no other
	// members, rule declarations or grouping policy to the selection.
	selectedGroups := make(map[string]bool, len(kept))
	for _, group := range kept {
		selectedGroups[group.ID] = true
	}
	for _, group := range view.Groups {
		if selectedGroups[group.ID] || !keptMembers[memberID(group.Origin, group.Method, group.Route, group.Structural...)] {
			continue
		}
		declarations := []Declaration{}
		for _, declaration := range group.Declarations {
			if !declaration.Rule {
				declarations = append(declarations, declaration)
			}
		}
		if len(declarations) > 0 {
			group.Members, group.Declarations, group.Omitted = []Member{}, declarations, 0
			kept = append(kept, group)
		}
	}
	view.Groups = kept
	used := map[string]bool{}
	for _, rule := range view.Rules {
		used[rule.SourceID] = true
	}
	for _, group := range view.Groups {
		for _, declaration := range group.Declarations {
			used[declaration.Provenance.SourceID] = true
		}
		for _, member := range group.Members {
			for _, summary := range member.Summaries {
				used[summary.Provenance.SourceID] = true
			}
			for _, annotation := range member.Annotations {
				used[annotation.Provenance.SourceID] = true
			}
		}
	}
	markers := []Marker{}
	for _, marker := range view.Markers {
		marker.Members = slices.DeleteFunc(marker.Members, func(id string) bool { return !keptMembers[id] })
		if len(marker.Members) > 0 {
			markers = append(markers, marker)
			used[marker.Provenance.SourceID] = true
		}
	}
	view.Markers = markers
	view.Sources = slices.DeleteFunc(view.Sources, func(source Source) bool { return !used[source.Provenance.SourceID] })
	if len(view.Gaps) > 0 {
		view.Gaps = []Gap{{Reason: "projection", Count: 1}}
	}
	for i := range view.Sources {
		source := &view.Sources[i]
		if source.Artifact == nil {
			continue
		}
		index := slices.IndexFunc(inputs, func(input contracts.ExactArtifact) bool {
			return input.Digest == source.Artifact.Digest && input.MediaType == source.Artifact.MediaType && input.SizeBytes == source.Artifact.SizeBytes
		})
		if index < 0 {
			source.Artifact = nil
			addGap(&view, source.Provenance.SourceID, "source-unavailable", 1)
		} else {
			input := inputs[index]
			source.Artifact = &input
		}
	}
	for i := range view.Groups {
		for j := range view.Groups[i].Members {
			member := &view.Groups[i].Members[j]
			before := len(member.Evidence)
			member.Evidence = slices.DeleteFunc(member.Evidence, func(e Evidence) bool {
				return e.RunID != runID || !slices.ContainsFunc(inputs, func(input contracts.ExactArtifact) bool {
					return input.Ref.SameExact(e.Artifact.Ref) && input.Digest == e.Artifact.Digest && input.MediaType == e.Artifact.MediaType && input.SizeBytes == e.Artifact.SizeBytes
				})
			})
			if len(member.Evidence) != before {
				addGap(&view, "", "source-unavailable", 1)
			}
		}
	}
	needed := []contracts.ExactArtifact{}
	for _, input := range inputs {
		if slices.ContainsFunc(view.Sources, func(source Source) bool { return source.Artifact != nil && source.Artifact.Ref.SameExact(input.Ref) }) || slices.ContainsFunc(view.Groups, func(group Group) bool {
			return slices.ContainsFunc(group.Members, func(member Member) bool {
				return slices.ContainsFunc(member.Evidence, func(e Evidence) bool { return e.Artifact.Ref.SameExact(input.Ref) })
			})
		}) {
			needed = append(needed, input)
		}
	}
	canonicalize(&view)
	if _, err := encode(view); err != nil {
		return Document{}, nil, err
	}
	return view, needed, nil
}

// retainSnapshotInputs forks only selected current-Run inputs through the
// existing exact lineage path. Every clone and the context commit together;
// a provenance mention alone never acquires a hold or reads a foreign scope.
func retainSnapshotInputs(ctx context.Context, repository *artifacts.PostgresTxRepository, access AllocationAccess, name string, view Document, inputs []contracts.ExactArtifact) (Document, []contracts.ExactArtifact, error) {
	service := artifacts.NewService(repository)
	scope, err := artifacts.RunScope(access.RunID)
	if err != nil {
		return Document{}, nil, err
	}
	consumer, err := service.Run(access.RunID)
	if err != nil {
		return Document{}, nil, err
	}
	retained := make([]contracts.ExactArtifact, 0, len(inputs))
	for _, input := range inputs {
		if _, err := consumer.ReadExact(ctx, input); err != nil {
			return Document{}, nil, err
		}
		fork, err := service.Fork(ctx, artifacts.Fork{Kind: artifacts.LineageContextProjection, Source: scope, SourceRef: input.Ref, Target: scope, TargetRef: contracts.ArtifactRef{Namespace: access.Namespace, Name: "sitemap-input-" + key("snapshot-input", name, string(jsonBytes(input)))}, Frozen: true, Replay: true})
		if err != nil {
			return Document{}, nil, err
		}
		metadata, err := consumer.Metadata(ctx, fork.TargetRef)
		if err != nil {
			return Document{}, nil, err
		}
		exact := metadata.Exact()
		retained = append(retained, exact)
		for i := range view.Sources {
			if view.Sources[i].Artifact != nil && view.Sources[i].Artifact.Ref.SameExact(input.Ref) {
				copy := exact
				view.Sources[i].Artifact = &copy
			}
		}
		for i := range view.Groups {
			for j := range view.Groups[i].Members {
				for k := range view.Groups[i].Members[j].Evidence {
					e := &view.Groups[i].Members[j].Evidence[k]
					if e.Artifact.Ref.SameExact(input.Ref) {
						e.Artifact = exact
					}
				}
			}
		}
	}
	return view, retained, nil
}
