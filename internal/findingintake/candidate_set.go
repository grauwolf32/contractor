package findingintake

import (
	"context"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// PublishCandidateSet consumes a flat ordinary material selection, at most 16
// candidate/evidence pairs. All envelopes validate before the first write.
func (s *Service) PublishCandidateSet(ctx context.Context, runID, stageID, claimID string, selection contracts.ArtifactRef) (CandidatePublicationSummary, error) {
	result := CandidatePublicationSummary{Complete: true, Sources: []CandidatePublicationSource{}}
	if selection.ValidateExact() != nil {
		return result, ErrInvalid
	}
	// Authenticate before even an empty set can succeed.
	if err := s.validateCandidateSetAuthority(ctx, runID, stageID, claimID, selection); err != nil {
		return result, err
	}
	store, _ := artifacts.NewService(s.storage.Repository(s.pool)).Run(runID)
	meta, err := store.Metadata(ctx, selection)
	if err != nil {
		return result, err
	}
	if meta.MediaType != "application/json" || meta.Size > contracts.MaxMaterialDescriptorBytes {
		return result, ErrInvalid
	}
	raw, err := store.ReadExact(ctx, meta.Exact())
	if err != nil {
		return result, err
	}
	set, err := contracts.DecodeMaterialSet(raw.Payload.Data)
	if err != nil {
		return result, ErrInvalid
	}
	candidates, evidence := []contracts.MaterialDescriptor{}, []contracts.MaterialDescriptor{}
	for _, material := range set.Materials {
		if material.ScopeKind != "run" || material.ScopeID != runID {
			return result, ErrAccessDenied
		}
		actual, err := store.Metadata(ctx, material.Ref)
		if err != nil {
			return result, err
		}
		if material.Exact().Verify(actual.Exact()) != nil {
			return result, ErrInvalid
		}
		switch material.MediaType {
		case CandidateMediaType:
			candidates = append(candidates, material)
		case CandidateEvidenceMediaType:
			evidence = append(evidence, material)
		default:
			return result, ErrInvalid
		}
	}
	if len(candidates) != len(evidence) {
		return result, ErrInvalid
	}
	prepared := []preparedCandidates{}
	used := map[int]bool{}
	scope, _ := artifacts.RunScope(runID)
	for _, material := range candidates {
		raw, err := store.ReadExact(ctx, material.Exact())
		if err != nil {
			return result, err
		}
		batch, err := decodeCandidateBatch(raw.Payload.Data)
		if err != nil {
			return result, err
		}
		found := -1
		for i, selected := range evidence {
			source, _, err := s.storage.Repository(s.pool).ExecutionOrigin(ctx, scope, selected.Ref)
			if err != nil {
				return result, err
			}
			if source != nil && source.Artifact.SameExact(batch.Evidence.Ref) && selected.Digest == batch.Evidence.Digest && selected.SizeBytes == batch.Evidence.SizeBytes {
				if found >= 0 {
					return result, ErrInvalid
				}
				found = i
			}
		}
		if found < 0 || used[found] {
			return result, ErrInvalid
		}
		used[found] = true
		item, err := s.prepareCandidates(ctx, CandidatePublication{RunID: runID, StageID: stageID, ClaimID: claimID, Candidates: material.Ref, Evidence: evidence[found].Ref, selection: &selection})
		if err != nil {
			return result, err
		}
		prepared = append(prepared, item)
	}
	for _, item := range prepared {
		summary, err := s.publishPreparedCandidates(ctx, item)
		result.Published += summary.Published
		result.Replayed += summary.Replayed
		result.Rejected += summary.Rejected
		result.Skipped += summary.Skipped
		result.Complete = result.Complete && summary.Complete
		for _, source := range summary.Sources {
			if len(result.Sources) < 128 {
				result.Sources = append(result.Sources, source)
			} else {
				result.SourcesTruncated = true
			}
		}
		if err != nil {
			result.Complete = false
			return result, err
		}
	}
	return result, nil
}
