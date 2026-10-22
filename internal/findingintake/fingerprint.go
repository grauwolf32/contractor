package findingintake

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// prepareFingerprint resolves only owner-pinned Stage inputs already selected
// as evidence. It runs before the intake transaction and cannot change the
// proposal, the client key or the model-visible tool response.
func (s *Service) prepareFingerprint(ctx context.Context, grant controlplane.AllocationGrant, input Submission) (auditdomain.FindingFingerprint, error) {
	absent := auditdomain.UnavailableFindingFingerprint
	if len(input.EvidenceRefs) == 0 || len(input.Proposal.Locations) == 0 {
		return absent("not_provided"), nil
	}
	stage, err := runstore.NewPostgresStore(s.pool).GetStageExecution(ctx, grant.StageExecutionID)
	if err != nil {
		return absent("not_provided"), err
	}
	if stage.RunID != grant.RunID {
		return absent("not_provided"), ErrAccessDenied
	}
	pinned := make(map[string]bool, len(stage.StageContext.Artifacts))
	key := func(ref contracts.ArtifactRef) string {
		revision := ""
		if ref.Revision != nil {
			revision = *ref.Revision
		}
		return ref.Namespace + "\x00" + ref.Name + "\x00" + revision
	}
	for _, binding := range stage.StageContext.Artifacts {
		if binding.Artifact != nil {
			pinned[key(*binding.Artifact)] = true
		}
	}
	selected := make([]contracts.ArtifactRef, 0)
	for _, ref := range input.EvidenceRefs {
		if pinned[key(ref)] {
			selected = append(selected, ref)
		}
	}
	if len(selected) > 32 {
		return absent("input_limit"), nil
	}
	runArtifacts, err := artifacts.NewService(artifacts.NewPostgresRepository(s.pool)).Run(grant.RunID)
	if err != nil {
		return absent("not_provided"), err
	}
	var source, report *artifacts.Metadata
	for _, ref := range selected {
		meta, readErr := runArtifacts.Metadata(ctx, ref)
		if errors.Is(readErr, artifacts.ErrArtifactNotFound) {
			return absent("not_provided"), ErrAccessDenied
		}
		if readErr != nil {
			return absent("not_provided"), readErr
		}
		if meta.GitSource != nil {
			if source != nil {
				return absent("asset_unavailable"), nil
			}
			value := meta
			source = &value
		}
		if meta.MediaType == "application/sarif+json" {
			if report != nil {
				return absent("producer_record_ambiguous"), nil
			}
			value := meta
			report = &value
		}
	}
	if report == nil {
		return absent("producer_key_unavailable"), nil
	}
	if source == nil {
		return absent("asset_unavailable"), nil
	}
	if report.Size > auditdomain.MaximumFingerprintReportBytes {
		return absent("input_limit"), nil
	}
	read, err := runArtifacts.ReadExact(ctx, report.Exact())
	if err != nil {
		return absent("producer_key_unavailable"), err
	}
	return auditdomain.FingerprintPinnedSARIF(read.Payload.Data, source.GitSource.RepositoryURL, source.GitSource.ResolvedCommit, source.Exact(), report.Exact(), input.Proposal.Locations), nil
}
