package findingintake

import (
	"archive/zip"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// prepareAnchors runs before mutation locks and reads only the named selected
// Stage input. The allocation caller is revalidated by the intake transaction.
func (s *Service) prepareAnchors(ctx context.Context, grant controlplane.AllocationGrant, input Submission) (auditdomain.FindingAnchorProjection, error) {
	absent := auditdomain.UnavailableFindingAnchors
	if len(input.DerivedAnchors) == 0 {
		return absent("not_provided"), nil
	}
	basis, err := auditdomain.DecodeFindingAnchorDerivation(input.DerivedAnchors)
	if err != nil {
		return auditdomain.FindingAnchorProjection{}, fmt.Errorf("%w: derived anchors", ErrInvalid)
	}
	if basis.Availability == "unavailable" {
		return absent(basis.Reason), nil
	}
	selected := false
	for _, ref := range input.EvidenceRefs {
		if ref.SameExact(*basis.SourceRef) {
			selected = true
			break
		}
	}
	if !selected {
		return auditdomain.FindingAnchorProjection{}, fmt.Errorf("%w: anchor source is not selected evidence", ErrInvalid)
	}
	stage, err := runstore.NewPostgresStore(s.pool).GetStageExecution(ctx, grant.StageExecutionID)
	if err != nil {
		return absent("asset_unavailable"), err
	}
	if stage.RunID != grant.RunID {
		return absent("asset_unavailable"), ErrAccessDenied
	}
	pinned := false
	for _, binding := range stage.StageContext.Artifacts {
		if binding.Artifact != nil && binding.Artifact.SameExact(*basis.SourceRef) {
			pinned = true
			break
		}
	}
	if !pinned {
		return absent("asset_unavailable"), nil
	}
	store, err := artifacts.NewService(artifacts.NewPostgresRepository(s.pool)).Run(grant.RunID)
	if err != nil {
		return absent("asset_unavailable"), err
	}
	metadata, err := store.Metadata(ctx, *basis.SourceRef)
	if errors.Is(err, artifacts.ErrArtifactNotFound) {
		return absent("asset_unavailable"), ErrAccessDenied
	}
	if err != nil {
		return absent("asset_unavailable"), err
	}
	if metadata.Digest != basis.SourceDigest || metadata.MediaType != "application/zip" || metadata.Size > auditdomain.MaximumFingerprintReportBytes {
		return auditdomain.FindingAnchorProjection{}, fmt.Errorf("%w: anchor source descriptor", ErrInvalid)
	}
	if metadata.GitSource == nil {
		return absent("asset_unavailable"), nil
	}
	read, err := store.ReadExact(ctx, metadata.Exact())
	if err != nil {
		return absent("asset_unavailable"), err
	}
	if err := verifyAnchorFiles(ctx, read.Payload.Data, basis); err != nil {
		return auditdomain.FindingAnchorProjection{}, err
	}
	result, err := auditdomain.ResolveFindingAnchors(basis, metadata.GitSource.RepositoryURL, metadata.GitSource.ResolvedCommit, metadata.Exact(), input.Proposal)
	if err != nil {
		return auditdomain.FindingAnchorProjection{}, fmt.Errorf("%w: anchor derivation", ErrInvalid)
	}
	return result, nil
}

func verifyAnchorFiles(ctx context.Context, data []byte, basis auditdomain.FindingAnchorDerivation) error {
	archive, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil || len(archive.File) > 20000 {
		return fmt.Errorf("%w: anchor source archive", ErrInvalid)
	}
	wanted := make(map[string]string, len(basis.Anchors))
	for _, a := range basis.Anchors {
		if digest, ok := wanted[a.File]; ok && digest != a.FileDigest {
			return fmt.Errorf("%w: anchor file digests disagree", ErrInvalid)
		}
		wanted[a.File] = a.FileDigest
	}
	found := make(map[string]bool, len(wanted))
	var total int
	for _, file := range archive.File {
		if err := ctx.Err(); err != nil {
			return err
		}
		digest, ok := wanted[file.Name]
		if !ok {
			continue
		}
		if found[file.Name] || !file.Mode().IsRegular() || file.UncompressedSize64 > 4<<20 {
			return fmt.Errorf("%w: anchor source member", ErrInvalid)
		}
		stream, err := file.Open()
		if err != nil {
			return fmt.Errorf("%w: anchor source member", ErrInvalid)
		}
		payload, err := io.ReadAll(io.LimitReader(stream, (4<<20)+1))
		closeErr := stream.Close()
		total += len(payload)
		if err != nil || closeErr != nil || len(payload) > 4<<20 || total > 8<<20 || contentdigest.Bytes(payload) != digest {
			return fmt.Errorf("%w: anchor file digest", ErrInvalid)
		}
		found[file.Name] = true
	}
	if len(found) != len(wanted) {
		return fmt.Errorf("%w: anchor file missing", ErrInvalid)
	}
	return nil
}
