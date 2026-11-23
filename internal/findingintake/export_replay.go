package findingintake

import (
	"context"
	"errors"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/findingexport"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

// readExportPublication verifies all four immutable bindings before returning
// the original bytes. It never asks current findings or the current renderer.
func readExportPublication(ctx context.Context, repository *artifactpostgres.Repository, scope artifacts.Scope, auditID, id, requestDigest string) (PublishedExport, map[string]artifacts.ReadResult, bool, error) {
	fail := func(err error) (PublishedExport, map[string]artifacts.ReadResult, bool, error) {
		return PublishedExport{}, nil, false, err
	}
	receiptRef := exportRef(auditID, id, "export-receipt")
	metadata, err := repository.Metadata(ctx, scope, receiptRef)
	if errors.Is(err, artifacts.ErrArtifactNotFound) {
		for _, part := range []string{"export-snapshot", "export", "export-diagnostics"} {
			_, err := repository.Metadata(ctx, scope, exportRef(auditID, id, part))
			if err == nil {
				return fail(ErrConflict)
			}
			if !errors.Is(err, artifacts.ErrArtifactNotFound) {
				return fail(err)
			}
		}
		return PublishedExport{}, nil, false, nil
	}
	if err != nil {
		return fail(err)
	}
	if !metadata.Frozen || metadata.MediaType != "application/json" || metadata.Size <= 0 || metadata.Size > 8192 {
		return fail(ErrConflict)
	}
	read, err := repository.Read(ctx, scope, metadata.Ref)
	if err != nil {
		return fail(err)
	}
	if int64(len(read.Payload.Data)) != metadata.Size || contentdigest.Bytes(read.Payload.Data) != metadata.Digest || read.Payload.MediaType != "application/json" {
		return fail(blob.ErrIntegrity)
	}
	var receipt exportPublication
	if strictjson.DecodeNonNull(read.Payload.Data, &receipt) != nil || receipt.Schema != exportPublicationSchema || !contentdigest.Valid(receipt.RequestDigest) {
		return fail(ErrConflict)
	}
	result := receipt.Published
	if requestDigest != "" && receipt.RequestDigest != requestDigest {
		return fail(ErrCollectionKeyConflict)
	}
	at, err := time.Parse(time.RFC3339Nano, result.SnapshotAt)
	if err != nil || at.UTC().Format(time.RFC3339Nano) != result.SnapshotAt || result.ExportID != id || result.SelectedCount < 0 || result.SelectedCount > auditdomain.MaximumCollectionEntries || result.ResultCount < 0 || result.ResultCount > result.SelectedCount || result.WarningCount < 0 || result.WarningCount > 4096 || result.Replayed {
		return fail(ErrConflict)
	}
	descriptors := []ExactArtifact{result.Snapshot, result.Artifact, result.Diagnostics}
	parts := []string{"export-snapshot", "export", "export-diagnostics"}
	requests := make([]artifactpostgres.ExactReadRequest, len(descriptors))
	for i, exact := range descriptors {
		maximum := int64(findingexport.MaxBytes)
		if i == 2 {
			maximum = findingexport.MaxDiagnosticsBytes
		}
		expected := exportRef(auditID, id, parts[i])
		if exact.Ref.ValidateExact() != nil || exact.Ref.Namespace != expected.Namespace || exact.Ref.Name != expected.Name || exact.MediaType != "application/json" || exact.SizeBytes <= 0 || exact.SizeBytes > maximum || !contentdigest.Valid(exact.Digest) {
			return fail(ErrConflict)
		}
		latest, err := repository.Metadata(ctx, scope, expected)
		if err != nil {
			return fail(err)
		}
		if !latest.Frozen || !latest.Ref.SameExact(exact.Ref) || latest.Digest != exact.Digest || latest.MediaType != exact.MediaType || latest.Size != exact.SizeBytes {
			return fail(blob.ErrIntegrity)
		}
		requests[i] = artifactpostgres.ExactReadRequest{Scope: scope, Ref: exact.Ref}
	}
	reads, err := repository.ReadExactBatch(ctx, requests)
	if err != nil {
		return fail(err)
	}
	contents := make(map[string]artifacts.ReadResult, len(parts))
	for i, value := range reads {
		exact := descriptors[i]
		if int64(len(value.Payload.Data)) != exact.SizeBytes || value.Payload.MediaType != exact.MediaType || contentdigest.Bytes(value.Payload.Data) != exact.Digest {
			return fail(blob.ErrIntegrity)
		}
		contents[parts[i]] = value
	}
	snapshot, err := findingexport.DecodeSnapshot(contents["export-snapshot"].Payload.Data)
	if err != nil || snapshot.AuditID != auditID || snapshot.RequestDigest != receipt.RequestDigest || snapshot.SnapshotAt != result.SnapshotAt || exportID(auditID, snapshot.Request.ClientKey) != id || len(snapshot.Results) != result.SelectedCount {
		return fail(blob.ErrIntegrity)
	}
	groups, err := snapshot.ResultGroups()
	if err != nil || len(groups) != result.ResultCount {
		return fail(blob.ErrIntegrity)
	}
	diagnostics, err := findingexport.DecodeDiagnostics(contents["export-diagnostics"].Payload.Data)
	if err != nil || diagnostics.AuditID != auditID || len(diagnostics.Items) != result.WarningCount {
		return fail(blob.ErrIntegrity)
	}
	selected := make(map[string]bool, len(snapshot.Results))
	for _, r := range snapshot.Results {
		selected[r.FindingID] = true
	}
	for _, d := range diagnostics.Items {
		if !selected[d.FindingID] {
			return fail(blob.ErrIntegrity)
		}
	}
	result.Replayed = true
	return result, contents, true, nil
}
