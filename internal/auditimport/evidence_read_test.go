package auditimport

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestImporterRetriesEvidenceStorageFailureWithoutCollectionReceipt(t *testing.T) {
	for _, failure := range []error{
		errors.New("temporary storage failure"),
		context.DeadlineExceeded,
		context.Canceled,
		artifacts.ErrArtifactIntegrity,
	} {
		t.Run(failure.Error(), func(t *testing.T) {
			harness := newImportHarness(t)
			access := configureExternalEvidence(t, &harness)
			access.err = fmt.Errorf("read evidence: %w", failure)
			worked, err := harness.importer.Collect(context.Background(), harness.claim, harness.snapshot, harness.execution)
			if worked || !errors.Is(err, failure) {
				t.Fatalf("failed evidence read = (%t, %v), want original error", worked, err)
			}
			if harness.store.collected.Disposition != "" || len(harness.artifacts.retainedTargets) != 0 {
				t.Fatalf("storage failure settled or retained result: %+v", harness.store.collected)
			}

			access.err = nil
			worked, err = harness.importer.Collect(context.Background(), harness.claim, harness.snapshot, harness.execution)
			if err != nil || !worked || harness.store.collected.Disposition != auditstore.CollectionAccepted {
				t.Fatalf("retry after storage recovery = (%t, %v, %+v)", worked, err, harness.store.collected)
			}
		})
	}
}

func TestImporterRejectsMissingEvidenceReference(t *testing.T) {
	harness := newImportHarness(t)
	access := configureExternalEvidence(t, &harness)
	access.err = fmt.Errorf("missing evidence: %w", artifacts.ErrArtifactNotFound)
	worked, err := harness.importer.Collect(context.Background(), harness.claim, harness.snapshot, harness.execution)
	collected := harness.store.collected
	if err != nil || !worked || collected.Disposition != auditstore.CollectionInvalidResult ||
		collected.ErrorCode == nil || *collected.ErrorCode != "evidence-reference-invalid" {
		t.Fatalf("missing evidence collection = (%t, %v, %+v)", worked, err, collected)
	}
}

type evidenceReadArtifacts struct {
	ArtifactAccess
	descriptor auditstore.ExactArtifact
	err        error
}

func (a *evidenceReadArtifacts) ReadRunExact(context.Context, string, contracts.ArtifactRef) (auditstore.ExactArtifact, []byte, error) {
	return a.descriptor, []byte("evidence"), a.err
}

func configureExternalEvidence(t *testing.T, harness *importHarness) *evidenceReadArtifacts {
	t.Helper()
	pkg, err := auditdomain.DecodeCheckResultPackage(harness.artifacts.runPayload)
	if err != nil {
		t.Fatal(err)
	}
	revision := "evidence-revision"
	ref := contracts.ArtifactRef{Namespace: "evidence", Name: "source", Revision: &revision}
	pkg.Evidence.Evidence[0].ContentMemberID = ""
	pkg.Evidence.Evidence[0].Artifact = &ref
	evidence, err := auditdomain.EncodeEvidence(pkg.Evidence)
	if err != nil {
		t.Fatal(err)
	}
	results, err := auditdomain.EncodeCheckResultSet(pkg.Results)
	if err != nil {
		t.Fatal(err)
	}
	payload, rebuilt, err := auditdomain.BuildPackage("external-evidence-results", auditdomain.PackageKindCheckResults, "", []auditdomain.PackageInput{
		{ID: auditdomain.CheckResultsMemberID, Path: "check-results.json", MediaType: auditdomain.JSONMediaType, Data: results},
		{ID: auditdomain.EvidenceMemberID, Path: "evidence.json", MediaType: auditdomain.JSONMediaType, Data: evidence},
	})
	if err != nil {
		t.Fatal(err)
	}
	harness.artifacts.runPayload = payload
	harness.artifacts.runDescriptor.Digest = rebuilt.Digest
	harness.artifacts.runDescriptor.SizeBytes = int64(len(payload))
	access := &evidenceReadArtifacts{
		ArtifactAccess: harness.artifacts,
		descriptor: auditstore.ExactArtifact{
			Ref: ref, Digest: digestBytes([]byte("evidence")), MediaType: "text/plain", SizeBytes: 8,
		},
	}
	harness.importer.artifacts = access
	return access
}
