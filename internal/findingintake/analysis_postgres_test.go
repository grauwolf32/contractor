//go:build integration

package findingintake

import (
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/findinganalysis"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

func TestPostgresCollectionValidatesOptionalAnalysisBeforePublication(t *testing.T) {
	f := newDeletionImportFixture(t)
	store, err := artifacts.NewService(artifacts.NewPostgresRepository(f.pool)).Run(f.request.RunID)
	if err != nil {
		t.Fatal(err)
	}
	data := []byte("α😀\r\nsink")
	written, err := store.Write(f.ctx, artifacts.ArtifactRef{Namespace: "evidence", Name: "analysis-source"}, artifacts.Payload{MediaType: "text/plain", Data: data}, nil)
	if err != nil {
		t.Fatal(err)
	}
	source := ExactArtifact{Ref: written.Ref, Digest: auditdomain.DigestBytes(data), MediaType: written.MediaType, SizeBytes: written.Size}
	publisher, err := NewCollectionPublisher(f.pool, emptyCollectionReviews{})
	if err != nil {
		t.Fatal(err)
	}
	var retainedReceipt Receipt
	for _, valid := range []bool{true, false} {
		key := "valid-analysis"
		digest := source.Digest
		if !valid {
			key = "invalid-analysis"
			digest = auditdomain.DigestBytes([]byte("not the exact source"))
		}
		native := findinganalysis.Document{Schema: findinganalysis.Schema, ClientKey: key, Sources: []findinganalysis.Source{{ID: "source", Kind: "file", ArtifactRef: source.Ref}}, Locations: []findinganalysis.Location{{ID: "sink", SourceID: "source", Path: "src/a.py", FileDigest: digest, Region: &findinganalysis.Region{StartLine: 2}}}, Flows: []findinganalysis.Flow{}, HTTPExchanges: []findinganalysis.HTTPExchange{}, PrimaryLocationID: "sink"}
		encoded, err := strictjson.Canonical(native)
		if err != nil {
			t.Fatal(err)
		}
		written, err := store.Write(f.ctx, artifacts.ArtifactRef{Namespace: "evidence", Name: key}, artifacts.Payload{MediaType: findinganalysis.MediaType, Data: encoded}, nil)
		if err != nil {
			t.Fatal(err)
		}
		analysis := ExactArtifact{Ref: written.Ref, Digest: auditdomain.DigestBytes(encoded), MediaType: written.MediaType, SizeBytes: written.Size}
		receipt := insertAuditChildReceiptWithEvidence(t, f, key, []ExactArtifact{source, analysis})
		result, err := publisher.PublishCollection(f.ctx, PublishCollectionParams{OwnerID: f.request.OwnerID, Request: PublishCollectionRequest{ClientKey: key, Sources: []CollectionSelection{{Kind: "run", ID: f.request.RunID, ReceiptIDs: []string{receipt.ReceiptID}, Findings: []CollectionFindingSelection{}}}}})
		if valid {
			retainedReceipt = receipt
			if err != nil || result.EntryCount != 1 {
				t.Fatalf("valid publication: %+v %v", result, err)
			}
		} else {
			if auditdomain.ErrorCode(err) != auditdomain.CodeInvalid {
				t.Fatalf("invalid exact source: %v", err)
			}
			user, err := artifacts.NewService(artifacts.NewPostgresRepository(f.pool)).User(f.request.OwnerID)
			if err != nil {
				t.Fatal(err)
			}
			for _, namespace := range []string{CollectionNamespace, CollectionReceiptNamespace} {
				if _, err := user.Metadata(f.ctx, artifacts.ArtifactRef{Namespace: namespace, Name: key}); !errors.Is(err, artifacts.ErrArtifactNotFound) {
					t.Fatalf("partial publication %s: %v", namespace, err)
				}
			}
		}
	}
	request := f.request
	request.Proposal = retainedReceipt.Proposal.Ref
	if _, _, err := f.intake.ImportIntoAudit(f.ctx, request); err != nil {
		t.Fatal(err)
	}
	if err := deleteRun(f.ctx, f.pool, f.request.OwnerID, f.request.RunID); err != nil {
		t.Fatal(err)
	}
	afterDeletion, err := publisher.PublishCollection(f.ctx, PublishCollectionParams{OwnerID: f.request.OwnerID, Request: PublishCollectionRequest{ClientKey: "analysis-after-delete", Sources: []CollectionSelection{{Kind: "audit", ID: f.request.AuditID, ReceiptIDs: []string{retainedReceipt.ReceiptID}, Findings: []CollectionFindingSelection{}}}}})
	if err != nil || afterDeletion.EntryCount != 1 {
		t.Fatalf("resolve original analysis refs through Audit hold: %+v %v", afterDeletion, err)
	}

}
