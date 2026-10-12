//go:build integration

package findingintake

import (
	"context"
	"fmt"
	"slices"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
)

// Publishing a collection of the most entries runs the statements of a
// two-entry publication plus one exact-read batch per MaxExactReadBatchSize
// further documents: sources, receipts, holds, the Audits of the holds and
// the source Runs are each read once per publication, not once per entry.
func TestPostgresCollectionPublicationReadsDoNotGrowPerEntry(t *testing.T) {
	f := newDeletionImportFixture(t)
	ctx, cancel := context.WithTimeout(t.Context(), 3*time.Minute)
	defer cancel()
	f.ctx = ctx
	const entries = auditdomain.MaximumCollectionEntries
	evidence := writeFixtureEvidence(t, f, entries)
	receiptIDs := make([]string, entries)
	requests := make([]ImportRequest, entries)
	for i := range entries {
		receipt := insertAuditChildReceiptWithEvidence(t, f, fmt.Sprintf("publish-%03d", i), evidence[i:i+1])
		receiptIDs[i] = receipt.ReceiptID
		requests[i] = f.request
		requests[i].Proposal = receipt.Proposal.Ref
	}
	// The fixture Audit holds every receipt, so publication checks the
	// ownership of each hold's Audit.
	for start := 0; start < entries; start += MaxAuditReceiptBatchSize {
		if err := f.intake.RetainAuditCollectionBatch(ctx, requests[start:min(start+MaxAuditReceiptBatchSize, entries)]); err != nil {
			t.Fatal(err)
		}
	}
	slices.Sort(receiptIDs)
	trace := &collectionQueryCounter{}
	actor, _ := deletionActorPool(t, f, trace)
	publisher, err := NewCollectionPublisher(actor, emptyCollectionReviews{})
	if err != nil {
		t.Fatal(err)
	}
	publish := func(clientKey string, ids []string) (int64, auditdomain.FindingCollection) {
		t.Helper()
		trace.count.Store(0)
		published, err := publisher.PublishCollection(ctx, PublishCollectionParams{
			OwnerID: f.request.OwnerID, Request: PublishCollectionRequest{
				ClientKey: clientKey, Sources: []CollectionSelection{{
					Kind: "run", ID: f.request.RunID, ReceiptIDs: ids, Findings: []CollectionFindingSelection{},
				}},
			},
		})
		statements := trace.count.Load()
		if err != nil || published.EntryCount != len(ids) || published.Replayed {
			t.Fatalf("publish %d entries = (%+v, %v)", len(ids), published, err)
		}
		user, err := artifacts.NewService(artifacts.NewPostgresRepository(f.pool)).User(f.request.OwnerID)
		if err != nil {
			t.Fatal(err)
		}
		archive, err := user.Read(ctx, published.Artifact.Ref)
		if err != nil {
			t.Fatal(err)
		}
		collection, _, err := auditdomain.DecodeFindingCollectionPackage(archive.Payload.Data)
		if err != nil {
			t.Fatal(err)
		}
		return statements, collection
	}
	small, _ := publish("publish-small", receiptIDs[:2])
	large, collection := publish("publish-large", receiptIDs)
	t.Logf("statements: %d for 2 entries, %d for %d entries", small, large, entries)
	// Each entry has two documents of its own: its proposal and its evidence.
	batches := func(documents int64) int64 {
		return (documents + artifacts.MaxExactReadBatchSize - 1) / artifacts.MaxExactReadBatchSize
	}
	if extra := batches(2*entries) - batches(2*2); large > small+extra {
		t.Fatalf("a %d-entry publication ran %d statements and a 2-entry one %d; want at most %d more, one per exact-read batch",
			entries, large, small, extra)
	}
	if len(collection.Entries) != entries || len(collection.Documents) != 2*entries {
		t.Fatalf("published %d entries and %d documents", len(collection.Entries), len(collection.Documents))
	}
	for _, entry := range collection.Entries {
		if len(entry.Evidence) != 1 || entry.Evidence[0].EvidenceID != "evidence-1" ||
			!slices.Equal(entry.AuditHolds, []string{f.request.AuditID}) || entry.Retention != "audit-held" {
			t.Fatalf("published entry = %+v", entry)
		}
	}
}
