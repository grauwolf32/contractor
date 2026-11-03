//go:build integration

package findingintake

import (
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
)

func TestPostgresHydratesNineMaximumSizeProposalsAcrossBatches(t *testing.T) {
	f := newFindingImportFixture(t, false, true)
	ids := make([]string, 9)
	for n := range ids {
		receipt := insertAuditChildReceiptWithEvidence(t, f, fmt.Sprintf("large-%d", n), []ExactArtifact{}, auditdomain.MaximumDocumentBytes)
		ids[n] = receipt.ReceiptID
	}
	receipts, err := f.intake.ListAuditInbox(f.ctx, f.request.OwnerID, f.request.AuditID, ListQuery{Limit: len(ids) + 1})
	if err != nil || len(receipts) != len(ids)+1 {
		t.Fatalf("hydrate nine 8 MiB proposals: count=%d error=%v", len(receipts), err)
	}
	seen := map[string]bool{}
	for _, receipt := range receipts {
		seen[receipt.Document.ClientKey] = true
	}
	for n := range ids {
		if !seen[fmt.Sprintf("large-%d", n)] {
			t.Errorf("proposal %d lost its exact document identity", n)
		}
	}
	page, err := f.intake.ListAuditCollection(f.ctx, f.request.OwnerID, f.request.AuditID, f.request.RunID, ListQuery{Limit: 200})
	// The fixture also owns one original proposal.
	if err != nil || len(page) != len(ids)+1 {
		t.Fatalf("next-round collection page: count=%d error=%v", len(page), err)
	}
}
