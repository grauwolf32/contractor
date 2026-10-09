package auditstore

import (
	"errors"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
)

func TestPostgresCollectsMaximumProposalsAndOriginAssessment(t *testing.T) {
	f := newCollectingAuditFixture(t, "association-limit", 1<<20)
	associations := make([]FindingAssociation, auditdomain.MaximumProposalsPerItem+1)
	for n := range associations {
		id := fmt.Sprintf("receipt-limit-%d", n)
		proposal := testExact("audit-findings", fmt.Sprintf("candidate-%d", n), "proposal-r1")
		proposal.MediaType, proposal.SizeBytes = "application/json", 128
		insertAuditChildFinding(t, f, id, proposal)
		associations[n] = FindingAssociation{AssessmentID: "assessment-" + id, ReceiptID: id,
			Proposal: proposal, SemanticAssessment: "supported"}
	}
	result := testExact("outputs", "result", "result-r1")
	params := CollectParams{
		Claim: f.claim, ReceiptID: "collection-limit", ExecutionID: f.execution.ExecutionID,
		Disposition: CollectionAccepted, SourceOutput: &result, RequestDigest: testDigest("9"),
		Items: []CollectionItem{{ExecutionItemID: f.memberID, Disposition: CollectionAccepted,
			FinalDisposition: FinalAccepted, Result: &result, Coverage: emptyCoverage(), FindingAssociations: associations}},
	}
	params.Items[0].FindingAssociations = append(associations, associations[0])
	if _, _, err := f.store.Collect(f.ctx, params); !errors.Is(err, ErrInvalid) {
		t.Fatalf("over-limit collection = %v", err)
	}
	params.Items[0].FindingAssociations = associations
	if _, inserted, err := f.store.Collect(f.ctx, params); err != nil || !inserted {
		t.Fatalf("maximum proposals plus origin = (%t, %v)", inserted, err)
	}
	var count int
	if err := f.pool.QueryRow(f.ctx, `SELECT count(*) FROM audit_finding_assessments WHERE audit_id=$1`, f.audit.AuditID).Scan(&count); err != nil {
		t.Fatal(err)
	}
	if count != len(associations) {
		t.Fatalf("collected %d assessments, want %d", count, len(associations))
	}
}
