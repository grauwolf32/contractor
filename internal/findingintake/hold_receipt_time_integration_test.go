//go:build integration

package findingintake

import (
	"testing"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

// A hold written by an owner import and one written by collection retention
// both keep their receipt's creation time, which orders the Audit inbox
// (V414-019, FND-R186); no update can change it, and a hold that names
// another time or a missing receipt is rejected.
func TestPostgresAuditHoldKeepsItsReceiptTime(t *testing.T) {
	f := newDeletionImportFixture(t)
	if _, _, err := f.intake.ImportIntoAudit(f.ctx, f.request); err != nil {
		t.Fatal(err)
	}
	collected := insertAuditChildReceipt(t, f, "collected-candidate")
	request := f.request
	request.Proposal = collected.Proposal.Ref
	if err := f.intake.RetainAuditCollectionBatch(f.ctx, []ImportRequest{request}); err != nil {
		t.Fatal(err)
	}
	rows, err := f.pool.Query(f.ctx, `
SELECT hold.receipt_id, hold.receipt_created_at = receipt.created_at
  FROM finding_proposal_audit_holds AS hold
  JOIN finding_proposal_receipts AS receipt USING (receipt_id)
 WHERE hold.audit_id = $1
 ORDER BY hold.receipt_id`, f.request.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	held := map[string]bool{}
	for rows.Next() {
		var receiptID string
		var equal bool
		if err := rows.Scan(&receiptID, &equal); err != nil {
			t.Fatal(err)
		}
		held[receiptID] = equal
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if len(held) != 2 || !held[f.receiptID] || !held[collected.ReceiptID] {
		t.Fatalf("imported and collected holds keep their receipt times = %v", held)
	}
	if _, err := f.pool.Exec(f.ctx, `
UPDATE finding_proposal_audit_holds SET receipt_created_at = receipt_created_at + interval '1 second'
 WHERE audit_id = $1`, f.request.AuditID); persistencepostgres.Classify(err) != persistencepostgres.ClassRejected {
		t.Fatalf("hold receipt time update = %v, want a rejection", err)
	}
	direct := insertAuditChildReceipt(t, f, "direct-candidate")
	insertHold := func(receiptID string, receiptTime *time.Time) error {
		_, err := f.pool.Exec(f.ctx, `
INSERT INTO finding_proposal_audit_holds (receipt_id, audit_id, project_id, proposal_ref, evidence, receipt_created_at)
VALUES ($1, $2, 'delete-project', '{"namespace":"audit-copy","name":"direct","revision":"r1"}', '[]', $3)`,
			receiptID, f.request.AuditID, receiptTime)
		return err
	}
	later := direct.CreatedAt.Add(time.Second)
	if err := insertHold(direct.ReceiptID, &later); persistencepostgres.Classify(err) != persistencepostgres.ClassRejected {
		t.Fatalf("hold naming another receipt time = %v, want a rejection", err)
	}
	if err := insertHold("missing-receipt", nil); persistencepostgres.Classify(err) != persistencepostgres.ClassReferenceViolation {
		t.Fatalf("hold of a missing receipt = %v, want a reference violation", err)
	}
	if err := insertHold(direct.ReceiptID, &direct.CreatedAt); err != nil {
		t.Fatalf("hold naming its receipt's own time = %v", err)
	}
}
