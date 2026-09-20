//go:build integration

package findingintake

import (
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/runstore"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

func TestPostgresSameReceiptImportsIntoIndependentAudits(t *testing.T) {
	f := newDeletionImportFixture(t)
	const secondAudit = "second-import-audit"
	_, _, err := auditstore.NewPostgresStore(f.pool).CreateDraft(f.ctx, auditstore.CreateDraftParams{
		AuditID: secondAudit, OwnerID: f.request.OwnerID, ProjectID: "delete-project",
		Profile:         auditstore.ProfileIdentity{Name: "profile", Version: "1", Digest: digestBytes([]byte("profile"))},
		ProfileSnapshot: json.RawMessage(`{"interaction":{"findingConfirmation":"human-required"}}`),
		InputSelection:  json.RawMessage(`{}`),
		Limits: auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 100, MaxItemsTotal: 100,
			MaxSubmittedRuns: 100, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1 << 20},
		IdempotencyKey: secondAudit, RequestDigest: digestBytes([]byte(secondAudit)),
	})
	if err != nil {
		t.Fatal(err)
	}
	holds := make(map[string]AuditHold)
	findingIDs := make(map[string]string)
	for _, auditID := range []string{f.request.AuditID, secondAudit} {
		request := f.request
		request.AuditID = auditID
		hold, replayed, err := f.intake.ImportIntoAudit(f.ctx, request)
		if err != nil || replayed {
			t.Fatalf("first import into %s: replayed=%v err=%v", auditID, replayed, err)
		}
		holds[auditID] = hold
		var findingID string
		if err := f.pool.QueryRow(f.ctx, `SELECT finding_id FROM audit_findings
WHERE audit_id=$1 AND first_receipt_id=$2`, auditID, f.receiptID).Scan(&findingID); err != nil {
			t.Fatal(err)
		}
		findingIDs[auditID] = findingID
		if _, replayed, err := f.intake.ImportIntoAudit(f.ctx, request); err != nil || !replayed {
			t.Fatalf("repeat import into %s: replayed=%v err=%v", auditID, replayed, err)
		}
	}
	if findingIDs[f.request.AuditID] == findingIDs[secondAudit] || sameRef(holds[f.request.AuditID].Proposal.Ref, holds[secondAudit].Proposal.Ref) {
		t.Fatal("destination Audits shared finding identity or retained proposal")
	}
	if err := runstore.NewPostgresStore(f.pool).DeleteReleasedTerminalRun(f.ctx, f.request.OwnerID, f.request.RunID); err != nil {
		t.Fatal(err)
	}
	for auditID := range holds {
		receipt, err := f.intake.GetAuditReceipt(f.ctx, f.request.OwnerID, auditID, f.receiptID)
		if err != nil || !receipt.Origin.RunDeleted || receipt.Retention != RetentionAuditHeld || len(receipt.AuditHolds) != 2 {
			t.Fatalf("retained receipt in %s after source deletion: %+v, %v", auditID, receipt, err)
		}
	}
	audits := auditstore.NewPostgresStore(f.pool)
	first, err := audits.Get(f.ctx, f.request.OwnerID, f.request.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := audits.RequestDelete(f.ctx, auditstore.DeleteParams{
		OwnerID: f.request.OwnerID, AuditID: first.AuditID, ExpectedRevision: first.Revision,
		IdempotencyKey: "delete-first", RequestDigest: digestBytes([]byte("delete-first")),
	}); err != nil {
		t.Fatal(err)
	}
	claims, err := audits.Claim(f.ctx, auditstore.ClaimParams{HolderID: "purge-first", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("purge claim: %+v %v", claims, err)
	}
	if err := audits.PurgeClaimed(f.ctx, claims[0], auditdomain.ArtifactNamespace(first.AuditID)); err != nil {
		t.Fatal(err)
	}
	receipt, err := f.intake.GetAuditReceipt(f.ctx, f.request.OwnerID, secondAudit, f.receiptID)
	if err != nil || receipt.Retention != RetentionAuditHeld || len(receipt.AuditHolds) != 1 || receipt.AuditHolds[0].AuditID != secondAudit {
		t.Fatalf("surviving Audit lost its receipt after other purge: %+v %v", receipt, err)
	}
	projectArtifacts, err := artifacts.NewService(artifacts.NewPostgresRepository(f.pool)).Project("delete-project")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := projectArtifacts.Read(f.ctx, holds[secondAudit].Proposal.Ref); err != nil {
		t.Fatalf("surviving retained proposal: %v", err)
	}
	if _, err := projectArtifacts.Read(f.ctx, holds[first.AuditID].Proposal.Ref); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("purged proposal: %v", err)
	}
}

func TestPostgresLegacyDirectAssessmentReplayIsAuditScoped(t *testing.T) {
	f := newDeletionImportFixture(t)
	if _, _, err := f.intake.ImportIntoAudit(f.ctx, f.request); err != nil {
		t.Fatal(err)
	}
	legacyID := deterministicID("direct-assessment", f.receiptID)
	resultDigest, contractDigest := digestBytes([]byte("result")), digestBytes([]byte("contract"))
	// Install a valid historical row using the pre-upgrade writer identity.
	// The immutable row and its accepted timestamp must survive replay unchanged.
	if _, err := f.pool.Exec(f.ctx, `INSERT INTO audit_finding_assessments (
 assessment_id, finding_id, audit_id, receipt_id, semantic_assessment,
 result_ref, result_digest, direct_verification, contract_ref, contract_digest)
SELECT $1,finding_id,audit_id,first_receipt_id,'supported',
 '{"namespace":"legacy","name":"result","revision":"r1"}'::jsonb,$4,true,
 '{"namespace":"legacy","name":"contract","revision":"r1"}'::jsonb,$5
FROM audit_findings WHERE audit_id=$2 AND first_receipt_id=$3`, legacyID, f.request.AuditID, f.receiptID, resultDigest, contractDigest); err != nil {
		t.Fatal(err)
	}
	var before string
	if err := f.pool.QueryRow(f.ctx, `SELECT row_to_json(a)::text FROM audit_finding_assessments a WHERE assessment_id=$1`, legacyID).Scan(&before); err != nil {
		t.Fatal(err)
	}
	tx, err := f.pool.Begin(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback(f.ctx)
	input := directVerificationInput{AuditID: f.request.AuditID, ReceiptID: f.receiptID}
	newID := deterministicID("direct-assessment", input.AuditID, input.ReceiptID)
	if replayed, err := directAssessmentReplay(f.ctx, tx, newID, input, "supported", resultDigest, contractDigest); err != nil || !replayed {
		t.Fatalf("legacy owning Audit replay=%v err=%v", replayed, err)
	}
	if _, err := directAssessmentReplay(f.ctx, tx, newID, input, "refuted", resultDigest, contractDigest); !errors.Is(err, ErrConflict) {
		t.Fatalf("mismatched legacy content: %v", err)
	}
	input.AuditID = "other-audit"
	newID = deterministicID("direct-assessment", input.AuditID, input.ReceiptID)
	if replayed, err := directAssessmentReplay(f.ctx, tx, newID, input, "supported", resultDigest, contractDigest); err != nil || replayed {
		t.Fatalf("foreign legacy row blocked independent import: replayed=%v err=%v", replayed, err)
	}
	var after string
	if err := tx.QueryRow(f.ctx, `SELECT row_to_json(a)::text FROM audit_finding_assessments a WHERE assessment_id=$1`, legacyID).Scan(&after); err != nil || after != before {
		t.Fatalf("legacy history changed: %v", err)
	}
}
