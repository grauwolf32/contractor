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

func TestPostgresDirectAssessmentReplayRequiresAuditScopedIdentity(t *testing.T) {
	for _, historical := range []bool{false, true} {
		name := "current"
		if historical {
			name = "receipt-only"
		}
		t.Run(name, func(t *testing.T) {
			f := newDeletionImportFixture(t)
			if _, _, err := f.intake.ImportIntoAudit(f.ctx, f.request); err != nil {
				t.Fatal(err)
			}
			input := directVerificationInput{AuditID: f.request.AuditID, ReceiptID: f.receiptID}
			currentID := deterministicID("direct-assessment", input.AuditID, input.ReceiptID)
			storedID := currentID
			if historical {
				storedID = deterministicID("direct-assessment", input.ReceiptID)
			}
			resultDigest, contractDigest := digestBytes([]byte("result")), digestBytes([]byte("contract"))
			if _, err := f.pool.Exec(f.ctx, `INSERT INTO audit_finding_assessments (
 assessment_id, finding_id, audit_id, receipt_id, semantic_assessment,
 result_ref, result_digest, direct_verification, contract_ref, contract_digest)
SELECT $1,finding_id,audit_id,first_receipt_id,'supported',
 '{"namespace":"verification","name":"result","revision":"r1"}'::jsonb,$4,true,
 '{"namespace":"verification","name":"contract","revision":"r1"}'::jsonb,$5
FROM audit_findings WHERE audit_id=$2 AND first_receipt_id=$3`, storedID, input.AuditID, input.ReceiptID, resultDigest, contractDigest); err != nil {
				t.Fatal(err)
			}
			var before string
			if err := f.pool.QueryRow(f.ctx, `SELECT row_to_json(a)::text FROM audit_finding_assessments a WHERE assessment_id=$1`, storedID).Scan(&before); err != nil {
				t.Fatal(err)
			}
			tx, err := f.pool.Begin(f.ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer tx.Rollback(f.ctx)
			if replayed, err := directAssessmentReplay(f.ctx, tx, currentID, input, "supported", resultDigest, contractDigest); err != nil || replayed == historical {
				t.Fatalf("owning Audit replay=%v err=%v", replayed, err)
			}
			for _, mismatch := range []string{"assessment", "result", "contract", "receipt"} {
				t.Run(mismatch, func(t *testing.T) {
					changedInput := input
					semantic, result, contract := "supported", resultDigest, contractDigest
					switch mismatch {
					case "assessment":
						semantic = "refuted"
					case "result":
						result = digestBytes([]byte("other-result"))
					case "contract":
						contract = digestBytes([]byte("other-contract"))
					case "receipt":
						changedInput.ReceiptID = "other-receipt"
					}
					replayed, err := directAssessmentReplay(f.ctx, tx, currentID, changedInput, semantic, result, contract)
					if historical {
						if err != nil || replayed {
							t.Fatalf("receipt-only ID was considered for replay: replayed=%v err=%v", replayed, err)
						}
					} else if !replayed || !errors.Is(err, ErrConflict) {
						t.Fatalf("changed current assessment: replayed=%v err=%v", replayed, err)
					}
				})
			}
			input.AuditID = "other-audit"
			for _, id := range []string{currentID, deterministicID("direct-assessment", input.AuditID, input.ReceiptID)} {
				if replayed, err := directAssessmentReplay(f.ctx, tx, id, input, "supported", resultDigest, contractDigest); err != nil || replayed {
					t.Fatalf("foreign assessment reused: replayed=%v err=%v", replayed, err)
				}
			}
			var after string
			if err := tx.QueryRow(f.ctx, `SELECT row_to_json(a)::text FROM audit_finding_assessments a WHERE assessment_id=$1`, storedID).Scan(&after); err != nil || after != before {
				t.Fatalf("assessment history changed: %v", err)
			}
			var count int
			if err := tx.QueryRow(f.ctx, `SELECT count(*) FROM audit_finding_assessments`).Scan(&count); err != nil || count != 1 {
				t.Fatalf("replay created an assessment: count=%d err=%v", count, err)
			}
		})
	}
}

func TestPostgresCollectionRetentionAdmitsClosingAudit(t *testing.T) {
	for _, state := range []string{"finalizing", "cancelling", "cancelled", "completed"} {
		t.Run(state, func(t *testing.T) {
			f := newDeletionImportFixture(t)
			if _, err := f.pool.Exec(f.ctx, `
UPDATE audits
   SET state = $2, baseline_snapshot = '{}'::jsonb, started_at = clock_timestamp(),
       finished_at = CASE WHEN $2 IN ('cancelled', 'completed') THEN clock_timestamp() END
 WHERE audit_id = $1`, f.request.AuditID, state); err != nil {
				t.Fatal(err)
			}
			if _, _, err := f.intake.ImportIntoAudit(f.ctx, f.request); !errors.Is(err, ErrNotFound) {
				t.Fatalf("owner import into %s Audit error = %v", state, err)
			}
			_, _, err := f.intake.RetainAuditCollection(f.ctx, f.request)
			if state == "cancelled" || state == "completed" {
				if !errors.Is(err, ErrAuditClosed) {
					t.Fatalf("collection retention into %s Audit error = %v", state, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("collection retention into %s Audit error = %v", state, err)
			}
			if _, replayed, err := f.intake.RetainAuditCollection(f.ctx, f.request); err != nil || !replayed {
				t.Fatalf("collection retention replay = (%t, %v)", replayed, err)
			}
			receipt, err := f.intake.GetAuditReceipt(f.ctx, f.request.OwnerID, f.request.AuditID, f.receiptID)
			if err != nil || receipt.Retention != RetentionAuditHeld || len(receipt.AuditHolds) != 1 {
				t.Fatalf("retained receipt = (%+v, %v)", receipt, err)
			}
		})
	}
}
