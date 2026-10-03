//go:build integration

package findingintake

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

func TestPostgresSameReceiptImportsIntoIndependentAudits(t *testing.T) {
	f := newDeletionImportFixture(t)
	const secondAudit = "second-import-audit"
	createFixtureAudit(t, f, secondAudit)
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
	if findingIDs[f.request.AuditID] == findingIDs[secondAudit] || holds[f.request.AuditID].Proposal.Ref.SameExact(holds[secondAudit].Proposal.Ref) {
		t.Fatal("destination Audits shared finding identity or retained proposal")
	}
	if err := runstore.NewPostgresStore(f.pool).DeleteReleasedTerminalRun(f.ctx, f.request.OwnerID, f.request.RunID); err != nil {
		t.Fatal(err)
	}
	for auditID, hold := range holds {
		receipt, err := f.intake.GetAuditReceipt(f.ctx, f.request.OwnerID, auditID, f.receiptID)
		if err != nil || !receipt.Origin.RunDeleted || receipt.Retention != RetentionAuditHeld {
			t.Fatalf("retained receipt in %s after source deletion: %+v, %v", auditID, receipt, err)
		}
		batch, err := f.intake.GetAuditReceipts(f.ctx, f.request.OwnerID, auditID, []string{f.receiptID})
		if err != nil || len(batch) != 1 {
			t.Fatalf("retained receipt page in %s after source deletion: %+v, %v", auditID, batch, err)
		}
		inbox, err := f.intake.ListAuditInbox(f.ctx, f.request.OwnerID, auditID, ListQuery{Limit: 10})
		if err != nil || len(inbox) != 1 || inbox[0].ReceiptID != f.receiptID {
			t.Fatalf("Audit inbox of %s after source deletion: %+v, %v", auditID, inbox, err)
		}
		// Each Audit sees, and reads the proposal through, only its own copy.
		for _, read := range []Receipt{receipt, batch[0], inbox[0]} {
			if len(read.AuditHolds) != 1 || read.AuditHolds[0].AuditID != auditID ||
				!read.AuditHolds[0].Proposal.Ref.SameExact(hold.Proposal.Ref) {
				t.Fatalf("receipt holds read through %s = %+v", auditID, read.AuditHolds)
			}
		}
	}
	audits := auditstore.NewPostgresStore(f.pool)
	first, err := audits.Get(f.ctx, f.request.OwnerID, f.request.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := audits.RequestDelete(f.ctx, auditstore.DeleteParams{
		OwnerID: f.request.OwnerID, AuditID: first.AuditID, ExpectedRevision: first.Revision,
		IdempotencyKey: "delete-first", RequestDigest: auditdomain.DigestBytes([]byte("delete-first")),
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

// After source Run deletion an Audit inbox reads the proposal only from that
// Audit's retained copy. The destination that imported first has a copy that
// no longer verifies, so reading it through the other Audit would fail.
func TestPostgresAuditInboxReadsOwnRetainedCopyAfterRunDeletion(t *testing.T) {
	f := newDeletionImportFixture(t)
	const firstAudit = "first-import-audit"
	createFixtureAudit(t, f, firstAudit)
	holds := make(map[string]AuditHold)
	for _, auditID := range []string{firstAudit, f.request.AuditID} {
		request := f.request
		request.AuditID = auditID
		hold, replayed, err := f.intake.ImportIntoAudit(f.ctx, request)
		if err != nil || replayed {
			t.Fatalf("import into %s: replayed=%v err=%v", auditID, replayed, err)
		}
		holds[auditID] = hold
	}
	if err := runstore.NewPostgresStore(f.pool).DeleteReleasedTerminalRun(f.ctx, f.request.OwnerID, f.request.RunID); err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(f.ctx, `ALTER TABLE finding_proposal_audit_holds DISABLE TRIGGER finding_proposal_audit_holds_protect_update`); err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(f.ctx, `
UPDATE finding_proposal_audit_holds
   SET proposal_ref = jsonb_set(proposal_ref, '{digest}', to_jsonb('sha256:' || repeat('0', 64)))
 WHERE receipt_id = $1 AND audit_id = $2`, f.receiptID, firstAudit); err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(f.ctx, `ALTER TABLE finding_proposal_audit_holds ENABLE TRIGGER finding_proposal_audit_holds_protect_update`); err != nil {
		t.Fatal(err)
	}
	own := holds[f.request.AuditID]
	inbox, err := f.intake.ListAuditInbox(f.ctx, f.request.OwnerID, f.request.AuditID, ListQuery{Limit: 10})
	if err != nil || len(inbox) != 1 || !inbox[0].Origin.RunDeleted || inbox[0].Document.ClientKey != "candidate" ||
		len(inbox[0].AuditHolds) != 1 || inbox[0].AuditHolds[0].AuditID != f.request.AuditID ||
		!inbox[0].AuditHolds[0].Proposal.Ref.SameExact(own.Proposal.Ref) {
		t.Fatalf("Audit inbox after source deletion = (%+v, %v)", inbox, err)
	}
	if _, err := f.intake.ListAuditInbox(f.ctx, f.request.OwnerID, firstAudit, ListQuery{Limit: 10}); !errors.Is(err, artifacts.ErrArtifactIntegrity) {
		t.Fatalf("first destination inbox over its unverifiable copy = %v", err)
	}
}

// Collection rejects one Audit child proposal and retains its sibling. Once
// the source Run is deleted only the retained copy is readable, so the
// rejected receipt leaves the inbox rather than failing every page at it.
func TestPostgresAuditInboxOmitsUnretainedChildReceiptAfterRunDeletion(t *testing.T) {
	f := newDeletionImportFixture(t)
	rejected := insertAuditChildReceipt(t, f, "rejected-candidate")
	retained := insertAuditChildReceipt(t, f, "retained-candidate")
	request := f.request
	request.Proposal = retained.Proposal.Ref
	if _, _, err := f.intake.RetainAuditCollection(f.ctx, request); err != nil {
		t.Fatal(err)
	}
	request.Proposal = rejected.Proposal.Ref
	if err := f.intake.RejectAuditCollection(f.ctx, request, "finding-proposal-standard-invalid"); err != nil {
		t.Fatal(err)
	}
	owner, auditID := f.request.OwnerID, f.request.AuditID
	live, err := f.intake.ListAuditInbox(f.ctx, owner, auditID, ListQuery{Limit: 10})
	if err != nil || len(live) != 2 || live[0].ReceiptID != rejected.ReceiptID || len(live[0].AuditHolds) != 0 ||
		live[1].ReceiptID != retained.ReceiptID || len(live[1].AuditHolds) != 1 {
		t.Fatalf("Audit inbox while the source Run exists = (%+v, %v)", live, err)
	}
	if err := runstore.NewPostgresStore(f.pool).DeleteReleasedTerminalRun(f.ctx, owner, f.request.RunID); err != nil {
		t.Fatal(err)
	}
	for name, list := range map[string]func(context.Context, string, string, ListQuery) ([]Receipt, error){
		"owner": f.intake.ListAuditInbox, "held": f.intake.ListAuditHeldInbox,
	} {
		for _, limit := range []int{1, 10} {
			listed := make([]Receipt, 0)
			query := ListQuery{Limit: limit}
			for range 4 {
				page, err := list(f.ctx, owner, auditID, query)
				if err != nil {
					t.Fatalf("%s inbox page at limit %d: %v", name, limit, err)
				}
				listed = append(listed, page...)
				if len(page) < limit {
					break
				}
				last := page[len(page)-1]
				query.AfterCreatedAt, query.AfterReceiptID = &last.CreatedAt, last.ReceiptID
			}
			if len(listed) != 1 || listed[0].ReceiptID != retained.ReceiptID ||
				listed[0].Retention != RetentionAuditHeld || !listed[0].Origin.RunDeleted ||
				listed[0].Document.ClientKey != "retained-candidate" ||
				len(listed[0].AuditHolds) != 1 || listed[0].AuditHolds[0].AuditID != auditID {
				t.Fatalf("%s inbox at limit %d after source deletion = %+v", name, limit, listed)
			}
		}
	}
}

func createFixtureAudit(t *testing.T, f deletionImportFixture, auditID string) {
	t.Helper()
	if _, _, err := auditstore.NewPostgresStore(f.pool).CreateDraft(f.ctx, auditstore.CreateDraftParams{
		AuditID: auditID, OwnerID: f.request.OwnerID, ProjectID: "delete-project",
		Profile:         auditstore.ProfileIdentity{Name: "profile", Version: "1", Digest: auditdomain.DigestBytes([]byte("profile"))},
		ProfileSnapshot: json.RawMessage(`{"interaction":{"findingConfirmation":"human-required"}}`),
		InputSelection:  json.RawMessage(`{}`),
		Limits: auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 100, MaxItemsTotal: 100,
			MaxSubmittedRuns: 100, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1 << 20},
		IdempotencyKey: auditID, RequestDigest: auditdomain.DigestBytes([]byte(auditID)),
	}); err != nil {
		t.Fatal(err)
	}
}

// insertAuditChildReceipt records one more receipt of the fixture's source Run
// with the fixture Audit as its origin, as intake does for an Audit child Run.
func insertAuditChildReceipt(t *testing.T, f deletionImportFixture, clientKey string) Receipt {
	t.Helper()
	source, err := readReceiptByID(f.ctx, f.pool, f.receiptID)
	if err != nil {
		t.Fatal(err)
	}
	canonical, err := canonicalize(testSubmission(clientKey+"-invocation", clientKey, []contracts.ArtifactRef{}))
	if err != nil {
		t.Fatal(err)
	}
	written, err := artifacts.NewService(artifacts.NewPostgresRepository(f.pool)).WriteFindingProposal(
		f.ctx, f.request.RunID, clientKey+"-proposal",
		artifacts.Payload{MediaType: proposalMediaType, Data: canonical.proposalBytes},
	)
	if err != nil {
		t.Fatal(err)
	}
	proposal := ExactArtifact{Ref: written.Ref, Digest: auditdomain.DigestBytes(canonical.proposalBytes), MediaType: written.MediaType, SizeBytes: written.Size}
	origin := source.Origin
	origin.Audit = &AuditOrigin{AuditID: f.request.AuditID, ExecutionID: "delete-execution", Role: "discovery"}
	project := "delete-project"
	grant := controlplane.AllocationGrant{RunID: f.request.RunID, AllocationID: "allocation", StageExecutionID: "stage",
		RuntimeAgentID: "runtime", RuntimeInstanceID: "instance", LogicalAgentName: "worker"}
	receiptID := clientKey + "-receipt"
	if err := persistencepostgres.InTx(f.ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		return insertReceipt(f.ctx, tx, receiptID, clientKey+"-proposal", canonical, proposal, []ExactArtifact{}, origin, f.request.OwnerID, &project, grant)
	}); err != nil {
		t.Fatal(err)
	}
	receipt, err := readReceiptByID(f.ctx, f.pool, receiptID)
	if err != nil {
		t.Fatal(err)
	}
	return receipt
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
			resultDigest, contractDigest := auditdomain.DigestBytes([]byte("result")), auditdomain.DigestBytes([]byte("contract"))
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
						result = auditdomain.DigestBytes([]byte("other-result"))
					case "contract":
						contract = auditdomain.DigestBytes([]byte("other-contract"))
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
