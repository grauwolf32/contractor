//go:build integration

package findingintake

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
)

func TestPostgresSameReceiptImportsIntoIndependentAudits(t *testing.T) {
	f := newDeletionImportFixture(t)
	const secondAudit = "second-import-audit"
	createFixtureAudit(t, f, secondAudit)
	holds := make(map[string]AuditHold)
	for _, auditID := range []string{f.request.AuditID, secondAudit} {
		request := f.request
		request.AuditID = auditID
		hold, replayed, err := f.intake.ImportIntoAudit(f.ctx, request)
		if err != nil || replayed {
			t.Fatalf("first import into %s: replayed=%v err=%v", auditID, replayed, err)
		}
		holds[auditID] = hold
		if _, replayed, err := f.intake.ImportIntoAudit(f.ctx, request); err != nil || !replayed {
			t.Fatalf("repeat import into %s: replayed=%v err=%v", auditID, replayed, err)
		}
	}
	if !holds[f.request.AuditID].Proposal.Ref.SameExact(holds[secondAudit].Proposal.Ref) {
		t.Fatal("destination Audits did not reuse the Project-retained proposal")
	}
	if err := deleteRun(f.ctx, f.pool, f.request.OwnerID, f.request.RunID); err != nil {
		t.Fatal(err)
	}
	for auditID, hold := range holds {
		receipt, err := f.intake.GetAuditReceipt(f.ctx, f.request.OwnerID, auditID, f.receiptID)
		if err != nil || !receipt.Origin.RunDeleted || receipt.Retention != RetentionProjectHeld {
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
		if !reflect.DeepEqual(receipt, batch[0]) {
			t.Fatalf("single and batch reads through %s differ: %+v != %+v", auditID, receipt, batch[0])
		}
		// Each Audit sees, and reads the proposal through, only its own copy.
		for _, read := range []Receipt{receipt, batch[0], inbox[0]} {
			if len(read.AuditHolds) != 1 || read.AuditHolds[0].AuditID != auditID ||
				!read.AuditHolds[0].Proposal.Ref.SameExact(hold.Proposal.Ref) {
				t.Fatalf("receipt holds read through %s = %+v", auditID, read.AuditHolds)
			}
		}
	}
	audits := auditstoretest.NewStore(f.pool).WithPurgeDependencies(DeleteAuditHoldsBatch)
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
	if err != nil || receipt.Retention != RetentionProjectHeld || len(receipt.AuditHolds) != 1 || receipt.AuditHolds[0].AuditID != secondAudit {
		t.Fatalf("surviving Audit lost its receipt after other purge: %+v %v", receipt, err)
	}
	projectArtifacts, err := artifacts.NewService(artifactpostgres.NewRepository(f.pool)).Project("delete-project")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := projectArtifacts.Read(f.ctx, holds[secondAudit].Proposal.Ref); err != nil {
		t.Fatalf("surviving retained proposal: %v", err)
	}
	if _, err := projectArtifacts.Read(f.ctx, holds[first.AuditID].Proposal.Ref); err != nil {
		t.Fatalf("Project material lost after Audit purge: %v", err)
	}
}

// A hold admits a receipt to an Audit only in that Audit's Project, the same
// condition under which the hold itself is read.
func TestPostgresAuditReceiptReadsRequireHoldInAuditProject(t *testing.T) {
	f := newDeletionImportFixture(t)
	const otherProject, otherAudit = "other-project", "other-project-audit"
	if _, _, err := projectstore.NewPostgresStore(f.pool).Create(f.ctx, projectstore.CreateParams{
		ProjectID: otherProject, OwnerID: f.request.OwnerID, Kind: projectstore.KindProject,
		Name: "Other project", IdempotencyKey: "other-project",
		RequestDigest: auditdomain.DigestBytes([]byte("other-project")),
	}); err != nil {
		t.Fatal(err)
	}
	if _, _, err := auditstoretest.NewStore(f.pool).WithPurgeDependencies(DeleteAuditHoldsBatch).CreateDraft(f.ctx, auditstore.CreateDraftParams{
		AuditID: otherAudit, OwnerID: f.request.OwnerID, ProjectID: otherProject,
		Profile:         auditstore.ProfileIdentity{Name: "profile", Version: "1", Digest: auditdomain.DigestBytes([]byte("profile"))},
		ProfileSnapshot: json.RawMessage(`{"interaction":{}}`),
		InputSelection:  json.RawMessage(`{}`),
		Limits: auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1,
			MaxSubmittedRuns: 1, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1 << 20},
		IdempotencyKey: otherAudit, RequestDigest: auditdomain.DigestBytes([]byte(otherAudit)),
	}); err != nil {
		t.Fatal(err)
	}
	receipt, err := readReceiptByID(f.ctx, f.pool, f.receiptID)
	if err != nil {
		t.Fatal(err)
	}
	proposal, err := json.Marshal(receipt.Proposal)
	if err != nil {
		t.Fatal(err)
	}
	// A hold naming the source Project instead of its Audit's Project.
	if _, err := f.pool.Exec(f.ctx, `
INSERT INTO finding_proposal_audit_holds (receipt_id, audit_id, project_id, proposal_ref, evidence)
VALUES ($1, $2, 'delete-project', $3::jsonb, '[]'::jsonb)`, f.receiptID, otherAudit, proposal); err != nil {
		t.Fatal(err)
	}
	if _, err := f.intake.GetAuditReceipt(f.ctx, f.request.OwnerID, otherAudit, f.receiptID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("receipt read through a foreign-Project hold = %v", err)
	}
	if _, err := f.intake.GetAuditReceipts(f.ctx, f.request.OwnerID, otherAudit, []string{f.receiptID}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("receipt page read through a foreign-Project hold = %v", err)
	}
}

// After source Run deletion, publishing a receipt selected through an Audit
// reads that Audit's own retained copy, not another destination's.
func TestPostgresCollectionPublicationReadsRequestingAuditHoldAfterRunDeletion(t *testing.T) {
	f := newDeletionImportFixture(t)
	// The other destination sorts first among the receipt's holds.
	const otherAudit = "a-first-audit"
	createFixtureAudit(t, f, otherAudit)
	for _, auditID := range []string{otherAudit, f.request.AuditID} {
		request := f.request
		request.AuditID = auditID
		if _, _, err := f.intake.ImportIntoAudit(f.ctx, request); err != nil {
			t.Fatalf("import into %s: %v", auditID, err)
		}
	}
	if err := deleteRun(f.ctx, f.pool, f.request.OwnerID, f.request.RunID); err != nil {
		t.Fatal(err)
	}
	publisher, err := NewCollectionPublisher(f.pool, emptyCollectionReviews{})
	if err != nil {
		t.Fatal(err)
	}
	published, err := publisher.PublishCollection(f.ctx, PublishCollectionParams{
		OwnerID: f.request.OwnerID, Request: PublishCollectionRequest{
			ClientKey: "requesting-audit-copy", Sources: []CollectionSelection{{
				Kind: "audit", ID: f.request.AuditID,
				ReceiptIDs: []string{f.receiptID}, Findings: []CollectionFindingSelection{},
			}},
		},
	})
	if err != nil || published.EntryCount != 1 {
		t.Fatalf("publish retained receipt = (%+v, %v)", published, err)
	}
	user, err := artifacts.NewService(artifactpostgres.NewRepository(f.pool)).User(f.request.OwnerID)
	if err != nil {
		t.Fatal(err)
	}
	archive, err := user.Read(f.ctx, published.Artifact.Ref)
	if err != nil {
		t.Fatal(err)
	}
	collection, _, err := auditdomain.DecodeFindingCollectionPackage(archive.Payload.Data)
	if err != nil || len(collection.Entries) != 1 {
		t.Fatalf("published collection = (%+v, %v)", collection, err)
	}
	for _, document := range collection.Documents {
		if document.ID != collection.Entries[0].ProposalDocumentID {
			continue
		}
		if document.Scope.Kind != "project" ||
			document.Ref.Namespace != artifactpolicy.FindingMaterialsNamespace {
			t.Fatalf("published proposal copy = %+v, want the requesting Audit's", document)
		}
		return
	}
	t.Fatalf("published collection has no proposal document: %+v", collection)
}

type emptyCollectionReviews struct{}

func (emptyCollectionReviews) SelectCollectionFindings(
	context.Context, pgx.Tx, string, string, []CollectionFindingSelection,
) ([]string, error) {
	return []string{}, nil
}

func (emptyCollectionReviews) ReadCollectionFindings(
	context.Context, pgx.Tx, string, []string,
) (map[string]*auditdomain.FindingCollectionFinding, error) {
	return map[string]*auditdomain.FindingCollectionFinding{}, nil
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
	if err := deleteRun(f.ctx, f.pool, f.request.OwnerID, f.request.RunID); err != nil {
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
	if _, err := f.intake.ListAuditInbox(f.ctx, f.request.OwnerID, firstAudit, ListQuery{Limit: 10}); !errors.Is(err, blob.ErrIntegrity) {
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
	if err := f.intake.RetainAuditCollectionBatch(f.ctx, []ImportRequest{request}); err != nil {
		t.Fatal(err)
	}
	audits := auditstoretest.NewStore(f.pool).WithPurgeDependencies(DeleteAuditHoldsBatch)
	beforeRejection, err := audits.Get(f.ctx, request.OwnerID, request.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	request.Proposal = rejected.Proposal.Ref
	if err := f.intake.RejectAuditCollection(f.ctx, request, "finding-proposal-standard-invalid"); err != nil {
		t.Fatal(err)
	}
	afterRejection, err := audits.Get(f.ctx, request.OwnerID, request.AuditID)
	if err != nil || afterRejection.Revision != beforeRejection.Revision+1 ||
		afterRejection.EventSequence != beforeRejection.EventSequence+1 {
		t.Fatalf("rejected proposal Audit revision = (%+v, %v)", afterRejection, err)
	}
	rejectionEvents, err := audits.EventsAfter(f.ctx, request.OwnerID, request.AuditID, beforeRejection.EventSequence, 1)
	if err != nil || len(rejectionEvents) != 1 ||
		rejectionEvents[0].Kind != "finding.proposal_rejected" ||
		rejectionEvents[0].EntityID != rejected.ReceiptID {
		t.Fatalf("rejected proposal event = (%+v, %v)", rejectionEvents, err)
	}
	if err := f.intake.RejectAuditCollection(f.ctx, request, "finding-proposal-standard-invalid"); err != nil {
		t.Fatalf("rejected proposal replay: %v", err)
	}
	afterReplay, err := audits.Get(f.ctx, request.OwnerID, request.AuditID)
	if err != nil || afterReplay.Revision != afterRejection.Revision ||
		afterReplay.EventSequence != afterRejection.EventSequence {
		t.Fatalf("rejected proposal replay Audit revision = (%+v, %v)", afterReplay, err)
	}
	owner, auditID := f.request.OwnerID, f.request.AuditID
	live, err := f.intake.ListAuditInbox(f.ctx, owner, auditID, ListQuery{Limit: 10})
	if err != nil || len(live) != 2 || live[0].ReceiptID != rejected.ReceiptID || len(live[0].AuditHolds) != 0 ||
		live[1].ReceiptID != retained.ReceiptID || len(live[1].AuditHolds) != 1 {
		t.Fatalf("Audit inbox while the source Run exists = (%+v, %v)", live, err)
	}
	if err := deleteRun(f.ctx, f.pool, owner, f.request.RunID); err != nil {
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
				listed[0].Retention != RetentionProjectHeld || !listed[0].Origin.RunDeleted ||
				listed[0].Document.ClientKey != "retained-candidate" ||
				len(listed[0].AuditHolds) != 1 || listed[0].AuditHolds[0].AuditID != auditID {
				t.Fatalf("%s inbox at limit %d after source deletion = %+v", name, limit, listed)
			}
		}
	}
}

func createFixtureAudit(t *testing.T, f deletionImportFixture, auditID string) {
	t.Helper()
	if _, _, err := auditstoretest.NewStore(f.pool).WithPurgeDependencies(DeleteAuditHoldsBatch).CreateDraft(f.ctx, auditstore.CreateDraftParams{
		AuditID: auditID, OwnerID: f.request.OwnerID, ProjectID: "delete-project",
		Profile:         auditstore.ProfileIdentity{Name: "profile", Version: "1", Digest: auditdomain.DigestBytes([]byte("profile"))},
		ProfileSnapshot: json.RawMessage(`{"interaction":{}}`),
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
// readReceiptByID reads proposal receipt receiptID with all its Audit holds,
// for tests that inspect a receipt regardless of the Audit that holds it.
func readReceiptByID(ctx context.Context, db querier, receiptID string) (Receipt, error) {
	return scanReceipt(ctx, db, db.QueryRow(ctx, `
SELECT `+receiptProjection+`
  FROM finding_proposal_receipts AS receipt
 WHERE receipt.receipt_id = $1`, receiptID))
}

func insertAuditChildReceipt(t *testing.T, f deletionImportFixture, clientKey string) Receipt {
	return insertAuditChildReceiptWithEvidence(t, f, clientKey, []ExactArtifact{})
}

// insertAuditChildReceiptWithEvidence is insertAuditChildReceipt for a
// proposal citing exact evidence revisions of the source Run, given in
// canonical reference order.
func insertAuditChildReceiptWithEvidence(
	t *testing.T, f deletionImportFixture, clientKey string, evidence []ExactArtifact, padding ...int,
) Receipt {
	t.Helper()
	receipt, err := recordAuditChildReceipt(t, f, clientKey, evidence, padding...)
	if err != nil {
		t.Fatal(err)
	}
	return receipt
}

// recordAuditChildReceipt records the receipt of insertAuditChildReceipt-
// WithEvidence through the intake's own insert and returns its error.
func recordAuditChildReceipt(
	t *testing.T, f deletionImportFixture, clientKey string, evidence []ExactArtifact, padding ...int,
) (Receipt, error) {
	t.Helper()
	source, err := readReceiptByID(f.ctx, f.pool, f.receiptID)
	if err != nil {
		t.Fatal(err)
	}
	refs := make([]contracts.ArtifactRef, len(evidence))
	for index := range evidence {
		refs[index] = evidence[index].Ref
	}
	canonical, err := canonicalize(testSubmission(clientKey+"-invocation", clientKey, refs))
	if err != nil {
		t.Fatal(err)
	}
	if len(padding) > 0 {
		canonical.proposalBytes = append(canonical.proposalBytes,
			bytes.Repeat([]byte(" "), padding[0]-len(canonical.proposalBytes))...)
		projection, err := auditdomain.BuildFindingChecksProjection(canonical.proposalBytes, canonical.request.Proposal)
		if err != nil {
			t.Fatal(err)
		}
		canonical.checksProjection, err = auditdomain.EncodeFindingChecksProjection(projection)
		if err != nil {
			t.Fatal(err)
		}
	}
	written, err := artifactstest.WriteFindingProposal(
		f.ctx, artifacts.NewService(artifactpostgres.NewRepository(f.pool)), f.request.RunID, clientKey+"-proposal",
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
	fingerprint, err := auditdomain.EncodeFindingFingerprint(auditdomain.UnavailableFindingFingerprint("not_provided"))
	if err != nil {
		t.Fatal(err)
	}
	if err := persistencepostgres.InTx(f.ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		return insertReceipt(f.ctx, tx, receiptID, clientKey+"-proposal", canonical, fingerprint, proposal, evidence, origin, f.request.OwnerID, &project, grant, nil, nil, "")
	}); err != nil {
		return Receipt{}, err
	}
	return readReceiptByID(f.ctx, f.pool, receiptID)
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
			err := f.intake.RetainAuditCollectionBatch(f.ctx, []ImportRequest{f.request})
			if state == "cancelled" || state == "completed" {
				if !errors.Is(err, ErrAuditClosed) {
					t.Fatalf("collection retention into %s Audit error = %v", state, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("collection retention into %s Audit error = %v", state, err)
			}
			receipt, err := f.intake.GetAuditReceipt(f.ctx, f.request.OwnerID, f.request.AuditID, f.receiptID)
			if err != nil || receipt.Retention != RetentionProjectHeld || len(receipt.AuditHolds) != 1 {
				t.Fatalf("retained receipt = (%+v, %v)", receipt, err)
			}
			if err := f.intake.RetainAuditCollectionBatch(f.ctx, []ImportRequest{f.request}); err != nil {
				t.Fatalf("collection retention replay: %v", err)
			}
			replayed, err := f.intake.GetAuditReceipt(f.ctx, f.request.OwnerID, f.request.AuditID, f.receiptID)
			if err != nil || !reflect.DeepEqual(replayed.AuditHolds, receipt.AuditHolds) {
				t.Fatalf("replayed retention changed holds: %+v, %v", replayed.AuditHolds, err)
			}
		})
	}
}

// Replay reconciles the ordinary frozen history artifact. A different receipt
// or Audit operation never finds or mutates another operation's history.
func TestPostgresDirectAssessmentReplayRequiresAuditScopedIdentity(t *testing.T) {
	f := newDeletionImportFixture(t)
	if _, _, err := f.intake.ImportIntoAudit(f.ctx, f.request); err != nil {
		t.Fatal(err)
	}
	input := directVerificationInput{ProjectID: "delete-project", AuditID: f.request.AuditID, ReceiptID: f.receiptID}
	id := deterministicID("direct-assessment", input.AuditID, input.ReceiptID)
	resultDigest, contractDigest := auditdomain.DigestBytes([]byte("result")), auditdomain.DigestBytes([]byte("contract"))
	fact := directHistoryFactForTest{Schema: auditdomain.FindingSourceHistorySchema, ReceiptID: input.ReceiptID, Kind: "direct-verification"}
	fact.Original.AuditID, fact.Original.RecordID, fact.Original.Semantic = input.AuditID, id, "supported"
	fact.Materials.Result.Digest, fact.Materials.Contract.Digest = resultDigest, contractDigest
	body, err := json.Marshal(fact)
	if err != nil {
		t.Fatal(err)
	}
	store := artifacts.NewService(artifactpostgres.NewRepository(f.pool))
	scope, err := artifacts.ProjectScope(input.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	ref := contracts.ArtifactRef{Namespace: artifactpolicy.FindingHistoryNamespace, Name: auditdomain.FindingSourceHistoryName(input.ReceiptID, "direct-verification", id)}
	if _, err := store.CreateFrozen(f.ctx, scope, ref, artifacts.Payload{MediaType: "application/json", Data: body}); err != nil {
		t.Fatal(err)
	}
	tx, err := f.pool.Begin(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback(f.ctx)
	if replayed, err := directAssessmentReplay(f.ctx, tx, id, input, "supported", resultDigest, contractDigest); err != nil || !replayed {
		t.Fatalf("own replay=%v %v", replayed, err)
	}
	for _, change := range []string{"assessment", "result", "contract", "receipt", "audit", "operation"} {
		t.Run(change, func(t *testing.T) {
			changed := input
			operation := id
			semantic, result, contract := "supported", resultDigest, contractDigest
			switch change {
			case "assessment":
				semantic = "refuted"
			case "result":
				result = auditdomain.DigestBytes([]byte("other-result"))
			case "contract":
				contract = auditdomain.DigestBytes([]byte("other-contract"))
			case "receipt":
				changed.ReceiptID = "other-receipt"
			case "audit":
				changed.AuditID = "other-audit"
			case "operation":
				changed.AuditID = "other-audit"
				operation = deterministicID("direct-assessment", changed.AuditID, changed.ReceiptID)
			}
			replayed, err := directAssessmentReplay(f.ctx, tx, operation, changed, semantic, result, contract)
			if change == "receipt" || change == "operation" {
				if replayed || err != nil {
					t.Fatalf("foreign lookup=%v %v", replayed, err)
				}
			} else if !replayed || !errors.Is(err, ErrConflict) {
				t.Fatalf("changed replay=%v %v", replayed, err)
			}
		})
	}
	project, err := store.Project(input.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	read, err := project.Read(f.ctx, ref)
	if err != nil || !bytes.Equal(read.Payload.Data, body) {
		t.Fatalf("history changed: %v", err)
	}
}
