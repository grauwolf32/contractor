//go:build integration

package findingintake

import (
	"context"
	"reflect"
	"slices"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
)

// createFixtureRun creates another terminal Run of the fixture's owner and
// Project with the fixture Run's Workflow.
func createFixtureRun(t *testing.T, f deletionImportFixture, runID string) {
	t.Helper()
	runs := runstore.NewPostgresStore(f.pool)
	source, err := runs.GetRun(f.ctx, f.request.RunID)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runstoretest.CreateRun(f.ctx, runs, runstore.CreateRunParams{
		LaneClass: resolved.LaneClassModel,
		RunID:     runID, OwnerID: f.request.OwnerID, ProjectID: source.ProjectID,
		WorkflowName: source.WorkflowName, WorkflowVersion: source.WorkflowVersion,
		WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: source.WorkflowSnapshot,
		Parameters: map[string]string{}, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(f.ctx, runID, runstore.RunInitializing, runstore.RunFailed,
		runstore.Reason{Code: "fixture_done"}); err != nil {
		t.Fatal(err)
	}
}

// insertInboxReceipt records owner's receipt clientKey-receipt of Run runID
// with its exact proposal, originated by Audit auditID unless it is empty.
func insertInboxReceipt(t *testing.T, f deletionImportFixture, owner, runID, auditID, clientKey string) Receipt {
	t.Helper()
	receiptID := clientKey + "-receipt"
	if err := tryInsertInboxReceipt(t, f, owner, runID, auditID, clientKey); err != nil {
		t.Fatal(err)
	}
	receipt, err := readReceiptByID(f.ctx, f.pool, receiptID)
	if err != nil {
		t.Fatal(err)
	}
	return receipt
}

// tryInsertInboxReceipt is insertInboxReceipt up to the receipt write, whose
// error it returns.
func tryInsertInboxReceipt(t *testing.T, f deletionImportFixture, owner, runID, auditID, clientKey string) error {
	t.Helper()
	source, err := readReceiptByID(f.ctx, f.pool, f.receiptID)
	if err != nil {
		t.Fatal(err)
	}
	canonical, err := canonicalize(testSubmission(clientKey+"-invocation", clientKey, []contracts.ArtifactRef{}))
	if err != nil {
		t.Fatal(err)
	}
	written, err := artifactstest.WriteFindingProposal(
		f.ctx, artifacts.NewService(artifactpostgres.NewRepository(f.pool)), runID, clientKey+"-proposal",
		artifacts.Payload{MediaType: proposalMediaType, Data: canonical.proposalBytes},
	)
	if err != nil {
		t.Fatal(err)
	}
	proposal := ExactArtifact{Ref: written.Ref, Digest: auditdomain.DigestBytes(canonical.proposalBytes),
		MediaType: written.MediaType, SizeBytes: written.Size}
	origin := source.Origin
	origin.RunID, origin.Audit = runID, nil
	if auditID != "" {
		origin.Audit = &AuditOrigin{AuditID: auditID, ExecutionID: auditID + "-execution", Role: "discovery"}
	}
	project := "delete-project"
	grant := controlplane.AllocationGrant{RunID: runID, AllocationID: "allocation", StageExecutionID: "stage",
		RuntimeAgentID: "runtime", RuntimeInstanceID: "instance", LogicalAgentName: "worker"}
	fingerprint, err := auditdomain.EncodeFindingFingerprint(auditdomain.UnavailableFindingFingerprint("not_provided"))
	if err != nil {
		t.Fatal(err)
	}
	receiptID := clientKey + "-receipt"
	return persistencepostgres.InTx(f.ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		return insertReceipt(f.ctx, tx, receiptID, clientKey+"-proposal", canonical, fingerprint, proposal,
			[]ExactArtifact{}, origin, owner, &project, grant, nil, nil, "")
	})
}

// pageAuditInbox pages list from the start at limit and returns every page.
func pageAuditInbox(
	t *testing.T, ctx context.Context, list func(context.Context, string, string, ListQuery) ([]Receipt, error),
	owner, auditID string, limit int,
) [][]Receipt {
	t.Helper()
	var pages [][]Receipt
	query := ListQuery{Limit: limit}
	for {
		page, err := list(ctx, owner, auditID, query)
		if err != nil {
			t.Fatalf("inbox page at limit %d: %v", limit, err)
		}
		pages = append(pages, page)
		if len(page) == 0 {
			return pages
		}
		if len(pages) > 64 {
			t.Fatalf("inbox at limit %d does not end", limit)
		}
		last := page[len(page)-1]
		query.AfterCreatedAt, query.AfterReceiptID = &last.CreatedAt, last.ReceiptID
	}
}

// Every page the service returns, of the inbox and of the held-only inbox,
// holds exactly the receipts the original statements list, in their order,
// each hydrated as GetAuditReceipts reads it through the Audit: only the
// Audit's own hold and the proposal from that hold's copy once the source Run
// is gone. The fixture has native-only, held-only and overlapping receipts,
// receipts sharing one creation time, an old receipt imported later, a
// sibling Audit's receipt imported, a hold in another Project, another
// owner's receipt naming the Audit, and native receipts of a deleted Run with
// and without a hold.
func TestPostgresAuditInboxServicePagesKeepOrderAndProjections(t *testing.T) {
	f := newDeletionImportFixture(t)
	owner, auditID := f.request.OwnerID, f.request.AuditID
	const siblingAudit, liveRun, goneRun = "inbox-sibling", "inbox-live-run", "delete-run"
	createFixtureAudit(t, f, siblingAudit)
	createFixtureRun(t, f, liveRun)
	createInboxProject(t, f.ctx, f.pool, owner, "inbox-other-project")
	receipts := map[string]Receipt{}
	for _, seed := range []struct{ owner, run, audit, key string }{
		{owner, goneRun, auditID, "gone-held"},
		{owner, goneRun, auditID, "gone-unheld"},
		{owner, liveRun, auditID, "native-a"},
		{owner, liveRun, auditID, "native-b"},
		{owner, liveRun, auditID, "overlap"},
		{owner, liveRun, siblingAudit, "sibling-held"},
		{owner, liveRun, siblingAudit, "sibling-only"},
		{"intruder-owner", liveRun, auditID, "intruder"},
	} {
		receipts[seed.key] = insertInboxReceipt(t, f, seed.owner, seed.run, seed.audit, seed.key)
	}
	// native-b, overlap and sibling-held share one creation time, and the
	// fixture's ordinary receipt is older than all of them.
	if _, err := f.pool.Exec(f.ctx, `ALTER TABLE finding_proposal_receipts DISABLE TRIGGER finding_proposal_receipts_protect_immutable`); err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(f.ctx, `
UPDATE finding_proposal_receipts
   SET created_at = (SELECT created_at FROM finding_proposal_receipts WHERE receipt_id = 'native-b-receipt')
 WHERE receipt_id IN ('overlap-receipt', 'sibling-held-receipt')`); err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(f.ctx, `ALTER TABLE finding_proposal_receipts ENABLE TRIGGER finding_proposal_receipts_protect_immutable`); err != nil {
		t.Fatal(err)
	}
	retain := func(key, runID string) ImportRequest {
		return ImportRequest{OwnerID: owner, AuditID: auditID, RunID: runID, Proposal: receipts[key].Proposal.Ref}
	}
	if err := f.intake.RetainAuditCollectionBatch(f.ctx, []ImportRequest{retain("gone-held", goneRun)}); err != nil {
		t.Fatal(err)
	}
	if err := f.intake.RetainAuditCollectionBatch(f.ctx, []ImportRequest{retain("overlap", liveRun)}); err != nil {
		t.Fatal(err)
	}
	for _, request := range []ImportRequest{retain("sibling-held", liveRun), f.request} {
		if _, _, err := f.intake.ImportIntoAudit(f.ctx, request); err != nil {
			t.Fatal(err)
		}
	}
	// A hold of the Audit naming another Project admits nothing.
	if _, err := f.pool.Exec(f.ctx, `
INSERT INTO finding_proposal_audit_holds (receipt_id, audit_id, project_id, proposal_ref, evidence)
SELECT receipt_id, $2, 'inbox-other-project', proposal_ref, '[]'::jsonb
  FROM finding_proposal_receipts WHERE receipt_id = $1`, "sibling-only-receipt", auditID); err != nil {
		t.Fatal(err)
	}
	if err := deleteRun(f.ctx, f.pool, owner, goneRun); err != nil {
		t.Fatal(err)
	}

	for _, view := range []struct {
		name, auditID string
		list          func(context.Context, string, string, ListQuery) ([]Receipt, error)
		original      string
		want          []string
	}{
		{"inbox", auditID, f.intake.ListAuditInbox, originalAuditInboxSQL,
			[]string{f.receiptID, "gone-held-receipt", "native-a-receipt", "native-b-receipt", "overlap-receipt", "sibling-held-receipt"}},
		{"held inbox", auditID, f.intake.ListAuditHeldInbox, originalAuditHeldInboxSQL,
			[]string{f.receiptID, "gone-held-receipt", "overlap-receipt", "sibling-held-receipt"}},
		{"sibling inbox", siblingAudit, f.intake.ListAuditInbox, originalAuditInboxSQL,
			[]string{"sibling-held-receipt", "sibling-only-receipt"}},
		{"sibling held inbox", siblingAudit, f.intake.ListAuditHeldInbox, originalAuditHeldInboxSQL, nil},
	} {
		// The expected order comes from the database's own ordering of the
		// expected receipts, independently of either inbox statement.
		var want []string
		if err := f.pool.QueryRow(f.ctx, `
SELECT coalesce(array_agg(receipt_id ORDER BY created_at, receipt_id), '{}')
  FROM finding_proposal_receipts WHERE receipt_id = ANY($1)`, view.want).Scan(&want); err != nil {
			t.Fatal(err)
		}
		if original := receiptIDs(inboxPage(t, f.ctx, f.pool, view.original, owner, view.auditID, nil, 201)); !slices.Equal(original, want) {
			t.Fatalf("original %s = %v, want %v", view.name, original, want)
		}
		for limit := 1; limit <= len(want)+1; limit++ {
			pages := pageAuditInbox(t, f.ctx, view.list, owner, view.auditID, limit)
			var listed []string
			var after *Receipt
			for index, page := range pages {
				original := inboxPage(t, f.ctx, f.pool, view.original, owner, view.auditID, after, limit)
				if !slices.Equal(receiptIDs(page), receiptIDs(original)) {
					t.Fatalf("%s at limit %d after %d receipts = %v, original %v",
						view.name, limit, len(listed), receiptIDs(page), receiptIDs(original))
				}
				if len(page) == 0 {
					break
				}
				if len(page) < limit && index != len(pages)-2 {
					// Only the final non-empty page may be short.
					t.Fatalf("%s at limit %d returned a short page %v before the last", view.name, limit, receiptIDs(page))
				}
				read, err := f.intake.GetAuditReceipts(f.ctx, owner, view.auditID, receiptIDs(page))
				if err != nil {
					t.Fatalf("%s receipts read through the Audit: %v", view.name, err)
				}
				if !reflect.DeepEqual(page, read) {
					t.Fatalf("%s page at limit %d = %+v, GetAuditReceipts = %+v", view.name, limit, page, read)
				}
				for _, receipt := range page {
					for _, hold := range receipt.AuditHolds {
						if hold.AuditID != view.auditID {
							t.Fatalf("%s exposes hold of %s on %s", view.name, hold.AuditID, receipt.ReceiptID)
						}
					}
					if receipt.Document.ClientKey != receipt.ClientKey {
						t.Fatalf("%s hydrated %s with document %q", view.name, receipt.ReceiptID, receipt.Document.ClientKey)
					}
				}
				listed = append(listed, receiptIDs(page)...)
				after = &page[len(page)-1]
			}
			if !slices.Equal(listed, want) {
				t.Fatalf("%s at limit %d listed %v, want %v", view.name, limit, listed, want)
			}
		}
	}
	gone, err := f.intake.GetAuditReceipt(f.ctx, owner, auditID, "gone-held-receipt")
	if err != nil || !gone.Origin.RunDeleted || gone.Retention != RetentionProjectHeld || len(gone.AuditHolds) != 1 {
		t.Fatalf("held receipt of the deleted Run = %+v, %v", gone, err)
	}
}
