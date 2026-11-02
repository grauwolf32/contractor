//go:build integration

package findingintake

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
)

// resolveStatements is the documented statement count of one
// ResolveAuditProposals call for keys small proposal documents: one statement
// resolves every key with its Audit hold, and one exact batch of at most
// artifacts.MaxExactReadBatchSize documents reads the proposals (AUX-R98).
func resolveStatements(keys int) int64 {
	return 1 + int64((keys+artifacts.MaxExactReadBatchSize-1)/artifacts.MaxExactReadBatchSize)
}

// A source Run with 4,096 retained proposals is resolved through 32 calls of
// the maximum 128 keys, each in the same bounded number of statements
// whatever the Run's size.
func TestPostgresResolveAuditProposalsUsesBoundedSetReads(t *testing.T) {
	f := newDeletionImportFixture(t)
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Minute)
	defer cancel()
	f.ctx = ctx
	const proposals = auditdomain.MaximumItems
	keys := retainAuditChildProposals(t, f, proposals)
	counter := &collectionQueryCounter{}
	actor, _ := deletionActorPool(t, f, counter)
	resolver, err := New(actor, artifactpgtest.NewStorage())
	if err != nil {
		t.Fatal(err)
	}
	calls := 0
	counter.count.Store(0)
	for start := 0; start < proposals; start += auditdomain.MaximumProposalsPerItem {
		page := keys[start : start+auditdomain.MaximumProposalsPerItem]
		before := counter.count.Load()
		resolved, err := resolver.ResolveAuditProposals(ctx, f.request.OwnerID, f.request.AuditID,
			"delete-execution", f.request.RunID, page)
		if err != nil || len(resolved) != len(page) {
			t.Fatalf("resolve keys %d..%d = %d proposals, %v", start, start+len(page), len(resolved), err)
		}
		if got := counter.count.Load() - before; got != resolveStatements(len(page)) {
			t.Fatalf("resolving %d keys used %d statements, want %d", len(page), got, resolveStatements(len(page)))
		}
		for index, proposal := range resolved {
			key := page[index]
			if proposal.ReceiptID != key.ClientKey+"-receipt" || proposal.Document.ClientKey != key.ClientKey ||
				proposal.Origin.InvocationID != key.InvocationID || proposal.Proposal.Ref.ValidateExact() != nil {
				t.Fatalf("key %+v resolved to %+v", key, proposal)
			}
		}
		calls++
	}
	if total := counter.count.Load(); calls != proposals/auditdomain.MaximumProposalsPerItem ||
		total != int64(calls)*resolveStatements(auditdomain.MaximumProposalsPerItem) {
		t.Fatalf("%d calls used %d statements, want %d", calls, total,
			int64(calls)*resolveStatements(auditdomain.MaximumProposalsPerItem))
	}
}

// Every rejected selection keeps its outcome: a request that is invalid as a
// whole fails before any read; a missing, unretained, foreign or stale key is
// ErrNotFound and a repeated or ambiguous key ErrConflict, and no partial
// result is returned.
func TestPostgresResolveAuditProposalsPreservesSelectionOutcomes(t *testing.T) {
	f := newDeletionImportFixture(t)
	keys := retainAuditChildProposals(t, f, 3)
	unretained := insertAuditChildReceipt(t, f, "unretained")
	ambiguous := insertAmbiguousAuditChildReceipt(t, f, keys[2].ClientKey)
	counter := &collectionQueryCounter{}
	actor, _ := deletionActorPool(t, f, counter)
	resolver, err := New(actor, artifactpgtest.NewStorage())
	if err != nil {
		t.Fatal(err)
	}
	owner, audit, execution, run := f.request.OwnerID, f.request.AuditID, "delete-execution", f.request.RunID
	resolved, err := resolver.ResolveAuditProposals(f.ctx, owner, audit, execution, run, keys[:2])
	if err != nil || len(resolved) != 2 || resolved[0].ReceiptID != keys[0].ClientKey+"-receipt" ||
		resolved[1].ReceiptID != keys[1].ClientKey+"-receipt" {
		t.Fatalf("valid selection = %+v, %v", resolved, err)
	}
	if resolved, err := resolver.ResolveAuditProposals(f.ctx, owner, audit, execution, run, nil); err != nil || len(resolved) != 0 {
		t.Fatalf("empty selection = %+v, %v", resolved, err)
	}
	tooMany := make([]ProposalKey, auditdomain.MaximumProposalsPerItem+1)
	for index := range tooMany {
		tooMany[index] = ProposalKey{InvocationID: "invocation", ClientKey: fmt.Sprintf("key-%03d", index)}
	}
	missing := ProposalKey{InvocationID: "missing-invocation", ClientKey: "missing"}
	unretainedKey := ProposalKey{InvocationID: unretained.Origin.InvocationID, ClientKey: unretained.ClientKey}
	for _, test := range []struct {
		name                           string
		owner, audit, execution, runID string
		keys                           []ProposalKey
		want                           error
		reads                          bool
	}{
		{name: "too many keys", keys: tooMany, want: ErrInvalid},
		{name: "invalid identity", keys: []ProposalKey{keys[0], {InvocationID: " invocation", ClientKey: "key"}}, want: ErrInvalid},
		{name: "invalid identity after a missing key", keys: []ProposalKey{missing, {InvocationID: "invocation", ClientKey: ""}}, want: ErrInvalid},
		{name: "repeated key", keys: []ProposalKey{keys[0], keys[1], keys[0]}, want: ErrConflict},
		{name: "missing key", keys: []ProposalKey{keys[0], missing}, want: ErrNotFound, reads: true},
		{name: "unretained receipt", keys: []ProposalKey{unretainedKey}, want: ErrNotFound, reads: true},
		{name: "foreign Audit", audit: "another-audit", keys: keys[:1], want: ErrNotFound, reads: true},
		{name: "foreign owner", owner: "another-owner", keys: keys[:1], want: ErrNotFound, reads: true},
		{name: "foreign Run", runID: "another-run", keys: keys[:1], want: ErrNotFound, reads: true},
		{name: "stale execution", execution: "earlier-execution", keys: keys[:1], want: ErrNotFound, reads: true},
		{name: "ambiguous key", keys: []ProposalKey{keys[0], keys[2]}, want: ErrConflict, reads: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := struct{ owner, audit, execution, run string }{owner, audit, execution, run}
			if test.owner != "" {
				request.owner = test.owner
			}
			if test.audit != "" {
				request.audit = test.audit
			}
			if test.execution != "" {
				request.execution = test.execution
			}
			if test.runID != "" {
				request.run = test.runID
			}
			before := counter.count.Load()
			resolved, err := resolver.ResolveAuditProposals(f.ctx, request.owner, request.audit, request.execution, request.run, test.keys)
			if resolved != nil || !errors.Is(err, test.want) {
				t.Fatalf("resolve = (%+v, %v), want %v", resolved, err, test.want)
			}
			// A rejected key fails before any proposal document is read.
			want := int64(0)
			if test.reads {
				want = 1
			}
			if got := counter.count.Load() - before; got != want {
				t.Fatalf("rejected selection used %d statements, want %d", got, want)
			}
		})
	}
	if ambiguous == keys[2].ClientKey+"-receipt" {
		t.Fatal("ambiguous fixture reused the original receipt")
	}
}

// retainAuditChildProposals inserts count proposals of the fixture's Audit
// execution, retains them in collection pages and returns their keys.
func retainAuditChildProposals(t *testing.T, f deletionImportFixture, count int) []ProposalKey {
	t.Helper()
	keys := make([]ProposalKey, count)
	requests := make([]ImportRequest, 0, MaxAuditReceiptBatchSize)
	for index := range keys {
		clientKey := fmt.Sprintf("candidate-%04d", index)
		receipt := insertAuditChildReceipt(t, f, clientKey)
		keys[index] = ProposalKey{InvocationID: receipt.Origin.InvocationID, ClientKey: clientKey}
		request := f.request
		request.Proposal = receipt.Proposal.Ref
		requests = append(requests, request)
		if len(requests) == MaxAuditReceiptBatchSize || index == count-1 {
			if err := f.intake.RetainAuditCollectionBatch(f.ctx, requests); err != nil {
				t.Fatal(err)
			}
			requests = requests[:0]
		}
	}
	return keys
}

// insertAmbiguousAuditChildReceipt stores a second retained receipt of the
// retained proposal clientKey under the same invocation and client key, which
// Runtime never submits but no constraint excludes, and returns its ID.
func insertAmbiguousAuditChildReceipt(t *testing.T, f deletionImportFixture, clientKey string) string {
	t.Helper()
	original, duplicate := clientKey+"-receipt", clientKey+"-ambiguous-receipt"

	var columns, selected string
	if err := f.pool.QueryRow(f.ctx, `SELECT string_agg(quote_ident(attname),',' ORDER BY attnum),string_agg('copy.'||quote_ident(attname),',' ORDER BY attnum) FROM pg_attribute WHERE attrelid='finding_proposal_receipts'::regclass AND attnum>0 AND NOT attisdropped AND attgenerated=''`).Scan(&columns, &selected); err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(f.ctx, `INSERT INTO finding_proposal_receipts (`+columns+`) SELECT `+selected+`
 FROM finding_proposal_receipts AS receipt CROSS JOIN LATERAL jsonb_populate_record(NULL::finding_proposal_receipts,
 to_jsonb(receipt)||jsonb_build_object('receipt_id',$2::text,'proposal_id',$2::text,'submission_id',$2::text)) AS copy
 WHERE receipt.receipt_id=$1`, original, duplicate); err != nil {
		t.Fatal(err)
	}

	if _, err := f.pool.Exec(f.ctx, `
INSERT INTO finding_proposal_audit_holds
SELECT (jsonb_populate_record(NULL::finding_proposal_audit_holds, to_jsonb(hold) || jsonb_build_object(
           'receipt_id', $2::text))).*
  FROM finding_proposal_audit_holds AS hold WHERE hold.receipt_id = $1 AND hold.audit_id = $3`,
		original, duplicate, f.request.AuditID); err != nil {
		t.Fatal(err)
	}
	var matches int
	if err := f.pool.QueryRow(context.WithoutCancel(f.ctx), `
SELECT count(*) FROM finding_proposal_receipts AS receipt
  JOIN finding_proposal_audit_holds AS hold ON hold.receipt_id = receipt.receipt_id AND hold.audit_id = $2
 WHERE receipt.run_id = $1 AND receipt.client_key = $3`, f.request.RunID, f.request.AuditID, clientKey).Scan(&matches); err != nil {
		t.Fatal(err)
	}
	if matches != 2 {
		t.Fatalf("ambiguous fixture has %d retained receipts for %s, want 2", matches, clientKey)
	}
	return duplicate
}
