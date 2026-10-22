//go:build integration

package findingintake

import (
	"context"
	"errors"
	"sort"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/auditstore/querystore"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestPostgresAutomaticCorrelationPreservesIndependentObservations(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedFindingPool(t, ctx)
	const owner, project, run = "correlation-owner", "delete-project", "correlation-run"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: project, OwnerID: owner, Kind: projectstore.KindProject, Name: "Correlation", IdempotencyKey: "correlation", RequestDigest: auditdomain.DigestBytes([]byte("correlation"))}); err != nil {
		t.Fatal(err)
	}
	var refs []contracts.ArtifactRef
	grant := quotaSubmissionGrantWithInputs(t, ctx, pool, owner, project, run, func() map[string]runstore.PinnedContextArtifact {
		refs = writeFingerprintInputs(t, ctx, pool, run)
		return map[string]runstore.PinnedContextArtifact{"source": {Artifact: &refs[0]}, "report": {Artifact: &refs[1]}}
	})
	intake, err := New(pool)
	if err != nil {
		t.Fatal(err)
	}
	submit := func(key, severity string) Receipt {
		input := testSubmission("invocation-"+key, key, refs)
		line := int64(3)
		input.Proposal.Title = "rule/sql"
		input.Proposal.Description = "Native issue message"
		input.Proposal.Subject = nil
		input.Proposal.SeveritySuggestion = severity
		input.Proposal.Locations = []auditdomain.FindingLocation{{File: "src/a.py", Line: &line}}
		receipt, _, err := intake.Submit(ctx, grant, input)
		if err != nil || receipt.Fingerprint.Availability != "available" {
			t.Fatalf("qualified receipt=%+v %v", receipt.Fingerprint, err)
		}
		return receipt
	}
	first, second := submit("first", "high"), submit("second", "low")
	fixture := deletionImportFixture{ctx: ctx, pool: pool, intake: intake, receiptID: first.ReceiptID, request: ImportRequest{OwnerID: owner, RunID: run, Proposal: first.Proposal.Ref}}
	for _, id := range []string{"correlation-a", "correlation-b", "correlation-c", "correlation-d", "correlation-e"} {
		createFixtureAudit(t, fixture, id)
	}
	readRevision := func() int64 {
		var value int64
		if err := pool.QueryRow(ctx, "SELECT finding_correlation_revision FROM projects WHERE project_id=$1", project).Scan(&value); err != nil {
			t.Fatal(err)
		}
		return value
	}
	importReceipt := func(auditID string, receipt Receipt) string {
		request := ImportRequest{OwnerID: owner, AuditID: auditID, RunID: run, Proposal: receipt.Proposal.Ref}
		if _, replayed, err := intake.ImportIntoAudit(ctx, request); err != nil || replayed {
			t.Fatalf("admit=%v %v", replayed, err)
		}
		var id string
		if err := pool.QueryRow(ctx, "SELECT finding_id FROM audit_findings WHERE audit_id=$1 AND first_receipt_id=$2", auditID, receipt.ReceiptID).Scan(&id); err != nil {
			t.Fatal(err)
		}
		return id
	}
	firstID := importReceipt("correlation-a", first)
	reads := querystore.New(pool)
	initial, err := reads.GetFinding(ctx, owner, "correlation-a", firstID)
	if err != nil || initial.Correlation == nil || initial.Correlation.AutomaticDuplicate || initial.Correlation.HasOtherObservations {
		t.Fatalf("unique presentation=%+v %v", initial.Correlation, err)
	}
	beforeReplay := readRevision()
	request := ImportRequest{OwnerID: owner, AuditID: "correlation-a", RunID: run, Proposal: first.Proposal.Ref}
	if _, replayed, err := intake.ImportIntoAudit(ctx, request); err != nil || !replayed {
		t.Fatalf("replay=%v %v", replayed, err)
	}
	if after := readRevision(); after != beforeReplay {
		t.Fatalf("replay advanced correlation revision: %d -> %d", beforeReplay, after)
	}
	ids := []string{firstID, importReceipt("correlation-b", first), importReceipt("correlation-b", second)}
	// A Project deletion/metadata writer must never wait on an Audit that is
	// itself waiting on the Project: admission aborts using the existing class.
	parent, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := parent.Exec(ctx, "SELECT project_id FROM projects WHERE project_id=$1 FOR UPDATE", project); err != nil {
		t.Fatal(err)
	}
	contested := ImportRequest{OwnerID: owner, AuditID: "correlation-c", RunID: run, Proposal: first.Proposal.Ref}
	if _, _, err := intake.ImportIntoAudit(ctx, contested); persistencepostgres.Classify(err) != persistencepostgres.ClassTransactionConflict {
		_ = parent.Rollback(ctx)
		t.Fatalf("contested parent lock=%v", err)
	}
	if _, err := parent.Exec(ctx, "SELECT audit_id FROM audits WHERE audit_id=$1 FOR UPDATE NOWAIT", "correlation-c"); err != nil {
		_ = parent.Rollback(ctx)
		t.Fatalf("aborted admission retained Audit lock: %v", err)
	}
	if err := parent.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	ids = append(ids, importReceipt("correlation-c", first))
	// Simultaneous independent Audit admissions share the Project fence. An
	// exhausted finite conflict is reconciled only after both writers join.
	type admissionResult struct {
		audit string
		err   error
	}
	start, completed := make(chan struct{}), make(chan admissionResult, 2)
	for _, auditID := range []string{"correlation-d", "correlation-e"} {
		go func() {
			<-start
			_, _, err := intake.ImportIntoAudit(ctx, ImportRequest{OwnerID: owner, AuditID: auditID, RunID: run, Proposal: first.Proposal.Ref})
			completed <- admissionResult{auditID, err}
		}()
	}
	close(start)
	results := []admissionResult{<-completed, <-completed}
	for _, result := range results {
		if result.err != nil && persistencepostgres.Classify(result.err) != persistencepostgres.ClassTransactionConflict {
			t.Fatal(result.err)
		}
		if result.err != nil {
			if _, _, err := intake.ImportIntoAudit(ctx, ImportRequest{OwnerID: owner, AuditID: result.audit, RunID: run, Proposal: first.Proposal.Ref}); err != nil {
				t.Fatal(err)
			}
		}
		var id string
		if err := pool.QueryRow(ctx, "SELECT finding_id FROM audit_findings WHERE audit_id=$1 AND first_receipt_id=$2", result.audit, first.ReceiptID).Scan(&id); err != nil {
			t.Fatal(err)
		}
		ids = append(ids, id)
	}
	sort.Strings(ids)
	var representative auditdomain.FindingCorrelationReference
	for _, auditID := range []string{"correlation-a", "correlation-b", "correlation-c", "correlation-d", "correlation-e"} {
		rows, err := reads.ListFindings(ctx, auditstore.FindingListParams{OwnerID: owner, AuditID: auditID, Limit: 201})
		if err != nil {
			t.Fatal(err)
		}
		for _, row := range rows {
			c := row.Correlation
			if c == nil || c.Availability != "available" || c.ProjectRevision != uint64(readRevision()) || c.Representative.FindingID != ids[0] || !c.HasOtherObservations || c.AutomaticDuplicate != (row.FindingID != ids[0]) || row.State != auditstore.FindingProposed || row.Revision != 1 || row.CurrentDecisionID != nil {
				t.Fatalf("group presentation=%+v row=%+v", c, row)
			}
			representative = *c.Representative
			if _, err := reads.GetFinding(ctx, "foreign-owner", auditID, row.FindingID); !errors.Is(err, auditstore.ErrNotFound) {
				t.Fatalf("foreign read=%v", err)
			}
		}
	}
	var count int
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM audit_findings").Scan(&count); err != nil || count != 6 {
		t.Fatalf("observations collapsed/count=%d %v", count, err)
	}
	if _, err := pool.Exec(ctx, "UPDATE audit_findings SET correlation_value=$2 WHERE finding_id=$1", firstID, first.Fingerprint.Value); err != nil {
		t.Fatalf("identical immutable assignment=%v", err)
	}
	if _, err := pool.Exec(ctx, "UPDATE audit_findings SET correlation_group_key=repeat('f',64) WHERE finding_id=$1", firstID); persistencepostgres.Classify(err) != persistencepostgres.ClassRejected {
		t.Fatalf("frozen primary mutation=%v", err)
	}
	if _, err := pool.Exec(ctx, "UPDATE audit_findings SET audit_id='correlation-c' WHERE finding_id=$1", firstID); persistencepostgres.Classify(err) != persistencepostgres.ClassRejected {
		t.Fatalf("frozen Audit mutation=%v", err)
	}
	// Source availability changes fence both scope and existing Audit pages,
	// while immutable matching membership and every observation survive.
	runs := runstore.NewPostgresStore(pool)
	if err := runs.MarkStageAllocationReleased(ctx, grant.AllocationID); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(ctx, run, runstore.RunRunning, runstore.RunFailed, runstore.Reason{Code: "fixture_finished"}); err != nil {
		t.Fatal(err)
	}
	beforeDelete := readRevision()
	if err := deleteRun(ctx, pool, owner, run); err != nil {
		t.Fatal(err)
	}
	if readRevision() <= beforeDelete {
		t.Fatal("retained origin change did not advance Project fence")
	}
	// Purging the representative's Audit promotes a retained peer through the
	// indexed read; no peer rating, receipt identity or revision is rewritten.
	audits := auditstoretest.NewStore(pool).WithPurgeDependencies(DeleteAuditHoldsBatch)
	audit, err := audits.Get(ctx, owner, representative.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := audits.RequestDelete(ctx, auditstore.DeleteParams{OwnerID: owner, AuditID: representative.AuditID, ExpectedRevision: audit.Revision, IdempotencyKey: "purge-representative", RequestDigest: auditdomain.DigestBytes([]byte("purge-representative"))}); err != nil {
		t.Fatal(err)
	}
	claims, err := audits.Claim(ctx, auditstore.ClaimParams{HolderID: "correlation-purge", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claims=%+v %v", claims, err)
	}
	beforePurge := readRevision()
	if err := audits.PurgeClaimed(ctx, claims[0], auditdomain.ArtifactNamespace(representative.AuditID)); err != nil {
		t.Fatal(err)
	}
	if readRevision() <= beforePurge {
		t.Fatal("membership purge did not advance Project fence")
	}
	var remaining []auditstore.FindingRecord
	for _, auditID := range []string{"correlation-a", "correlation-b", "correlation-c", "correlation-d", "correlation-e"} {
		if auditID == representative.AuditID {
			continue
		}
		rows, err := reads.ListFindings(ctx, auditstore.FindingListParams{OwnerID: owner, AuditID: auditID, Limit: 201})
		if err != nil {
			t.Fatal(err)
		}
		remaining = append(remaining, rows...)
	}
	sort.Slice(remaining, func(i, j int) bool { return remaining[i].FindingID < remaining[j].FindingID })
	if len(remaining) < 2 {
		t.Fatalf("remaining=%d", len(remaining))
	}
	for _, row := range remaining {
		if row.Correlation.Representative.FindingID != remaining[0].FindingID || row.Revision != 1 || row.State != auditstore.FindingProposed {
			t.Fatalf("surviving observation changed=%+v", row)
		}
	}
}
