package pgstore

import (
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

func TestPostgresAuditPurgeCommitsBoundedBatchesAndResumesAfterTakeover(t *testing.T) {
	f := newCollectingAuditFixture(t, "purge-batches", 1<<20)
	confirmedID := collectPurgeFinding(t, f, "batch-confirmed")
	duplicateProposal := testExact("audit-findings", "candidate-batch-duplicate", "proposal-r1")
	duplicateProposal.MediaType, duplicateProposal.SizeBytes = "application/json", 128
	duplicateID := insertAuditChildFinding(t, f, "batch-duplicate", duplicateProposal)
	seedDecidedPurgeReviews(t, f, confirmedID, duplicateID)
	namespace := auditdomain.ArtifactNamespace(f.audit.AuditID)
	scope, _ := artifacts.ProjectScope(f.project.ProjectID)
	service := artifacts.NewService(artifacts.NewPostgresRepository(f.pool))
	for index := range 2005 {
		if _, err := service.CreateFrozen(f.ctx, scope,
			contracts.ArtifactRef{Namespace: namespace, Name: fmt.Sprintf("material-%05d", index)},
			artifacts.Payload{MediaType: "text/plain", Data: fmt.Appendf(nil, "material %d", index)}); err != nil {
			t.Fatal(err)
		}
	}
	// This storage fixture deliberately avoids a huge model-visible package.
	// Reserve the same event sequence the production allocator owns.
	if _, err := f.pool.Exec(f.ctx, `WITH reserved AS (
  UPDATE audits SET next_event_sequence = next_event_sequence + 4005 WHERE audit_id = $1 RETURNING next_event_sequence
)
INSERT INTO audit_events (audit_id, sequence_number, kind, entity_id, summary)
SELECT $1, next_event_sequence - 4006 + n, 'fixture.changed', $1, '{}'::jsonb
FROM reserved CROSS JOIN generate_series(1,4005) AS n`, f.audit.AuditID); err != nil {
		t.Fatal(err)
	}
	claim := drainDeletedAuditForPurge(t, f)
	revisions := func() int {
		t.Helper()
		var n int
		if err := f.pool.QueryRow(f.ctx, `SELECT count(*) FROM artifact_binding_revisions
WHERE scope_kind = 'project' AND scope_id = $1 AND namespace = $2`, f.project.ProjectID, namespace).Scan(&n); err != nil {
			t.Fatal(err)
		}
		return n
	}
	previous := revisions()
	if err := f.store.PurgeClaimed(f.ctx, claim, namespace); err != nil {
		t.Fatal(err)
	}
	if left := revisions(); previous-left != persistencepostgres.CleanupBatchRows || left == 0 {
		t.Fatalf("first purge: removed %d revisions, left %d", previous-left, left)
	}
	stale := claim
	if _, err := f.pool.Exec(f.ctx, `UPDATE audit_controller_claims
SET claimed_at = clock_timestamp() - interval '2 seconds', expires_at = clock_timestamp() - interval '1 second'
WHERE audit_id = $1`, f.audit.AuditID); err != nil {
		t.Fatal(err)
	}
	claims, err := f.store.Claim(f.ctx, auditstore.ClaimParams{HolderID: "purge-replacement", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("purge takeover: %v %v", claims, err)
	}
	claim = claims[0]
	if err := f.store.PurgeClaimed(f.ctx, stale, namespace); !errors.Is(err, auditstore.ErrClaimLost) {
		t.Fatalf("stale purge: %v", err)
	}
	tables := auditOwnedTables(t, f)
	for pass := range 20 {
		before := revisions()
		var eventsBefore int
		if err := f.pool.QueryRow(f.ctx, `SELECT count(*) FROM audit_events WHERE audit_id = $1`, claim.AuditID).Scan(&eventsBefore); err != nil {
			t.Fatal(err)
		}
		if err := f.store.PurgeClaimed(f.ctx, claim, namespace); err != nil {
			t.Fatalf("purge pass %d: %v", pass, err)
		}
		if removed := before - revisions(); removed > persistencepostgres.CleanupBatchRows {
			t.Fatalf("pass %d removed %d revisions", pass, removed)
		}
		var eventsAfter int
		if err := f.pool.QueryRow(f.ctx, `SELECT count(*) FROM audit_events WHERE audit_id = $1`, claim.AuditID).Scan(&eventsAfter); err != nil {
			t.Fatal(err)
		}
		if eventsBefore-eventsAfter > persistencepostgres.CleanupBatchRows {
			t.Fatalf("pass %d exceeded domain row budget", pass)
		}
		audit, err := f.store.Get(f.ctx, f.audit.OwnerID, claim.AuditID)
		if errors.Is(err, auditstore.ErrNotFound) {
			for _, table := range tables {
				assertPurgedTable(t, f, table)
			}
			return
		}
		if err != nil || audit.State != auditstore.AuditDeleting {
			t.Fatalf("partial purge lost deleting Audit: %v %v", audit, err)
		}
	}
	t.Fatal("bounded purge did not finish")
}
