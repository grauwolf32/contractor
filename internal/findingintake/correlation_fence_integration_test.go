//go:build integration

package findingintake

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

// Project deletion locks its claimed Project row FOR UPDATE first in each of
// its transactions. An Audit purge is a finding writer that runs while its
// Project is deleting, and it takes the Project row only shared and without
// waiting (FND-R177). When the deletion holds its claim first, the purge aborts
// at once with a serialization failure and changes nothing. When the purge
// holds the fence first, the deletion waits for it and then purges the
// Project's receipts. Neither order ends in a deadlock abort.
func TestPostgresFindingPurgeSerializesWithProjectDeletionClaim(t *testing.T) {
	for _, purgeFirst := range []bool{false, true} {
		name := "deletion claim first"
		if purgeFirst {
			name = "purge holds the fence first"
		}
		t.Run(name, func(t *testing.T) {
			f := newDeletionImportFixture(t)
			if _, _, err := f.intake.ImportIntoAudit(f.ctx, f.request); err != nil {
				t.Fatal(err)
			}
			projects := projectstore.NewPostgresStore(f.pool)
			beginFixtureProjectDeletion(t, f, projects)
			claim, claimed, err := projects.ClaimDeletion(f.ctx, projectstore.ClaimDeletionParams{
				ClaimID: "project-deleter", Duration: time.Minute, PurgeDuration: time.Minute,
			})
			if err != nil || !claimed {
				t.Fatalf("claim the Project deletion = (%t, %v)", claimed, err)
			}
			if requested, err := hookstore.NewProjectDeletion(f.pool).RequestProjectOwnedDeletion(f.ctx, claim); err != nil || !requested {
				t.Fatalf("request the Project-owned Audit deletion = (%t, %v)", requested, err)
			}
			gate := &findingPurgeGate{held: make(chan struct{}), resume: make(chan struct{})}
			t.Cleanup(gate.release)
			var trace pgx.QueryTracer
			if purgeFirst {
				trace = gate
			}
			purgePool, purgePID := deletionActorPool(t, f, trace)
			audits := auditstoretest.NewStore(purgePool).WithPurgeDependencies(DeleteAuditHoldsBatch)
			auditClaims, err := audits.Claim(f.ctx, auditstore.ClaimParams{
				HolderID: "audit-controller", Lease: time.Minute, Limit: 1,
			})
			if err != nil || len(auditClaims) != 1 || auditClaims[0].AuditID != f.request.AuditID {
				t.Fatalf("claim the deleting Audit = (%+v, %v)", auditClaims, err)
			}
			namespace := auditdomain.ArtifactNamespace(f.request.AuditID)
			deletionPool, deletionPID := deletionActorPool(t, f, nil)
			deletion, err := deletionPool.Begin(f.ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = deletion.Rollback(context.Background()) }()
			before := readIntakeState(t, f)

			if !purgeFirst {
				if err := projectstore.NewPostgresTxStore(deletion).LockDeletionClaim(f.ctx, claim); err != nil {
					t.Fatal(err)
				}
				err := audits.PurgeClaimed(f.ctx, auditClaims[0], namespace)
				requireSerializationFailure(t, "Audit purge under the deletion's claim", err)
				if after := readIntakeState(t, f); after != before {
					t.Fatalf("aborted purge changed state %+v to %+v", before, after)
				}
				if err := deletion.Commit(f.ctx); err != nil {
					t.Fatal(err)
				}
				if err := audits.PurgeClaimed(f.ctx, auditClaims[0], namespace); err != nil {
					t.Fatalf("purge after the deletion's transaction = %v", err)
				}
				requireAuditPurged(t, f, before.correlationRevision+1)
				return
			}

			purged := make(chan error, 1)
			go func() { purged <- audits.PurgeClaimed(f.ctx, auditClaims[0], namespace) }()
			select {
			case <-gate.held:
			case err := <-purged:
				t.Fatalf("purge finished before it held the fence: %v", err)
			case <-f.ctx.Done():
				t.Fatal("purge never held the fence")
			}
			locked := make(chan error, 1)
			go func() { locked <- projectstore.NewPostgresTxStore(deletion).LockDeletionClaim(f.ctx, claim) }()
			awaitDeletionDatabaseWait(t, f, deletionPID, purgePID, "FOR UPDATE")
			gate.release()
			awaitDeletionOperation(t, f.ctx, purged)
			awaitDeletionOperation(t, f.ctx, locked)
			gone, err := PurgeProjectReceipts(f.ctx, deletion, "delete-project")
			if err != nil || !gone {
				t.Fatalf("receipt purge after the Audit purge = (%t, %v)", gone, err)
			}
			if err := deletion.Commit(f.ctx); err != nil {
				t.Fatal(err)
			}
			requireAuditPurged(t, f, before.correlationRevision+1)
			var receipts int
			if err := f.pool.QueryRow(f.ctx, `SELECT count(*) FROM finding_proposal_receipts WHERE project_id = $1`,
				"delete-project").Scan(&receipts); err != nil || receipts != 0 {
				t.Fatalf("receipts after the Project's purge = %d, %v", receipts, err)
			}
		})
	}
}

// A Project-bound Run's receipts exist whether or not its Project has an
// Audit. A Project that never had an Audit has no finding correlation fence
// row, and deleting such a Run fences nothing. The Project's first Audit then
// creates the row at revision 1, and a later Audit keeps it.
func TestPostgresSourceRunDeletionWithoutAuditFencesNothing(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedFindingPool(t, ctx)
	const owner, project, run = "unaudited-owner", "delete-project", "unaudited-run"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: project, OwnerID: owner, Kind: projectstore.KindProject, Name: "Unaudited",
		IdempotencyKey: "unaudited", RequestDigest: auditdomain.DigestBytes([]byte("unaudited")),
	}); err != nil {
		t.Fatal(err)
	}
	grant := quotaSubmissionGrant(t, ctx, pool, owner, project, run)
	intake, err := New(pool)
	if err != nil {
		t.Fatal(err)
	}
	receipt, _, err := intake.Submit(ctx, grant, testSubmission("unaudited-invocation", "unaudited", nil))
	if err != nil {
		t.Fatal(err)
	}
	runs := runstore.NewPostgresStore(pool)
	if err := runs.MarkStageAllocationReleased(ctx, grant.AllocationID); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(ctx, run, runstore.RunRunning, runstore.RunFailed, runstore.Reason{Code: "fixture_finished"}); err != nil {
		t.Fatal(err)
	}
	if err := deleteRun(ctx, pool, owner, run); err != nil {
		t.Fatalf("delete a Run of a Project without an Audit: %v", err)
	}
	var retention string
	if err := pool.QueryRow(ctx, `SELECT retention_state FROM finding_proposal_receipts WHERE receipt_id = $1`,
		receipt.ReceiptID).Scan(&retention); err != nil || retention != string(RetentionDiscarded) {
		t.Fatalf("receipt after its source Run's deletion = %q, %v", retention, err)
	}
	readFence := func() (int, int64) {
		t.Helper()
		var rows int
		var revision int64
		if err := pool.QueryRow(ctx, `SELECT count(*), coalesce(max(revision), 0) FROM finding_correlation_fences
 WHERE project_id = $1`, project).Scan(&rows, &revision); err != nil {
			t.Fatal(err)
		}
		return rows, revision
	}
	if rows, _ := readFence(); rows != 0 {
		t.Fatalf("a Project without an Audit has %d fence rows", rows)
	}
	fixture := deletionImportFixture{ctx: ctx, pool: pool, request: ImportRequest{OwnerID: owner}}
	for audit, audits := range []string{"unaudited-first", "unaudited-second"} {
		createFixtureAudit(t, fixture, audits)
		if rows, revision := readFence(); rows != 1 || revision != 1 {
			t.Fatalf("fence after Audit %d = %d rows at revision %d", audit+1, rows, revision)
		}
	}
}

// findingPurgeGate pauses an Audit purge right after the batch that deletes
// the Audit's findings, while its transaction holds the Project row FOR
// SHARE and the Project's finding correlation fence.
type findingPurgeGate struct {
	held, resume chan struct{}
	holdOnce     sync.Once
	releaseOnce  sync.Once
}

type findingPurgeGateKey struct{}

func (g *findingPurgeGate) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	if strings.Contains(data.SQL, "DELETE FROM audit_findings") {
		return context.WithValue(ctx, findingPurgeGateKey{}, true)
	}
	return ctx
}

func (g *findingPurgeGate) TraceQueryEnd(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryEndData) {
	if gated, _ := ctx.Value(findingPurgeGateKey{}).(bool); gated && data.Err == nil {
		g.holdOnce.Do(func() {
			close(g.held)
			<-g.resume
		})
	}
}

func (g *findingPurgeGate) release() { g.releaseOnce.Do(func() { close(g.resume) }) }

// requireSerializationFailure fails unless err is PostgreSQL's serialization
// failure: a definite transaction conflict that is not a deadlock abort.
func requireSerializationFailure(t *testing.T, operation string, err error) {
	t.Helper()
	var postgresError *pgconn.PgError
	if !errors.As(err, &postgresError) || postgresError.Code != persistencepostgres.SQLStateSerializationFailure {
		t.Fatalf("%s = %v, want a serialization failure", operation, err)
	}
}

// requireAuditPurged fails unless the fixture Audit and its findings are gone
// and the Project's fence row stays at revision.
func requireAuditPurged(t *testing.T, f deletionImportFixture, revision int64) {
	t.Helper()
	var audits, findings int
	var fence int64
	if err := f.pool.QueryRow(f.ctx, `
SELECT (SELECT count(*) FROM audits WHERE audit_id = $1),
       (SELECT count(*) FROM audit_findings WHERE audit_id = $1),
       (SELECT revision FROM finding_correlation_fences WHERE project_id = $2)`,
		f.request.AuditID, "delete-project").Scan(&audits, &findings, &fence); err != nil {
		t.Fatal(err)
	}
	if audits != 0 || findings != 0 || fence != revision {
		t.Fatalf("after the Audit purge: %d Audits, %d findings, fence revision %d, want revision %d",
			audits, findings, fence, revision)
	}
}
