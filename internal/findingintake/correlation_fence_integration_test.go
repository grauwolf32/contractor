//go:build integration

package findingintake

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// Audit purge releases execution references. It does not mutate current
// Project Findings or need the Project deletion transaction's row lock.
func TestPostgresAuditPurgeLeavesTheCanonicalProjectFenceAlone(t *testing.T) {
	f := newDeletionImportFixture(t)
	if _, _, err := f.intake.ImportIntoAudit(f.ctx, f.request); err != nil {
		t.Fatal(err)
	}
	projects := projectstore.NewPostgresStore(f.pool)
	beginFixtureProjectDeletion(t, f, projects)
	claim, claimed, err := projects.ClaimDeletion(f.ctx, projectstore.ClaimDeletionParams{ClaimID: "project-deleter", Duration: time.Minute, PurgeDuration: time.Minute})
	if err != nil || !claimed {
		t.Fatalf("claim: %t %v", claimed, err)
	}
	if _, err := hookstore.NewProjectDeletion(f.pool, artifactpgtest.NewStorage()).RequestProjectOwnedDeletion(f.ctx, claim); err != nil {
		t.Fatal(err)
	}
	audits := auditstoretest.NewStore(f.pool, artifactpgtest.NewStorage()).WithPurgeDependencies(DeleteAuditHoldsBatch)
	claims, err := audits.Claim(f.ctx, auditstore.ClaimParams{HolderID: "audit-controller", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("Audit claim: %+v %v", claims, err)
	}
	before := readIntakeState(t, f)
	tx, err := f.pool.Begin(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.WithoutCancel(f.ctx)) }()
	if err := projectstore.NewPostgresTxStore(tx).LockDeletionClaim(f.ctx, claim); err != nil {
		t.Fatal(err)
	}
	if err := audits.PurgeClaimed(f.ctx, claims[0], auditdomain.ArtifactNamespace(f.request.AuditID)); err != nil {
		t.Fatal(err)
	}
	var afterRevision int64
	if err := f.pool.QueryRow(f.ctx, "SELECT revision FROM finding_correlation_fences WHERE project_id=$1", "delete-project").Scan(&afterRevision); err != nil {
		t.Fatal(err)
	}
	if afterRevision != before.correlationRevision {
		t.Fatal("Audit purge mutated the canonical Project state fence")
	}
	if _, err := audits.Get(f.ctx, f.request.OwnerID, f.request.AuditID); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatal("Audit survived purge", err)
	}
	receipt, err := readReceiptByID(f.ctx, f.pool, f.receiptID)
	if err != nil || receipt.ProjectProposal == nil || receipt.Retention != RetentionProjectHeld {
		t.Fatalf("Project source lost: %+v %v", receipt, err)
	}
}

func TestPostgresSourceRunDeletionKeepsTheProjectFenceWithoutAudits(t *testing.T) {
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
	intake, err := New(pool, artifactpgtest.NewStorage())
	if err != nil {
		t.Fatal(err)
	}
	receipt, _, err := intake.Submit(ctx, grant, testSubmission("unaudited-invocation", "unaudited", fixtureEvidence(t, ctx, pool, grant)))
	if err != nil {
		t.Fatal(err)
	}
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
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
		receipt.ReceiptID).Scan(&retention); err != nil || retention != string(RetentionProjectHeld) {
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
	rows, initialRevision := readFence()
	if rows != 1 || initialRevision < 1 {
		t.Fatalf("missing Project-owned fence: %d/%d", rows, initialRevision)
	}
	fixture := deletionImportFixture{ctx: ctx, pool: pool, request: ImportRequest{OwnerID: owner}}
	for audit, audits := range []string{"unaudited-first", "unaudited-second"} {
		createFixtureAudit(t, fixture, audits)
		if rows, revision := readFence(); rows != 1 || revision != initialRevision {
			t.Fatalf("fence after Audit %d = %d rows at revision %d", audit+1, rows, revision)
		}
	}
}
