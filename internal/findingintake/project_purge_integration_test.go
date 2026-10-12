//go:build integration

package findingintake

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// The receipts of a deleting Project go only through its purge, in batches
// bounded by count and by stored bytes; every other receipt stays immutable
// (DEL-R61, LIM-DEL-3).
func TestPostgresProjectReceiptPurgeIsBoundedAndScoped(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	pool := isolatedFindingPool(t, ctx)
	projects := projectstore.NewPostgresStore(pool)
	create := func(projectID string) projectstore.Project {
		t.Helper()
		project, _, err := projects.Create(ctx, projectstore.CreateParams{
			ProjectID: projectID, OwnerID: "purge-owner", Kind: projectstore.KindProject,
			Name: projectID, IdempotencyKey: projectID,
			RequestDigest: auditdomain.DigestBytes([]byte(projectID)),
		})
		if err != nil {
			t.Fatal(err)
		}
		return project
	}
	deleting := create("receipt-purge-deleting")
	active := create("receipt-purge-active")
	seedPurgeReceipts(t, ctx, pool, "deleting", &deleting.ProjectID, 7)
	seedPurgeReceipts(t, ctx, pool, "active", &active.ProjectID, 2)
	seedPurgeReceipts(t, ctx, pool, "standalone", nil, 2)

	purge := func() (bool, error) {
		var done bool
		err := persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			var err error
			done, err = PurgeProjectReceipts(ctx, tx, deleting.ProjectID)
			return err
		})
		return done, err
	}
	if _, err := purge(); persistencepostgres.SQLState(err) != persistencepostgres.SQLStateCheckViolation {
		t.Fatalf("receipt purge of an active Project = %v, want the immutability check", err)
	}
	if _, _, err := projects.BeginDeletion(ctx, projectstore.BeginDeletionParams{
		ProjectID: deleting.ProjectID, OwnerID: deleting.OwnerID, ExpectedRevision: deleting.Revision,
	}); err != nil {
		t.Fatal(err)
	}
	// Outside the Project purge mode, and for a receipt of another Project or
	// of a Run outside any Project, the trigger still refuses the deletion.
	for _, attempt := range []struct {
		receiptID string
		purgeMode bool
	}{
		{"deleting-0001", false}, {"active-0001", true}, {"standalone-0001", true},
	} {
		err := persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			if attempt.purgeMode {
				if _, err := tx.Exec(ctx, `SELECT set_config('contractor.lifecycle_purge', 'project', true)`); err != nil {
					return err
				}
			}
			_, err := tx.Exec(ctx, `DELETE FROM finding_proposal_receipts WHERE receipt_id = $1`, attempt.receiptID)
			return err
		})
		if persistencepostgres.SQLState(err) != persistencepostgres.SQLStateCheckViolation {
			t.Fatalf("delete receipt %s (purge mode %t) = %v, want the immutability check",
				attempt.receiptID, attempt.purgeMode, err)
		}
	}

	var firstTwo int64
	if err := pool.QueryRow(ctx, `
SELECT sum(pg_column_size(report_projection) + pg_column_size(evidence))
  FROM (SELECT report_projection, evidence FROM finding_proposal_receipts
         WHERE project_id = $1 ORDER BY receipt_id LIMIT 2) AS first`,
		deleting.ProjectID).Scan(&firstTwo); err != nil {
		t.Fatal(err)
	}
	restoreSize, restoreBytes := receiptPurgeBatchSize, receiptPurgeBatchBytes
	t.Cleanup(func() { receiptPurgeBatchSize, receiptPurgeBatchBytes = restoreSize, restoreBytes })
	for _, step := range []struct {
		size  int
		bytes int64
		want  []string
		done  bool
	}{
		// The byte bound ends the batch after the receipts that fit in it.
		{1000, firstTwo, []string{"deleting-0001", "deleting-0002"}, false},
		// The first receipt goes even when it alone exceeds the byte bound.
		{1000, 1, []string{"deleting-0003"}, false},
		{3, 64 << 20, []string{"deleting-0004", "deleting-0005", "deleting-0006"}, false},
		{3, 64 << 20, []string{"deleting-0007"}, true},
		{3, 64 << 20, nil, true},
	} {
		receiptPurgeBatchSize, receiptPurgeBatchBytes = step.size, step.bytes
		before := projectReceiptIDs(t, ctx, pool, &deleting.ProjectID)
		done, err := purge()
		if err != nil {
			t.Fatal(err)
		}
		after := projectReceiptIDs(t, ctx, pool, &deleting.ProjectID)
		removed := before[:len(before)-len(after)]
		if done != step.done || fmt.Sprint(removed) != fmt.Sprint(step.want) ||
			fmt.Sprint(after) != fmt.Sprint(before[len(removed):]) {
			t.Fatalf("purge with %d receipts and %d bytes = (%t, removed %v, left %v), want (%t, %v)",
				step.size, step.bytes, done, removed, after, step.done, step.want)
		}
	}
	if left := projectReceiptIDs(t, ctx, pool, &active.ProjectID); len(left) != 2 {
		t.Fatalf("receipts of another Project = %v", left)
	}
	if left := projectReceiptIDs(t, ctx, pool, nil); len(left) != 2 {
		t.Fatalf("receipts of Runs outside any Project = %v", left)
	}
	if _, err := PurgeProjectReceipts(ctx, nil, deleting.ProjectID); err == nil {
		t.Fatal("receipt purge without a transaction was accepted")
	}
	err := persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		_, err := PurgeProjectReceipts(ctx, tx, " ")
		return err
	})
	if !errors.Is(err, ErrInvalid) {
		t.Fatalf("receipt purge without a Project = %v", err)
	}
}

// seedPurgeReceipts inserts count receipts named prefix-0001 onwards for the
// Project, or for a Run outside any Project when projectID is nil.
func seedPurgeReceipts(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, prefix string, projectID *string, count int,
) {
	t.Helper()
	if _, err := pool.Exec(ctx, `
INSERT INTO finding_proposal_receipts (
    receipt_id, proposal_id, allocation_id, runtime_agent_id,
    runtime_instance_id, stage_execution_id, logical_agent_name,
    invocation_id, submission_id, client_key, request_digest,
    run_id, owner_id, project_id,
    workflow_name, workflow_version, workflow_schema_version,
    workflow_configuration_ref, workflow_closure_digest,
    proposal_ref, proposal_digest, proposal_media_type,
    proposal_size_bytes, evidence, report_projection
)
SELECT $1 || '-' || lpad(n::text, 4, '0'), $1 || '-proposal-' || n, $1 || '-allocation',
       'runtime', 'instance', 'stage', 'worker', 'invocation', $1 || '-submission-' || n,
       'candidate-' || n, 'sha256:' || repeat('a', 64), $1 || '-run', 'purge-owner', $2,
       'finding-source', '1', 'contractor/v1alpha1', '{"name":"finding-source","version":"1"}'::jsonb,
       'sha256:' || repeat('b', 64),
       jsonb_build_object('namespace', 'findings', 'name', 'candidate-' || n, 'revision', 'r1'),
       'sha256:' || repeat('c', 64), 'application/json', 128, '[]'::jsonb,
       '{"description":"Test finding ' || n || '.","limitations":[],"severity_suggestion":"","standard_refs":[],"subject":null,"title":"Test finding."}'
  FROM generate_series(1, $3::integer) AS n`, prefix, projectID, count); err != nil {
		t.Fatalf("seed finding proposal receipts: %v", err)
	}
}

// projectReceiptIDs lists the receipts of the Project in purge order, or of
// Runs outside any Project when projectID is nil.
func projectReceiptIDs(t *testing.T, ctx context.Context, pool *pgxpool.Pool, projectID *string) []string {
	t.Helper()
	rows, err := pool.Query(ctx, `
SELECT receipt_id FROM finding_proposal_receipts
 WHERE project_id IS NOT DISTINCT FROM $1 ORDER BY receipt_id`, projectID)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := pgx.CollectRows(rows, pgx.RowTo[string])
	if err != nil {
		t.Fatal(err)
	}
	return ids
}
