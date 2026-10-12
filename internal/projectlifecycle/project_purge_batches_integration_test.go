//go:build integration

package projectlifecycle

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// purgeBatchRevisions is LIM-DEL-2: binding revisions one ProjectScope purge
// transaction removes.
const purgeBatchRevisions = 1000

// purgeBatchReceipts is LIM-DEL-3: finding proposal receipts one Project purge
// transaction deletes.
const purgeBatchReceipts = 1000

// A Project that accumulated 50,000 binding revisions is purged in bounded
// transactions: every controller step commits at most one purge batch, and
// the step that finds the scope empty deletes the Project.
func TestProjectDeletionPurgesLargeScopeInBoundedTransactions(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()
	pool := isolatedPool(t, ctx)
	projects := projectstore.NewPostgresStore(pool)
	runs := runstore.NewPostgresStore(pool)
	project, _, err := projects.Create(ctx, projectstore.CreateParams{
		ProjectID: "project-large", OwnerID: "user-1", Kind: projectstore.KindProject,
		Name: "Large scope", IdempotencyKey: "create-project-large",
		RequestDigest: "sha256:" + strings.Repeat("a", 64),
	})
	if err != nil {
		t.Fatal(err)
	}
	const revisions = 50000
	// About three revisions per binding, so binding groups straddle batches.
	for _, statement := range []string{`
INSERT INTO artifact_scopes (scope_kind, scope_id) VALUES ('project', $1)`, `
INSERT INTO artifact_blobs (sha256, payload, size_bytes, backend)
SELECT sha256(convert_to($1 || '-' || i, 'UTF8')), convert_to($1 || '-' || i, 'UTF8'),
       octet_length(convert_to($1 || '-' || i, 'UTF8')), 'postgresql'
FROM generate_series(1, $2::int) AS i`, `
INSERT INTO artifact_versions (version_id, blob_sha256, media_type)
SELECT $1 || '-version-' || i, sha256(convert_to($1 || '-' || i, 'UTF8')), 'text/plain'
FROM generate_series(1, $2::int) AS i`, `
INSERT INTO artifact_binding_revisions (scope_kind, scope_id, namespace, name, revision, version_id)
SELECT 'project', $1, 'notes', 'item-' || (i % 16667), 'rev-' || i, $1 || '-version-' || i
FROM generate_series(1, $2::int) AS i`, `
INSERT INTO artifact_bindings (scope_kind, scope_id, namespace, name, current_revision)
SELECT DISTINCT ON (namespace, name) scope_kind, scope_id, namespace, name, revision
FROM artifact_binding_revisions
WHERE scope_kind = 'project' AND scope_id = $1
ORDER BY namespace, name, revision DESC`} {
		arguments := []any{project.ProjectID}
		if strings.Contains(statement, "$2") {
			arguments = append(arguments, revisions)
		}
		if _, err := pool.Exec(ctx, statement, arguments...); err != nil {
			t.Fatalf("seed Project revisions: %v", err)
		}
	}
	if _, _, err := projects.BeginDeletion(ctx, projectstore.BeginDeletionParams{
		ProjectID: project.ProjectID, OwnerID: project.OwnerID, ExpectedRevision: project.Revision,
	}); err != nil {
		t.Fatal(err)
	}
	controller := newTestController(t, pool, runs, &recordingNotifier{}, "large")
	remaining := func() int {
		t.Helper()
		var count int
		if err := pool.QueryRow(ctx, `
SELECT count(*) FROM artifact_binding_revisions WHERE scope_kind = 'project' AND scope_id = $1`,
			project.ProjectID).Scan(&count); err != nil {
			t.Fatal(err)
		}
		return count
	}
	purgeSteps := 0
	for step := 0; ; step++ {
		if step > revisions/purgeBatchRevisions+10 {
			t.Fatalf("Project deletion did not finish after %d steps", step)
		}
		before := remaining()
		worked, err := controller.RunOnce(ctx)
		if err != nil || !worked {
			t.Fatalf("Project deletion step %d = (%t, %v)", step, worked, err)
		}
		removed := before - remaining()
		if removed > purgeBatchRevisions {
			t.Fatalf("one Project purge transaction removed %d revisions, more than %d", removed, purgeBatchRevisions)
		}
		if removed > 0 {
			purgeSteps++
		}
		if _, err := projects.Get(ctx, project.OwnerID, project.ProjectID); errors.Is(err, projectstore.ErrNotFound) {
			break
		} else if err != nil {
			t.Fatal(err)
		}
	}
	if purgeSteps != revisions/purgeBatchRevisions {
		t.Fatalf("Project purge removed revisions in %d transactions, want %d", purgeSteps, revisions/purgeBatchRevisions)
	}
	var leftovers int
	if err := pool.QueryRow(ctx, `
SELECT (SELECT count(*) FROM artifact_scopes) + (SELECT count(*) FROM artifact_bindings)
     + (SELECT count(*) FROM artifact_binding_revisions) + (SELECT count(*) FROM artifact_versions)
     + (SELECT count(*) FROM artifact_blobs)`).Scan(&leftovers); err != nil {
		t.Fatal(err)
	}
	if leftovers != 0 {
		t.Fatalf("Project purge left %d Artifact rows", leftovers)
	}
}

// The finding proposal receipts a deleting Project's Runs left behind go in
// bounded transactions before its ProjectScope: every controller step deletes
// at most one receipt batch, and no Artifact is purged while a receipt is
// left. Receipts of another Project and of a Run outside any Project stay
// (DEL-R61).
func TestProjectDeletionPurgesReceiptsInBoundedTransactions(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	pool := isolatedPool(t, ctx)
	projects := projectstore.NewPostgresStore(pool)
	create := func(projectID string) projectstore.Project {
		t.Helper()
		project, _, err := projects.Create(ctx, projectstore.CreateParams{
			ProjectID: projectID, OwnerID: "user-1", Kind: projectstore.KindProject,
			Name: projectID, IdempotencyKey: "create-" + projectID,
			RequestDigest: "sha256:" + strings.Repeat("a", 64),
		})
		if err != nil {
			t.Fatal(err)
		}
		return project
	}
	project := create("project-receipts")
	other := create("project-receipts-other")
	const receipts = 2500
	seedFindingReceipts(t, ctx, pool, "deleted", "run-deleted", &project.ProjectID, receipts)
	// Run deletion has left them as discarded tombstones (DEL-R28).
	if _, err := pool.Exec(ctx, `
UPDATE finding_proposal_receipts
   SET retention_state = 'discarded', source_run_deleted_at = clock_timestamp(),
       discarded_at = clock_timestamp(), retention_updated_at = clock_timestamp()
 WHERE run_id = 'run-deleted'`); err != nil {
		t.Fatal(err)
	}
	seedFindingReceipts(t, ctx, pool, "other", "run-other", &other.ProjectID, 2)
	seedFindingReceipts(t, ctx, pool, "standalone", "run-standalone", nil, 1)
	projectArtifacts, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"first", "second"} {
		if _, err := projectArtifacts.Write(ctx, artifacts.ArtifactRef{Namespace: "sources", Name: name},
			artifacts.Payload{MediaType: "text/plain", Data: []byte(name)}, nil); err != nil {
			t.Fatal(err)
		}
	}
	if _, _, err := projects.BeginDeletion(ctx, projectstore.BeginDeletionParams{
		ProjectID: project.ProjectID, OwnerID: project.OwnerID, ExpectedRevision: project.Revision,
	}); err != nil {
		t.Fatal(err)
	}
	controller := newTestController(t, pool, runstore.NewPostgresStore(pool), &recordingNotifier{}, "receipts")
	count := func(query string) int {
		t.Helper()
		var value int
		if err := pool.QueryRow(ctx, query, project.ProjectID).Scan(&value); err != nil {
			t.Fatal(err)
		}
		return value
	}
	const receiptQuery = `SELECT count(*) FROM finding_proposal_receipts WHERE project_id = $1`
	const revisionQuery = `
SELECT count(*) FROM artifact_binding_revisions WHERE scope_kind = 'project' AND scope_id = $1`
	receiptSteps := 0
	for step := 0; ; step++ {
		if step > receipts/purgeBatchReceipts+10 {
			t.Fatalf("Project deletion did not finish after %d steps", step)
		}
		receiptsBefore, revisionsBefore := count(receiptQuery), count(revisionQuery)
		worked, err := controller.RunOnce(ctx)
		if err != nil || !worked {
			t.Fatalf("Project deletion step %d = (%t, %v)", step, worked, err)
		}
		receiptsAfter, revisionsAfter := count(receiptQuery), count(revisionQuery)
		if removed := receiptsBefore - receiptsAfter; removed > purgeBatchReceipts {
			t.Fatalf("one Project purge transaction deleted %d receipts, more than %d", removed, purgeBatchReceipts)
		} else if removed > 0 {
			receiptSteps++
		}
		if receiptsAfter > 0 && revisionsAfter != revisionsBefore {
			t.Fatalf("step %d purged Artifacts while %d receipts were left", step, receiptsAfter)
		}
		if _, err := projects.Get(ctx, project.OwnerID, project.ProjectID); errors.Is(err, projectstore.ErrNotFound) {
			break
		} else if err != nil {
			t.Fatal(err)
		}
	}
	if receiptSteps != 3 {
		t.Fatalf("Project purge deleted %d receipts in %d transactions, want 3", receipts, receiptSteps)
	}
	if left := count(receiptQuery) + count(revisionQuery); left != 0 {
		t.Fatalf("Project purge left %d receipts and revisions", left)
	}
	var kept int
	if err := pool.QueryRow(ctx, `
SELECT count(*) FROM finding_proposal_receipts
 WHERE project_id = $1 OR (project_id IS NULL AND run_id = 'run-standalone')`,
		other.ProjectID).Scan(&kept); err != nil || kept != 3 {
		t.Fatalf("receipts outside the deleted Project = %d, error = %v", kept, err)
	}
}
