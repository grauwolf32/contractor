//go:build integration

package projectlifecycle

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// purgeBatchRevisions is LIM-DEL-2: binding revisions one ProjectScope purge
// transaction removes.
const purgeBatchRevisions = 1000

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
