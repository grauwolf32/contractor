//go:build integration

package projectlifecycle

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/rundeletion"
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
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
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
INSERT INTO artifact_bindings (scope_kind, scope_id, namespace, name, current_revision, retained_revisions, retained_bytes)
SELECT DISTINCT ON (namespace, name) scope_kind, scope_id, namespace, name, revision, 1, 0
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
	projectArtifacts, err := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)).Project(project.ProjectID)
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
	controller := newTestController(t, pool, runstore.NewPostgresStore(pool, artifactpgtest.NewStorage()), &recordingNotifier{}, "receipts")
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

// evalPurgeBatchRows is the shared CleanupBatchRows bound of one Eval purge
// pass (EVM-R243), counting the two collection rows each deleted dataset
// revision rewrites.
const evalPurgeBatchRows = 1000

// A deleting evaluation Project's private Eval data goes first, one bounded
// pass per controller transaction: its fenced experiment, then its dataset
// revisions, collection row and Eval mutation receipts. No finding receipt or
// Artifact is purged while an Eval row is left, the Project row goes last,
// and another evaluation Project of the owner keeps all its Eval rows
// (EVM-R243, DEL-R60).
func TestProjectDeletionPurgesEvalDataInBoundedTransactions(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	pool := isolatedPool(t, ctx)
	projects := projectstore.NewPostgresStore(pool)
	create := func(projectID string) projectstore.Project {
		t.Helper()
		project, _, err := projects.Create(ctx, projectstore.CreateParams{
			ProjectID: projectID, OwnerID: "user-1", Kind: projectstore.KindEvaluation,
			Name: projectID, IdempotencyKey: "create-" + projectID,
			RequestDigest: "sha256:" + strings.Repeat("a", 64),
		})
		if err != nil {
			t.Fatal(err)
		}
		return project
	}
	project := create("evaluation-deleted")
	other := create("evaluation-kept")
	seedEval := func(projectID string, datasets, receipts int) {
		t.Helper()
		for _, seed := range []struct {
			statement string
			arguments []any
		}{{`
INSERT INTO eval_dataset_revisions (owner_id, project_id, dataset_id, revision, document, metadata)
SELECT 'user-1', $1, 'dataset', 'r' || i, convert_to('{"revision":' || i || '}', 'UTF8'), '{}'
FROM generate_series(1, $2::int) AS i`, []any{projectID, datasets}}, {`
INSERT INTO mutation_receipts (key_scope, idempotency_key, operation, request_digest, resource_id, response, owner_id, project_id)
SELECT ARRAY['eval', 'user-1', $1, 'experiments', 'experiment-create'], 'create-' || i, 'experiment-create',
       'sha256:' || repeat('b', 64), 'experiments', convert_to('{"experimentId":"exp-' || i || '"}', 'UTF8'), 'user-1', $1
FROM generate_series(1, $2::int) AS i`, []any{projectID, receipts}}, {`
INSERT INTO eval_experiments (
    experiment_id, owner_id, project_id, portable_id, control_mode, name, state, max_in_flight, wall_ms, deletion_requested_at
) VALUES ($1 || '-experiment', 'user-1', $1, 'experiment', 'external', 'Fenced', 'cancelling', 1, 1000, clock_timestamp())`,
			[]any{projectID}}, {`INSERT INTO eval_controller_claims (experiment_id) VALUES ($1 || '-experiment')`, []any{projectID}}} {
			if _, err := pool.Exec(ctx, seed.statement, seed.arguments...); err != nil {
				t.Fatalf("seed Eval data: %v", err)
			}
		}
	}
	const datasets, evalReceipts = 1500, 1200
	seedEval(project.ProjectID, datasets, evalReceipts)
	seedEval(other.ProjectID, 10, 5)
	seedFindingReceipts(t, ctx, pool, "deleted", "run-deleted", &project.ProjectID, 10)
	if _, err := pool.Exec(ctx, `
UPDATE finding_proposal_receipts
   SET retention_state = 'discarded', source_run_deleted_at = clock_timestamp(),
       discarded_at = clock_timestamp(), retention_updated_at = clock_timestamp()
 WHERE run_id = 'run-deleted'`); err != nil {
		t.Fatal(err)
	}
	projectArtifacts, err := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)).Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := projectArtifacts.Write(ctx, artifacts.ArtifactRef{Namespace: "sources", Name: "first"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("first")}, nil); err != nil {
		t.Fatal(err)
	}
	if _, _, err := projects.BeginDeletion(ctx, projectstore.BeginDeletionParams{
		ProjectID: project.ProjectID, OwnerID: project.OwnerID, ExpectedRevision: project.Revision,
	}); err != nil {
		t.Fatal(err)
	}
	type evalRows struct{ experiments, datasets, collections, receipts int }
	evalFootprint := func(projectID string) evalRows {
		t.Helper()
		var rows evalRows
		if err := pool.QueryRow(ctx, `
SELECT (SELECT count(*) FROM eval_experiments WHERE project_id = $1),
       (SELECT count(*) FROM eval_dataset_revisions WHERE project_id = $1),
       (SELECT count(*) FROM eval_collections WHERE project_id = $1),
       (SELECT count(*) FROM mutation_receipts WHERE key_scope[1] = 'eval' AND project_id = $1)`,
			projectID).Scan(&rows.experiments, &rows.datasets, &rows.collections, &rows.receipts); err != nil {
			t.Fatal(err)
		}
		return rows
	}
	laterPhases := func() int {
		t.Helper()
		var rows int
		if err := pool.QueryRow(ctx, `
SELECT (SELECT count(*) FROM finding_proposal_receipts WHERE project_id = $1)
     + (SELECT count(*) FROM artifact_binding_revisions WHERE scope_kind = 'project' AND scope_id = $1)`,
			project.ProjectID).Scan(&rows); err != nil {
			t.Fatal(err)
		}
		return rows
	}
	otherBefore := evalFootprint(other.ProjectID)
	var logs bytes.Buffer
	controller, err := New(pool, runstore.NewPostgresStore(pool, artifactpgtest.NewStorage()), rundeletion.New(pool, artifactpgtest.NewStorage()), &recordingNotifier{}, Options{Storage: artifactpgtest.NewStorage(),
		PollInterval: time.Millisecond, ClaimDuration: time.Minute, OperationTimeout: 5 * time.Second,
		Logger: slog.New(slog.NewJSONHandler(&logs, nil)),
	})
	if err != nil {
		t.Fatal(err)
	}
	evalSteps := 0
	for step := 0; ; step++ {
		if step > (3*datasets+evalReceipts)/evalPurgeBatchRows+20 {
			t.Fatalf("Project deletion did not finish after %d steps", step)
		}
		before, laterBefore := evalFootprint(project.ProjectID), laterPhases()
		worked, err := controller.RunOnce(ctx)
		if err != nil || !worked {
			t.Fatalf("Project deletion step %d = (%t, %v)", step, worked, err)
		}
		after, laterAfter := evalFootprint(project.ProjectID), laterPhases()
		// An experiment costs its row, its claim and its two collection rows.
		cost := 4*(before.experiments-after.experiments) + 3*(before.datasets-after.datasets) +
			(before.collections - after.collections) + (before.receipts - after.receipts)
		if cost > evalPurgeBatchRows {
			t.Fatalf("step %d spent %d Eval row mutations, more than %d", step, cost, evalPurgeBatchRows)
		}
		if cost > 0 {
			evalSteps++
		}
		if after != (evalRows{}) && laterAfter != laterBefore {
			t.Fatalf("step %d purged finding receipts or Artifacts while Eval rows %+v were left", step, after)
		}
		if after.receipts < before.receipts && (after.experiments > 0 || after.datasets > 0) {
			t.Fatalf("step %d deleted Eval receipts while %+v were left", step, after)
		}
		if _, err := projects.Get(ctx, project.OwnerID, project.ProjectID); errors.Is(err, projectstore.ErrNotFound) {
			if after != (evalRows{}) || laterAfter != 0 {
				t.Fatalf("Project deleted with Eval rows %+v and %d later-phase rows left", after, laterAfter)
			}
			break
		} else if err != nil {
			t.Fatal(err)
		}
	}
	if want := (3*datasets + evalReceipts + evalPurgeBatchRows - 1) / evalPurgeBatchRows; evalSteps < want {
		t.Fatalf("Eval purge took %d transactions, want at least %d", evalSteps, want)
	}
	if got := evalFootprint(other.ProjectID); got != otherBefore {
		t.Fatalf("another evaluation Project's Eval rows = %+v, want %+v", got, otherBefore)
	}
	// The purge meter logged the Eval purge's start and its end with every
	// Eval pass.
	var records []string
	for decoder := json.NewDecoder(&logs); decoder.More(); {
		var record struct {
			Msg       string `json:"msg"`
			ProjectID string `json:"project_id"`
			Passes    int64  `json:"passes"`
		}
		if err := decoder.Decode(&record); err != nil {
			t.Fatal(err)
		}
		if record.ProjectID == project.ProjectID && strings.HasPrefix(record.Msg, "Eval purge") {
			records = append(records, fmt.Sprintf("%s after %d passes", record.Msg, record.Passes))
		}
	}
	if want := []string{"Eval purge started after 1 passes", fmt.Sprintf("Eval purge finished after %d passes", evalSteps)}; !slices.Equal(records, want) {
		t.Fatalf("Eval purge records = %q, want %q", records, want)
	}
}
