package runstore

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

// Every Run creation path stores through one statement: a keyed public Run
// and an Audit Run differ only in the columns their path sets, and a
// duplicate identity is ErrConflict on every path, whichever unique key the
// statement met, and stores nothing.
func TestPostgresRunCreationPathsShareOneInsert(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool)
	projectID := "project-run-creation"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: projectID, OwnerID: "user-1", Kind: projectstore.KindProject,
		Name: projectID, IdempotencyKey: projectID, RequestDigest: "sha256:" + strings.Repeat("c", 64),
	}); err != nil {
		t.Fatal(err)
	}
	type storedKey struct{ key, digest *string }
	readKey := func(runID string) storedKey {
		t.Helper()
		var stored storedKey
		if err := pool.QueryRow(ctx, `SELECT request_idempotency_key, request_digest
FROM workflow_runs WHERE run_id = $1`, runID).Scan(&stored.key, &stored.digest); err != nil {
			t.Fatal(err)
		}
		return stored
	}

	params := testRunParams("run-keyed")
	params.ProjectID = &projectID
	digest := "sha256:" + strings.Repeat("d", 64)
	keyed, inserted, err := store.CreateRunIdempotent(ctx, CreateRunIdempotentParams{
		CreateRunParams: params, IdempotencyKey: "creation-key", RequestDigest: digest,
	})
	if err != nil || !inserted || keyed.State != RunInitializing || keyed.PublicationMode != PublicationOrdinary ||
		keyed.AuditExecutionID != nil || keyed.AuditSubmissionKey != nil ||
		keyed.ProjectID == nil || *keyed.ProjectID != projectID {
		t.Fatalf("keyed Run = (%+v, %t, %v)", keyed, inserted, err)
	}
	if stored := readKey(keyed.RunID); stored.key == nil || *stored.key != "creation-key" ||
		stored.digest == nil || *stored.digest != digest {
		t.Fatalf("keyed Run key = %+v", stored)
	}

	duplicateKeyed := CreateRunIdempotentParams{
		CreateRunParams: params, IdempotencyKey: "other-creation-key", RequestDigest: digest,
	}
	if _, _, err := store.CreateRunIdempotent(ctx, duplicateKeyed); !errors.Is(err, ErrConflict) {
		t.Fatalf("keyed create of a taken Run ID = %v, want ErrConflict", err)
	}
	if _, found, err := store.LookupRunIdempotency(ctx, "user-1", "other-creation-key", digest); err != nil || found {
		t.Fatalf("conflicting keyed create stored its key: found=%t, %v", found, err)
	}
	// The Audit path meets the same Run ID before its AuditExecution is
	// checked, so it conflicts without an Audit fixture.
	if _, err := store.CreateAuditRun(ctx, CreateAuditRunParams{
		CreateRunParams: params, AuditExecutionID: "execution-run-creation", AuditSubmissionKey: "submission-run-creation",
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("Audit create of a taken Run ID = %v, want ErrConflict", err)
	}
	stored, err := store.GetRun(ctx, params.RunID)
	if err != nil || stored.PublicationMode != PublicationOrdinary || stored.AuditExecutionID != nil ||
		stored.AuditSubmissionKey != nil {
		t.Fatalf("conflicting creates changed the stored Run: (%+v, %v)", stored, err)
	}
	if storedKey := readKey(params.RunID); storedKey.key == nil || *storedKey.key != "creation-key" {
		t.Fatalf("conflicting creates changed the stored key: %+v", storedKey)
	}
}

// Run creation answers a Project that its owner does not have, unknown or
// another owner's, as ErrNotFound from the Project foreign key alone, and
// stores nothing. A deleting Project stays ErrProjectDeleting, a replay
// returns the stored Run, and the reference violation of the trusted Audit
// association keeps its cause instead of becoming a missing Project.
func TestPostgresRunCreationClassifiesTheMissingProjectParent(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool)
	projects := projectstore.NewPostgresStore(pool)
	for index, project := range []struct{ id, owner string }{
		{"project-present", "user-1"}, {"project-foreign", "user-2"}, {"project-deleting", "user-1"},
	} {
		if _, _, err := projects.Create(ctx, projectstore.CreateParams{
			ProjectID: project.id, OwnerID: project.owner, Kind: projectstore.KindProject, Name: project.id,
			IdempotencyKey: project.id, RequestDigest: fmt.Sprintf("sha256:%064x", index+1),
		}); err != nil {
			t.Fatal(err)
		}
	}
	if _, _, err := projects.BeginDeletion(ctx, projectstore.BeginDeletionParams{
		ProjectID: "project-deleting", OwnerID: "user-1", ExpectedRevision: 1,
	}); err != nil {
		t.Fatal(err)
	}
	keyed := func(runID, projectID string) CreateRunIdempotentParams {
		params := testRunParams(runID)
		params.ProjectID = &projectID
		return CreateRunIdempotentParams{
			CreateRunParams: params, IdempotencyKey: "key-" + runID,
			RequestDigest: contentdigest.Bytes([]byte(runID)),
		}
	}
	// Nothing of a rejected Run remains: no row, no lifecycle event, no
	// RunScope and no claimed create-Run key.
	requireNothingStored := func(t *testing.T, params CreateRunIdempotentParams) {
		t.Helper()
		var rows int
		if err := pool.QueryRow(ctx, `
SELECT (SELECT count(*) FROM workflow_runs WHERE run_id = $1)
     + (SELECT count(*) FROM workflow_run_events WHERE run_id = $1)
     + (SELECT count(*) FROM artifact_scopes WHERE scope_kind = 'run' AND scope_id = $1)`,
			params.RunID,
		).Scan(&rows); err != nil {
			t.Fatal(err)
		}
		if rows != 0 {
			t.Fatalf("rejected Run %q stored %d rows", params.RunID, rows)
		}
		if _, found, err := store.LookupRunIdempotency(
			ctx, params.OwnerID, params.IdempotencyKey, params.RequestDigest,
		); err != nil || found {
			t.Fatalf("rejected Run %q claimed its key: found=%t, %v", params.RunID, found, err)
		}
	}

	for _, test := range []struct{ name, projectID string }{
		{"unknown Project", "project-missing"},
		{"another owner's Project", "project-foreign"},
	} {
		t.Run(test.name, func(t *testing.T) {
			params := keyed("run-"+test.projectID, test.projectID)
			_, inserted, err := store.CreateRunIdempotent(ctx, params)
			if !errors.Is(err, ErrNotFound) || errors.Is(err, ErrProjectDeleting) || inserted {
				t.Fatalf("create Run in %s = (%t, %v), want ErrNotFound", test.name, inserted, err)
			}
			requireNothingStored(t, params)
		})
	}

	t.Run("deleting Project", func(t *testing.T) {
		params := keyed("run-project-deleting", "project-deleting")
		if _, _, err := store.CreateRunIdempotent(ctx, params); !errors.Is(err, ErrProjectDeleting) ||
			errors.Is(err, ErrNotFound) {
			t.Fatalf("create Run in a deleting Project = %v, want ErrProjectDeleting", err)
		}
		requireNothingStored(t, params)
	})

	t.Run("present Project and replay", func(t *testing.T) {
		params := keyed("run-project-present", "project-present")
		created, inserted, err := store.CreateRunIdempotent(ctx, params)
		if err != nil || !inserted || created.ProjectID == nil || *created.ProjectID != "project-present" {
			t.Fatalf("create Run in a present Project = (%+v, %t, %v)", created, inserted, err)
		}
		replayed, inserted, err := store.CreateRunIdempotent(ctx, params)
		if err != nil || inserted || replayed.RunID != created.RunID {
			t.Fatalf("replay = (%+v, %t, %v), want the stored Run", replayed, inserted, err)
		}
	})

	// The Project exists, so only the Audit association can be missing: a
	// broken trusted invariant, not a Project the caller named.
	t.Run("missing Audit association", func(t *testing.T) {
		params := keyed("run-audit-association", "project-present")
		_, err := store.CreateAuditRun(ctx, CreateAuditRunParams{
			CreateRunParams:  params.CreateRunParams,
			AuditExecutionID: "execution-missing", AuditSubmissionKey: "submission-missing",
		})
		if errors.Is(err, ErrNotFound) ||
			persistencepostgres.Classify(err) != persistencepostgres.ClassReferenceViolation ||
			persistencepostgres.ConstraintName(err) != "workflow_runs_audit_submission_fkey" {
			t.Fatalf("create Audit Run without its AuditExecution = %v, want its reference violation", err)
		}
		requireNothingStored(t, params)
	})
}
