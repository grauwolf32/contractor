package runstore

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

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
