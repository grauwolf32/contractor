package runstore

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/projectstore"
)

func TestPostgresRunProjectContextUsesCallerTransaction(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	base := isolatedRunStorePool(t, ctx)
	pool := singleConnectionPool(t, ctx, base)
	store := NewPostgresStore(pool, artifactpgtest.NewStorage())
	projectID := "project-context"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: projectID, OwnerID: "user-1", Kind: projectstore.KindProject, Name: projectID, IdempotencyKey: projectID, RequestDigest: "sha256:" + strings.Repeat("a", 64)}); err != nil {
		t.Fatal(err)
	}
	params := testRunParams("project-run")
	params.ProjectID = &projectID
	if _, err := createRun(ctx, store, params); err != nil {
		t.Fatal(err)
	}
	createTestRun(t, ctx, store, "projectless")
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback(ctx) //nolint:errcheck // committed below
	reader := NewPostgresStore(tx, artifactpgtest.NewStorage())
	got, err := reader.ProjectContext(ctx, "project-run")
	if err != nil || got.OwnerID != "user-1" || got.ProjectID != projectID {
		t.Fatalf("Project identity: %+v %v", got, err)
	}
	for _, id := range []string{"missing", "projectless"} {
		if _, err := reader.ProjectContext(ctx, id); !errors.Is(err, ErrNotFound) {
			t.Fatalf("%s: %v", id, err)
		}
	}
	if _, err := reader.ProjectContext(ctx, "bad/run"); !errors.Is(err, ErrInvalid) {
		t.Fatalf("invalid identity: %v", err)
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
}
