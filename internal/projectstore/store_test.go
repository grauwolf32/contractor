package projectstore

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestProjectValidationBounds(t *testing.T) {
	t.Parallel()
	valid := CreateParams{
		ProjectID: "project-1", OwnerID: "owner-1", Kind: KindProject,
		Name: "Project", Description: "Description", IdempotencyKey: "request-1",
		RequestDigest: "sha256:" + strings.Repeat("a", 64),
	}
	if err := validateCreate(valid); err != nil {
		t.Fatalf("valid Project: %v", err)
	}
	for name, mutate := range map[string]func(*CreateParams){
		"kind":        func(value *CreateParams) { value.Kind = "other" },
		"blank name":  func(value *CreateParams) { value.Name = "  " },
		"long name":   func(value *CreateParams) { value.Name = strings.Repeat("x", MaxNameBytes+1) },
		"description": func(value *CreateParams) { value.Description = strings.Repeat("x", MaxDescriptionBytes+1) },
		"idempotency": func(value *CreateParams) { value.IdempotencyKey = " bad" },
		"digest":      func(value *CreateParams) { value.RequestDigest = "sha256:no" },
	} {
		t.Run(name, func(t *testing.T) {
			candidate := valid
			mutate(&candidate)
			if err := validateCreate(candidate); !errors.Is(err, ErrInvalid) {
				t.Fatalf("validation error = %v", err)
			}
		})
	}
}

func TestPostgresProjectIdempotencyOwnershipPaginationAndCAS(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedProjectPool(t, ctx, databaseURL)
	store := NewPostgresStore(pool)
	digest := "sha256:" + strings.Repeat("a", 64)
	created, inserted, err := store.Create(ctx, CreateParams{
		ProjectID: "project-one", OwnerID: "owner-1", Kind: KindProject,
		Name: "One", Description: "first", IdempotencyKey: "create-one", RequestDigest: digest,
	})
	if err != nil || !inserted || created.Revision != 1 {
		t.Fatalf("create Project = (%+v, %t, %v)", created, inserted, err)
	}
	replayed, inserted, err := store.Create(ctx, CreateParams{
		ProjectID: "project-unused", OwnerID: "owner-1", Kind: KindProject,
		Name: "One", Description: "first", IdempotencyKey: "create-one", RequestDigest: digest,
	})
	if err != nil || inserted || replayed.ProjectID != created.ProjectID {
		t.Fatalf("replay Project = (%+v, %t, %v)", replayed, inserted, err)
	}
	_, _, err = store.Create(ctx, CreateParams{
		ProjectID: "project-conflict", OwnerID: "owner-1", Kind: KindEvaluation,
		Name: "Changed", IdempotencyKey: "create-one",
		RequestDigest: "sha256:" + strings.Repeat("b", 64),
	})
	if !errors.Is(err, ErrConflict) {
		t.Fatalf("changed replay error = %v", err)
	}
	if _, err := store.Get(ctx, "owner-2", created.ProjectID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("foreign read error = %v", err)
	}
	updated, err := store.Update(ctx, UpdateParams{
		ProjectID: created.ProjectID, OwnerID: "owner-1", ExpectedRevision: 1,
		Name: "Renamed", Description: "current",
	})
	if err != nil || updated.Revision != 2 || updated.Name != "Renamed" || !updated.UpdatedAt.After(created.UpdatedAt) {
		t.Fatalf("update Project = (%+v, %v)", updated, err)
	}
	if _, err := store.Update(ctx, UpdateParams{
		ProjectID: created.ProjectID, OwnerID: "owner-1", ExpectedRevision: 1,
		Name: "Stale", Description: "stale",
	}); !errors.Is(err, ErrPrecondition) {
		t.Fatalf("stale update error = %v", err)
	}
	if _, _, err := store.Create(ctx, CreateParams{
		ProjectID: "evaluation-one", OwnerID: "owner-1", Kind: KindEvaluation,
		Name: "Evaluation", IdempotencyKey: "create-evaluation",
		RequestDigest: "sha256:" + strings.Repeat("c", 64),
	}); err != nil {
		t.Fatal(err)
	}
	kind := KindProject
	projects, err := store.List(ctx, ListParams{OwnerID: "owner-1", Kind: &kind, Limit: 10})
	if err != nil || len(projects) != 1 || projects[0].ProjectID != created.ProjectID {
		t.Fatalf("Project page = (%+v, %v)", projects, err)
	}
}

func isolatedProjectPool(t *testing.T, ctx context.Context, databaseURL string) *pgxpool.Pool {
	t.Helper()
	adminConfig, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	admin, err := pgxpool.NewWithConfig(ctx, adminConfig)
	if err != nil {
		t.Fatal(err)
	}
	if err := admin.Ping(ctx); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	random := make([]byte, 8)
	if _, err := rand.Read(random); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	schema := "contractor_project_test_" + hex.EncodeToString(random)
	identifier := pgx.Identifier{schema}.Sanitize()
	if _, err := admin.Exec(ctx, `CREATE SCHEMA `+identifier); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		admin.Close()
		t.Fatal(err)
	}
	config.ConnConfig.RuntimeParams["search_path"] = schema
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		admin.Close()
		t.Fatal(err)
	}
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		pool.Close()
		admin.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		pool.Close()
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cleanupCancel()
		_, _ = admin.Exec(cleanupCtx, `DROP SCHEMA `+identifier+` CASCADE`)
		admin.Close()
	})
	return pool
}
