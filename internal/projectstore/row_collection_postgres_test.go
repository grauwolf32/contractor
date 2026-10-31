package projectstore

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/pagelimit"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/persistence/postgrestest"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresProjectCollectionPreservesPagesAndReleasesRows(t *testing.T) {
	ctx := t.Context()
	pool := postgrestest.NewPool(t, ctx, "", postgrestest.Options{Configure: func(config *pgxpool.Config) {
		config.MaxConns = 1
	}})
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool, persistencepostgres.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	store := NewPostgresStore(pool)
	params := ListParams{OwnerID: "owner", Limit: pagelimit.MaxFetchRows}
	if empty, err := store.List(ctx, params); err != nil || empty == nil || len(empty) != 0 {
		t.Fatalf("empty Project page = %v, %v", empty, err)
	}
	for index := range pagelimit.MaxFetchRows + 1 {
		id := fmt.Sprintf("project-%03d", index)
		if _, _, err := store.Create(ctx, CreateParams{
			ProjectID: id, OwnerID: "owner", Kind: KindProject, Name: id,
			IdempotencyKey: id, RequestDigest: "sha256:" + strings.Repeat("a", 64),
		}); err != nil {
			t.Fatal(err)
		}
	}
	// Fixed database timestamps exercise the tie-breaker without clock assertions.
	if _, err := pool.Exec(ctx, "ALTER TABLE projects DISABLE TRIGGER USER"); err != nil {
		t.Fatal(err)
	}
	at := time.Date(2026, time.October, 31, 0, 0, 0, 0, time.UTC)
	if _, err := pool.Exec(ctx, "UPDATE projects SET created_at=$1, updated_at=$1", at); err != nil {
		t.Fatal(err)
	}
	page, err := store.List(ctx, params)
	if err != nil || len(page) != pagelimit.MaxFetchRows {
		t.Fatalf("look-ahead Project page has %d rows: %v", len(page), err)
	}
	for index, project := range page {
		if want := fmt.Sprintf("project-%03d", pagelimit.MaxFetchRows-index); project.ProjectID != want {
			t.Fatalf("Project %d = %s, want %s", index, project.ProjectID, want)
		}
	}
	tail, err := store.List(ctx, ListParams{OwnerID: "owner", Limit: pagelimit.MaxFetchRows, BeforeCreatedAt: &at, BeforeProjectID: page[len(page)-1].ProjectID})
	if err != nil || len(tail) != 1 || tail[0].ProjectID != "project-000" {
		t.Fatalf("Project page continuation = %v, %v", tail, err)
	}
	// Put a malformed row after a full public page of valid rows. The decoder must
	// reject it and release the only connection instead of returning partial data.
	if _, err := pool.Exec(ctx, "ALTER TABLE projects DROP CONSTRAINT projects_active_checks_policy"); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, "UPDATE projects SET active_checks='invalid' WHERE project_id='project-001'"); err != nil {
		t.Fatal(err)
	}
	if partial, err := store.List(ctx, params); partial != nil || err == nil || !strings.Contains(err.Error(), "active checks") {
		t.Fatalf("malformed Project page = %v, %v", partial, err)
	}
	if got := pool.Stat().AcquiredConns(); got != 0 {
		t.Fatalf("decoder failure left %d connections acquired", got)
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if partial, err := store.List(cancelled, params); partial != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled Project page = %v, %v", partial, err)
	}
	if got := pool.Stat().AcquiredConns(); got != 0 {
		t.Fatalf("cancellation left %d connections acquired", got)
	}
	if _, err := store.Get(ctx, "owner", page[0].ProjectID); err != nil {
		t.Fatalf("reuse the single connection after failure: %v", err)
	}
}
