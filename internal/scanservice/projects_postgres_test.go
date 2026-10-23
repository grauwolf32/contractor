package scanservice

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresSweepProjectCaptureBoundsContainingPortfolio(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	admin, err := pgxpool.New(ctx, url)
	if err != nil {
		t.Fatal(err)
	}
	defer admin.Close()
	if err := admin.Ping(ctx); err != nil {
		t.Fatal(err)
	}
	random := make([]byte, 8)
	if _, err := rand.Read(random); err != nil {
		t.Fatal(err)
	}
	schema := "sweep_capture_" + hex.EncodeToString(random)
	identifier := pgx.Identifier{schema}.Sanitize()
	if _, err := admin.Exec(ctx, `CREATE SCHEMA `+identifier); err != nil {
		t.Fatal(err)
	}
	defer func() {
		cleanup, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
		if _, err := admin.Exec(cleanup, `DROP SCHEMA `+identifier+` CASCADE`); err != nil {
			t.Error(err)
		}
	}()
	cfg, err := pgxpool.ParseConfig(url)
	if err != nil {
		t.Fatal(err)
	}
	cfg.ConnConfig.RuntimeParams["search_path"] = schema
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool, persistencepostgres.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	digest := contentdigest.Bytes([]byte("capture"))
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: "template", OwnerID: "large-owner", Kind: projectstore.KindProject, Name: "template", IdempotencyKey: "template", RequestDigest: digest}); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO projects(project_id,owner_id,kind,name,request_idempotency_key,request_digest,active_checks)
SELECT 'project-'||lpad(i::text,5,'0'),'large-owner','project','Project '||i,'key-'||i,$1,
 (SELECT active_checks FROM projects WHERE project_id='template') FROM generate_series(1,10000) i`, digest); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `ANALYZE projects`); err != nil {
		t.Fatal(err)
	}
	ids := make([]string, 1000)
	for i := range ids {
		ids[i] = fmt.Sprintf("project-%05d", 9001+i)
	}
	for _, tc := range []struct {
		name     string
		request  Request
		count    int
		maxReads int
	}{{"all", Request{AllProjects: true}, 1001, 1100}, {"selected", Request{ProjectIDs: ids}, 1000, 2100}} {
		t.Run(tc.name, func(t *testing.T) {
			var reads int64
			if err := persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
				captured, err := captureProjects(ctx, tx, "large-owner", tc.request)
				if err != nil {
					return err
				}
				if len(captured) != tc.count {
					return fmt.Errorf("captured %d, want %d", len(captured), tc.count)
				}
				return tx.QueryRow(ctx, `SELECT coalesce(sum(seq_tup_read+idx_tup_fetch),0) FROM pg_stat_xact_user_tables WHERE relname='projects'`).Scan(&reads)
			}); err != nil {
				t.Fatal(err)
			}
			if reads > int64(tc.maxReads) {
				t.Fatalf("capture read %d rows of a 10001-Project portfolio for %d results", reads, tc.count)
			}
		})
	}
}
