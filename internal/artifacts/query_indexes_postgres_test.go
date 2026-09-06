package artifacts_test

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

// Capture the actual repository history SQL, including all metadata joins.
type historyQueryCapture struct {
	sql  string
	args []any
}

func (*historyQueryCapture) Exec(context.Context, string, ...any) (pgconn.CommandTag, error) {
	panic("unexpected Exec")
}
func (*historyQueryCapture) QueryRow(context.Context, string, ...any) pgx.Row {
	panic("unexpected QueryRow")
}
func (c *historyQueryCapture) Query(_ context.Context, sql string, args ...any) (pgx.Rows, error) {
	c.sql, c.args = sql, args
	return nil, fmt.Errorf("capture only")
}

func TestPostgresArtifactSelectiveQueryPlans(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	for _, sql := range []string{
		`INSERT INTO artifact_scopes (scope_kind,scope_id) VALUES ('user','plan-owner')`,
		`INSERT INTO artifact_blobs (sha256,payload,size_bytes) SELECT sha256(i::text::bytea),i::text::bytea,octet_length(i::text::bytea) FROM generate_series(1,50000) i`,
		`INSERT INTO artifact_versions (version_id,blob_sha256,media_type) SELECT 'v-'||i,sha256(i::text::bytea),'text/plain' FROM generate_series(1,50000) i`,
		`INSERT INTO artifact_binding_revisions (scope_kind,scope_id,namespace,name,revision,version_id,created_at)
SELECT 'user','plan-owner','history','long','r-'||i,'v-'||i,'2026-01-01'::timestamptz + i*interval '1 second' FROM generate_series(1,50000) i`,
		`INSERT INTO artifact_bindings (scope_kind,scope_id,namespace,name,current_revision) VALUES ('user','plan-owner','history','long','r-50000')`,
		`ANALYZE artifact_blobs`, `ANALYZE artifact_versions`, `ANALYZE artifact_binding_revisions`, `ANALYZE artifact_bindings`,
	} {
		if _, err := pool.Exec(ctx, sql); err != nil {
			t.Fatal(err)
		}
	}
	capture := &historyQueryCapture{}
	scope, _ := UserScope("plan-owner")
	_, _ = NewPostgresRepository(capture).ListVersions(ctx, scope, ArtifactRef{Namespace: "history", Name: "long"}, VersionPageQuery{Limit: 51})
	probes := []struct {
		name, sql, index string
		args             []any
	}{
		{"revision-reference", `SELECT 1 FROM artifact_binding_revisions WHERE version_id=$1`, "artifact_binding_revisions_version_idx", []any{"v-25000"}},
		{"blob-reference", `SELECT 1 FROM artifact_versions WHERE blob_sha256=sha256($1::bytea)`, "artifact_versions_blob_idx", []any{[]byte("25000")}},
		{"binary-digest", `SELECT 1 FROM artifact_blobs WHERE sha256=sha256($1::bytea)`, "artifact_blobs_pkey", []any{[]byte("25000")}},
		{"history", capture.sql, "artifact_binding_revisions_history_idx", capture.args},
	}
	explain := func(db interface {
		Query(context.Context, string, ...any) (pgx.Rows, error)
	}, sql string, args []any) string {
		t.Helper()
		rows, err := db.Query(ctx, "EXPLAIN (ANALYZE, BUFFERS) "+sql, args...)
		if err != nil {
			t.Fatal(err)
		}
		defer rows.Close()
		var lines []string
		for rows.Next() {
			var line string
			if err := rows.Scan(&line); err != nil {
				t.Fatal(err)
			}
			lines = append(lines, line)
		}
		if err := rows.Err(); err != nil {
			t.Fatal(err)
		}
		return strings.Join(lines, "\n")
	}
	// Roll back DDL after measuring the pre-migration access paths, without
	// touching checksums or disabling sequential scans in the planner.
	before, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer before.Rollback(context.Background())
	for _, name := range []string{"artifact_binding_revisions_version_idx", "artifact_versions_blob_idx", "artifact_binding_revisions_history_idx"} {
		if _, err := before.Exec(ctx, "DROP INDEX "+name); err != nil {
			t.Fatal(err)
		}
	}
	for _, probe := range probes {
		t.Logf("BEFORE %s\n%s", probe.name, explain(before, probe.sql, probe.args))
	}
	t.Logf("BEFORE encoded-digest\n%s", explain(before, `SELECT 1 FROM artifact_blobs WHERE encode(sha256,'hex')=encode(sha256($1::bytea),'hex')`, []any{[]byte("25000")}))
	if err := before.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	for _, probe := range probes {
		plan := explain(pool, probe.sql, probe.args)
		t.Logf("AFTER %s\n%s", probe.name, plan)
		if !strings.Contains(plan, probe.index) {
			t.Errorf("%s missing selective index %s", probe.name, probe.index)
		}
		if strings.Contains(plan, "Seq Scan on artifact_binding_revisions") || strings.Contains(plan, "Seq Scan on artifact_versions") || strings.Contains(plan, "Seq Scan on artifact_blobs") || strings.Contains(plan, "Sort Method") {
			t.Errorf("%s scans/sorts retained history", probe.name)
		}
	}
	var duplicate bool
	if err := pool.QueryRow(ctx, `SELECT to_regclass('artifact_bindings_list_idx') IS NOT NULL`).Scan(&duplicate); err != nil || duplicate {
		t.Fatalf("duplicate binding index remains: %v", err)
	}
	page, err := NewPostgresRepository(pool).ListVersions(ctx, scope, ArtifactRef{Namespace: "history", Name: "long"}, VersionPageQuery{Limit: 51})
	if err != nil || len(page) != 51 || *page[0].Ref.Revision != "r-50000" {
		t.Fatalf("history page size=%d: %v", len(page), err)
	}
	last := page[len(page)-1]
	next, err := NewPostgresRepository(pool).ListVersions(ctx, scope, ArtifactRef{Namespace: "history", Name: "long"}, VersionPageQuery{Limit: 51, BeforeCreatedAt: &last.CreatedAt, BeforeRevision: *last.Ref.Revision})
	if err != nil || len(next) != 51 || *next[0].Ref.Revision != "r-49949" {
		t.Fatalf("next history page size=%d: %v", len(next), err)
	}
}
