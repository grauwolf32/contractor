package postgres

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func TestPostgresExplicitAuditHTTPMigrationRefusesSeedPinsWithoutRewritingThem(t *testing.T) {
	dsn := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	for name, baseline := range map[string]string{
		"seed-pins": `{"engagement":{"document":{"target":{"origins":["https://target.example"],"seedUrls":["https://target.example/start"]}}}}`,
	} {
		t.Run(name, func(t *testing.T) {
			ctx := t.Context()
			pool, _ := isolatedPools(t, ctx, dsn)
			previous, _ := embeddedMigration(t, "explicit_audit_http_credentials")
			installMigrationPrefix(t, ctx, pool, previous)
			digest := "sha256:" + strings.Repeat("a", 64)
			if _, err := pool.Exec(ctx, `INSERT INTO projects(project_id,owner_id,kind,name,request_idempotency_key,request_digest) VALUES('manual-migration','owner','project','Migration','create',$1)`, digest); err != nil {
				t.Fatal(err)
			}
			if _, err := pool.Exec(ctx, `INSERT INTO audits(audit_id,owner_id,project_id,profile_name,profile_version,profile_digest,profile_snapshot,input_selection,baseline_snapshot,state,started_at,deadline_at,max_rounds,batch_size,max_items_per_round,max_items_total,max_submitted_runs,max_item_run_attempts,max_evidence_bytes) VALUES('manual-audit','owner','manual-migration','profile','1',$1,'{}','{}',$2,'paused',clock_timestamp()-interval '1 hour',clock_timestamp()+interval '1 hour',1,1,1,1,1,1,1)`, digest, baseline); err != nil {
				t.Fatal(err)
			}
			var before json.RawMessage
			if err := pool.QueryRow(ctx, `SELECT baseline_snapshot FROM audits WHERE audit_id='manual-audit'`).Scan(&before); err != nil {
				t.Fatal(err)
			}
			if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err == nil || !strings.Contains(err.Error(), "explicit operator resolution") {
				t.Fatalf("unsupported pin was not refused: %v", err)
			}
			var after json.RawMessage
			if err := pool.QueryRow(ctx, `SELECT baseline_snapshot FROM audits WHERE audit_id='manual-audit'`).Scan(&after); err != nil || !bytes.Equal(before, after) {
				t.Fatalf("refused migration changed an immutable pin: %v", err)
			}
		})
	}
}
