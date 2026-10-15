package postgres

import (
	"encoding/json"
	"os"
	"testing"
)

func TestPostgresProjectActiveChecksMigrationPreservesBaselines(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	installMigrationPrefix(t, ctx, pool, 18)
	if _, err := pool.Exec(ctx, `INSERT INTO projects(project_id,owner_id,kind,name,request_idempotency_key,request_digest)
 VALUES('policy-upgrade','owner','project','Keep business data','create','sha256:'||repeat('a',64))`); err != nil {
		t.Fatal(err)
	}
	for _, policy := range []string{"prohibited", "approval-required"} {
		if _, err := pool.Exec(ctx, `INSERT INTO audits(audit_id,owner_id,project_id,profile_name,profile_version,profile_digest,
 profile_snapshot,input_selection,baseline_snapshot,state,phase,started_at,max_rounds,batch_size,max_items_per_round,max_items_total,
 max_submitted_runs,max_item_run_attempts,max_evidence_bytes)
 VALUES($1,'owner','policy-upgrade','profile','1','sha256:'||repeat('a',64),
 jsonb_build_object('interaction',jsonb_build_object('activeChecks',$1::text)), '{}',
 '{"schema":"contractor.audit.baseline.v1","inputs":{},"scope":{"target":"https://target.example"}}',
 'active','rounds',clock_timestamp(),1,1,10,10,10,1,1024)`, policy); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	var mode, name string
	if err := pool.QueryRow(ctx, "SELECT active_checks,name FROM projects WHERE project_id='policy-upgrade'").Scan(&mode, &name); err != nil || mode != "automatic" || name != "Keep business data" {
		t.Fatalf("Project upgrade=%q %q, %v", mode, name, err)
	}
	for _, policy := range []string{"prohibited", "approval-required"} {
		var raw []byte
		if err := pool.QueryRow(ctx, "SELECT baseline_snapshot FROM audits WHERE audit_id=$1", policy).Scan(&raw); err != nil {
			t.Fatal(err)
		}
		var baseline map[string]json.RawMessage
		if err := json.Unmarshal(raw, &baseline); err != nil {
			t.Fatal(err)
		}
		if string(baseline["activeChecks"]) != "\""+policy+"\"" || len(baseline) != 4 || string(baseline["scope"]) != `{"target": "https://target.example"}` {
			t.Fatalf("baseline changed: %s", raw)
		}
	}
	var enabled string
	if err := pool.QueryRow(ctx, "SELECT tgenabled FROM pg_trigger WHERE tgrelid='audits'::regclass AND tgname='audits_protect_preparation_baseline'").Scan(&enabled); err != nil || enabled != "O" {
		t.Fatalf("baseline protection=%q, %v", enabled, err)
	}
	if _, err := pool.Exec(ctx, "UPDATE projects SET active_checks='enabled', revision=revision+1, updated_at=updated_at+interval '1 second' WHERE project_id='policy-upgrade'"); Classify(err) != ClassInvalid {
		t.Fatalf("invalid policy reached storage: %v", err)
	}
}
