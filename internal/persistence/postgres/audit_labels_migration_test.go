package postgres

import (
	"encoding/json"
	"os"
	"testing"
)

func TestPostgresAuditLabelsMigrationPreservesSnapshots(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	installMigrationPrefix(t, ctx, pool, 32)
	if _, err := pool.Exec(ctx, `INSERT INTO projects(project_id,owner_id,kind,name,request_idempotency_key,request_digest)
 VALUES('labels-upgrade','owner','project','Keep data','create','sha256:'||repeat('a',64))`); err != nil {
		t.Fatal(err)
	}
	cases := []struct{ id, state, phase, members, labels, placement string }{
		{"draft", "draft", "not-started", "", `{}`, `[]`},
		{"completed", "completed", "rounds", "", `{}`, `[]`},
		{"only-placement", "draft", "not-started", `,"requiredPlacementLabels":[]`, `{}`, `[]`},
		{"only-labels", "draft", "not-started", `,"labels":{}`, `{}`, `[]`},
		{"preparing", "active", "preparing", "", `{}`, `[]`},
		// Existing values and missing derived projections exercise separate paths.
		{"selected", "active", "rounds", `,"labels":{"team":"appsec","cloud":"aws"},"requiredPlacementLabels":["cloud/aws"]`, `{"cloud":"aws","team":"appsec"}`, `["cloud/aws"]`},
		{"derived", "active", "rounds", `,"labels":{"cloud":""}`, `{"cloud":""}`, `["cloud"]`},
		{"null", "active", "rounds", `,"labels":null,"requiredPlacementLabels":null`, `null`, `null`},
	}
	before := map[string]string{}
	original := func(id string) string {
		t.Helper()
		var value string
		if err := pool.QueryRow(ctx, `SELECT jsonb_build_array(
 to_jsonb(a)-ARRAY['input_selection','baseline_snapshot','engagement_active_us','engagement_active_since','engagement_clock_valid','engagement_action_count','engagement_request_count','engagement_held_count','engagement_cooldown_generation','engagement_recovery_after'],
 input_selection-ARRAY['labels','requiredPlacementLabels'],
 baseline_snapshot-ARRAY['labels','requiredPlacementLabels'])::text
 FROM audits a WHERE audit_id=$1`, id).Scan(&value); err != nil {
			t.Fatal(err)
		}
		return value
	}
	for _, c := range cases {
		selection := `{"schema":"contractor.audit.draft-selection.v1","inputs":{},"scope":{"objective":"keep"}` + c.members + `}`
		var baseline *string
		if c.state != "draft" {
			value := `{"schema":"contractor.audit.baseline.v1","inputs":{},"scope":{"objective":"keep"}` + c.members + `}`
			baseline = &value
		}
		if _, err := pool.Exec(ctx, `INSERT INTO audits(audit_id,owner_id,project_id,profile_name,profile_version,profile_digest,
 profile_snapshot,input_selection,baseline_snapshot,state,phase,started_at,finished_at,max_rounds,batch_size,max_items_per_round,max_items_total,
 max_submitted_runs,max_item_run_attempts,max_evidence_bytes)
 VALUES($1,'owner','labels-upgrade','profile','1','sha256:'||repeat('a',64),
 '{"workflows":{"prepare":{"kind":"prepare"}}}',$2::jsonb,$3::jsonb,$4,$5,
 CASE WHEN $4<>'draft' THEN clock_timestamp() END,CASE WHEN $4='completed' THEN clock_timestamp() END,
 1,1,10,10,10,1,1024)`, c.id, selection, baseline, c.state, c.phase); err != nil {
			t.Fatal(err)
		}
		before[c.id] = original(c.id)
	}
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		if original(c.id) != before[c.id] {
			t.Fatalf("%s: existing business data changed", c.id)
		}
		var selection, baseline []byte
		if err := pool.QueryRow(ctx, `SELECT input_selection,baseline_snapshot FROM audits WHERE audit_id=$1`, c.id).Scan(&selection, &baseline); err != nil {
			t.Fatal(err)
		}
		if c.state == "draft" && baseline != nil {
			t.Fatalf("%s: draft gained a baseline", c.id)
		}
		for _, raw := range [][]byte{selection, baseline} {
			if raw == nil {
				if c.state != "draft" {
					t.Fatal("lost baseline")
				}
				continue
			}
			var value map[string]json.RawMessage
			if err := json.Unmarshal(raw, &value); err != nil {
				t.Fatal(err)
			}
			compact := func(raw json.RawMessage) string {
				var v any
				if err := json.Unmarshal(raw, &v); err != nil {
					t.Fatalf("%s: missing/invalid collection: %v", c.id, err)
				}
				b, _ := json.Marshal(v)
				return string(b)
			}
			if compact(value["labels"]) != c.labels || compact(value["requiredPlacementLabels"]) != c.placement {
				t.Fatalf("%s: labels=%s placement=%s", c.id, value["labels"], value["requiredPlacementLabels"])
			}
		}
	}
	var enabled string
	if err := pool.QueryRow(ctx, `SELECT tgenabled FROM pg_trigger WHERE tgrelid='audits'::regclass AND tgname='audits_protect_preparation_baseline'`).Scan(&enabled); err != nil || enabled != "O" {
		t.Fatalf("baseline guard=%q: %v", enabled, err)
	}
	if _, err := pool.Exec(ctx, `UPDATE audits SET baseline_snapshot=baseline_snapshot||'{"tampered":true}' WHERE audit_id='preparing'`); Classify(err) != ClassRejected {
		t.Fatalf("preparation snapshot mutation=%v", err)
	}
	result, err := ApplyMigrations(ctx, pool, MigrationBudgets{})
	if err != nil || len(result.AppliedVersions) != 0 {
		t.Fatalf("repeat migration=%+v: %v", result, err)
	}
}
