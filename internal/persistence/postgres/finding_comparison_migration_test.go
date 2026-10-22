package postgres

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func TestPostgresFindingComparisonProjectionIsBoundedImmutableAndDiscarded(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	installMigrationPrefix(t, ctx, pool, 24)
	if _, err := pool.Exec(ctx, `INSERT INTO finding_proposal_receipts(receipt_id,proposal_id,allocation_id,runtime_agent_id,runtime_instance_id,stage_execution_id,logical_agent_name,invocation_id,submission_id,client_key,request_digest,run_id,owner_id,workflow_name,workflow_version,workflow_schema_version,workflow_configuration_ref,workflow_closure_digest,proposal_ref,proposal_digest,proposal_media_type,proposal_size_bytes,evidence,report_projection)
 VALUES('cache-receipt','proposal','allocation','runtime','instance','stage','worker','invocation','submission','client','sha256:'||repeat('a',64),'run','owner','workflow','1','contractor/v1alpha1','{}','sha256:'||repeat('b',64),'{}','sha256:'||repeat('c',64),'application/json',2,'[]',$1)`, `{"title":"Retained title","severity_suggestion":"medium","subject":null,"standard_refs":[]}`); err != nil {
		t.Fatal(err)
	}
	var before, after string
	if err := pool.QueryRow(ctx, "SELECT to_jsonb(r)::text FROM finding_proposal_receipts r").Scan(&before); err != nil {
		t.Fatal(err)
	}
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	var cache []byte
	if err := pool.QueryRow(ctx, "SELECT (to_jsonb(r)-'comparison_projection')::text,comparison_projection FROM finding_proposal_receipts r").Scan(&after, &cache); err != nil {
		t.Fatal(err)
	}
	if before != after {
		t.Fatal("comparison migration changed existing receipt fields")
	}
	var projection map[string]any
	if err := json.Unmarshal(cache, &projection); err != nil {
		t.Fatal(err)
	}
	if projection["availability"] != "available" || projection["preview"].(map[string]any)["title"] != "Retained title" {
		t.Fatalf("bad cache=%s", cache)
	}
	if _, err := pool.Exec(ctx, "UPDATE finding_proposal_receipts SET comparison_projection='{}'"); Classify(err) != ClassRejected {
		t.Fatalf("forged cache mutation accepted: %v", err)
	}
	for _, count := range []int{32, 33} {
		refs := make([]map[string]string, count)
		for i := range refs {
			refs[i] = map[string]string{"scheme": "CWE", "version": "4.20", "requirement_id": "CWE-89"}
		}
		report, _ := json.Marshal(map[string]any{"title": strings.Repeat("界", 300), "subject": nil, "standard_refs": refs, "severity_suggestion": "medium"})
		if err := pool.QueryRow(ctx, `SELECT contractor_finding_comparison_projection($1,
 '{"availability":"unavailable"}','{"availability":"unavailable"}','source-held')`, string(report)).Scan(&cache); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(cache, &projection); err != nil {
			t.Fatal(err)
		}
		display := projection["preview"].(map[string]any)
		hints := projection["hints"].(map[string]any)
		if display["title"] != strings.Repeat("界", 256) || display["titleTruncated"] != true || hints["familiesAvailable"] != (count <= 32) || len(cache) > 32768 {
			t.Fatalf("cache bounds=%s", cache)
		}
	}
	if _, err := pool.Exec(ctx, "UPDATE finding_proposal_receipts SET retention_state='discarded',source_run_deleted_at=clock_timestamp(),discarded_at=clock_timestamp()"); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, "SELECT comparison_projection FROM finding_proposal_receipts").Scan(&cache); err != nil {
		t.Fatal(err)
	}
	projection = nil
	if err := json.Unmarshal(cache, &projection); err != nil {
		t.Fatal(err)
	}
	if len(projection) != 3 || projection["availability"] != "unavailable" || projection["reason"] != "discarded" {
		t.Fatalf("sensitive cache survived discard=%s", cache)
	}
}
