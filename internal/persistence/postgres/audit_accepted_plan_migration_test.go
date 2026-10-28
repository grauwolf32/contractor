package postgres

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/strictjson"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestSharedPostgresAcceptedPlanMigrationPreservesPreparedItemsAndHolds(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	previous, _ := embeddedMigration(t, "audit_accepted_plan_projection")
	// This fixture starts before the exact-hold counters, then reaches the
	// immediately preceding schema with the same complete native holders.
	installMigrationPrefix(t, ctx, pool, 42)
	seedHistoryHolders(t, ctx, pool, `{"artifacts":{}}`)
	applyMigrationsThrough(t, ctx, pool, "context_projection_lineage")
	var version int64
	if err := pool.QueryRow(ctx, `SELECT max(version) FROM contractor_schema_migrations`).Scan(&version); err != nil || version != previous {
		t.Fatalf("preceding schema=%d error=%v, want %d", version, err, previous)
	}
	seedAcceptedPlanMigrationItems(t, ctx, pool)
	snapshot := func(withDuplicate bool) []byte {
		t.Helper()
		columns := "to_jsonb(item)"
		if withDuplicate {
			columns += "-'accepted_plan'"
		}
		var data []byte
		err := pool.QueryRow(ctx, `SELECT jsonb_agg(jsonb_build_array(`+columns+`,item.prepared_execution::text,item.retained_dependencies::text) ORDER BY item_id)::text FROM audit_items AS item`).Scan(&data)
		if err != nil {
			t.Fatal(err)
		}
		return data
	}
	before := snapshot(true)
	beforeHolds := storedHolds(t, ctx, pool)
	ledger, err := readAppliedMigrations(ctx, pool)
	if err != nil {
		t.Fatal(err)
	}
	available, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	for index, item := range available {
		if item.version == previous+1 {
			available = available[:index+1]
			break
		}
	}
	result, err := applyMigrations(ctx, pool, MigrationBudgets{}, available)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.AppliedVersions) != 1 || result.AppliedVersions[0] != previous+1 {
		t.Fatalf("applied=%+v", result)
	}
	if !bytes.Equal(before, snapshot(false)) || !maps.Equal(beforeHolds, storedHolds(t, ctx, pool)) {
		t.Fatal("migration rewrote item content, exact projection text, dependencies or holds")
	}
	afterLedger, err := readAppliedMigrations(ctx, pool)
	if err != nil {
		t.Fatal(err)
	}
	for version, entry := range ledger {
		if !reflect.DeepEqual(entry, afterLedger[version]) {
			t.Fatalf("migration changed existing ledger version %d", version)
		}
	}
	var duplicate bool
	var held int
	if err = pool.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema=current_schema() AND table_name='audit_items' AND column_name='accepted_plan'),(SELECT retained_dependency_hold_count FROM audit_items WHERE item_id='typed-plan-item')`).Scan(&duplicate, &held); err != nil || duplicate || held != 1 {
		t.Fatalf("duplicate=%v retained holds=%d error=%v", duplicate, held, err)
	}
	if state, err := compareMigrationLedger(available, afterLedger); err != nil || len(state.pending) != 0 {
		t.Fatalf("migration ledger not complete: %+v %v", state, err)
	}
	repeated, err := applyMigrations(ctx, pool, MigrationBudgets{}, available)
	if err != nil || len(repeated.AppliedVersions) != 0 || repeated.CurrentVersion != result.CurrentVersion {
		t.Fatalf("repeat=%+v error=%v", repeated, err)
	}
}

func seedAcceptedPlanMigrationItems(t *testing.T, ctx context.Context, pool *pgxpool.Pool) {
	t.Helper()
	sourceDigest := migrationPlanDigest([]byte("x"))
	var namespace string
	if err := pool.QueryRow(ctx, `SELECT contractor_audit_dependency_namespace('audit-h')`).Scan(&namespace); err != nil {
		t.Fatal(err)
	}
	source := map[string]any{"ref": map[string]any{"namespace": "inputs", "name": "i", "revision": "i1"}, "digest": sourceDigest, "mediaType": "text/plain", "sizeBytes": 1}
	retained := map[string]any{"ref": map[string]any{"namespace": namespace, "name": "dependency", "revision": "dep-r1"}, "digest": sourceDigest, "mediaType": "text/plain", "sizeBytes": 1}
	baseline, _ := json.Marshal(map[string]any{"inputs": map[string]any{"k": source}, "skills": []any{}})
	if _, err := pool.Exec(ctx, `UPDATE audits SET baseline_snapshot=$1::jsonb,max_rounds=2 WHERE audit_id='audit-h'`, baseline); err != nil {
		t.Fatal(err)
	}
	var persisted []byte
	if err := pool.QueryRow(ctx, `SELECT baseline_snapshot FROM audits WHERE audit_id='audit-h'`).Scan(&persisted); err != nil {
		t.Fatal(err)
	}
	// Reuse the strict cross-language plan fixture without importing the Audit
	// domain back into its underlying PostgreSQL package.
	fixtureBytes := contractstest.ReadFile(t, "api", "testdata", "audit-check-plans", "accepted-plan-cases.json")
	var fixture struct {
		Base struct {
			Plan map[string]any `json:"plan"`
		} `json:"base"`
	}
	if err := json.Unmarshal(fixtureBytes, &fixture); err != nil {
		t.Fatal(err)
	}
	plan := fixture.Base.Plan
	pin := "sha256:" + repeat('a')
	itemKey := plan["item_key"].(string)
	plan["owner_id"], plan["project_id"], plan["audit_id"] = "owner", "proj", "audit-h"
	plan["authorization"].(map[string]any)["active_checks"] = "prohibited"
	plan["authorization"].(map[string]any)["approval_requirement"] = "none"
	plan["baseline_digest"], plan["profile_digest"] = migrationPlanDigest(persisted), pin
	plan["source"].(map[string]any)["retained_proposal"].(map[string]any)["ref"].(map[string]any)["namespace"] = namespace
	route := plan["route"].(map[string]any)
	route["workflow_role"], route["workflow_digest"], route["skills_digest"] = "check", pin, pin
	dependencies := map[string]any{"schema": "contractor.audit.retained-dependencies.v1", "owner_id": "owner", "project_id": "proj", "audit_id": "audit-h", "item_key": itemKey, "baseline_digest": plan["baseline_digest"], "profile_digest": pin, "workflow_digest": pin, "skills_digest": pin, "dependencies": []any{map[string]any{"name": "baseline", "kind": "baseline", "projection": "exact", "source": source, "retained": retained, "origin": map[string]any{"kind": "baseline", "baseline_input": "k"}, "requires": []string{}}}}
	dependencyBytes, err := strictjson.Canonical(dependencies)
	if err != nil {
		t.Fatal(err)
	}
	plan["dependencies_digest"] = migrationPlanDigest(dependencyBytes)
	planBytes, err := strictjson.Canonical(plan)
	if err != nil {
		t.Fatal(err)
	}
	taskRef := map[string]any{"namespace": namespace, "name": "task", "revision": "dep-r1"}
	execution, err := strictjson.Canonical(map[string]any{"accepted_plan": plan, "dependencies": dependencies, "item_key": itemKey, "ordinal": 0, "subject_key": "subject", "task_package_id": "task", "task_package_digest": pin, "task_ref": taskRef, "inputs": []any{}})
	if err != nil {
		t.Fatal(err)
	}
	err = InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, e := tx.Exec(ctx, `INSERT INTO artifact_binding_revisions(scope_kind,scope_id,namespace,name,revision,version_id) VALUES('project','proj',$1,'dependency','dep-r1','v-i1')`, namespace); e != nil {
			return e
		}
		if _, e := tx.Exec(ctx, `INSERT INTO artifact_bindings(scope_kind,scope_id,namespace,name,current_revision,frozen,retained_revisions,retained_bytes) VALUES('project','proj',$1,'dependency','dep-r1',true,1,1)`, namespace); e != nil {
			return e
		}
		if _, e := tx.Exec(ctx, `INSERT INTO artifact_lineage(target_scope_kind,target_scope_id,target_namespace,target_name,target_revision,source_scope_kind,source_scope_id,source_namespace,source_name,source_revision,lineage_kind) VALUES('project','proj',$1,'dependency','dep-r1','project','proj','inputs','i','i1','audit_import')`, namespace); e != nil {
			return e
		}
		if _, e := tx.Exec(ctx, `INSERT INTO audit_rounds(round_id,audit_id,ordinal,manifest_ref,manifest_digest,state,expected_item_count,prepared_generation) VALUES('ordinary-plan-round','audit-h',1,'{}',$1,'accepted',1,NULL),('typed-plan-round','audit-h',2,'{}',$1,'proposed',1,'plan-migration')`, pin); e != nil {
			return e
		}
		_, e := tx.Exec(ctx, `INSERT INTO audit_items(item_id,audit_id,round_id,item_key,ordinal,kind,subject_key,task_ref,task_digest,origin,workflow_role,state,coverage_status,coverage_requested,coverage_completed,coverage_gaps,coverage_rationale,accepted_plan,prepared_execution,prepared_page_ordinal,prepared_related_count,retained_dependencies)
VALUES('ordinary-plan-item','audit-h','ordinary-plan-round','ordinary',0,'checklist','subject',$1::jsonb,$2,jsonb_set($3::jsonb,'{entryKey}','"ordinary"'),'check','ready','not-tested','[]','[]','[]','',NULL,NULL,NULL,NULL,NULL),
('typed-plan-item','audit-h','typed-plan-round',$4,0,'finding-verification','subject',$1::jsonb,$2,jsonb_set($3::jsonb,'{entryKey}',to_jsonb($4::text)),'check','ready','not-tested','[]','[]','[]','',$5::text::json,$6::text::json,0,7,$7::text::json)`, mustMigrationJSON(t, taskRef), pin, `{"schema":"contractor.audit.item-origin.v1","entryKey":"migration","sourceRef":{"namespace":"inputs","name":"i","revision":"i1"},"sourceContentDigest":"`+sourceDigest+`","canonicalInventoryDigest":"`+pin+`","sourceMediaType":"application/json"}`, itemKey, string(planBytes), string(execution), string(dependencyBytes))
		return e
	})
	if err != nil {
		t.Fatal(err)
	}
}

func mustMigrationJSON(t *testing.T, value any) []byte {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func migrationPlanDigest(data []byte) string { return fmt.Sprintf("sha256:%x", sha256.Sum256(data)) }
