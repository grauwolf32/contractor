package postgres

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresExecutionManifestTransitionRefusesEveryLiveWorklist(t *testing.T) {
	for _, state := range []string{"active", "waiting_review", "paused", "finalizing", "cancelling", "deleting"} {
		t.Run(state, func(t *testing.T) {
			ctx, pool, previous := executionTransitionFixture(t)
			seedExecutionTransitionAudit(t, ctx, pool, state, state, true)
			before := executionTransitionData(t, ctx, pool)
			ledger, err := readAppliedMigrations(ctx, pool)
			if err != nil {
				t.Fatal(err)
			}
			_, err = ApplyMigrations(ctx, pool, MigrationBudgets{})
			var refusal *pgconn.PgError
			if !errors.As(err, &refusal) || refusal.Code != SQLStateObjectNotInPrerequisiteState {
				t.Fatalf("live worklist migration=%v, want operator prerequisite refusal", err)
			}
			if before != executionTransitionData(t, ctx, pool) {
				t.Fatal("refused migration changed Audit, Round, frozen worklist or counters")
			}
			after, err := readAppliedMigrations(ctx, pool)
			if err != nil || !reflect.DeepEqual(ledger, after) {
				t.Fatal("refused migration changed its ledger", err)
			}
			var current int64
			if err = pool.QueryRow(ctx, `SELECT max(version) FROM contractor_schema_migrations`).Scan(&current); err != nil || current != previous {
				t.Fatalf("ledger version=%d error=%v, want %d", current, err, previous)
			}
		})
	}
}

func TestPostgresExecutionManifestTransitionPreservesHistoryAndNoRoundAudits(t *testing.T) {
	ctx, pool, previous := executionTransitionFixture(t)
	for _, state := range []string{"completed", "cancelled", "failed"} {
		seedExecutionTransitionAudit(t, ctx, pool, "history-"+state, state, true)
	}
	for _, state := range []string{"draft", "active", "waiting_review", "paused", "finalizing", "cancelling", "deleting"} {
		seedExecutionTransitionAudit(t, ctx, pool, "no-round-"+state, state, false)
	}
	before := executionTransitionData(t, ctx, pool)
	ledger, err := readAppliedMigrations(ctx, pool)
	if err != nil {
		t.Fatal(err)
	}
	// Keep this migration's preservation proof independent of later transitions.
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
	if err != nil || len(result.AppliedVersions) != 1 || result.AppliedVersions[0] != previous+1 {
		t.Fatalf("allowed upgrade=%+v error=%v", result, err)
	}
	if before != executionTransitionData(t, ctx, pool) {
		t.Fatal("migration rewrote terminal evidence or no-Round Audit data")
	}
	after, err := readAppliedMigrations(ctx, pool)
	if err != nil {
		t.Fatal(err)
	}
	for version, entry := range ledger {
		if !reflect.DeepEqual(entry, after[version]) {
			t.Fatalf("migration changed existing ledger version %d", version)
		}
	}
	repeated, err := applyMigrations(ctx, pool, MigrationBudgets{}, available)
	if err != nil || len(repeated.AppliedVersions) != 0 {
		t.Fatalf("repeat=%+v error=%v", repeated, err)
	}
	// Existing history is still frozen after the new ledger entry commits.
	if _, err = pool.Exec(ctx, `UPDATE artifact_blobs SET payload='changed'::bytea WHERE size_bytes>0`); SQLState(err) != SQLStateCheckViolation {
		t.Fatalf("historical payload mutation=%v", err)
	}
}

func executionTransitionFixture(t *testing.T) (context.Context, *pgxpool.Pool, int64) {
	t.Helper()
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	previous, _ := embeddedMigration(t, "audit_execution_manifest_transition")
	installMigrationPrefix(t, ctx, pool, previous)
	return ctx, pool, previous
}

func seedExecutionTransitionAudit(t *testing.T, ctx context.Context, pool *pgxpool.Pool, id, state string, withRound bool) {
	t.Helper()
	// The preflight treats frozen old worklists as opaque bytes. A real ZIP
	// keeps the preservation proof independent of the new transport decoder.
	var content bytes.Buffer
	writer := zip.NewWriter(&content)
	entry, err := writer.Create("execution.json")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = entry.Write([]byte(`{"schema":"contractor.audit.execution-manifest.v1","items":[{"item_key":"old-item","ordinal":0,"subject_key":"old-subject","task_package_id":"old-task","accepted_plan":{"schema":"contractor.audit.accepted-plan.v1"},"dependencies":{"schema":"contractor.audit.retained-dependencies.v1"}}]}`)); err != nil {
		t.Fatal(err)
	}
	if err = writer.Close(); err != nil {
		t.Fatal(err)
	}
	payload := content.Bytes()
	project := "transition-project-" + id
	audit := "transition-audit-" + id
	var namespace string
	if err = pool.QueryRow(ctx, `SELECT contractor_audit_dependency_namespace($1)`, audit).Scan(&namespace); err != nil {
		t.Fatal(err)
	}
	ref, err := json.Marshal(map[string]string{"namespace": namespace, "name": "worklist", "revision": "r1"})
	if err != nil {
		t.Fatal(err)
	}
	err = InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, e := tx.Exec(ctx, `INSERT INTO projects(project_id,owner_id,kind,name,request_idempotency_key,request_digest) VALUES($1,'owner','project','Execution transition',$1,'sha256:'||repeat('a',64))`, project); e != nil {
			return e
		}
		if _, e := tx.Exec(ctx, `INSERT INTO audits(audit_id,owner_id,project_id,profile_name,profile_version,profile_digest,profile_snapshot,input_selection,baseline_snapshot,state,phase,started_at,finished_at,paused_at,deletion_requested_at,max_rounds,batch_size,max_items_per_round,max_items_total,max_submitted_runs,max_item_run_attempts,max_evidence_bytes,retained_evidence_bytes)
VALUES($1,'owner',$2,'profile','1','sha256:'||repeat('b',64),'{"workflows":{"prepare":{"kind":"prepare"}}}','{"inputs":{}}','{"schema":"contractor.audit.baseline.v1","inputs":{},"scope":{}}',$3,CASE WHEN $3='draft' THEN 'not-started' WHEN $4 THEN 'rounds' ELSE 'preparing' END,CASE WHEN $3<>'draft' THEN clock_timestamp() END,CASE WHEN $3 IN('completed','cancelled','failed') THEN clock_timestamp() END,CASE WHEN $3='paused' THEN clock_timestamp() END,CASE WHEN $3='deleting' THEN clock_timestamp() END,2,1,1,2,2,1,1048576,$5)`, audit, project, state, withRound, len(payload)); e != nil {
			return e
		}
		if _, e := tx.Exec(ctx, `INSERT INTO artifact_scopes(scope_kind,scope_id) VALUES('project',$1)`, project); e != nil {
			return e
		}
		if _, e := tx.Exec(ctx, `INSERT INTO artifact_blobs(sha256,payload,size_bytes,backend) VALUES(sha256($1::bytea),$1,$2,'postgresql') ON CONFLICT(sha256) DO NOTHING`, payload, len(payload)); e != nil {
			return e
		}
		if _, e := tx.Exec(ctx, `INSERT INTO artifact_versions(version_id,blob_sha256,media_type) VALUES($1,sha256($2::bytea),'application/zip')`, "version-"+id, payload); e != nil {
			return e
		}
		if _, e := tx.Exec(ctx, `INSERT INTO artifact_binding_revisions(scope_kind,scope_id,namespace,name,revision,version_id) VALUES('project',$1,$3,'worklist','r1',$2)`, project, "version-"+id, namespace); e != nil {
			return e
		}
		if _, e := tx.Exec(ctx, `INSERT INTO artifact_bindings(scope_kind,scope_id,namespace,name,current_revision,frozen,retained_revisions,retained_bytes) VALUES('project',$1,$3,'worklist','r1',true,1,$2)`, project, len(payload), namespace); e != nil {
			return e
		}
		if _, e := tx.Exec(ctx, `INSERT INTO audit_artifact_links(audit_id,logical_key,artifact_ref,artifact_digest,media_type,size_bytes,source_provenance) VALUES($1,'historical-worklist',$4::jsonb,'sha256:'||encode(sha256($2::bytea),'hex'),'application/zip',$3,'{"kind":"migration-fixture"}')`, audit, payload, len(payload), ref); e != nil {
			return e
		}
		if withRound {
			if _, e := tx.Exec(ctx, `INSERT INTO audit_rounds(round_id,audit_id,ordinal,manifest_ref,manifest_digest,state,expected_item_count) VALUES($1,$2,1,$4::jsonb,'sha256:'||encode(sha256($3::bytea),'hex'),'closed',1)`, "round-"+id, audit, payload, ref); e != nil {
				return e
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("seed %s with Round=%t: %v", state, withRound, err)
	}
}

func executionTransitionData(t *testing.T, ctx context.Context, pool *pgxpool.Pool) string {
	t.Helper()
	var data string
	if err := pool.QueryRow(ctx, executionTransitionDataSQL).Scan(&data); err != nil {
		t.Fatal(err)
	}
	return data
}
