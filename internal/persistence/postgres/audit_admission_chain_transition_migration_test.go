package postgres

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"os"
	"reflect"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresAdmissionChainTransitionRefusesEveryLiveTypedStage(t *testing.T) {
	for _, state := range []string{"active", "waiting_review", "paused", "finalizing", "cancelling", "deleting"} {
		for _, stage := range []string{"checkpoint", "partial-generation", "sealed"} {
			t.Run(state+"/"+stage, func(t *testing.T) {
				ctx, pool, previous := admissionChainTransitionFixture(t)
				seedAdmissionChainTransitionAudit(t, ctx, pool, state, state, stage)
				before, holds := admissionChainTransitionData(t, ctx, pool), storedHolds(t, ctx, pool)
				ledger, err := readAppliedMigrations(ctx, pool)
				if err != nil {
					t.Fatal(err)
				}
				_, err = ApplyMigrations(ctx, pool, MigrationBudgets{})
				var refusal *pgconn.PgError
				if !errors.As(err, &refusal) || refusal.Code != SQLStateObjectNotInPrerequisiteState {
					t.Fatalf("live typed stage migration=%v, want operator prerequisite refusal", err)
				}
				if before != admissionChainTransitionData(t, ctx, pool) || !maps.Equal(holds, storedHolds(t, ctx, pool)) {
					t.Fatal("refused migration changed data, literal checkpoint/projection bytes, frozen evidence or holds")
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
}

func TestPostgresAdmissionChainTransitionPreservesTerminalTypedAndOrdinaryAudits(t *testing.T) {
	ctx, pool, previous := admissionChainTransitionFixture(t)
	for _, state := range []string{"completed", "cancelled", "failed"} {
		for _, stage := range []string{"checkpoint", "partial-generation", "sealed"} {
			seedAdmissionChainTransitionAudit(t, ctx, pool, "history-"+state+"-"+stage, state, stage)
		}
	}
	for _, state := range []string{"active", "waiting_review", "paused", "finalizing", "cancelling", "deleting"} {
		seedAdmissionChainTransitionAudit(t, ctx, pool, "ordinary-"+state, state, "ordinary")
	}
	for _, state := range []string{"draft", "active", "waiting_review", "paused", "finalizing", "cancelling", "deleting"} {
		seedAdmissionChainTransitionAudit(t, ctx, pool, "no-round-"+state, state, "none")
	}
	before, holds := admissionChainTransitionData(t, ctx, pool), storedHolds(t, ctx, pool)
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
	if err != nil || len(result.AppliedVersions) != 1 || result.AppliedVersions[0] != previous+1 {
		t.Fatalf("allowed upgrade=%+v error=%v", result, err)
	}
	if before != admissionChainTransitionData(t, ctx, pool) || !maps.Equal(holds, storedHolds(t, ctx, pool)) {
		t.Fatal("migration rewrote historical typed evidence, exact holds/counters or ordinary Audit data")
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
	if err != nil || len(repeated.AppliedVersions) != 0 || repeated.CurrentVersion != result.CurrentVersion {
		t.Fatalf("repeat=%+v error=%v", repeated, err)
	}
	if _, err = pool.Exec(ctx, `UPDATE artifact_blobs SET payload='changed'::bytea WHERE size_bytes>0`); SQLState(err) != SQLStateCheckViolation {
		t.Fatalf("historical payload mutation=%v", err)
	}
}

func admissionChainTransitionFixture(t *testing.T) (context.Context, *pgxpool.Pool, int64) {
	t.Helper()
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	previous, _ := embeddedMigration(t, "audit_admission_chain_transition")
	installMigrationPrefix(t, ctx, pool, previous)
	return ctx, pool, previous
}

func seedAdmissionChainTransitionAudit(t *testing.T, ctx context.Context, pool *pgxpool.Pool, id, state, stage string) {
	t.Helper()
	seedExecutionTransitionAudit(t, ctx, pool, id, state, stage != "none")
	if stage == "none" || stage == "ordinary" {
		return
	}
	auditID, roundID, projectID := "transition-audit-"+id, "round-"+id, "transition-project-"+id
	var namespace string
	if err := pool.QueryRow(ctx, `SELECT contractor_audit_dependency_namespace($1)`, auditID).Scan(&namespace); err != nil {
		t.Fatal(err)
	}
	err := InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		refs := map[string]json.RawMessage{}
		for _, kind := range []string{"selection-page", "selection-root", "prepared-page", "prepared-root"} {
			schema := map[string]string{"selection-page": "round-selection-page", "selection-root": "round-selection", "prepared-page": "prepared-round-page", "prepared-root": "prepared-round"}[kind]
			payload := []byte(fmt.Sprintf("{ \"schema\": \"contractor.audit.%s.v1\", \"audit_id\": %q, \"items\": [] }\n", schema, auditID))
			ref, e := json.Marshal(map[string]string{"namespace": namespace, "name": kind, "revision": "r1"})
			if e != nil {
				return e
			}
			refs[kind] = ref
			versionID := id + "-" + kind
			if _, e = tx.Exec(ctx, `INSERT INTO artifact_blobs(sha256,payload,size_bytes,backend) VALUES(sha256($1::bytea),$1,$2,'postgresql')`, payload, len(payload)); e != nil {
				return e
			}
			if _, e = tx.Exec(ctx, `INSERT INTO artifact_versions(version_id,blob_sha256,media_type) VALUES($1,sha256($2::bytea),'application/json')`, versionID, payload); e != nil {
				return e
			}
			if _, e = tx.Exec(ctx, `INSERT INTO artifact_binding_revisions(scope_kind,scope_id,namespace,name,revision,version_id) VALUES('project',$1,$2,$3,'r1',$4)`, projectID, namespace, kind, versionID); e != nil {
				return e
			}
			if _, e = tx.Exec(ctx, `INSERT INTO artifact_bindings(scope_kind,scope_id,namespace,name,current_revision,frozen,retained_revisions,retained_bytes) VALUES('project',$1,$2,$3,'r1',true,1,$4)`, projectID, namespace, kind, len(payload)); e != nil {
				return e
			}
			if _, e = tx.Exec(ctx, `INSERT INTO audit_artifact_links(audit_id,logical_key,artifact_ref,artifact_digest,media_type,size_bytes,source_provenance) VALUES($1,$2,$3::jsonb,'sha256:'||encode(sha256($4::bytea),'hex'),'application/json',$5,'{"kind":"historical-chain"}')`, auditID, kind, ref, payload, len(payload)); e != nil {
				return e
			}
		}
		// Frozen Audit links preserve their own bytes without incrementing the
		// generic hold counter. Add a real native draft-input holder as well, so
		// the migration's hold preservation assertion cannot compare empty maps.
		inputs := map[string]any{}
		for kind, ref := range refs {
			inputs[kind] = map[string]any{"ref": ref}
		}
		inputs["worklist"] = map[string]any{"ref": map[string]string{"namespace": namespace, "name": "worklist", "revision": "r1"}}
		selection, e := json.Marshal(map[string]any{"inputs": inputs})
		if e != nil {
			return e
		}
		if _, e = tx.Exec(ctx, `INSERT INTO audits(audit_id,owner_id,project_id,profile_name,profile_version,profile_digest,profile_snapshot,input_selection,max_rounds,batch_size,max_items_per_round,max_items_total,max_submitted_runs,max_item_run_attempts,max_evidence_bytes) VALUES($1,'owner',$2,'holder','1','sha256:'||repeat('d',64),'{}',$3::jsonb,1,1,1,1,1,1,1048576)`, auditID+"-holder", projectID, selection); e != nil {
			return e
		}
		if _, e := tx.Exec(ctx, `UPDATE audits SET retained_evidence_bytes=(SELECT sum(size_bytes) FROM audit_artifact_links WHERE audit_id=$1) WHERE audit_id=$1`, auditID); e != nil {
			return e
		}
		// Deliberate formatting proves the migration preserves literal JSON, not
		// just a semantically equivalent re-encoding of an old checkpoint.
		if stage == "checkpoint" {
			_, e := tx.Exec(ctx, `UPDATE audit_rounds SET admission_checkpoint=$2::text::json WHERE round_id=$1`, roundID, "{ \"schema\": \"contractor.audit.admission.v1\", \"selection_root\": null, \"prepared_root\": null }\n")
			return e
		}
		// Each guard marker is tested independently. The partial-generation case
		// has no checkpoint/seal; the sealed case has no generation/checkpoint.
		expected := 1
		if stage == "partial-generation" {
			expected = 2
		}
		if _, e := tx.Exec(ctx, `UPDATE audit_rounds SET state='proposed',expected_item_count=$2,prepared_generation=$3 WHERE round_id=$1`, roundID, expected, "old-generation"); e != nil {
			return e
		}
		if _, e := tx.Exec(ctx, `INSERT INTO audit_items(item_id,audit_id,round_id,item_key,ordinal,kind,subject_key,task_ref,task_digest,origin,workflow_role,state,coverage_status,coverage_requested,coverage_completed,coverage_gaps,prepared_execution,prepared_page_ordinal,prepared_related_count) VALUES($1,$2,$3,'old-item',0,'finding-verification','old-subject',$4::jsonb,'sha256:'||repeat('a',64),'{"schema":"contractor.audit.item-origin.v1"}','check','ready','not-tested','[]','[]','[]',$5::text::json,0,1)`, "item-"+id, auditID, roundID, refs["prepared-page"], fmt.Sprintf("{ \"accepted_plan\": { \"schema\": \"contractor.audit.accepted-plan.v1\", \"audit_id\": %q, \"item_key\": \"old-item\" } }\n", auditID)); e != nil {
			return e
		}
		if _, e := tx.Exec(ctx, `UPDATE audit_rounds SET prepared_staged_count=1 WHERE round_id=$1`, roundID); e != nil {
			return e
		}
		if stage == "sealed" {
			_, e := tx.Exec(ctx, `UPDATE audit_rounds SET state='accepted',prepared_generation=NULL,prepared_page_count=1,prepared_pages_digest='sha256:'||repeat('c',64) WHERE round_id=$1`, roundID)
			return e
		}
		return nil
	})
	if err != nil {
		t.Fatalf("seed typed stage %s: %v", stage, err)
	}
	var held int
	if err = pool.QueryRow(ctx, `SELECT count(*) FROM artifact_binding_revisions WHERE scope_kind='project' AND scope_id=$1 AND hold_count>0`, projectID).Scan(&held); err != nil || held != 5 {
		t.Fatalf("historical worklist and both page/root pairs hold %d revisions, want 5: %v", held, err)
	}
}

func admissionChainTransitionData(t *testing.T, ctx context.Context, pool *pgxpool.Pool) string {
	t.Helper()
	var data string
	if err := pool.QueryRow(ctx, admissionChainTransitionDataSQL).Scan(&data); err != nil {
		t.Fatal(err)
	}
	return data
}
