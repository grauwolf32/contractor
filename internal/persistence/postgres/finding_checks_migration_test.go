package postgres

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/jackc/pgx/v5"
)

func TestPostgresFindingChecksMigrationPreservesInlineAndFilesystemSources(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	installMigrationPrefix(t, ctx, pool, 42)
	seedHistoryHolders(t, ctx, pool, `{"artifacts":{}}`)
	content := []byte(`{"client_key":"current","description":"Preserve current exact proposal bytes.","evidence_ids":[],"limitations":[],"preconditions":[],"proposed_checks":[{"method":"static-trace","objective":"Verify 世界"}],"schema":"contractor.audit.finding-proposal.v1","severity_suggestion":"","standard_refs":[],"subject":null,"title":"Current finding"}`)
	root := t.TempDir()
	key := "ab/" + "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	if err := os.MkdirAll(filepath.Join(root, "ab"), 0o750); err != nil {
		t.Fatal(err)
	}
	filename := filepath.Join(root, key)
	if err := os.WriteFile(filename, content, 0o600); err != nil {
		t.Fatal(err)
	}
	// Separate content identities exercise both physical registries. The
	// filesystem source remains outside PostgreSQL throughout schema migration.
	for _, backend := range []string{"postgresql", "filesystem"} {
		data := content
		if backend == "postgresql" {
			data = bytes.Replace(content, []byte("Current finding"), []byte("Inline current finding"), 1)
		}
		err := InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			var payload any = data
			var object any
			if backend == "filesystem" {
				payload, object = nil, key
			}
			if _, err := tx.Exec(ctx, `INSERT INTO artifact_blobs(sha256,payload,size_bytes,backend,object_key) VALUES(sha256($1::bytea),$2,$3,$4,$5)`, data, payload, len(data), backend, object); err != nil {
				return err
			}
			if _, err := tx.Exec(ctx, `INSERT INTO artifact_versions(version_id,blob_sha256,media_type) VALUES($1,sha256($2::bytea),'application/json')`, "checks-"+backend, data); err != nil {
				return err
			}
			if _, err := tx.Exec(ctx, `INSERT INTO artifact_bindings(scope_kind,scope_id,namespace,name,current_revision) VALUES('run','run-h','finding-proposals',$1,'checks-r1')`, backend); err != nil {
				return err
			}
			if _, err := tx.Exec(ctx, `INSERT INTO artifact_binding_revisions(scope_kind,scope_id,namespace,name,revision,version_id) VALUES('run','run-h','finding-proposals',$1,'checks-r1',$2)`, backend, "checks-"+backend); err != nil {
				return err
			}
			_, err := tx.Exec(ctx, `INSERT INTO finding_proposal_receipts(receipt_id,proposal_id,allocation_id,runtime_agent_id,runtime_instance_id,stage_execution_id,logical_agent_name,invocation_id,submission_id,client_key,request_digest,run_id,owner_id,workflow_name,workflow_version,workflow_schema_version,workflow_configuration_ref,workflow_closure_digest,proposal_ref,proposal_digest,proposal_media_type,proposal_size_bytes,evidence,report_projection)
VALUES($1,$1,'allocation','runtime','instance','stage','worker',$1,$1,$1,'sha256:'||repeat('a',64),'run-h','owner','workflow','1','contractor/v1alpha1','{}','sha256:'||repeat('b',64),jsonb_build_object('namespace','finding-proposals','name',$2::text,'revision','checks-r1'),$3,'application/json',$4,'[]','{}')`, "checks-"+backend, backend, fmt.Sprintf("sha256:%x", sha256.Sum256(data)), len(data))
			return err
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	// The shared holder fixture predates history counters. Bring its valid
	// current finding documents to the immediately preceding schema first.
	available, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	var preceding []migration
	for _, item := range available {
		if item.version <= 47 {
			preceding = append(preceding, item)
		}
	}
	if _, err := applyMigrations(ctx, pool, MigrationBudgets{}, preceding); err != nil {
		t.Fatal(err)
	}
	var before, after string
	if err := pool.QueryRow(ctx, `SELECT jsonb_agg(to_jsonb(receipt) ORDER BY receipt_id)::text FROM finding_proposal_receipts receipt`).Scan(&before); err != nil {
		t.Fatal(err)
	}
	// Preserve this historical fixture before the incompatible required-conditions transition.
	applyMigrationsThrough(t, ctx, pool, "audit_admission_chain_transition")
	if err := pool.QueryRow(ctx, `SELECT jsonb_agg(to_jsonb(receipt)-ARRAY['checks_projection','checks_projection_bytes','checks_projection_count'] ORDER BY receipt_id)::text FROM finding_proposal_receipts receipt`).Scan(&after); err != nil || before != after {
		t.Fatal("migration rewrote retained receipt data", err)
	}
	var pending int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM finding_proposal_receipts WHERE receipt_id LIKE 'checks-%' AND checks_projection IS NULL AND checks_projection_bytes IS NULL AND checks_projection_count IS NULL`).Scan(&pending); err != nil || pending != 2 {
		t.Fatalf("existing exact sources were guessed instead of marked pending: %d %v", pending, err)
	}
	fileAfter, err := os.ReadFile(filename)
	if err != nil || !bytes.Equal(content, fileAfter) {
		t.Fatal("migration changed filesystem content", err)
	}
	var persisted []byte
	if err := pool.QueryRow(ctx, `SELECT payload FROM artifact_blobs WHERE backend='postgresql' AND sha256=(SELECT blob_sha256 FROM artifact_versions WHERE version_id='checks-postgresql')`).Scan(&persisted); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(persisted, bytes.Replace(content, []byte("Current finding"), []byte("Inline current finding"), 1)) {
		t.Fatal("migration changed inline content")
	}
	if _, err := pool.Exec(ctx, `UPDATE finding_proposal_receipts SET checks_projection='{}' WHERE receipt_id='checks-postgresql'`); SQLState(err) != SQLStateCheckViolation {
		t.Fatalf("missing index schema bypassed SQL shape check: %v", err)
	}
}
