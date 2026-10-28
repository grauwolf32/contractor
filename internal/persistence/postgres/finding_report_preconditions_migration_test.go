package postgres

import (
	"context"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresFindingReportPreconditionsTransitionRefusesOldRetainedBytesAtomically(t *testing.T) {
	for _, state := range []string{"source-held", "audit-held"} {
		for _, value := range []string{"missing", "null", "string", "object", "number"} {
			t.Run(state+"/"+value, func(t *testing.T) {
				ctx, pool, previous := findingReportPreconditionsTransitionFixture(t)
				fields := map[string]string{"missing": "", "null": `,"preconditions":null`, "string": `,"preconditions":"condition"`, "object": `,"preconditions":{}`, "number": `,"preconditions":1`}
				old := "{ \"title\": \"Original retained finding\"" + fields[value] + " }\n"
				if err := insertFindingReportTransitionReceipt(t, ctx, pool, "old", state, old); err != nil {
					t.Fatal(err)
				}
				if err := insertFindingReportTransitionReceipt(t, ctx, pool, "valid", "source-held", currentFindingReportProjection); err != nil {
					t.Fatal(err)
				}
				if err := insertFindingReportTransitionReceipt(t, ctx, pool, "discarded", "discarded", "{}"); err != nil {
					t.Fatal(err)
				}
				seedFindingReportTransitionArtifact(t, ctx, pool)
				before := findingReportTransitionBytes(t, ctx, pool)
				ledger, err := readAppliedMigrations(ctx, pool)
				if err != nil {
					t.Fatal(err)
				}
				_, err = ApplyMigrations(ctx, pool, MigrationBudgets{})
				var refusal *pgconn.PgError
				if !errors.As(err, &refusal) || refusal.Code != SQLStateObjectNotInPrerequisiteState || !strings.Contains(refusal.Message, "preserve this database") || !strings.Contains(refusal.Message, "separate fresh database") {
					t.Fatalf("old %s projection migration=%v, want actionable prerequisite refusal", state, err)
				}
				afterLedger, err := readAppliedMigrations(ctx, pool)
				if err != nil || !reflect.DeepEqual(ledger, afterLedger) {
					t.Fatal("refusal changed the ledger", err)
				}
				if before != findingReportTransitionBytes(t, ctx, pool) {
					t.Fatal("refusal changed literal receipt/projection or immutable artifact bytes")
				}
				var current int64
				if err = pool.QueryRow(ctx, `SELECT max(version) FROM contractor_schema_migrations`).Scan(&current); err != nil || current != previous {
					t.Fatalf("ledger=%d error=%v want=%d", current, err, previous)
				}
				var constraints int
				if err = pool.QueryRow(ctx, `SELECT count(*) FROM pg_constraint WHERE conrelid='finding_proposal_receipts'::regclass AND conname='finding_proposal_receipts_report_preconditions_check'`).Scan(&constraints); err != nil || constraints != 0 {
					t.Fatalf("refusal installed write constraint: %d %v", constraints, err)
				}
			})
		}
	}
}

func TestPostgresFindingReportPreconditionsTransitionPreservesCurrentAndDiscardedBytes(t *testing.T) {
	ctx, pool, previous := findingReportPreconditionsTransitionFixture(t)
	for _, state := range []string{"source-held", "audit-held", "discarded"} {
		projection := currentFindingReportProjection
		if state == "discarded" {
			projection = "{}"
		}
		if err := insertFindingReportTransitionReceipt(t, ctx, pool, state, state, projection); err != nil {
			t.Fatal(err)
		}
	}
	seedFindingReportTransitionArtifact(t, ctx, pool)
	before := findingReportTransitionBytes(t, ctx, pool)
	result, err := ApplyMigrations(ctx, pool, MigrationBudgets{})
	if err != nil || len(result.AppliedVersions) == 0 || result.AppliedVersions[0] != previous+1 {
		t.Fatalf("current projections upgrade=%+v error=%v", result, err)
	}
	if before != findingReportTransitionBytes(t, ctx, pool) {
		t.Fatal("upgrade rewrote literal projections, tombstones or historical report artifact bytes")
	}
	repeated, err := ApplyMigrations(ctx, pool, MigrationBudgets{})
	if err != nil || len(repeated.AppliedVersions) != 0 || repeated.CurrentVersion != result.CurrentVersion {
		t.Fatalf("repeat=%+v error=%v", repeated, err)
	}
	// New retained writes cannot reintroduce the omitted field. Discarded rows
	// remain erased and do not acquire conditions just to satisfy the new codec.
	err = insertFindingReportTransitionReceipt(t, ctx, pool, "invalid-new", "source-held", "{}")
	if Classify(err) != ClassInvalid || ConstraintName(err) != "finding_proposal_receipts_report_preconditions_check" {
		t.Fatalf("new omitted conditions=%v, want constraint rejection", err)
	}
	_, err = pool.Exec(ctx, `UPDATE finding_proposal_receipts SET report_projection='{}' WHERE receipt_id='audit-held'`)
	if Classify(err) != ClassRejected {
		t.Fatalf("retained projection mutation=%v", err)
	}
}

// Deliberate whitespace and Unicode prove that the migration preserves literal
// projection text rather than decoding/re-encoding it with manufactured values.
const currentFindingReportProjection = "{ \"title\": \"Retained 世界\", \"description\": \"Original details\", \"subject\": null, \"preconditions\": [\"Только для читателя\", \"Owns an order\"], \"standard_refs\": [], \"severity_suggestion\": \"low\", \"limitations\": [] }\n"

func findingReportPreconditionsTransitionFixture(t *testing.T) (context.Context, *pgxpool.Pool, int64) {
	t.Helper()
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	previous, _ := embeddedMigration(t, "finding_report_preconditions_transition")
	installMigrationPrefix(t, ctx, pool, previous)
	return ctx, pool, previous
}

func insertFindingReportTransitionReceipt(t *testing.T, ctx context.Context, pool *pgxpool.Pool, id, state, projection string) error {
	t.Helper()
	_, err := pool.Exec(ctx, `INSERT INTO finding_proposal_receipts (
receipt_id,proposal_id,allocation_id,runtime_agent_id,runtime_instance_id,stage_execution_id,logical_agent_name,invocation_id,submission_id,client_key,request_digest,run_id,owner_id,
workflow_name,workflow_version,workflow_schema_version,workflow_configuration_ref,workflow_closure_digest,proposal_ref,proposal_digest,proposal_media_type,proposal_size_bytes,evidence,report_projection,
retention_state,source_run_deleted_at,discarded_at,checks_projection)
VALUES($1,$1,$1,'runtime','instance','stage','worker','invocation',$1,$1,'sha256:'||repeat('a',64),'run','owner',
'workflow','1','contractor/v1alpha1','{}','sha256:'||repeat('b',64),'{"namespace":"finding-proposals","name":"original","revision":"1"}','sha256:'||repeat('c',64),'application/json',128,
'[{"ref":{"namespace":"worker","name":"trace","revision":"1"},"digest":"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","mediaType":"text/plain","sizeBytes":42}]',$3,
$2,CASE WHEN $2='source-held' THEN NULL ELSE clock_timestamp() END,CASE WHEN $2='discarded' THEN clock_timestamp() ELSE NULL END,
CASE WHEN $2='discarded' THEN contractor_empty_finding_checks('sha256:'||repeat('c',64),128) ELSE NULL END)`, id, state, projection)
	return err
}

func seedFindingReportTransitionArtifact(t *testing.T, ctx context.Context, pool *pgxpool.Pool) {
	t.Helper()
	const oldReport = "{ \"schema\": \"historical-report\", \"findings\": [{\"title\": \"No original preconditions member\"}] }\n"
	if _, err := pool.Exec(ctx, `INSERT INTO artifact_blobs(sha256,payload,size_bytes,backend) VALUES(sha256($1::bytea),$1,$2,'postgresql')`, []byte(oldReport), len(oldReport)); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO artifact_versions(version_id,blob_sha256,media_type) VALUES('historical-report',sha256($1::bytea),'application/json')`, []byte(oldReport)); err != nil {
		t.Fatal(err)
	}
}

func findingReportTransitionBytes(t *testing.T, ctx context.Context, pool *pgxpool.Pool) string {
	t.Helper()
	var data string
	if err := pool.QueryRow(ctx, `SELECT jsonb_build_array(
(SELECT jsonb_agg(to_jsonb(r) ORDER BY receipt_id) FROM finding_proposal_receipts r),
(SELECT jsonb_agg(to_jsonb(b) ORDER BY sha256) FROM artifact_blobs b),
(SELECT jsonb_agg(to_jsonb(v) ORDER BY version_id) FROM artifact_versions v))::text`).Scan(&data); err != nil {
		t.Fatal(err)
	}
	return data
}
