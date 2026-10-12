package auditstore

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/jackc/pgx/v5"
)

const (
	purgeScaleExecutions           = 100_000
	purgeScaleBackgroundExecutions = 50_000
	purgeScaleAssessments          = 2_000
	purgeScaleFindings             = 2_000
	// purgeScaleIndexMinBlocks is the smallest index whose whole reads the
	// test tells apart from probes; a smaller index is cheap to read whole.
	purgeScaleIndexMinBlocks = 64
)

// purgeScaleReadsSQL returns, for each table of $1 and each of its indexes,
// the scans and fetched blocks that the current transaction has counted and
// the size in blocks. The scans of a table are its sequential scans.
const purgeScaleReadsSQL = `
SELECT relation::regclass::text, kind, pg_stat_get_xact_numscans(relation),
       pg_stat_get_xact_blocks_fetched(relation), pg_relation_size(relation) / current_setting('block_size')::bigint
  FROM (SELECT table_oid::oid, 'table' FROM unnest($1::text[]::regclass[]) AS table_oid
        UNION ALL
        SELECT indexrelid, 'index' FROM pg_index WHERE indrelid = ANY ($1::text[]::regclass[])) AS relations(relation, kind)`

type purgeScaleReads struct {
	kind                 string
	scans, fetched, size int64
}

// TestPostgresAuditPurgeAtScaleProbesForeignKeysThroughIndexes purges an
// Audit with 100,000 executions and 2,000 assessments while another Audit
// keeps 50,000 executions and 2,000 findings, and requires that the purge
// transaction, including every cascade and every deferred foreign-key check,
// reads none of those tables whole (V370-023). Statistics stay absent, so the
// planner treats each probe as selective, as it is in a database of many
// Audits. A probe without an index led by its key then scans the table
// sequentially or, when an index holds the key in a later column, reads that
// whole index. The test fails on either: on any sequential scan, and on an
// index of at least purgeScaleIndexMinBlocks blocks of which the purge read
// more than nine tenths per scan. The other Audit owns a third of the
// executions, so a probe for the purged Audit's own rows reads about two
// thirds of an index at most, and a whole read nearly all of it.
func TestPostgresAuditPurgeAtScaleProbesForeignKeysThroughIndexes(t *testing.T) {
	f := newCollectingAuditFixture(t, "purge-scale", 1<<20)
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Minute)
	defer cancel()
	findingID := collectPurgeFinding(t, f, "receipt-purge-scale")
	background, _, err := f.store.CreateDraft(ctx, CreateDraftParams{
		AuditID: "audit-purge-scale-background", OwnerID: f.audit.OwnerID, ProjectID: f.project.ProjectID,
		Profile: f.audit.Profile, ProfileSnapshot: f.audit.ProfileSnapshot,
		InputSelection: json.RawMessage(`{}`), Limits: f.audit.Limits,
		IdempotencyKey: "purge-scale-background", RequestDigest: testDigest("e"),
	})
	if err != nil {
		t.Fatal(err)
	}
	claim := drainDeletedAuditForPurge(t, f)

	scaled := []string{"audit_executions", "audit_finding_assessments", "audit_findings"}
	conn, err := f.pool.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Release()
	for _, table := range scaled {
		if _, err := conn.Exec(ctx, `ALTER TABLE `+pgx.Identifier{table}.Sanitize()+
			` SET (autovacuum_enabled = false)`); err != nil {
			t.Fatal(err)
		}
	}
	setup, err := conn.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = setup.Rollback(context.WithoutCancel(ctx)) }()
	// The copies only need to satisfy the table constraints; the triggers
	// guard the store's own writes.
	if _, err := setup.Exec(ctx, `SET LOCAL session_replication_role = replica`); err != nil {
		t.Fatal(err)
	}
	backgroundID := `'` + background.AuditID + `'`
	cloneRows(t, ctx, setup, "audit_executions", "execution_id", f.execution.ExecutionID, purgeScaleExecutions, map[string]string{
		"execution_id":          `src.execution_id || '-scale-' || n`,
		"submission_key":        `'scale-' || n`,
		"run_id":                `'run-scale-' || n`,
		"run_provenance":        `jsonb_set(src.run_provenance, '{runId}', to_jsonb('run-scale-' || n))`,
		"collection_receipt_id": `'collection-scale-' || n`,
	})
	// The other Audit's executions belong to no Round.
	cloneRows(t, ctx, setup, "audit_executions", "execution_id", f.execution.ExecutionID, purgeScaleBackgroundExecutions, map[string]string{
		"execution_id": `'background-execution-' || n`, "audit_id": backgroundID,
		"round_id": `NULL`, "role": `'discovery'`, "role_attempt": `n`,
		"submission_key":        `'background-' || n`,
		"run_id":                `'background-run-' || n`,
		"run_provenance":        `jsonb_set(src.run_provenance, '{runId}', to_jsonb('background-run-' || n))`,
		"collection_receipt_id": `'background-collection-' || n`,
	})
	cloneRows(t, ctx, setup, "audit_finding_assessments", "assessment_id", "assessment-receipt-purge-scale", purgeScaleAssessments, map[string]string{
		"assessment_id": `src.assessment_id || '-scale-' || n`,
		"result_digest": `'sha256:' || lpad(to_hex(n), 64, '0')`,
	})
	cloneRows(t, ctx, setup, "finding_proposal_receipts", "receipt_id", "receipt-purge-scale", purgeScaleFindings, map[string]string{
		"receipt_id":    `'background-receipt-' || n`,
		"proposal_id":   `'background-proposal-' || n`,
		"submission_id": `'background-submission-' || n`,
	})
	cloneRows(t, ctx, setup, "audit_findings", "finding_id", findingID, purgeScaleFindings, map[string]string{
		"finding_id": `'background-finding-' || n`, "audit_id": backgroundID,
		"first_receipt_id": `'background-receipt-' || n`, "state": `'proposed'`,
		"rejection_reason": `NULL`, "duplicate_target_id": `NULL`,
		"current_assessment_id": `NULL`, "current_decision_id": `NULL`,
	})
	// The copies take longer than the fixture's claim lease.
	if _, err := setup.Exec(ctx, `
UPDATE audit_controller_claims SET expires_at = clock_timestamp() + interval '10 minutes'
 WHERE audit_id = $1 AND holder_id = $2 AND epoch = $3`, claim.AuditID, claim.HolderID, claim.Epoch); err != nil {
		t.Fatal(err)
	}
	if err := setup.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	counts := map[string]int{}
	for _, table := range scaled {
		var count int
		if err := conn.QueryRow(ctx, `SELECT count(*) FROM `+pgx.Identifier{table}.Sanitize()).Scan(&count); err != nil {
			t.Fatal(err)
		}
		counts[table] = count
	}
	if counts["audit_executions"] <= purgeScaleExecutions+purgeScaleBackgroundExecutions || counts["audit_finding_assessments"] <= purgeScaleAssessments ||
		counts["audit_findings"] <= purgeScaleFindings {
		t.Fatalf("scaled fixture = %v", counts)
	}

	purge, err := conn.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = purge.Rollback(context.WithoutCancel(ctx)) }()
	// The counters can still hold work of earlier transactions on this
	// connection that is not flushed yet, so only differences count.
	reads := func() map[string]purgeScaleReads {
		rows, err := purge.Query(ctx, purgeScaleReadsSQL, scaled)
		if err != nil {
			t.Fatal(err)
		}
		relations := map[string]purgeScaleReads{}
		var name string
		var read purgeScaleReads
		if _, err := pgx.ForEachRow(rows, []any{&name, &read.kind, &read.scans, &read.fetched, &read.size}, func() error {
			relations[name] = read
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		return relations
	}
	before := reads()
	if err := NewPostgresStore(purge).PurgeClaimed(ctx, claim, auditdomain.ArtifactNamespace(f.audit.AuditID)); err != nil {
		t.Fatal(err)
	}
	// Run the deferred checks inside the purge transaction, where it still
	// counts what the purge did.
	if _, err := purge.Exec(ctx, `SET CONSTRAINTS ALL IMMEDIATE`); err != nil {
		t.Fatal(err)
	}
	largeIndexScans := int64(0)
	for name, after := range reads() {
		scans, fetched := after.scans-before[name].scans, after.fetched-before[name].fetched
		switch {
		case after.kind == "table" && scans != 0:
			t.Errorf("purge scanned %s (%d rows) sequentially %d times", name, counts[name], scans)
		case after.kind == "index" && after.size >= purgeScaleIndexMinBlocks && scans != 0:
			largeIndexScans += scans
			if 10*fetched > 9*scans*after.size {
				t.Errorf("purge read %d blocks of the %d-block index %s in %d scans", fetched, after.size, name, scans)
			}
		}
	}
	if largeIndexScans == 0 {
		t.Errorf("purge scanned no index of %d blocks or more; the fixture is too small to tell whole reads apart", purgeScaleIndexMinBlocks)
	}
	if err := purge.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	var executions, assessments, kept int
	if err := conn.QueryRow(ctx, `
SELECT (SELECT count(*) FROM audit_executions WHERE audit_id = $1), (SELECT count(*) FROM audit_finding_assessments),
       (SELECT count(*) FROM audit_findings WHERE audit_id = $2)`, f.audit.AuditID, background.AuditID).Scan(
		&executions, &assessments, &kept); err != nil {
		t.Fatal(err)
	}
	if executions != 0 || assessments != 0 || kept != purgeScaleFindings {
		t.Fatalf("after purge: %d executions, %d assessments, %d background findings", executions, assessments, kept)
	}
}

// cloneRows inserts count copies of the row of table whose keyColumn is key.
// Each copy takes the SQL expressions of overrides, over the source row src
// and the copy number n, in place of the copied column values.
func cloneRows(
	t *testing.T, ctx context.Context, tx pgx.Tx,
	table, keyColumn, key string, count int, overrides map[string]string,
) {
	t.Helper()
	rows, err := tx.Query(ctx, `
SELECT attname::text FROM pg_attribute
 WHERE attrelid = $1::regclass AND attnum > 0 AND NOT attisdropped AND attgenerated = ''
 ORDER BY attnum`, table)
	if err != nil {
		t.Fatal(err)
	}
	columns, err := pgx.CollectRows(rows, pgx.RowTo[string])
	if err != nil {
		t.Fatal(err)
	}
	names := make([]string, len(columns))
	values := make([]string, len(columns))
	used := 0
	for index, column := range columns {
		names[index] = pgx.Identifier{column}.Sanitize()
		if expression, overridden := overrides[column]; overridden {
			values[index] = expression
			used++
		} else {
			values[index] = "src." + names[index]
		}
	}
	if used != len(overrides) {
		t.Fatalf("%s: overrides %v name a missing column", table, overrides)
	}
	identifier := pgx.Identifier{table}.Sanitize()
	tag, err := tx.Exec(ctx, `INSERT INTO `+identifier+` (`+strings.Join(names, ", ")+`)
SELECT `+strings.Join(values, ", ")+`
  FROM `+identifier+` AS src, generate_series(1, $2::integer) AS n
 WHERE src.`+pgx.Identifier{keyColumn}.Sanitize()+` = $1`, key, count)
	if err != nil {
		t.Fatalf("clone %s: %v", table, err)
	}
	if tag.RowsAffected() != int64(count) {
		t.Fatalf("clone %s inserted %d rows, want %d", table, tag.RowsAffected(), count)
	}
}
