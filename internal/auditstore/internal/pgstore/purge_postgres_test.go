package pgstore

import (
	"errors"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
)

// The bound child Run references its AuditExecution from outside the Audit.
// Purge requires the Controller to delete that Run first instead of
// cascading into a Run it does not own.
const auditChildRunForeignKey = "workflow_runs_audit_submission_fkey"

func TestPostgresPurgeRemovesCollectedFindingAssessments(t *testing.T) {
	f := newCollectingAuditFixture(t, "purge-assessment", 1<<20)
	findingID := collectPurgeFinding(t, f, "receipt-purge-assessment")
	claim := drainDeletedAuditForPurge(t, f)
	if err := purgeFixtureUntilAbsent(t, f, claim); err != nil {
		t.Fatalf("purge Audit with collected finding %s: %v", findingID, err)
	}
	for _, table := range auditOwnedTables(t, f) {
		assertPurgedTable(t, f, table)
	}
}

// TestPostgresPurgeCoversEveryAuditOwnedForeignKey keeps every restricting
// key into an Audit-owned table both safe for the cascading purge and
// exercised by data. PostgreSQL cascades breadth-first and checks a RESTRICT
// key one cascade level after it deleted the referenced row, so a referencing
// row must be deleted at the same or an earlier level. Deferred keys are
// checked at commit, after the whole Audit is gone.
func TestPostgresPurgeCoversEveryAuditOwnedForeignKey(t *testing.T) {
	f := newCollectingAuditFixture(t, "purge-keys", 1<<20)
	confirmedID := collectPurgeFinding(t, f, "receipt-purge-confirmed")
	duplicateProposal := testExact("audit-findings", "candidate-purge-duplicate", "proposal-r1")
	duplicateProposal.MediaType, duplicateProposal.SizeBytes = "application/json", 128
	duplicateID := insertAuditChildFinding(t, f, "receipt-purge-duplicate", duplicateProposal)
	seedPurgeFindingAssessments(t, f, confirmedID, duplicateID)

	if _, err := f.pool.Exec(f.ctx, `INSERT INTO artifact_scopes(scope_kind,scope_id) VALUES('audit',$1) ON CONFLICT DO NOTHING`, f.audit.AuditID); err != nil {
		t.Fatal(err)
	}
	keys := auditOwnedForeignKeys(t, f)
	for _, key := range keys {
		switch {
		case key.name == auditChildRunForeignKey || key.name == "artifact_scopes_audit_id_fkey":
			// Child Runs are guarded by the drain precondition. The Artifact
			// owner explicitly purges Audit scope before removing its parent.
		case key.referencingDepth < 0:
			t.Errorf("%s: %s rows have no cascade path from audits and block purge",
				key.name, key.referencing)
		case !key.deferred && key.referencingDepth > key.referencedDepth:
			t.Errorf("%s: %s rows are deleted at cascade level %d, after %s rows at level %d are checked",
				key.name, key.referencing, key.referencingDepth, key.referenced, key.referencedDepth)
		}
	}
	claim := drainDeletedAuditForPurge(t, f, purgeDrainOptions{keepRun: true})
	for _, key := range keys {
		if key.name == "audit_tasks_last_execution_fk" {
			// The common Task/attempt topology is exercised through its real
			// owner in TestSharedPostgresCommonTaskCollectionAndPurge. This
			// retired item fixture must not fabricate a mixed-format Task.
			continue
		}
		if count := countForeignKeyRows(t, f, key); count == 0 {
			t.Errorf("%s: fixture holds no %s row referencing %s", key.name, key.referencing, key.referenced)
		}
	}
	namespace := auditdomain.ArtifactNamespace(f.audit.AuditID)
	if err := f.store.PurgeClaimed(f.ctx, claim, namespace); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("purge with a live child Run = %v, want precondition failure", err)
	}
	deleteChildRunForPurge(t, f)
	if err := purgeFixtureUntilAbsent(t, f, claim); err != nil {
		t.Fatalf("purge Audit with decided reviews and collected assessments: %v", err)
	}
	for _, table := range auditOwnedTables(t, f) {
		assertPurgedTable(t, f, table)
	}
}

type auditOwnedForeignKey struct {
	name             string
	referencing      string
	referenced       string
	columns          []string
	deferred         bool
	referencingDepth int
	referencedDepth  int
}

// auditOwnedDepthCTE yields every table that cascades from audits with the
// breadth-first cascade level at which an Audit purge deletes its rows. Only
// cascades over NOT NULL keys count: they reach every referencing row.
const auditOwnedDepthCTE = `
WITH RECURSIVE owned (table_oid, depth) AS (
    SELECT 'audits'::regclass::oid, 0
    UNION ALL
    SELECT con.conrelid, owned.depth + 1
      FROM owned
      JOIN pg_constraint AS con
        ON con.confrelid = owned.table_oid AND con.contype = 'f' AND con.confdeltype = 'c'
     WHERE con.conrelid <> owned.table_oid AND owned.depth < 8
       AND NOT EXISTS (
           SELECT 1 FROM pg_attribute AS attribute
            WHERE attribute.attrelid = con.conrelid
              AND attribute.attnum = ANY (con.conkey) AND NOT attribute.attnotnull
       )
), depth AS (
    SELECT table_oid, min(depth) AS depth FROM owned GROUP BY table_oid
)`

func auditOwnedTables(t *testing.T, f collectingAuditFixture) []string {
	t.Helper()
	rows, err := f.pool.Query(f.ctx, auditOwnedDepthCTE+`
SELECT table_oid::regclass::text FROM depth ORDER BY depth, 1`)
	if err != nil {
		t.Fatal(err)
	}
	tables, err := pgx.CollectRows(rows, pgx.RowTo[string])
	if err != nil {
		t.Fatal(err)
	}
	return tables
}

// auditOwnedForeignKeys lists every restricting foreign key into an
// Audit-owned table with the cascade level of both tables. A referencing
// table without a cascade path from audits has depth -1.
func auditOwnedForeignKeys(t *testing.T, f collectingAuditFixture) []auditOwnedForeignKey {
	t.Helper()
	rows, err := f.pool.Query(f.ctx, auditOwnedDepthCTE+`
SELECT con.conname::text, con.conrelid::regclass::text, con.confrelid::regclass::text,
       ARRAY(
           SELECT attribute.attname::text
             FROM unnest(con.conkey) WITH ORDINALITY AS key (attnum, position)
             JOIN pg_attribute AS attribute
               ON attribute.attrelid = con.conrelid AND attribute.attnum = key.attnum
            ORDER BY key.position
       ),
       con.condeferrable AND con.condeferred,
       COALESCE(referencing.depth, -1), referenced.depth
  FROM pg_constraint AS con
  JOIN depth AS referenced ON referenced.table_oid = con.confrelid
  LEFT JOIN depth AS referencing ON referencing.table_oid = con.conrelid
 WHERE con.contype = 'f' AND con.confdeltype IN ('r', 'a')
 ORDER BY con.conname`)
	if err != nil {
		t.Fatal(err)
	}
	keys, err := pgx.CollectRows(rows, func(row pgx.CollectableRow) (auditOwnedForeignKey, error) {
		var key auditOwnedForeignKey
		err := row.Scan(&key.name, &key.referencing, &key.referenced, &key.columns,
			&key.deferred, &key.referencingDepth, &key.referencedDepth)
		return key, err
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(keys) == 0 {
		t.Fatal("no restricting foreign keys reference Audit-owned tables")
	}
	return keys
}

func countForeignKeyRows(t *testing.T, f collectingAuditFixture, key auditOwnedForeignKey) int {
	t.Helper()
	predicates := make([]string, len(key.columns))
	for index, column := range key.columns {
		predicates[index] = pgx.Identifier{column}.Sanitize() + " IS NOT NULL"
	}
	var count int
	if err := f.pool.QueryRow(f.ctx, `SELECT count(*) FROM `+pgx.Identifier{key.referencing}.Sanitize()+
		` WHERE `+strings.Join(predicates, " AND ")).Scan(&count); err != nil {
		t.Fatal(err)
	}
	return count
}

func assertPurgedTable(t *testing.T, f collectingAuditFixture, table string) {
	t.Helper()
	var count int
	if err := f.pool.QueryRow(f.ctx, `SELECT count(*) FROM `+pgx.Identifier{table}.Sanitize()).Scan(&count); err != nil {
		t.Fatal(err)
	}
	if count != 0 {
		t.Errorf("%s keeps %d rows after the only Audit was purged", table, count)
	}
}

// collectPurgeFinding admits a finding from the fixture's child Run and
// collects the Run's accepted result with an assessment of that finding.
func collectPurgeFinding(t *testing.T, f collectingAuditFixture, receiptID string) string {
	t.Helper()
	proposal := testExact("audit-findings", "candidate-"+receiptID, "proposal-r1")
	proposal.MediaType, proposal.SizeBytes = "application/json", 128
	findingID := insertAuditChildFinding(t, f, receiptID, proposal)
	gap := "fixture-no-output"
	if _, inserted, err := f.store.Collect(f.ctx, auditstore.CollectParams{Claim: f.claim, ReceiptID: "collection-" + receiptID, ExecutionID: f.execution.ExecutionID, Disposition: auditstore.CollectionMissingOutput, ErrorCode: &gap, RequestDigest: testDigest("9"), Tasks: []auditstore.TaskCollection{{ExecutionItemID: f.memberID, TaskID: f.itemID, GapCode: &gap}}}); err != nil || !inserted {
		t.Fatalf("collect common Task: %t %v", inserted, err)
	}

	return findingID
}

// seedPurgeFindingAssessments records independent canonical assessments that
// survive their Audit and exercise the Project duplicate graph.
func seedPurgeFindingAssessments(t *testing.T, f collectingAuditFixture, confirmedID, duplicateID string) {
	t.Helper()
	_ = seedItemRows(t, f.ctx, f.pool, f.audit.AuditID, f.itemID)
	if _, err := f.pool.Exec(f.ctx, `UPDATE audit_items SET state='settled',final_disposition='excluded' WHERE audit_id=$1`, f.audit.AuditID); err != nil {
		t.Fatal(err)
	}
	err := persistencepostgres.InTx(f.ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		severity := auditstore.SeverityHigh
		if _, err := NewOwnerTxStore(tx, artifactpgtest.NewStorage()).AssessProjectFinding(f.ctx, auditstore.ProjectFindingEdit{OwnerID: f.audit.OwnerID, ProjectID: f.project.ProjectID, FindingID: confirmedID, ExpectedRevision: 1, Verdict: auditstore.VerdictTruePositive, Severity: &severity, Rationale: "Confirmed before Audit deletion."}); err != nil {
			return err
		}
		_, err := NewOwnerTxStore(tx, artifactpgtest.NewStorage()).AssessProjectFinding(f.ctx, auditstore.ProjectFindingEdit{OwnerID: f.audit.OwnerID, ProjectID: f.project.ProjectID, FindingID: duplicateID, ExpectedRevision: 1, Verdict: auditstore.VerdictDuplicate, DuplicateTargetID: &confirmedID, Rationale: "Linked before Audit deletion."})
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
}

type purgeDrainOptions struct {
	keepRun bool
}

// drainDeletedAuditForPurge requests owner deletion of the fixture Audit and
// repeats the Controller drain that precedes purge: enter deleting, delete
// the collected child Run and release the dispatch hold.
func drainDeletedAuditForPurge(
	t *testing.T, f collectingAuditFixture, options ...purgeDrainOptions,
) auditstore.ControllerClaim {
	t.Helper()
	current, err := f.store.Get(f.ctx, f.audit.OwnerID, f.audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	cancelling, changed, err := f.store.RequestDelete(f.ctx, auditstore.DeleteParams{
		OwnerID: current.OwnerID, AuditID: current.AuditID, ExpectedRevision: current.Revision,
		IdempotencyKey: "purge-delete", RequestDigest: testDigest("a"),
	})
	if err != nil || !changed || cancelling.State != auditstore.AuditCancelling {
		t.Fatalf("request Audit deletion = (%+v, %t, %v)", cancelling, changed, err)
	}
	// The deletion key reused with another request digest is permanent.
	if _, changed, err := f.store.RequestDelete(f.ctx, auditstore.DeleteParams{
		OwnerID: current.OwnerID, AuditID: current.AuditID, ExpectedRevision: current.Revision,
		IdempotencyKey: "purge-delete", RequestDigest: testDigest("b"),
	}); !errors.Is(err, contracts.ErrIdempotencyConflict) || errors.Is(err, auditstore.ErrConflict) || changed {
		t.Fatalf("Audit deletion idempotency drift = (%t, %v)", changed, err)
	}
	if _, err := f.store.TransitionClaimed(f.ctx, auditstore.ClaimedTransitionParams{
		Claim: f.claim, ExpectedRevision: cancelling.Revision,
		ExpectedState: auditstore.AuditCancelling, TargetState: auditstore.AuditDeleting, Reason: cancelling.StopReason,
	}); err != nil {
		t.Fatal(err)
	}
	if len(options) == 0 || !options[0].keepRun {
		deleteChildRunForPurge(t, f)
	}
	if _, released, err := f.store.ReleaseDispatchHold(f.ctx, f.claim); err != nil || !released {
		t.Fatalf("release Audit dispatch hold = (%t, %v)", released, err)
	}
	return f.claim
}

func deleteChildRunForPurge(t *testing.T, f collectingAuditFixture) {
	t.Helper()
	if _, err := f.pool.Exec(f.ctx, `DELETE FROM workflow_runs WHERE run_id = $1`, *f.execution.RunID); err != nil {
		t.Fatal(err)
	}
}

// Populate the journal's audit ownership and composite action identity keys;
// parent purge must remove attempts before intents without cascading a batch.

func TestProjectFindingPurgeBoundsLargeAnalystMetadata(t *testing.T) {
	f := newCollectingAuditFixture(t, "finding-metadata-purge", 1<<20)
	base := collectPurgeFinding(t, f, "metadata-source")
	// Size is the concern here; clone a schema-valid canonical identity, preserving
	// its real retained first receipt. No legacy Finding schema participates.
	_, err := f.pool.Exec(f.ctx, `INSERT INTO project_findings
 SELECT (jsonb_populate_record(NULL::project_findings,to_jsonb(seed)||jsonb_build_object(
 'finding_id','metadata-'||n,'title',repeat('"',65536),'state','duplicate','analyst_verdict','duplicate',
 'rationale',repeat('"',65536),'actor_id','owner','assessed_at',clock_timestamp(),'duplicate_target_id',$1::text))).*
 FROM project_findings AS seed CROSS JOIN generate_series(1,8) AS n WHERE finding_id=$1`, base)
	if err != nil {
		t.Fatal(err)
	}
	sizes := map[string]int{}
	rows, err := f.pool.Query(f.ctx, `SELECT finding_id,octet_length(to_jsonb(finding)::text) FROM project_findings AS finding WHERE project_id=$1 AND duplicate_target_id IS NOT NULL`, f.project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	for rows.Next() {
		var id string
		var bytes int
		if err := rows.Scan(&id, &bytes); err != nil {
			t.Fatal(err)
		}
		sizes[id] = bytes
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	rows.Close()
	if _, _, err := projectstore.NewPostgresStore(f.pool).BeginDeletion(f.ctx, projectstore.BeginDeletionParams{OwnerID: f.project.OwnerID, ProjectID: f.project.ProjectID, ExpectedRevision: f.project.Revision}); err != nil {
		t.Fatal(err)
	}
	if err := persistencepostgres.InTx(f.ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		_, err := NewOwnerTxStore(tx, artifactpgtest.NewStorage()).PurgeProjectFindings(f.ctx, f.project.ProjectID)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	rows, err = f.pool.Query(f.ctx, `SELECT finding_id FROM project_findings WHERE project_id=$1 AND finding_id LIKE 'metadata-%' AND state IS NULL`, f.project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	records, bytes := 0, 0
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			t.Fatal(err)
		}
		records++
		bytes += sizes[id]
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	rows.Close()
	if records == 0 || records > 128 || bytes > 1<<20 {
		t.Fatalf("purge slice changed %d records/%d metadata bytes", records, bytes)
	}
}

func purgeFixtureUntilAbsent(t *testing.T, f collectingAuditFixture, claim auditstore.ControllerClaim) error {
	t.Helper()
	for range 16 {
		if err := f.store.PurgeClaimed(f.ctx, claim, auditdomain.ArtifactNamespace(f.audit.AuditID)); err != nil {
			return err
		}
		var exists bool
		if err := f.pool.QueryRow(f.ctx, `SELECT EXISTS(SELECT 1 FROM audits WHERE audit_id=$1)`, f.audit.AuditID).Scan(&exists); err != nil {
			return err
		}
		if !exists {
			return nil
		}
	}
	t.Fatal("bounded fixture purge did not converge")
	return nil
}
