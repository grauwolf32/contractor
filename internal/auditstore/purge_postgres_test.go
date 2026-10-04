package auditstore

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
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
	if err := f.store.PurgeClaimed(f.ctx, claim, auditdomain.ArtifactNamespace(f.audit.AuditID)); err != nil {
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
	seedDecidedPurgeReviews(t, f, confirmedID, duplicateID)

	keys := auditOwnedForeignKeys(t, f)
	for _, key := range keys {
		switch {
		case key.name == auditChildRunForeignKey:
			// Guarded by the purge precondition, exercised below.
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
		if count := countForeignKeyRows(t, f, key); count == 0 {
			t.Errorf("%s: fixture holds no %s row referencing %s", key.name, key.referencing, key.referenced)
		}
	}
	namespace := auditdomain.ArtifactNamespace(f.audit.AuditID)
	if err := f.store.PurgeClaimed(f.ctx, claim, namespace); !errors.Is(err, ErrPrecondition) {
		t.Fatalf("purge with a live child Run = %v, want precondition failure", err)
	}
	deleteChildRunForPurge(t, f)
	if err := f.store.PurgeClaimed(f.ctx, claim, namespace); err != nil {
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
	result := testExact(f.audit.AuditID, "result", "result-r1")
	if _, inserted, err := f.store.Collect(f.ctx, CollectParams{
		Claim: f.claim, ReceiptID: "collection-" + receiptID, ExecutionID: f.execution.ExecutionID,
		Disposition: CollectionAccepted, SourceOutput: &result, RequestDigest: testDigest("9"),
		Retained: []ArtifactLink{{
			LogicalKey: "result/" + f.memberID, Artifact: result,
			SourceProvenance: json.RawMessage(`{"runId":"` + *f.execution.RunID + `"}`),
		}},
		Items: []CollectionItem{{
			ExecutionItemID: f.memberID, Disposition: CollectionAccepted,
			FinalDisposition: FinalAccepted, Result: &result,
			Coverage: Coverage{Status: CoverageSatisfied, Requested: []string{}, Completed: []string{}, Gaps: []string{}},
			FindingAssociations: []FindingAssociation{{
				AssessmentID: "assessment-" + receiptID, ReceiptID: receiptID,
				Proposal: proposal, SemanticAssessment: "supported",
			}},
		}},
	}); err != nil || !inserted {
		t.Fatalf("collect finding assessment = (%t, %v)", inserted, err)
	}
	return findingID
}

// seedDecidedPurgeReviews records the decided review ledger an owner leaves
// behind: a confirmed finding, a duplicate of it, an approved item action
// and a rejected report candidate.
func seedDecidedPurgeReviews(t *testing.T, f collectingAuditFixture, confirmedID, duplicateID string) {
	t.Helper()
	digest := testDigest("c")
	link := func(key, name, media string) []byte {
		artifact := testExact(f.audit.AuditID, name, "report-r1")
		artifact.MediaType, artifact.SizeBytes = media, 16
		encoded, err := json.Marshal(ArtifactLink{
			LogicalKey: key, Artifact: artifact,
			SourceProvenance: json.RawMessage(`{"schema":"contractor.audit.report-provenance.v1"}`),
		})
		if err != nil {
			t.Fatal(err)
		}
		return encoded
	}
	machine := link(ReportMachineLogicalKey, "report.json", "application/json")
	summary := link(ReportSummaryLogicalKey, "report.md", "text/markdown")
	err := persistencepostgres.InTx(f.ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		for _, request := range []struct {
			id, findingID, subjectKind, subjectID, kind, actions string
		}{
			{"review-purge-confirmed", confirmedID, "finding", confirmedID, "finding-triage", `["true_positive"]`},
			{"review-purge-duplicate", duplicateID, "finding", duplicateID, "finding-triage", `["duplicate"]`},
			{"review-purge-item", "", "audit-item-action", f.itemID, "active-check-approval", `["approve","reject"]`},
			{"review-purge-report", "", "audit-report", f.audit.AuditID, "report-acceptance", `["approve","reject"]`},
		} {
			if _, err := tx.Exec(f.ctx, `
INSERT INTO audit_review_requests (
    request_id, audit_id, finding_id, subject_kind, subject_id, kind,
    subject_revision, subject_digest, requested_actions, state,
    idempotency_key, request_digest,
    report_round_id, report_machine_link, report_summary_link
) VALUES ($1, $2, NULLIF($3, ''), $4, $5, $6, 1, $7, $8::jsonb, 'decided', $1, $7,
          CASE WHEN $4 = 'audit-report' THEN $9 END,
          CASE WHEN $4 = 'audit-report' THEN $10::jsonb END,
          CASE WHEN $4 = 'audit-report' THEN $11::jsonb END)`,
				request.id, f.audit.AuditID, request.findingID, request.subjectKind,
				request.subjectID, request.kind, digest, request.actions,
				*f.audit.CurrentRoundID, machine, summary); err != nil {
				return err
			}
		}
		for _, decision := range []struct {
			id, requestID, findingID, action, verdict, severity, duplicateTarget string
		}{
			{"decision-purge-confirmed", "review-purge-confirmed", confirmedID, "true_positive", "true_positive", "high", ""},
			{"decision-purge-duplicate", "review-purge-duplicate", duplicateID, "duplicate", "duplicate", "", confirmedID},
			{"decision-purge-item", "review-purge-item", "", "approve", "", "", ""},
			{"decision-purge-report", "review-purge-report", "", "reject", "", "", ""},
		} {
			if _, err := tx.Exec(f.ctx, `
INSERT INTO audit_review_decisions (
    decision_id, request_id, audit_id, finding_id, actor_id, action, verdict,
    severity, rationale, duplicate_target_id, subject_revision, subject_digest,
    idempotency_key, request_digest
) VALUES ($1, $2, $3, NULLIF($4, ''), $5, $6, NULLIF($7, ''), NULLIF($8, ''),
          'Decided before deletion.', NULLIF($9, ''), 1, $10, $1, $10)`,
				decision.id, decision.requestID, f.audit.AuditID, decision.findingID,
				f.audit.OwnerID, decision.action, decision.verdict, decision.severity,
				decision.duplicateTarget, digest); err != nil {
				return err
			}
		}
		if _, err := tx.Exec(f.ctx, `
UPDATE audit_findings
   SET state = 'confirmed', current_decision_id = 'decision-purge-confirmed',
       revision = revision + 1
 WHERE finding_id = $1`, confirmedID); err != nil {
			return err
		}
		_, err := tx.Exec(f.ctx, `
UPDATE audit_findings
   SET state = 'duplicate', duplicate_target_id = $2, revision = revision + 1
 WHERE finding_id = $1`, duplicateID, confirmedID)
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
) ControllerClaim {
	t.Helper()
	current, err := f.store.Get(f.ctx, f.audit.OwnerID, f.audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	cancelling, changed, err := f.store.RequestDelete(f.ctx, DeleteParams{
		OwnerID: current.OwnerID, AuditID: current.AuditID, ExpectedRevision: current.Revision,
		IdempotencyKey: "purge-delete", RequestDigest: testDigest("a"),
	})
	if err != nil || !changed || cancelling.State != AuditCancelling {
		t.Fatalf("request Audit deletion = (%+v, %t, %v)", cancelling, changed, err)
	}
	if _, err := f.store.TransitionClaimed(f.ctx, ClaimedTransitionParams{
		Claim: f.claim, ExpectedRevision: cancelling.Revision,
		ExpectedState: AuditCancelling, TargetState: AuditDeleting, Reason: cancelling.StopReason,
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
