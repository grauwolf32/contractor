package pgstore

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/jackc/pgx/v5"
)

const (
	// findingScaleFindings findings, each with a pending triage review,
	// and findingScaleExecutions collected executions belong to the read
	// Audit. Each of findingScaleBackground other Audits of the same owner
	// holds a quarter as many, so the read Audit owns a third of each table
	// and the planner's statistics see several Audits, as in a database of
	// many.
	findingScaleFindings   = 4_000
	findingScaleExecutions = 20_000
	findingScaleBackground = 8
	// findingScaleEmptyAudits Audits without findings or executions make
	// the audits table large enough that the planner probes it by key, as
	// it does in a database of many Audits.
	findingScaleEmptyAudits = 2_000
)

// findingScaleFixture is a collected Audit with findingScaleFindings + 1
// unreviewed findings, findingScaleFindings pending triage reviews of which
// half are past their expiry without anything having stored it,
// findingScaleExecutions + 1 collected executions, and background and empty
// Audits of the same owner. The copies are read-model fixtures: they satisfy
// the schema, not every store rule.
type findingScaleFixture struct {
	collectingAuditFixture
	findingID string
}

func newFindingScaleFixture(t *testing.T) findingScaleFixture {
	t.Helper()
	f := newCollectingAuditFixture(t, "finding-scale", 1<<20)
	// The copies and the reads outlast the collecting fixture's deadline.
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	t.Cleanup(cancel)
	f.ctx = ctx
	findingID := collectPurgeFinding(t, f, "receipt-finding-scale")
	type copySet struct {
		audit, prefix        string
		findings, executions int
	}
	copies := []copySet{{f.audit.AuditID, "scale", findingScaleFindings, findingScaleExecutions}}
	for index := range findingScaleBackground {
		background, _, err := f.store.CreateDraft(f.ctx, auditstore.CreateDraftParams{
			AuditID: fmt.Sprintf("audit-finding-scale-background-%d", index), OwnerID: f.audit.OwnerID,
			ProjectID: f.project.ProjectID, Profile: f.audit.Profile, ProfileSnapshot: f.audit.ProfileSnapshot,
			InputSelection: json.RawMessage(`{}`), Limits: f.audit.Limits,
			IdempotencyKey: fmt.Sprintf("finding-scale-background-%d", index), RequestDigest: testDigest("e"),
		})
		if err != nil {
			t.Fatal(err)
		}
		copies = append(copies, copySet{background.AuditID, fmt.Sprintf("background-%d", index),
			findingScaleFindings / 4, findingScaleExecutions / 4})
	}
	setup, err := f.pool.Begin(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = setup.Rollback(context.WithoutCancel(f.ctx)) }()
	// The copied executions satisfy every constraint and foreign key, but
	// not the triggers that guard the store's own execution writes; the test
	// owns its schema, so it switches them off for the copies.
	if _, err := setup.Exec(f.ctx, `ALTER TABLE audit_executions DISABLE TRIGGER USER`); err != nil {
		t.Fatal(err)
	}
	for _, copy := range copies {
		audit := `'` + copy.audit + `'`
		cloneRows(t, f.ctx, setup, "finding_proposal_receipts", "receipt_id", "receipt-finding-scale", copy.findings, map[string]string{
			"receipt_id":    `'` + copy.prefix + `-receipt-' || n`,
			"proposal_id":   `'` + copy.prefix + `-proposal-' || n`,
			"submission_id": `'` + copy.prefix + `-submission-' || n`,
		})
		cloneRows(t, f.ctx, setup, "audit_findings", "finding_id", findingID, copy.findings, map[string]string{
			"finding_id": `'` + copy.prefix + `-finding-' || lpad(n::text, 6, '0')`, "audit_id": audit,
			"first_receipt_id": `'` + copy.prefix + `-receipt-' || n`, "state": `'proposed'`,
			"rejection_reason": `NULL`, "duplicate_target_id": `NULL`,
			"current_assessment_id": `NULL`, "current_decision_id": `NULL`,
			"created_at": `src.created_at + n * interval '1 millisecond'`,
		})
		// Odd copies' reviews expired an hour ago, even ones expire in an
		// hour.
		if _, err := setup.Exec(f.ctx, `
INSERT INTO audit_review_requests (request_id, audit_id, finding_id, subject_kind, subject_id, kind,
       subject_revision, subject_digest, requested_actions, state, expires_at, idempotency_key, request_digest)
SELECT 'review-' || finding_id, audit_id, finding_id, 'finding', finding_id, 'finding-triage',
       1, $2, '["true_positive","false_positive"]'::jsonb, 'pending',
       clock_timestamp() + CASE WHEN right(finding_id, 1)::integer % 2 = 1
                                THEN interval '-1 hour' ELSE interval '1 hour' END,
       'review-' || finding_id, $2
  FROM audit_findings WHERE audit_id = $1 AND finding_id <> $3`, copy.audit, testDigest("d"), findingID); err != nil {
			t.Fatal(err)
		}
		cloneRows(t, f.ctx, setup, "audit_executions", "execution_id", f.execution.ExecutionID, copy.executions, map[string]string{
			"execution_id": `'` + copy.prefix + `-execution-' || n`, "audit_id": audit,
			"round_id": `NULL`, "role": `'discovery'`, "role_attempt": `n`,
			"submission_key":        `'` + copy.prefix + `-' || n`,
			"run_id":                `'` + copy.prefix + `-run-' || n`,
			"run_provenance":        `jsonb_set(src.run_provenance, '{runId}', to_jsonb('` + copy.prefix + `-run-' || n))`,
			"collection_receipt_id": `'` + copy.prefix + `-collection-' || n`,
		})
	}
	cloneRows(t, f.ctx, setup, "audits", "audit_id", copies[1].audit, findingScaleEmptyAudits, map[string]string{
		"audit_id": `'audit-finding-scale-empty-' || n`, "next_event_sequence": `1`,
	})
	if _, err := setup.Exec(f.ctx, `ALTER TABLE audit_executions ENABLE TRIGGER USER`); err != nil {
		t.Fatal(err)
	}
	if err := setup.Commit(f.ctx); err != nil {
		t.Fatal(err)
	}
	for _, table := range []string{"audit_findings", "audit_review_requests", "audit_executions", "audits", "finding_proposal_receipts"} {
		if _, err := f.pool.Exec(f.ctx, `ANALYZE `+pgx.Identifier{table}.Sanitize()); err != nil {
			t.Fatal(err)
		}
	}
	return findingScaleFixture{collectingAuditFixture: f, findingID: findingID}
}
