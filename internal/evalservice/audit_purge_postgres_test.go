package evalservice

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// TestPostgresEvalAuditWithDecidedHistoryPurges drains an Eval-managed Audit
// that holds an owner decision and a collected finding assessment through the
// ordinary purge, which must also record the Eval execution tombstone.
func TestPostgresEvalAuditWithDecidedHistoryPurges(t *testing.T) {
	h := newHarness(t)
	e := h.prepared(t, "audit")
	h.command(t, e, "start")
	c := h.coordinator(t, "controller")
	tick(t, c)
	tick(t, c)
	var auditID string
	if err := h.pool.QueryRow(t.Context(), `SELECT audit_id FROM audits`).Scan(&auditID); err != nil {
		t.Fatal("Eval-managed Audit was not created", err)
	}
	audit, err := h.audit.Get(t.Context(), h.scope.OwnerID, auditID)
	if err != nil {
		t.Fatal(err)
	}
	seedEvalAuditDecidedHistory(t, h.pool, audit)
	e = h.get(t, e.ID)
	h.command(t, e, "cancel")
	tick(t, c)
	store := auditstore.NewPostgresStore(h.pool)
	claims, err := store.Claim(t.Context(), auditstore.ClaimParams{HolderID: "audit-controller", Lease: time.Minute, Limit: 10})
	if err != nil || len(claims) != 1 || claims[0].AuditID != auditID {
		t.Fatal(claims, err)
	}
	if err = store.PurgeClaimed(t.Context(), claims[0], auditdomain.ArtifactNamespace(auditID)); err != nil {
		t.Fatalf("purge Eval-managed Audit with decided history: %v", err)
	}
	tick(t, c)
	e = h.get(t, e.ID)
	if e.State != "cancelled" || e.Outstanding != 0 || count(t, h.pool, "audits") != 0 ||
		count(t, h.pool, "audit_review_decisions") != 0 || count(t, h.pool, "audit_finding_assessments") != 0 ||
		count(t, h.pool, "eval_execution_tombstones") != 1 {
		t.Fatal("Eval-managed Audit with decided history failed to drain", e.State)
	}
}

// seedEvalAuditDecidedHistory records the rows an Audit keeps after its child
// Run is deleted: a collected attempt whose result assessed a retained
// finding, and the owner's decision confirming that finding.
func seedEvalAuditDecidedHistory(t *testing.T, pool *pgxpool.Pool, audit auditstore.Audit) {
	t.Helper()
	digest := "sha256:" + strings.Repeat("d", 64)
	sourceRevision := "checks-r1"
	origin, err := json.Marshal(auditstore.ItemOrigin{
		Schema: auditstore.ItemOriginSchema, EntryKey: "check-decided",
		SourceRef:           &contracts.ArtifactRef{Namespace: "inputs", Name: "checks", Revision: &sourceRevision},
		SourceContentDigest: digest, SourceMediaType: "application/json",
		CanonicalInventoryDigest: digest,
	})
	if err != nil {
		t.Fatal(err)
	}
	workflow := &auditstore.WorkflowProvenance{
		Name: "check", Version: "1", SchemaVersion: contracts.APIVersion, ClosureDigest: digest,
	}
	workflow.ConfigurationRef.Name, workflow.ConfigurationRef.Version = workflow.Name, workflow.Version
	runProvenance, err := json.Marshal(auditstore.RunProvenance{
		Schema: "contractor.audit.run-provenance.v1", RunID: "deleted-run-decided", Workflow: workflow,
	})
	if err != nil {
		t.Fatal(err)
	}
	proposalRef := `{"namespace":"audit-finding-proposals","name":"candidate-decided","revision":"proposal-r1"}`
	taskRef := `{"namespace":"audit-task-packages","name":"check-decided","revision":"task-r1"}`
	resultRef := `{"namespace":"audit-results","name":"check-decided","revision":"result-r1"}`
	statements := []struct {
		sql  string
		args []any
	}{
		{`
INSERT INTO finding_proposal_receipts (
    receipt_id, proposal_id, allocation_id, runtime_agent_id,
    runtime_instance_id, stage_execution_id, logical_agent_name,
    invocation_id, submission_id, client_key, request_digest,
    run_id, owner_id, project_id,
    workflow_name, workflow_version, workflow_schema_version,
    workflow_configuration_ref, workflow_closure_digest,
    proposal_ref, proposal_digest, proposal_media_type,
    proposal_size_bytes, evidence
) VALUES ('receipt-decided', 'proposal-decided', 'allocation-decided', 'runtime-decided',
          'instance-decided', 'stage-decided', 'worker', 'invocation-decided',
          'submission-decided', 'candidate-decided', $1, 'deleted-run-decided', $2, $3,
          'finding-source', '1', 'contractor/v1alpha1',
          '{"name":"finding-source","version":"1"}'::jsonb, $1,
          $4::jsonb, $1, 'application/json', 128, '[]'::jsonb)`,
			[]any{digest, audit.OwnerID, audit.ProjectID, proposalRef}},
		{`
UPDATE finding_proposal_receipts
   SET retention_state = 'audit-held', source_run_deleted_at = clock_timestamp()
 WHERE receipt_id = 'receipt-decided'`, nil},
		{`
INSERT INTO finding_proposal_audit_holds (receipt_id, audit_id, project_id, proposal_ref, evidence)
VALUES ('receipt-decided', $1, $2,
        jsonb_build_object('ref', $3::jsonb, 'digest', $4::text,
                           'mediaType', 'application/json', 'sizeBytes', 128),
        '[]'::jsonb)`, []any{audit.AuditID, audit.ProjectID, proposalRef, digest}},
		{`
INSERT INTO audit_rounds (
    round_id, audit_id, ordinal, manifest_ref, manifest_digest, state, expected_item_count
) VALUES ('round-decided', $1, 1,
          '{"namespace":"audit-rounds","name":"round-decided","revision":"round-r1"}'::jsonb,
          $2, 'closed', 1)`, []any{audit.AuditID, digest}},
		{`
INSERT INTO audit_items (
    item_id, audit_id, round_id, item_key, ordinal, kind, subject_key,
    task_ref, task_digest, origin, workflow_role, state
) VALUES ('item-decided', $1, 'round-decided', 'check-decided', 0, 'check',
          'component-decided', $2::jsonb, $3, $4::jsonb, 'check-role', 'ready')`,
			[]any{audit.AuditID, taskRef, digest, origin}},
		{`
INSERT INTO audit_executions (
    execution_id, audit_id, round_id, role, workflow_role, manifest_ref, manifest_digest,
    submission_key, request_digest, run_id, state, terminal_outcome,
    terminal_run_generation, terminal_run_sequence, terminal_observed_at,
    run_provenance, run_deleted_at
) VALUES ('execution-decided', $1, 'round-decided', 'check', 'check-role',
          '{"namespace":"audit-executions","name":"check-decided","revision":"execution-r1"}'::jsonb,
          $2, 'submission-decided', $2, 'deleted-run-decided', 'collected', 'succeeded',
          'generation-one', 1, clock_timestamp(), $3::jsonb, clock_timestamp())`,
			[]any{audit.AuditID, digest, runProvenance}},
		{`
INSERT INTO audit_execution_items (
    execution_item_id, execution_id, audit_id, round_id, item_id,
    batch_ordinal, item_attempt, task_ref, task_digest, input_refs,
    state, collection_disposition, result_ref, result_digest, collected_at
) VALUES ('execution-item-decided', 'execution-decided', $1, 'round-decided', 'item-decided',
          0, 1, $2::jsonb, $3, '[]'::jsonb, 'settled', 'accepted-result',
          $4::jsonb, $3, clock_timestamp())`, []any{audit.AuditID, taskRef, digest, resultRef}},
		{`
INSERT INTO audit_collection_receipts (
    receipt_id, audit_id, execution_id, run_id, terminal_outcome,
    terminal_run_generation, terminal_run_sequence, disposition,
    source_output_ref, source_output_digest, retained_refs, request_digest
) VALUES ('collection-decided', $1, 'execution-decided', 'deleted-run-decided', 'succeeded',
          'generation-one', 1, 'accepted-result', $2::jsonb, $3, '[]'::jsonb, $3)`,
			[]any{audit.AuditID, resultRef, digest}},
		{`
INSERT INTO audit_finding_assessments (
    assessment_id, finding_id, audit_id, receipt_id, item_id,
    execution_item_id, collection_receipt_id, semantic_assessment,
    result_ref, result_digest
)
SELECT 'assessment-decided', finding_id, audit_id, first_receipt_id,
       'item-decided', 'execution-item-decided', 'collection-decided', 'supported',
       $2::jsonb, $3
  FROM audit_findings WHERE audit_id = $1`, []any{audit.AuditID, resultRef, digest}},
		{`
UPDATE audit_items
   SET state = 'settled', final_disposition = 'accepted-result',
       accepted_result_ref = $2::jsonb, accepted_result_digest = $3,
       last_execution_item_id = 'execution-item-decided'
 WHERE audit_id = $1 AND item_id = 'item-decided'`, []any{audit.AuditID, resultRef, digest}},
		{`
INSERT INTO audit_review_requests (
    request_id, audit_id, finding_id, subject_kind, subject_id, kind,
    subject_revision, subject_digest, requested_actions, state,
    idempotency_key, request_digest
)
SELECT 'review-decided', audit_id, finding_id, 'finding', finding_id,
       'finding-triage', 1, $2, '["true_positive","false_positive"]'::jsonb, 'decided',
       'review-decided', $2
  FROM audit_findings WHERE audit_id = $1`, []any{audit.AuditID, digest}},
		{`
INSERT INTO audit_review_decisions (
    decision_id, request_id, audit_id, finding_id, actor_id, action, verdict,
    severity, rationale, subject_revision, subject_digest, idempotency_key, request_digest
)
SELECT 'decision-decided', 'review-decided', audit_id, finding_id, $2,
       'true_positive', 'true_positive', 'high', 'Confirmed before deletion.',
       1, $3, 'decision-decided', $3
  FROM audit_findings WHERE audit_id = $1`, []any{audit.AuditID, audit.OwnerID, digest}},
		{`
UPDATE audit_findings
   SET state = 'confirmed', current_decision_id = 'decision-decided',
       current_assessment_id = 'assessment-decided', revision = revision + 1
 WHERE audit_id = $1`, []any{audit.AuditID}},
	}
	err = pg.InTx(t.Context(), pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		for _, statement := range statements {
			if _, err := tx.Exec(t.Context(), statement.sql, statement.args...); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
