//go:build integration

package auditimport

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

func TestImporterRunDeletionInvalidatesNativeReceiptAuditOnce(t *testing.T) {
	f, snapshot := newReportDeletionFixture(t, completionPool(t), "automatic")
	// Storage-boundary fixture: two native receipt tombstones belong to the
	// same collected execution. No intake/publication behavior is simulated;
	// those paths have separate real-service import/delete regressions.
	for _, suffix := range []string{"one", "two"} {
		id := f.id + "-" + suffix
		_, err := f.pool.Exec(f.ctx, `
INSERT INTO finding_proposal_receipts (
    receipt_id, proposal_id, allocation_id, runtime_agent_id, runtime_instance_id,
    stage_execution_id, logical_agent_name, invocation_id, submission_id,
    client_key, request_digest, run_id, owner_id, project_id,
    audit_execution_id, audit_id, audit_role, workflow_name, workflow_version,
    workflow_schema_version, workflow_configuration_ref, workflow_closure_digest,
    proposal_ref, proposal_digest, proposal_media_type, proposal_size_bytes, evidence
) VALUES (
    $1, $1, $1, 'runtime', 'instance', 'stage', 'worker', $1, $1,
    $1, $2, $3, 'owner', $4, $5, $4, 'check', 'workflow', '1',
    'contractor/v1alpha1', '{}', $2,
    '{"namespace":"finding-proposals","name":"candidate","revision":"retained-source"}',
    $2, 'application/json', 2, '[]'
)`, id, completionDigest([]byte(id)), f.run.RunID, f.id, f.execution.ExecutionID)
		mustCompletion(t, err)
		_, err = f.pool.Exec(f.ctx, `
INSERT INTO finding_proposal_retention (receipt_id, state)
VALUES ($1, 'source-held')`, id)
		mustCompletion(t, err)
	}
	access, err := NewArtifactAccess(f.artifacts)
	mustCompletion(t, err)
	importer, err := New(f.audits, f.runs, access)
	mustCompletion(t, err)
	if worked, err := importer.Finalize(f.ctx, f.claim, snapshot); err != nil || !worked {
		t.Fatalf("complete Audit before source deletion: worked=%t err=%v", worked, err)
	}
	before, err := f.audits.Get(f.ctx, "owner", f.id)
	mustCompletion(t, err)
	if before.State != auditstore.AuditCompleted {
		t.Fatal("fixture must cover source deletion after the Audit becomes terminal")
	}
	var original []byte
	mustCompletion(t, f.pool.QueryRow(f.ctx, `
SELECT jsonb_agg(to_jsonb(receipt) ORDER BY receipt_id)
FROM finding_proposal_receipts AS receipt WHERE run_id = $1`, f.run.RunID).Scan(&original))
	mustCompletion(t, f.runs.DeleteReleasedTerminalRun(f.ctx, "owner", f.run.RunID))
	after, err := f.audits.Get(f.ctx, "owner", f.id)
	mustCompletion(t, err)
	if after.Revision != before.Revision+1 || !after.UpdatedAt.After(before.UpdatedAt) {
		t.Fatal("execution and multiple native receipts must invalidate their Audit exactly once")
	}
	var retained []byte
	var discarded int
	mustCompletion(t, f.pool.QueryRow(f.ctx, `
SELECT jsonb_agg(to_jsonb(receipt) ORDER BY receipt_id)
FROM finding_proposal_receipts AS receipt WHERE run_id = $1`, f.run.RunID).Scan(&retained))
	// jsonb has a canonical encoding, including the original exact refs.
	if string(original) != string(retained) {
		t.Fatal("Run deletion changed immutable receipt identity or source provenance")
	}
	mustCompletion(t, f.pool.QueryRow(f.ctx, `
SELECT count(*) FROM finding_proposal_retention AS retention
JOIN finding_proposal_receipts AS receipt USING (receipt_id)
WHERE receipt.run_id = $1 AND retention.state = 'discarded'
  AND retention.source_run_deleted_at IS NOT NULL
  AND retention.discarded_at IS NOT NULL`, f.run.RunID).Scan(&discarded))
	if discarded != 2 {
		t.Fatalf("unheld native tombstones = %d, want 2", discarded)
	}
}
