//go:build integration

package auditimport

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestPostgresContractInvalidCollectionRetainsProposalAfterRunDeletion(t *testing.T) {
	f := newCompletionFixtureWithInteraction(t, completionPool(t), "automatic", "human-required")
	proposal, err := auditdomain.EncodeFindingProposal(auditdomain.FindingProposal{
		Schema: auditdomain.FindingProposalSchema, ClientKey: "candidate",
		Title: "Candidate", Description: "Check the source behavior",
		Subject:        &auditdomain.FindingSubject{Kind: "code", Key: "handler"},
		Preconditions:  []string{},
		StandardRefs:   []auditdomain.StandardReference{},
		EvidenceIDs:    []string{},
		ProposedChecks: []auditdomain.ProposedCheck{},
		Limitations:    []string{},
	})
	mustCompletion(t, err)
	written, err := f.artifacts.WriteFindingProposal(f.ctx, f.run.RunID,
		"candidate", artifacts.Payload{MediaType: "application/json", Data: proposal})
	mustCompletion(t, err)
	proposalRef, err := json.Marshal(written.Ref)
	mustCompletion(t, err)
	receiptID := f.id + "-finding-receipt"
	_, err = f.pool.Exec(f.ctx, `
INSERT INTO finding_proposal_receipts (
    receipt_id, proposal_id, allocation_id, runtime_agent_id, runtime_instance_id,
    stage_execution_id, logical_agent_name, invocation_id, submission_id,
    client_key, request_digest, run_id, owner_id, project_id,
    audit_execution_id, audit_id, audit_role, workflow_name, workflow_version,
    workflow_schema_version, workflow_configuration_ref, workflow_closure_digest,
    proposal_ref, proposal_digest, proposal_media_type, proposal_size_bytes, evidence
) VALUES (
    $1, $2, $3, 'runtime', 'instance', 'stage', 'checker', $3, $3,
    'candidate', $4, $5, 'owner', $6, $7, $6, 'check', $8, $9,
    'contractor/v1alpha1', '{}', $10, $11::jsonb, $4, 'application/json', $12, '[]'
)`, receiptID, f.id+"-proposal", f.id+"-submission", completionDigest(proposal),
		f.run.RunID, f.id, f.execution.ExecutionID, f.run.WorkflowName,
		f.run.WorkflowVersion, completionDigest(f.run.WorkflowSnapshot),
		string(proposalRef), len(proposal))
	mustCompletion(t, err)

	mustCompletion(t, f.artifacts.FreezeRunOutputs(f.ctx, f.run.RunID))
	_, err = f.runs.TransitionRun(f.ctx, f.run.RunID, runstore.RunRunning,
		runstore.RunFailed, runstore.Reason{Code: "fixture-terminal"})
	mustCompletion(t, err)
	cursor, err := f.runs.GetRunEventCursor(f.ctx, f.run.RunID)
	mustCompletion(t, err)
	execution, err := f.audits.ObserveTerminal(f.ctx, auditstore.ObserveTerminalParams{
		Claim: f.claim, ExecutionID: f.execution.ExecutionID,
		RunID: f.run.RunID, Generation: cursor.Generation, Sequence: uint64(cursor.Sequence),
	})
	mustCompletion(t, err)
	snapshot, err := f.audits.GetReconcileSnapshot(f.ctx, f.claim)
	mustCompletion(t, err)
	access, err := NewArtifactAccess(f.artifacts)
	mustCompletion(t, err)
	intake, err := findingintake.New(f.pool)
	mustCompletion(t, err)
	importer, err := New(f.audits, f.runs,
		invalidTaskArtifactAccess{ArtifactAccess: access, task: f.tasks[0].Ref}, intake)
	mustCompletion(t, err)
	changed, err := importer.Collect(f.ctx, f.claim, snapshot, execution)
	if err != nil || !changed {
		t.Fatalf("contract-invalid collection = (%t, %v)", changed, err)
	}
	var disposition string
	mustCompletion(t, f.pool.QueryRow(f.ctx, `
SELECT disposition FROM audit_collection_receipts WHERE execution_id = $1`,
		execution.ExecutionID).Scan(&disposition))
	if disposition != string(auditstore.CollectionContractInvalid) {
		t.Fatalf("collection disposition = %q", disposition)
	}
	mustCompletion(t, f.runs.DeleteReleasedTerminalRun(f.ctx, "owner", f.run.RunID))
	var retention string
	mustCompletion(t, f.pool.QueryRow(f.ctx, `
SELECT retention_state FROM finding_proposal_receipts WHERE receipt_id = $1`, receiptID).Scan(&retention))
	if retention != string(findingintake.RetentionAuditHeld) {
		t.Fatalf("proposal retention after source Run deletion = %q", retention)
	}
}

type invalidTaskArtifactAccess struct {
	ArtifactAccess
	task contracts.ArtifactRef
}

func (a invalidTaskArtifactAccess) ReadProjectExact(
	ctx context.Context, projectID string, expected auditstore.ExactArtifact,
) ([]byte, error) {
	if expected.Ref.SameExact(a.task) {
		return []byte("not an Audit task package"), nil
	}
	return a.ArtifactAccess.ReadProjectExact(ctx, projectID, expected)
}

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
SELECT jsonb_agg(to_jsonb(receipt) - ARRAY['retention_state', 'source_run_deleted_at', 'discarded_at', 'retention_updated_at']
                 ORDER BY receipt_id)
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
SELECT jsonb_agg(to_jsonb(receipt) - ARRAY['retention_state', 'source_run_deleted_at', 'discarded_at', 'retention_updated_at']
                 ORDER BY receipt_id)
FROM finding_proposal_receipts AS receipt WHERE run_id = $1`, f.run.RunID).Scan(&retained))
	// jsonb has a canonical encoding, including the original exact refs.
	if string(original) != string(retained) {
		t.Fatal("Run deletion changed immutable receipt identity or source provenance")
	}
	mustCompletion(t, f.pool.QueryRow(f.ctx, `
SELECT count(*) FROM finding_proposal_receipts AS receipt
WHERE receipt.run_id = $1 AND receipt.retention_state = 'discarded'
  AND receipt.source_run_deleted_at IS NOT NULL
  AND receipt.discarded_at IS NOT NULL`, f.run.RunID).Scan(&discarded))
	if discarded != 2 {
		t.Fatalf("unheld native tombstones = %d, want 2", discarded)
	}
}
