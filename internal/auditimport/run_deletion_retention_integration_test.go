//go:build integration

package auditimport

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestPostgresContractInvalidCollectionRetainsProposalAfterRunDeletion(t *testing.T) {
	f := newCompletionFixtureWithInteraction(t, completionPool(t), "automatic", "human-required")
	receiptID := f.proposal(t, "candidate", auditdomain.FindingProposal{
		Schema: auditdomain.FindingProposalSchema, ClientKey: "candidate",
		Title: "Candidate", Description: "Check the source behavior",
		Subject:        &auditdomain.FindingSubject{Kind: "code", Key: "handler"},
		Preconditions:  []string{},
		StandardRefs:   []auditdomain.StandardReference{},
		EvidenceIDs:    []string{},
		ProposedChecks: []auditdomain.ProposedCheck{},
		Limitations:    []string{},
	})

	execution, snapshot := f.terminal(t, runstore.RunFailed)
	access, err := NewArtifactAccess(f.artifacts, hookstore.NewClaimedArtifacts(f.pool))
	mustCompletion(t, err)
	intake, err := findingintake.New(f.pool)
	mustCompletion(t, err)
	importer, err := New(f.audits, f.audits, f.runs,
		invalidTaskArtifactAccess{ArtifactAccess: access, task: f.tasks[0].Ref}, intake)
	mustCompletion(t, err)
	changed, err := importer.Collect(f.ctx, f.claim, snapshot, execution)
	if err != nil || !changed {
		t.Fatalf("contract-invalid collection = (%t, %v)", changed, err)
	}
	var disposition string
	mustCompletion(t, f.pool.QueryRow(f.ctx, `
SELECT collection_disposition FROM audit_executions WHERE execution_id = $1`,
		execution.ExecutionID).Scan(&disposition))
	if disposition != string(auditstore.CollectionContractInvalid) {
		t.Fatalf("collection disposition = %q", disposition)
	}
	mustCompletion(t, rundeletion.New(f.pool).DeleteReleasedTerminalRun(f.ctx, "owner", f.run.RunID))
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

func (a invalidTaskArtifactAccess) WithClaim(claim auditstore.ControllerClaim) ArtifactAccess {
	a.ArtifactAccess = a.ArtifactAccess.WithClaim(claim)
	return a
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
	// those paths have separate real-service import/delete regressions. A
	// receipt holds its proposal, which must therefore be a stored revision.
	candidate, err := artifactstest.WriteFindingProposal(f.ctx, f.artifacts, f.run.RunID, "candidate",
		artifacts.Payload{MediaType: "application/json", Data: []byte("{}")})
	mustCompletion(t, err)
	proposalRef, err := json.Marshal(candidate.Ref)
	mustCompletion(t, err)
	for _, suffix := range []string{"one", "two"} {
		id := f.id + "-" + suffix
		_, err := f.pool.Exec(f.ctx, `
INSERT INTO finding_proposal_receipts (
    receipt_id, proposal_id, allocation_id, runtime_agent_id, runtime_instance_id,
    stage_execution_id, logical_agent_name, invocation_id, submission_id,
    client_key, request_digest, run_id, owner_id, project_id,
    audit_execution_id, audit_id, audit_role, workflow_name, workflow_version,
    workflow_schema_version, workflow_configuration_ref, workflow_closure_digest,
    proposal_ref, proposal_digest, proposal_media_type, proposal_size_bytes, evidence,
    report_projection
) VALUES (
    $1, $1, $1, 'runtime', 'instance', 'stage', 'worker', $1, $1,
    $1, $2, $3, 'owner', $4, $5, $4, 'check', 'workflow', '1',
    'contractor/v1alpha1', '{}', $2,
    $6::jsonb,
    $2, 'application/json', 2, '[]', '{"description":"Test finding.","limitations":[],"severity_suggestion":"","standard_refs":[],"subject":null,"title":"Test finding."}'
)`, id, completionDigest([]byte(id)), f.run.RunID, f.id, f.execution.ExecutionID, string(proposalRef))
		mustCompletion(t, err)
	}
	access, err := NewArtifactAccess(f.artifacts, hookstore.NewClaimedArtifacts(f.pool))
	mustCompletion(t, err)
	importer, err := New(f.audits, f.audits, f.runs, access)
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
SELECT jsonb_agg(to_jsonb(receipt) - ARRAY['retention_state', 'source_run_deleted_at', 'discarded_at', 'retention_updated_at', 'report_projection', 'fingerprint_projection', 'anchor_projection', 'comparison_projection']
                 ORDER BY receipt_id)
FROM finding_proposal_receipts AS receipt WHERE run_id = $1`, f.run.RunID).Scan(&original))
	mustCompletion(t, rundeletion.New(f.pool).DeleteReleasedTerminalRun(f.ctx, "owner", f.run.RunID))
	after, err := f.audits.Get(f.ctx, "owner", f.id)
	mustCompletion(t, err)
	if after.Revision != before.Revision+1 || !after.UpdatedAt.After(before.UpdatedAt) {
		t.Fatal("execution and multiple native receipts must invalidate their Audit exactly once")
	}
	var retained []byte
	var discarded int
	var incorrectDiscardCache int
	mustCompletion(t, f.pool.QueryRow(f.ctx, `
SELECT count(*) FROM finding_proposal_receipts
WHERE run_id=$1 AND retention_state='discarded'
AND (comparison_projection->>'availability' IS DISTINCT FROM 'unavailable'
 OR comparison_projection->>'reason' IS DISTINCT FROM 'discarded')`, f.run.RunID).Scan(&incorrectDiscardCache))
	if incorrectDiscardCache != 0 {
		t.Fatal("discarded receipts retained comparison matching hints")
	}
	mustCompletion(t, f.pool.QueryRow(f.ctx, `
SELECT jsonb_agg(to_jsonb(receipt) - ARRAY['retention_state', 'source_run_deleted_at', 'discarded_at', 'retention_updated_at', 'report_projection', 'fingerprint_projection', 'anchor_projection', 'comparison_projection']
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
  AND receipt.discarded_at IS NOT NULL
  AND receipt.report_projection = '{}'
  AND receipt.anchor_projection::jsonb = '{"schema":"contractor.finding.anchors-projection.v1","availability":"unavailable","reason":"discarded"}'::jsonb
  AND receipt.fingerprint_projection::jsonb = '{"schema":"contractor.finding.fingerprint-projection.v1","availability":"unavailable","reason":"discarded"}'::jsonb`, f.run.RunID).Scan(&discarded))
	if discarded != 2 {
		t.Fatalf("unheld native tombstones = %d, want 2", discarded)
	}
}

func TestImporterFinalizesHeldFindingReportAfterSourceRunDeletion(t *testing.T) {
	f := newCompletionFixtureWithInteraction(t, completionPool(t), "automatic", "human-required")
	document := auditdomain.FindingProposal{
		Schema: auditdomain.FindingProposalSchema, ClientKey: "retained-candidate",
		Title: "Retained candidate", Description: "Exact report details survive the source Run",
		Subject:       &auditdomain.FindingSubject{Kind: "code", Key: "handler"},
		Preconditions: []string{}, StandardRefs: []auditdomain.StandardReference{},
		EvidenceIDs: []string{}, ProposedChecks: []auditdomain.ProposedCheck{}, Limitations: []string{},
	}
	receiptID := f.proposal(t, document.ClientKey, document)
	access, err := NewArtifactAccess(f.artifacts, hookstore.NewClaimedArtifacts(f.pool))
	mustCompletion(t, err)
	intake, err := findingintake.New(f.pool)
	mustCompletion(t, err)
	importer, err := New(f.audits, f.audits, f.runs, access, intake)
	mustCompletion(t, err)
	execution, snapshot := f.terminal(t, runstore.RunFailed)
	if worked, err := importer.Collect(f.ctx, f.claim, snapshot, execution); err != nil || !worked {
		t.Fatalf("collect retained candidate: worked=%v, err=%v", worked, err)
	}
	finishReportDeletionFixture(t, f)
	mustCompletion(t, rundeletion.New(f.pool).DeleteReleasedTerminalRun(f.ctx, "owner", f.run.RunID))
	var retention string
	mustCompletion(t, f.pool.QueryRow(f.ctx, "SELECT retention_state FROM finding_proposal_receipts WHERE receipt_id=$1", receiptID).Scan(&retention))
	if retention != string(findingintake.RetentionAuditHeld) {
		t.Fatalf("imported candidate after deletion = %s", retention)
	}
	snapshot, err = f.audits.GetReconcileSnapshot(f.ctx, f.claim)
	mustCompletion(t, err)
	if worked, err := importer.Finalize(f.ctx, f.claim, snapshot); err != nil || !worked {
		t.Fatalf("finalize retained candidate: worked=%v, err=%v", worked, err)
	}
	link, err := f.audits.GetArtifactLink(f.ctx, f.id, auditstore.ReportMachineLogicalKey)
	mustCompletion(t, err)
	data, err := access.ReadProjectExact(f.ctx, f.id, link.Artifact)
	mustCompletion(t, err)
	var report struct {
		Findings struct {
			Proposed []reportFinding `json:"proposed"`
		} `json:"findings"`
	}
	mustCompletion(t, json.Unmarshal(data, &report))
	if len(report.Findings.Proposed) != 1 || report.Findings.Proposed[0].Title != document.Title ||
		report.Findings.Proposed[0].Description != document.Description {
		t.Fatalf("final report lost imported finding: %+v", report.Findings.Proposed)
	}
}
