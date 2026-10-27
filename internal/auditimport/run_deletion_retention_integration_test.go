//go:build integration

package auditimport

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
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
    $2, 'application/json', 2, '[]', '{"description":"Test finding.","limitations":[],"preconditions":[],"severity_suggestion":"","standard_refs":[],"subject":null,"title":"Test finding."}'
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
SELECT jsonb_agg(to_jsonb(receipt) - ARRAY['retention_state', 'source_run_deleted_at', 'discarded_at', 'retention_updated_at', 'report_projection', 'fingerprint_projection', 'anchor_projection', 'comparison_projection', 'checks_projection', 'checks_projection_bytes', 'checks_projection_count']
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
SELECT jsonb_agg(to_jsonb(receipt) - ARRAY['retention_state', 'source_run_deleted_at', 'discarded_at', 'retention_updated_at', 'report_projection', 'fingerprint_projection', 'anchor_projection', 'comparison_projection', 'checks_projection', 'checks_projection_bytes', 'checks_projection_count']
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
	// The new private index is another erased cache, not immutable receipt
	// provenance. Both unheld tombstones retain only their exact digest/size
	// header, and their generated cost headers describe those stored bytes.
	rows, err := f.pool.Query(f.ctx, `
SELECT receipt_id, checks_projection, checks_projection_count, checks_projection_bytes,
       proposal_digest, proposal_size_bytes
  FROM finding_proposal_receipts WHERE run_id=$1 ORDER BY receipt_id`, f.run.RunID)
	mustCompletion(t, err)
	defer rows.Close()
	indexedTombstones := 0
	for rows.Next() {
		var id, digest string
		var data []byte
		var count, bytes int
		var size int64
		mustCompletion(t, rows.Scan(&id, &data, &count, &bytes, &digest, &size))
		projection, err := auditdomain.DecodeFindingChecksProjection(data)
		if err != nil || !projection.Discarded || projection.Subject != nil || projection.Limitations != nil || len(projection.Checks) != 0 ||
			projection.ProposalDigest != digest || projection.SizeBytes != size || count != 0 || bytes != len(data) || bytes < 1 || bytes > auditdomain.MaximumFindingChecksProjectionBytes {
			t.Fatalf("discarded check index %s: discarded=%v checks=%d count=%d bytes=%d actual=%d error=%v", id, projection.Discarded, len(projection.Checks), count, bytes, len(data), err)
		}
		indexedTombstones++
	}
	mustCompletion(t, rows.Err())
	if indexedTombstones != 2 {
		t.Fatalf("discarded check index tombstones = %d, want 2", indexedTombstones)
	}
}

func TestImporterFinalizesHeldFindingReportAfterSourceRunDeletion(t *testing.T) {
	f := newCompletionFixtureWithInteraction(t, completionPool(t), "automatic", "human-required")
	document := auditdomain.FindingProposal{
		Schema: auditdomain.FindingProposalSchema, ClientKey: "retained-candidate",
		Title: "Retained candidate", Description: "Exact report details survive the source Run",
		Subject:       &auditdomain.FindingSubject{Kind: "code", Key: "handler"},
		Preconditions: []string{"Доступна роль читателя", "The account already owns an order."}, StandardRefs: []auditdomain.StandardReference{},
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
		report.Findings.Proposed[0].Description != document.Description ||
		!slices.Equal(report.Findings.Proposed[0].Preconditions, document.Preconditions) {
		t.Fatalf("final report lost imported finding: %+v", report.Findings.Proposed)
	}
}

// The report reader selects the first receipt even when another retained
// contribution has different conditions. Intake and human confirmation run
// through the real services; only the existing contribution relation is seeded.
func TestImporterReportsFirstReceiptPreconditionsAfterHumanConfirmation(t *testing.T) {
	f := newCompletionFixtureWithOptions(t, completionPool(t), completionOptions{acceptance: "automatic", findingConfirmation: "human-required", findingWriter: true})
	first := auditdomain.FindingProposal{
		Schema: auditdomain.FindingProposalSchema, ClientKey: "first-condition",
		Title: "Candidate", Description: "The original observation.",
		Subject:       &auditdomain.FindingSubject{Kind: "code", Key: "handler"},
		Preconditions: []string{"Только для приглашённого читателя", "The account owns an order."},
		StandardRefs:  []auditdomain.StandardReference{}, EvidenceIDs: []string{},
		ProposedChecks: []auditdomain.ProposedCheck{}, Limitations: []string{},
	}
	submit, release := realFindingReportIntake(t, f)
	receiptID := submit(first)
	later := first
	later.ClientKey, later.Title = "later-condition", "Later contribution"
	later.Preconditions = []string{"A different condition from a later source."}
	laterReceipt := submit(later)
	release()
	access, err := NewArtifactAccess(f.artifacts, hookstore.NewClaimedArtifacts(f.pool))
	mustCompletion(t, err)
	intake, err := findingintake.New(f.pool)
	mustCompletion(t, err)
	importer, err := New(f.audits, f.audits, f.runs, access, intake)
	mustCompletion(t, err)
	execution, snapshot := f.terminal(t, runstore.RunFailed)
	if worked, err := importer.Collect(f.ctx, f.claim, snapshot, execution); err != nil || !worked {
		t.Fatalf("collect actual conditions: worked=%t err=%v", worked, err)
	}
	var findingID, laterID string
	mustCompletion(t, f.pool.QueryRow(f.ctx, "SELECT finding_id FROM audit_findings WHERE audit_id=$1 AND first_receipt_id=$2", f.id, receiptID).Scan(&findingID))
	mustCompletion(t, f.pool.QueryRow(f.ctx, "SELECT finding_id FROM audit_findings WHERE audit_id=$1 AND first_receipt_id=$2", f.id, laterReceipt).Scan(&laterID))
	// Match the same historical contributing-source fixture used by the
	// correlation/collection readers, without changing first-receipt authority.
	_, err = f.pool.Exec(f.ctx, "DELETE FROM audit_findings WHERE finding_id=$1", laterID)
	mustCompletion(t, err)
	_, err = f.pool.Exec(f.ctx, `INSERT INTO audit_finding_contributions(finding_id,audit_id,receipt_id,relation,proposal_ref)
SELECT $1,$2,receipt_id,'contributing',proposal_ref FROM finding_proposal_audit_holds WHERE receipt_id=$3 AND audit_id=$2`, findingID, f.id, laterReceipt)
	mustCompletion(t, err)
	review := reportDeletionReviewService(t, f)
	request, err := review.CreateFindingReview(f.ctx, auditservice.CreateFindingReviewParams{
		OwnerID: "owner", AuditID: f.id, FindingID: findingID, ExpectedRevision: 1,
		RequestID: "confirm-conditions", IdempotencyKey: "confirm-conditions", RequestDigest: completionDigest([]byte("confirm-conditions")),
	})
	mustCompletion(t, err)
	severity := auditstore.SeverityHigh
	decision, err := review.DecideFinding(f.ctx, auditservice.DecideFindingParams{
		OwnerID: "owner", AuditID: f.id, RequestID: request.Request.RequestID,
		ExpectedRequestRevision: request.Request.Revision, DecisionID: "confirmed-conditions",
		Verdict: auditstore.VerdictTruePositive, Severity: &severity, Rationale: "Independent human verification.",
		IdempotencyKey: "confirmed-conditions", RequestDigest: completionDigest([]byte("confirmed-conditions")),
	})
	mustCompletion(t, err)
	if decision.Finding.State != auditstore.FindingConfirmed {
		t.Fatal("finding was not human-confirmed")
	}
	finishReportDeletionFixture(t, f)
	mustCompletion(t, rundeletion.New(f.pool).DeleteReleasedTerminalRun(f.ctx, "owner", f.run.RunID))
	snapshot, err = f.audits.GetReconcileSnapshot(f.ctx, f.claim)
	mustCompletion(t, err)
	if worked, err := importer.Finalize(f.ctx, f.claim, snapshot); err != nil || !worked {
		t.Fatalf("finalize human-confirmed conditions: worked=%t err=%v", worked, err)
	}
	link, err := f.audits.GetArtifactLink(f.ctx, f.id, auditstore.ReportMachineLogicalKey)
	mustCompletion(t, err)
	data, err := access.ReadProjectExact(f.ctx, f.id, link.Artifact)
	mustCompletion(t, err)
	var report struct {
		Findings struct {
			Confirmed []reportFinding `json:"confirmed"`
		} `json:"findings"`
	}
	mustCompletion(t, json.Unmarshal(data, &report))
	if len(report.Findings.Confirmed) != 1 || !slices.Equal(report.Findings.Confirmed[0].Preconditions, first.Preconditions) || report.Findings.Confirmed[0].AnalystDecision == nil {
		t.Fatalf("confirmed report lost first conditions: %+v", report.Findings.Confirmed)
	}
	// Review may continue after publication. Its new state never rewrites the
	// already accepted report artifact or its original conditions.
	reopened, err := review.CreateFindingReview(f.ctx, auditservice.CreateFindingReviewParams{
		OwnerID: "owner", AuditID: f.id, FindingID: findingID, ExpectedRevision: decision.Finding.Revision,
		RequestID: "reopen-conditions", IdempotencyKey: "reopen-conditions", RequestDigest: completionDigest([]byte("reopen-conditions")),
	})
	mustCompletion(t, err)
	_, err = review.DecideFinding(f.ctx, auditservice.DecideFindingParams{
		OwnerID: "owner", AuditID: f.id, RequestID: reopened.Request.RequestID,
		ExpectedRequestRevision: reopened.Request.Revision, DecisionID: "reopened-conditions",
		Verdict: auditstore.VerdictReopen, Rationale: "Request additional investigation after publication.",
		IdempotencyKey: "reopened-conditions", RequestDigest: completionDigest([]byte("reopened-conditions")),
	})
	mustCompletion(t, err)
	after, err := access.ReadProjectExact(f.ctx, f.id, link.Artifact)
	mustCompletion(t, err)
	if !bytes.Equal(data, after) {
		t.Fatal("later review rewrote immutable accepted report bytes")
	}
}

// realFindingReportIntake admits proposals through the production service with
// a real persisted StageExecution and selected allocation, rather than seeding
// the report projection under test. Only Scheduler/Control Plane admission is
// supplied by this deterministic single-writer fixture.
func realFindingReportIntake(t *testing.T, f *completionFixture) (func(auditdomain.FindingProposal) string, func()) {
	t.Helper()
	stage := f.profile.Workflows["check"].Workflow.Stages["check"]
	binding := stage.Agents["checker"]
	encoded, err := json.Marshal(stage)
	mustCompletion(t, err)
	execution, err := f.runs.CreateStageExecution(f.ctx, runstore.CreateStageExecutionParams{
		LaneClass: resolved.LaneClassModel, StageExecutionID: f.id + "-finding-stage", RunID: f.run.RunID, StageName: "check", Attempt: 1,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: encoded,
		StageContextSchemaVersion: contracts.APIVersion,
		StageContext: runstore.StageContextSnapshot{Parameters: map[string]string{}, Artifacts: map[string]runstore.PinnedContextArtifact{
			"task":     {Required: true, Artifact: &f.run.AuditCompletion.Contract.Task},
			"manifest": {Required: true, Artifact: &f.run.AuditCompletion.Contract.ExecutionManifest},
		}},
	})
	mustCompletion(t, err)
	gateway := stage.ExecutionConfig.Agents["checker"].LLMGateway.Ref
	allocation := runstore.StageAllocation{
		AllocationID: f.id + "-finding-allocation", StageExecutionID: execution.StageExecutionID,
		LogicalAgentName: "checker", Namespace: binding.Namespace, AgentTemplateRef: binding.Template.Ref, WorkerRuntimeRef: binding.Template.Runtime,
		RuntimeAgentID: strings.Repeat("a", 64), RuntimeAgentInstanceID: "runtime-conditions", RuntimeAgentLabelRevision: 1,
		RuntimeConfigurationSchemaVersion: runstore.AllocationRuntimeConfigurationSchemaVersion,
		PerformanceCollectionPolicy:       reporting.PerformanceCollectionDisabled, CompletionContract: &f.run.AuditCompletion.Contract,
		RuntimeConfiguration: &runstore.AllocationRuntimeConfiguration{
			ModelPolicy: stage.ExecutionConfig.Agents["checker"].ModelPolicy.Ref,
			Origins:     runtimeconfig.ResolvedRuntimeConfigOrigins{},
			Provenance: runtimesettings.ResolvedRuntimeConfigProvenance{
				Default:   runtimesettings.RuntimeLabelBindingProvenance{Label: "default", BindingRevision: 1, Config: runtimesettings.RuntimeConfigRef{Name: runtimeconfig.BuiltInName, Version: runtimeconfig.BuiltInVersion, Digest: runtimeconfig.BuiltInDigest}},
				RunLabels: []runtimesettings.RuntimeLabelBindingProvenance{}, AgentLabels: []runtimesettings.RuntimeLabelBindingProvenance{},
				RuntimeAdapters: []runtimesettings.RuntimeAdapterRef{}, RuntimeCredentialRefs: []runtimesettings.RuntimeCredentialRef{}, LLMGatewayConfig: &gateway,
			},
		},
	}
	mustCompletion(t, f.runs.RecordStageAllocation(f.ctx, allocation))
	grant := controlplane.AllocationGrant{AllocationID: allocation.AllocationID, RuntimeAgentID: allocation.RuntimeAgentID, RuntimeInstanceID: allocation.RuntimeAgentInstanceID,
		RunID: f.run.RunID, StageExecutionID: execution.StageExecutionID, LogicalAgentName: "checker", Namespace: binding.Namespace}
	intake, err := findingintake.New(f.pool)
	mustCompletion(t, err)
	submit := func(document auditdomain.FindingProposal) string {
		t.Helper()
		submissionDigest := sha256.Sum256([]byte("contractor.finding.submission.v1\x00report-conditions\x00" + document.ClientKey))
		receipt, replayed, err := intake.Submit(f.ctx, grant, findingintake.Submission{
			APIVersion: contracts.APIVersion, InvocationID: "report-conditions", SubmissionID: "finding-" + hex.EncodeToString(submissionDigest[:]),
			Proposal: document, EvidenceRefs: []contracts.ArtifactRef{},
		})
		mustCompletion(t, err)
		if replayed {
			t.Fatal("new condition proposal unexpectedly replayed")
		}
		return receipt.ReceiptID
	}
	return submit, func() { mustCompletion(t, f.runs.MarkStageAllocationReleased(f.ctx, allocation.AllocationID)) }
}
