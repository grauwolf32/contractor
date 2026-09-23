package auditstore

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresCollectionReportsEvidenceBudgetExhaustion(t *testing.T) {
	f := newCollectingAuditFixture(t, "budget", 64)
	result := testExact("audit-budget", "result", "result-r1")
	result.SizeBytes = 48
	evidence := testExact("audit-budget", "evidence", "evidence-r1")
	evidence.SizeBytes = 32
	provenance := json.RawMessage(`{"runId":"run-budget"}`)
	_, inserted, err := f.store.Collect(f.ctx, CollectParams{
		Claim: f.claim, ReceiptID: "receipt-budget", ExecutionID: f.execution.ExecutionID,
		Disposition: CollectionAccepted, SourceOutput: &result, RequestDigest: testDigest("5"),
		Retained: []ArtifactLink{
			{LogicalKey: "result/" + f.memberID, Artifact: result, SourceProvenance: provenance},
			{LogicalKey: "evidence/budget/one", Artifact: evidence, SourceProvenance: provenance},
			{LogicalKey: "evidence/budget/two", Artifact: evidence, SourceProvenance: provenance},
		},
		Items: []CollectionItem{{
			ExecutionItemID: f.memberID, Disposition: CollectionAccepted,
			FinalDisposition: FinalAccepted, Result: &result,
			Coverage: Coverage{Status: CoverageSatisfied, Requested: []string{}, Completed: []string{}, Gaps: []string{}},
		}},
	})
	if !errors.Is(err, ErrEvidenceBudgetExhausted) || !errors.Is(err, ErrPrecondition) || inserted {
		t.Fatalf("over-budget collection = (%t, %v), want evidence budget precondition", inserted, err)
	}
	code := "evidence-budget-exhausted"
	receipt, inserted, err := f.store.Collect(f.ctx, CollectParams{
		Claim: f.claim, ReceiptID: "receipt-budget", ExecutionID: f.execution.ExecutionID,
		Disposition: CollectionInvalidResult, SourceOutput: &result, ErrorCode: &code,
		RequestDigest: testDigest("6"), Retained: []ArtifactLink{},
		Items: []CollectionItem{{
			ExecutionItemID: f.memberID, Disposition: CollectionInvalidResult,
			FinalDisposition: FinalInvalidResult,
			Coverage:         Coverage{Status: CoverageInconclusive, Requested: []string{}, Completed: []string{}, Gaps: []string{code}},
		}},
	})
	if err != nil || !inserted || receipt.Disposition != CollectionInvalidResult {
		t.Fatalf("terminal budget receipt = (%+v, %t, %v)", receipt, inserted, err)
	}
	audit, err := f.store.Get(f.ctx, f.audit.OwnerID, f.audit.AuditID)
	if err != nil || audit.RetainedEvidenceBytes != 0 {
		t.Fatalf("budget receipt charged evidence = (%+v, %v)", audit, err)
	}
}

func TestPostgresCollectionExpiresStaleFindingTriageRequests(t *testing.T) {
	f := newCollectingAuditFixture(t, "triage", 1<<20)
	proposal := testExact("audit-findings", "candidate-triage", "proposal-r1")
	proposal.MediaType = "application/json"
	proposal.SizeBytes = 128
	findingID := insertAuditChildFinding(t, f, "receipt-triage", proposal)
	if _, err := f.pool.Exec(f.ctx, `
INSERT INTO audit_review_requests (
    request_id, audit_id, finding_id, subject_kind, subject_id, kind,
    subject_revision, subject_digest, requested_actions, idempotency_key, request_digest
) VALUES ('review-triage', $1, $2, 'finding', $2, 'finding-triage', 1, $3,
          '["true_positive","false_positive"]'::jsonb, 'review-triage', $3)`,
		f.audit.AuditID, findingID, testDigest("8")); err != nil {
		t.Fatal(err)
	}
	result := testExact("audit-triage", "result", "result-r1")
	if _, inserted, err := f.store.Collect(f.ctx, CollectParams{
		Claim: f.claim, ReceiptID: "receipt-collection-triage", ExecutionID: f.execution.ExecutionID,
		Disposition: CollectionAccepted, SourceOutput: &result, RequestDigest: testDigest("9"),
		Retained: []ArtifactLink{{
			LogicalKey: "result/" + f.memberID, Artifact: result,
			SourceProvenance: json.RawMessage(`{"runId":"run-triage"}`),
		}},
		Items: []CollectionItem{{
			ExecutionItemID: f.memberID, Disposition: CollectionAccepted,
			FinalDisposition: FinalAccepted, Result: &result,
			Coverage: Coverage{Status: CoverageSatisfied, Requested: []string{}, Completed: []string{}, Gaps: []string{}},
			FindingAssociations: []FindingAssociation{{
				AssessmentID: "assessment-triage", ReceiptID: "receipt-triage",
				Proposal: proposal, SemanticAssessment: "supported",
			}},
		}},
	}); err != nil || !inserted {
		t.Fatalf("accepted collection = (%t, %v)", inserted, err)
	}
	var requestState string
	var requestRevision, findingRevision int
	if err := f.pool.QueryRow(f.ctx, `
SELECT request.state, request.revision, finding.revision
  FROM audit_review_requests AS request
  JOIN audit_findings AS finding USING (finding_id)
 WHERE request.request_id = 'review-triage'`).Scan(
		&requestState, &requestRevision, &findingRevision,
	); err != nil {
		t.Fatal(err)
	}
	if requestState != "expired" || requestRevision != 2 || findingRevision != 2 {
		t.Fatalf("stale triage request = %s revision %d, finding revision %d",
			requestState, requestRevision, findingRevision)
	}
	audit, err := f.store.Get(f.ctx, f.audit.OwnerID, f.audit.AuditID)
	if err != nil || audit.Revision != audit.EventSequence {
		t.Fatalf("Audit revision/event sequence = (%+v, %v)", audit, err)
	}
	events, err := f.store.ListEvents(f.ctx, f.audit.AuditID, 0, MaxPageSize)
	if err != nil || len(events) < 2 {
		t.Fatalf("Audit events = (%+v, %v)", events, err)
	}
	for index, event := range events {
		if event.Sequence != uint64(index+1) {
			t.Fatalf("Audit event %d has sequence %d", index, event.Sequence)
		}
	}
	collected, expired := events[len(events)-2], events[len(events)-1]
	if collected.Kind != "execution.collected" || expired.Kind != "review.expired" ||
		expired.EntityID != "review-triage" ||
		!strings.Contains(string(expired.Summary), `"findingId": "`+findingID+`"`) {
		t.Fatalf("collection events = %+v, %+v (%s)", collected, expired, expired.Summary)
	}
}

// insertAuditChildFinding records a proposal receipt from the fixture's child
// Run and its Audit hold, which admits the finding the collection assesses.
func insertAuditChildFinding(
	t *testing.T, f collectingAuditFixture, receiptID string, proposal ExactArtifact,
) string {
	t.Helper()
	proposalJSON, err := json.Marshal(proposal)
	if err != nil {
		t.Fatal(err)
	}
	proposalRefJSON, err := json.Marshal(proposal.Ref)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(f.ctx, `
INSERT INTO finding_proposal_receipts (
    receipt_id, proposal_id, allocation_id, runtime_agent_id,
    runtime_instance_id, stage_execution_id, logical_agent_name,
    invocation_id, submission_id, client_key, request_digest,
    run_id, owner_id, project_id, audit_execution_id, audit_id, audit_role,
    workflow_name, workflow_version, workflow_schema_version,
    workflow_configuration_ref, workflow_closure_digest,
    proposal_ref, proposal_digest, proposal_media_type,
    proposal_size_bytes, evidence
) VALUES (
    $1, $1 || '-proposal', $1 || '-allocation', 'runtime-agent', 'runtime-instance',
    'stage-execution', 'worker', 'invocation', $1 || '-submission', 'candidate', $2,
    $3, $4, $5, $6, $7, 'check',
    'audit-check', '1', 'contractor/v1alpha1',
    '{"name":"audit-check","version":"1"}'::jsonb, $8,
    $9::jsonb, $10, 'application/json', $11, '[]'::jsonb
)`,
		receiptID, testDigest("d"), *f.execution.RunID, f.audit.OwnerID, f.project.ProjectID,
		f.execution.ExecutionID, f.audit.AuditID, testDigest("e"),
		proposalRefJSON, proposal.Digest, proposal.SizeBytes,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(f.ctx, `
INSERT INTO finding_proposal_retention (receipt_id) VALUES ($1)`, receiptID); err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(f.ctx, `
INSERT INTO finding_proposal_audit_holds (
    receipt_id, audit_id, project_id, proposal_ref, evidence
) VALUES ($1, $2, $3, $4::jsonb, '[]'::jsonb)`,
		receiptID, f.audit.AuditID, f.project.ProjectID, proposalJSON); err != nil {
		t.Fatal(err)
	}
	var findingID string
	if err := f.pool.QueryRow(f.ctx, `
SELECT finding_id FROM audit_findings WHERE audit_id = $1 AND first_receipt_id = $2`,
		f.audit.AuditID, receiptID).Scan(&findingID); err != nil {
		t.Fatal(err)
	}
	return findingID
}

type collectingAuditFixture struct {
	ctx       context.Context
	pool      *pgxpool.Pool
	store     *PostgresStore
	project   projectstore.Project
	audit     Audit
	claim     ControllerClaim
	execution Execution
	memberID  string
	itemID    string
}

// newCollectingAuditFixture starts a one-item Audit and leaves its only
// execution collecting a succeeded Run under a live Controller claim.
func newCollectingAuditFixture(t *testing.T, name string, maxEvidenceBytes int64) collectingAuditFixture {
	t.Helper()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	t.Cleanup(cancel)
	pool := isolatedAuditPool(t, ctx, databaseURL)
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "project-" + name, OwnerID: "owner-" + name, Kind: projectstore.KindProject,
		Name: "Audit " + name, IdempotencyKey: "project-create", RequestDigest: testDigest("1"),
	})
	if err != nil {
		t.Fatal(err)
	}
	store := NewPostgresStore(pool)
	audit, _, err := store.CreateDraft(ctx, CreateDraftParams{
		AuditID: "audit-" + name, OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:         ProfileIdentity{Name: "checklist", Version: "1", Digest: testDigest("2")},
		ProfileSnapshot: json.RawMessage(`{"name":"checklist","workflows":{"check":{"kind":"check"}}}`),
		InputSelection:  json.RawMessage(`{}`),
		Limits: Limits{
			MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1,
			MaxSubmittedRuns: 2, MaxItemRunAttempts: 2, MaxEvidenceBytes: maxEvidenceBytes,
		},
		IdempotencyKey: "audit-create", RequestDigest: testDigest("3"),
	})
	if err != nil {
		t.Fatal(err)
	}
	roundID := "round-" + name
	task := testExact("audits", name+"-task", "task-r1")
	item := MaterializedItem{
		ItemID: "item-" + name, ItemKey: "check-" + name, Ordinal: 0, Kind: "checklist",
		SubjectKey: "subject-" + name, Task: task, Origin: testOrigin("check-" + name),
		WorkflowRole: "check", InitialState: ItemReady, Coverage: emptyCoverage(),
	}
	audit, _, err = store.MaterializeRound(ctx, MaterializeRoundParams{
		OwnerID: audit.OwnerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision,
		RoundID: roundID, RoundOrdinal: 1, Manifest: testExact("audits", name+"-worklist", "worklist-r1"),
		BaselineSnapshot: json.RawMessage(`{"inputs":[],"skills":[]}`),
		DeadlineAt:       time.Now().Add(time.Hour), IdempotencyKey: "audit-start",
		RequestDigest: testDigest("4"), Items: []MaterializedItem{item},
	})
	if err != nil {
		t.Fatal(err)
	}
	claims, err := store.Claim(ctx, ClaimParams{HolderID: name + "-controller", Lease: 30 * time.Second, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim Audit = (%+v, %v)", claims, err)
	}
	claim := claims[0]
	if _, err := store.TransitionRound(ctx, RoundTransitionParams{
		Claim: claim, RoundID: roundID, ExpectedRevision: 1,
		ExpectedState: RoundAccepted, TargetState: RoundExecuting,
	}); err != nil {
		t.Fatal(err)
	}
	memberID := "member-" + name
	execution, _, err := store.CreateExecutionIntent(ctx, CreateExecutionIntentParams{
		Claim: claim, ExecutionID: "execution-" + name, RoundID: &roundID, Role: ExecutionCheck,
		WorkflowRole: "check", Manifest: testExact("audits", name+"-execution", "manifest-r1"),
		SubmissionKey: "submission-" + name, RequestDigest: testDigest("7"),
		Members: []ExecutionMemberIntent{{
			ExecutionItemID: memberID, ItemID: item.ItemID, BatchOrdinal: 0, ItemAttempt: 1,
			Task: task, Inputs: []ExactArtifact{},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	runID := "run-" + name
	execution, err = insertAndBindTestRun(t, ctx, pool, claim, execution, runID, audit.OwnerID, project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	generation, sequence := terminateTestRun(t, ctx, pool, runID, "succeeded")
	execution, err = store.ObserveTerminal(ctx, ObserveTerminalParams{
		Claim: claim, ExecutionID: execution.ExecutionID, RunID: runID,
		Generation: generation, Sequence: sequence,
	})
	if err != nil {
		t.Fatal(err)
	}
	audit, err = store.Get(ctx, audit.OwnerID, audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	return collectingAuditFixture{
		ctx: ctx, pool: pool, store: store, project: project, audit: audit,
		claim: claim, execution: execution, memberID: memberID, itemID: item.ItemID,
	}
}
