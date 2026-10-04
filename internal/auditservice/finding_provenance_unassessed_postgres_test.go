package auditservice

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

func TestFindingProvenanceIncludesUnassessedProposalAttempts(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, databaseURL)
	const owner, projectID, auditID = "owner-unassessed", "project-unassessed", "audit-unassessed"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: projectID, OwnerID: owner, Kind: projectstore.KindProject,
		Name: "Unassessed provenance", IdempotencyKey: "project",
		RequestDigest: serviceTestDigest("project"),
	}); err != nil {
		t.Fatal(err)
	}
	if _, _, err := auditstore.NewPostgresStore(pool).CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: auditID, OwnerID: owner, ProjectID: projectID,
		Profile:         auditstore.ProfileIdentity{Name: "profile", Version: "1", Digest: serviceTestDigest("profile")},
		ProfileSnapshot: json.RawMessage(`{"interaction":{"findingConfirmation":"human-required"}}`),
		InputSelection:  json.RawMessage(`{}`),
		Limits: auditstore.Limits{
			MaxRounds: 2, BatchSize: 1, MaxItemsPerRound: 4, MaxItemsTotal: 4,
			MaxSubmittedRuns: 5, MaxItemRunAttempts: 2, MaxEvidenceBytes: 1 << 20,
		},
		IdempotencyKey: "audit", RequestDigest: serviceTestDigest("audit"),
	}); err != nil {
		t.Fatal(err)
	}
	checks := make([]auditdomain.ProposedCheck, 4)
	for ordinal := range checks {
		checks[ordinal] = auditdomain.ProposedCheck{
			Objective: fmt.Sprintf("Verify condition %d", ordinal), Method: "run-check",
		}
	}
	findingID := seedAuditFinding(t, ctx, pool, projectID, owner, auditID, "unassessed", checks...)
	if _, err := pool.Exec(ctx, `
INSERT INTO audit_rounds (
    round_id, audit_id, ordinal, manifest_ref, manifest_digest, state, expected_item_count
) VALUES (
    'round-unassessed', $1, 2,
    '{"namespace":"audit-rounds","name":"unassessed","revision":"round-r2"}'::jsonb,
    $2, 'closed', 4
)`, auditID, serviceTestDigest("round")); err != nil {
		t.Fatal(err)
	}
	type attemptCase struct {
		item        string
		attempt     int
		disposition auditstore.CollectionDisposition
		outcome     auditstore.TerminalOutcome
		runDeleted  bool
	}
	attempts := []attemptCase{
		{item: "failed", attempt: 1, disposition: auditstore.CollectionExecutionFailed, outcome: auditstore.TerminalFailed, runDeleted: true},
		{item: "failed", attempt: 2, disposition: auditstore.CollectionExecutionFailed, outcome: auditstore.TerminalFailed, runDeleted: true},
		{item: "missing", attempt: 1, disposition: auditstore.CollectionMissingOutput, outcome: auditstore.TerminalSucceeded, runDeleted: true},
		{item: "invalid", attempt: 1, disposition: auditstore.CollectionInvalidResult, outcome: auditstore.TerminalSucceeded, runDeleted: true},
		{item: "cancelled", attempt: 1, disposition: auditstore.CollectionExecutionCancelled, outcome: auditstore.TerminalCancelled, runDeleted: true},
	}
	for ordinal, item := range []struct {
		key   string
		final auditstore.FinalDisposition
	}{
		{"failed", auditstore.FinalExecutionFailed},
		{"missing", auditstore.FinalMissingOutput},
		{"invalid", auditstore.FinalInvalidResult},
		{"cancelled", auditstore.FinalExecutionCancelled},
	} {
		itemID := "item-unassessed-" + item.key
		taskRef := fmt.Sprintf(`{"namespace":"audit-task-packages","name":"check-%s","revision":"task-r1"}`, item.key)
		sourceRevision := "checks-r1"
		origin, err := json.Marshal(auditstore.ItemOrigin{
			Schema: auditstore.ItemOriginSchema, EntryKey: "check-" + item.key,
			SourceRef: &contracts.ArtifactRef{
				Namespace: "inputs", Name: "checks", Revision: &sourceRevision,
			},
			SourceContentDigest: serviceTestDigest("source"), SourceMediaType: "application/json",
			CanonicalInventoryDigest: serviceTestDigest("inventory"),
		})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(ctx, `
INSERT INTO audit_items (
    item_id, audit_id, round_id, item_key, ordinal, kind, subject_key,
    task_ref, task_digest, origin, workflow_role, state, final_disposition,
    coverage_status, coverage_requested, coverage_completed, coverage_gaps,
    proposal_receipt_id, proposal_check_ordinal, proposal_ref, proposal_digest
)
SELECT $1, $2, 'round-unassessed', $3, $4, 'check', $5,
       $6::jsonb, $7, $8::jsonb, 'check-role', 'settled', $9,
       'not-tested', '[]', '[]', '[]',
       hold.receipt_id, $4, hold.proposal_ref->'ref', hold.proposal_ref->>'digest'
  FROM finding_proposal_audit_holds AS hold
 WHERE hold.audit_id = $2 AND hold.receipt_id = $10`, itemID, auditID, "check-"+item.key, ordinal, "subject-"+item.key,
			taskRef, serviceTestDigest("task-"+item.key), origin, string(item.final), "receipt-unassessed"); err != nil {
			t.Fatal(err)
		}
	}
	for _, attempt := range attempts {
		key := fmt.Sprintf("%s-%d", attempt.item, attempt.attempt)
		itemID := "item-unassessed-" + attempt.item
		executionID, memberID, runID := "execution-"+key, "member-"+key, "run-"+key
		taskRef := fmt.Sprintf(`{"namespace":"audit-task-packages","name":"check-%s","revision":"task-r1"}`, attempt.item)
		var deletedAt *time.Time
		if attempt.runDeleted {
			now := time.Now()
			deletedAt = &now
		}
		workflow := &auditstore.WorkflowProvenance{
			Name: "check", Version: "1", SchemaVersion: contracts.APIVersion,
			ClosureDigest: serviceTestDigest("workflow"),
		}
		workflow.ConfigurationRef.Name, workflow.ConfigurationRef.Version = workflow.Name, workflow.Version
		runProvenance, err := json.Marshal(auditstore.RunProvenance{
			Schema: "contractor.audit.run-provenance.v1", RunID: runID, Workflow: workflow,
		})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(ctx, `
INSERT INTO audit_executions (
    execution_id, audit_id, round_id, role, workflow_role,
    manifest_ref, manifest_digest, submission_key, request_digest,
    run_id, state, terminal_outcome, terminal_run_generation,
    terminal_run_sequence, terminal_observed_at, run_deleted_at, run_provenance
) VALUES (
    $1, $2, 'round-unassessed', 'check', 'check-role',
    $3::jsonb, $4, $5, $6,
    $7, 'collected', $8, 'generation-one', $9, clock_timestamp(), $10, $11::jsonb
)`, executionID, auditID, taskRef, serviceTestDigest("manifest-"+key),
			"submission-"+key, serviceTestDigest("request-"+key), runID,
			string(attempt.outcome), attempt.attempt, deletedAt, runProvenance); err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(ctx, `
INSERT INTO audit_execution_items (
    execution_item_id, execution_id, audit_id, round_id, item_id,
    batch_ordinal, item_attempt, task_ref, task_digest, input_refs,
    state, collection_disposition, collected_at
) VALUES (
    $1, $2, $3, 'round-unassessed', $4,
    0, $5, $6::jsonb, $7, '[]'::jsonb,
    'settled', $8, clock_timestamp()
)`, memberID, executionID, auditID, itemID, attempt.attempt,
			taskRef, serviceTestDigest("task-"+attempt.item), string(attempt.disposition)); err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(ctx, `
UPDATE audit_items SET last_execution_item_id = $2 WHERE item_id = $1`, itemID, memberID); err != nil {
			t.Fatal(err)
		}
	}
	intake, err := findingintake.New(pool)
	if err != nil {
		t.Fatal(err)
	}
	service := &Service{pool: pool, findings: intake, now: time.Now}
	params := ProvenanceListParams{OwnerID: owner, AuditID: auditID, FindingID: findingID, Limit: 10}
	records, err := service.ListFindingProvenance(ctx, params)
	if err != nil || len(records) != 6 {
		t.Fatalf("unassessed provenance = (%+v, %v), want proposal and five attempts", records, err)
	}
	if records[0].Kind != ProvenanceSourceProposal || records[0].RecordID != "proposal:receipt-unassessed" {
		t.Fatalf("proposal provenance = %+v", records[0])
	}
	expected := make(map[string]attemptCase, len(attempts))
	for _, attempt := range attempts {
		key := fmt.Sprintf("%s-%d", attempt.item, attempt.attempt)
		expected["attempt:receipt-unassessed:member-"+key] = attempt
	}
	for _, record := range records[1:] {
		want, ok := expected[record.RecordID]
		if !ok {
			t.Fatalf("unexpected or repeated provenance record %q", record.RecordID)
		}
		delete(expected, record.RecordID)
		key := fmt.Sprintf("%s-%d", want.item, want.attempt)
		got := record.Attempt
		if record.Kind != ProvenanceCheckAttempt || record.ReceiptID != "receipt-unassessed" ||
			record.Relation != "verification" || record.Assessment != nil || record.SupportsCurrent ||
			got == nil || got.ItemID != "item-unassessed-"+want.item ||
			got.ExecutionID != "execution-"+key || got.ExecutionItemID != "member-"+key ||
			got.ItemAttempt != want.attempt || got.State != auditstore.ItemSettled ||
			got.CollectionDisposition == nil || *got.CollectionDisposition != want.disposition ||
			got.TerminalOutcome == nil || *got.TerminalOutcome != want.outcome ||
			got.RunID == nil || *got.RunID != "run-"+key || got.RunDeleted != want.runDeleted {
			t.Fatalf("unassessed attempt %q = %+v", record.RecordID, record)
		}
	}
	if len(expected) != 0 {
		t.Fatalf("missing attempts: %+v", expected)
	}
	revisions, err := service.readProvenanceRevisions(ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	params.AuditRevision, params.FindingRevision = &revisions.audit, &revisions.finding
	params.Limit = 2
	var paged []FindingProvenance
	for range 3 {
		page, err := service.ListFindingProvenance(ctx, params)
		if err != nil || len(page) != 2 {
			t.Fatalf("pinned provenance page = (%+v, %v)", page, err)
		}
		paged = append(paged, page...)
		last := page[len(page)-1]
		params.AfterCreatedAt, params.AfterRecordID = &last.CreatedAt, last.RecordID
	}
	if !reflect.DeepEqual(paged, records) {
		t.Fatalf("pinned keyset pages changed provenance: got %+v, want %+v", paged, records)
	}
}
