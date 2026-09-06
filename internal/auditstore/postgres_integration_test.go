package auditstore

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"sync"
	"testing"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const auditTestRuntimeSnapshot = `{"default":{"label":"default","explicit":false,"bindingRevision":1,"config":{"name":"contractor-empty","version":"1","digest":"sha256:80a1754c01f8443c29fdc8f650a2254b2461694819918b204a55a7ad3425dc5f"}},"labels":[],"llmCredentialIds":[],"runtimeCredentialIds":[]}`

func TestPostgresAuditLifecycleClaimsReceiptsAndProjectFence(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pool := isolatedAuditPool(t, ctx, databaseURL)
	projects := projectstore.NewPostgresStore(pool)
	project, _, err := projects.Create(ctx, projectstore.CreateParams{
		ProjectID: "project-audit", OwnerID: "owner-audit", Kind: projectstore.KindProject,
		Name: "Audit project", IdempotencyKey: "project-create", RequestDigest: testDigest("1"),
	})
	if err != nil {
		t.Fatal(err)
	}
	store := NewPostgresStore(pool)
	create := CreateDraftParams{
		AuditID: "audit-one", OwnerID: "owner-audit", ProjectID: project.ProjectID,
		Profile:         ProfileIdentity{Name: "checklist", Version: "1", Digest: testDigest("2")},
		ProfileSnapshot: json.RawMessage(`{"name":"checklist","workflows":{"check":{"kind":"check"},"discovery":{"kind":"discovery"}}}`),
		InputSelection:  json.RawMessage(`{"checklist":{"namespace":"docs","name":"checks","revision":"r1"}}`),
		Limits:          Limits{MaxRounds: 1, BatchSize: 2, MaxItemsPerRound: 10, MaxItemsTotal: 10, MaxSubmittedRuns: 10, MaxItemRunAttempts: 2, MaxEvidenceBytes: 1024},
		IdempotencyKey:  "audit-create", RequestDigest: testDigest("3"),
	}

	var wait sync.WaitGroup
	type createResult struct {
		audit    Audit
		inserted bool
		err      error
	}
	results := make([]createResult, 2)
	for index := range results {
		wait.Add(1)
		go func(index int) {
			defer wait.Done()
			results[index].audit, results[index].inserted, results[index].err = store.CreateDraft(ctx, create)
		}(index)
	}
	wait.Wait()
	insertions := 0
	for _, result := range results {
		if result.err != nil || result.audit.AuditID != create.AuditID {
			t.Fatalf("concurrent create = (audit=%+v, inserted=%t, err=%v)", result.audit, result.inserted, result.err)
		}
		if result.inserted {
			insertions++
		}
	}
	if insertions != 1 {
		t.Fatalf("insertions = %d, want 1", insertions)
	}
	driftedCreate := create
	driftedCreate.RequestDigest = testDigest("0")
	if _, inserted, err := store.CreateDraft(ctx, driftedCreate); !errors.Is(err, ErrConflict) || inserted {
		t.Fatalf("Audit create idempotency drift = (%t, %v)", inserted, err)
	}
	if _, err := store.Get(ctx, "foreign-owner", create.AuditID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("foreign Audit lookup error = %v", err)
	}

	manifest := testExact("audits", "worklist", "worklist-r1")
	itemOneTask := testExact("audits", "task-one", "task-r1")
	itemTwoTask := testExact("audits", "task-two", "task-r1")
	startParams := MaterializeRoundParams{
		OwnerID: create.OwnerID, AuditID: create.AuditID, ExpectedRevision: 1,
		RoundID: "round-one", RoundOrdinal: 1, Manifest: manifest,
		BaselineSnapshot: json.RawMessage(`{"inputs":[],"skills":[]}`),
		DeadlineAt:       time.Now().Add(time.Hour), IdempotencyKey: "audit-start", RequestDigest: testDigest("4"),
		Items: []MaterializedItem{
			{ItemID: "item-one", ItemKey: "check-one", Ordinal: 0, Kind: "checklist", SubjectKey: "subject-one", Task: itemOneTask, Origin: testOrigin("check-one"), WorkflowRole: "check", InitialState: ItemReady, Coverage: emptyCoverage()},
			{ItemID: "item-two", ItemKey: "check-two", Ordinal: 1, Kind: "checklist", SubjectKey: "subject-two", Task: itemTwoTask, Origin: testOrigin("check-two"), WorkflowRole: "check", InitialState: ItemReady, Coverage: emptyCoverage()},
		},
	}
	type startResult struct {
		audit    Audit
		inserted bool
		err      error
	}
	startResults := make([]startResult, 2)
	for index := range startResults {
		wait.Add(1)
		go func(index int) {
			defer wait.Done()
			startResults[index].audit, startResults[index].inserted, startResults[index].err = store.MaterializeRound(ctx, startParams)
		}(index)
	}
	wait.Wait()
	var started Audit
	startInsertions := 0
	for _, result := range startResults {
		if result.err != nil || result.audit.State != AuditActive || result.audit.CurrentRoundID == nil {
			t.Fatalf("concurrent Audit start = (%+v, %t, %v)", result.audit, result.inserted, result.err)
		}
		started = result.audit
		if result.inserted {
			startInsertions++
		}
	}
	if startInsertions != 1 {
		t.Fatalf("Audit start insertions = %d, want 1", startInsertions)
	}
	replayed, inserted, err := store.MaterializeRound(ctx, startParams)
	if err != nil || inserted || replayed.AuditID != started.AuditID {
		t.Fatalf("start replay = (%+v, %t, %v)", replayed, inserted, err)
	}
	driftedStart := startParams
	driftedStart.RequestDigest = testDigest("0")
	if _, inserted, err = store.MaterializeRound(ctx, driftedStart); !errors.Is(err, ErrConflict) || inserted {
		t.Fatalf("Audit start idempotency drift = (%t, %v)", inserted, err)
	}
	paused, changed, err := store.Transition(ctx, TransitionParams{
		OwnerID: create.OwnerID, AuditID: create.AuditID,
		ExpectedRevision: started.Revision, ExpectedState: AuditActive, TargetState: AuditPaused,
		Reason:         &StopReason{Code: "owner_paused", Message: "paused by test"},
		IdempotencyKey: "audit-pause", RequestDigest: testDigest("c"),
	})
	if err != nil || !changed || paused.State != AuditPaused || paused.StopReason == nil || paused.StopReason.Code != "owner_paused" {
		t.Fatalf("pause Audit = (%+v, %t, %v)", paused, changed, err)
	}
	resumed, changed, err := store.Transition(ctx, TransitionParams{
		OwnerID: create.OwnerID, AuditID: create.AuditID,
		ExpectedRevision: paused.Revision, ExpectedState: AuditPaused, TargetState: AuditActive,
		IdempotencyKey: "audit-resume", RequestDigest: testDigest("d"),
	})
	if err != nil || !changed || resumed.State != AuditActive || resumed.StopReason != nil {
		t.Fatalf("resume Audit = (%+v, %t, %v)", resumed, changed, err)
	}

	type claimResult struct {
		claims []ControllerClaim
		err    error
	}
	claimResults := make([]claimResult, 2)
	for index, holder := range []string{"controller-a", "controller-b"} {
		wait.Add(1)
		go func(index int, holder string) {
			defer wait.Done()
			claimResults[index].claims, claimResults[index].err = store.Claim(ctx, ClaimParams{HolderID: holder, Lease: 30 * time.Second, Limit: 10})
		}(index, holder)
	}
	wait.Wait()
	var claim ControllerClaim
	claimCount := 0
	for _, result := range claimResults {
		if result.err != nil {
			t.Fatalf("concurrent Audit claim: %v", result.err)
		}
		claimCount += len(result.claims)
		if len(result.claims) == 1 {
			claim = result.claims[0]
		}
	}
	if claimCount != 1 {
		t.Fatalf("concurrent Audit claim count = %d, want 1", claimCount)
	}
	roundID := "round-one"
	activeRound, err := store.TransitionRound(ctx, RoundTransitionParams{
		Claim: claim, RoundID: roundID, ExpectedRevision: 1,
		ExpectedState: RoundAccepted, TargetState: RoundExecuting,
	})
	if err != nil || activeRound.State != RoundExecuting || activeRound.Revision != 2 {
		t.Fatalf("activate Audit round = (%+v, %v)", activeRound, err)
	}
	execution, inserted, err := store.CreateExecutionIntent(ctx, CreateExecutionIntentParams{
		Claim: claim, ExecutionID: "execution-one", RoundID: &roundID, Role: ExecutionCheck,
		WorkflowRole:  "check",
		Manifest:      testExact("audits", "execution-one", "manifest-r1"),
		SubmissionKey: "audit-one-round-one-item-one-attempt-one", RequestDigest: testDigest("5"),
		Members: []ExecutionMemberIntent{{ExecutionItemID: "execution-item-one", ItemID: "item-one", BatchOrdinal: 0, ItemAttempt: 1, Task: itemOneTask, Inputs: []ExactArtifact{}}},
	})
	if err != nil || !inserted || execution.State != ExecutionIntent {
		t.Fatalf("create execution intent = (%+v, %t, %v)", execution, inserted, err)
	}
	execution, err = insertAndBindTestRun(t, ctx, pool, claim, execution, "run-one", create.OwnerID, project.ProjectID)
	if err != nil || execution.State != ExecutionSubmitted {
		t.Fatalf("bind execution Run = (%+v, %v)", execution, err)
	}
	generation, sequence := terminateTestRun(t, ctx, pool, "run-one", "succeeded")
	execution, err = store.ObserveTerminal(ctx, ObserveTerminalParams{
		Claim: claim, ExecutionID: execution.ExecutionID, RunID: "run-one",
		Generation: generation, Sequence: sequence,
	})
	if err != nil || execution.State != ExecutionCollecting {
		t.Fatalf("observe terminal Run = (%+v, %v)", execution, err)
	}
	receipt, inserted, err := store.Collect(ctx, CollectParams{
		Claim: claim, ReceiptID: "receipt-one", ExecutionID: execution.ExecutionID,
		Disposition: CollectionMissingOutput, ErrorCode: stringPointer("result_missing"),
		RequestDigest: testDigest("6"),
		Items: []CollectionItem{{
			ExecutionItemID: "execution-item-one", Disposition: CollectionMissingOutput,
			Retryable: true, FinalDisposition: FinalMissingOutput,
			Coverage: Coverage{Status: CoverageInconclusive, Requested: []string{}, Completed: []string{}, Gaps: []string{"missing-output"}},
		}},
	})
	if err != nil || !inserted || receipt.Disposition != CollectionMissingOutput {
		t.Fatalf("collect retryable attempt = (%+v, %t, %v)", receipt, inserted, err)
	}
	observedReplay, err := store.ObserveTerminal(ctx, ObserveTerminalParams{
		Claim: claim, ExecutionID: execution.ExecutionID, RunID: "run-one",
		Generation: generation, Sequence: sequence,
	})
	if err != nil || observedReplay.State != ExecutionCollected {
		t.Fatalf("terminal observation replay after collection = (%+v, %v)", observedReplay, err)
	}
	items, err := store.ListItems(ctx, create.AuditID)
	if err != nil || len(items) != 2 || items[0].State != ItemReady || items[0].FinalDisposition != nil {
		t.Fatalf("items after retryable collection = (%+v, %v)", items, err)
	}

	executionTwo, _, err := store.CreateExecutionIntent(ctx, CreateExecutionIntentParams{
		Claim: claim, ExecutionID: "execution-two", RoundID: &roundID, Role: ExecutionCheck,
		WorkflowRole:  "check",
		Manifest:      testExact("audits", "execution-two", "manifest-r1"),
		SubmissionKey: "audit-one-round-one-item-one-attempt-two", RequestDigest: testDigest("7"),
		Members: []ExecutionMemberIntent{{ExecutionItemID: "execution-item-two", ItemID: "item-one", BatchOrdinal: 0, ItemAttempt: 2, Task: itemOneTask, Inputs: []ExactArtifact{}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = insertAndBindTestRun(t, ctx, pool, claim, executionTwo, "run-two", create.OwnerID, project.ProjectID); err != nil {
		t.Fatal(err)
	}
	generation, sequence = terminateTestRun(t, ctx, pool, "run-two", "failed")
	if _, err = store.ObserveTerminal(ctx, ObserveTerminalParams{Claim: claim, ExecutionID: executionTwo.ExecutionID, RunID: "run-two", Generation: generation, Sequence: sequence}); err != nil {
		t.Fatal(err)
	}
	receiptTwoParams := CollectParams{
		Claim: claim, ReceiptID: "receipt-two", ExecutionID: executionTwo.ExecutionID,
		Disposition: CollectionExecutionFailed, ErrorCode: stringPointer("run_failed"), RequestDigest: testDigest("8"),
		Items: []CollectionItem{{ExecutionItemID: "execution-item-two", Disposition: CollectionExecutionFailed, Retryable: true, FinalDisposition: FinalExecutionFailed, Coverage: Coverage{Status: CoverageBlocked, Requested: []string{}, Completed: []string{}, Gaps: []string{"run-failed"}}}},
	}
	if _, inserted, err = store.Collect(ctx, receiptTwoParams); err != nil || !inserted {
		t.Fatalf("collect exhausted attempt = (%t, %v)", inserted, err)
	}
	if _, inserted, err = store.Collect(ctx, receiptTwoParams); err != nil || inserted {
		t.Fatalf("collection replay = (%t, %v)", inserted, err)
	}
	items, err = store.ListItems(ctx, create.AuditID)
	if err != nil || items[0].State != ItemSettled || items[0].FinalDisposition == nil || *items[0].FinalDisposition != FinalExecutionFailed {
		t.Fatalf("items after exhausted retry = (%+v, %v)", items, err)
	}
	firstAttempt, err := store.ListExecutionItems(ctx, execution.ExecutionID)
	if err != nil || len(firstAttempt) != 1 || firstAttempt[0].ItemAttempt != 1 ||
		firstAttempt[0].State != ItemSettled || firstAttempt[0].CollectionDisposition == nil ||
		*firstAttempt[0].CollectionDisposition != CollectionMissingOutput {
		t.Fatalf("preserved first execution-item attempt = (%+v, %v)", firstAttempt, err)
	}
	secondAttempt, err := store.ListExecutionItems(ctx, executionTwo.ExecutionID)
	if err != nil || len(secondAttempt) != 1 || secondAttempt[0].ItemAttempt != 2 ||
		secondAttempt[0].State != ItemSettled || secondAttempt[0].CollectionDisposition == nil ||
		*secondAttempt[0].CollectionDisposition != CollectionExecutionFailed {
		t.Fatalf("preserved second execution-item attempt = (%+v, %v)", secondAttempt, err)
	}

	newClaim := supersedeBlockedClaim(t, ctx, pool, claim, func(blockedStore *PostgresStore) error {
		_, _, mutationErr := blockedStore.CreateExecutionIntent(ctx, CreateExecutionIntentParams{
			Claim: claim, ExecutionID: "stale-execution", RoundID: &roundID, Role: ExecutionCheck,
			WorkflowRole: "check",
			Manifest:     testExact("audits", "stale", "manifest-r1"), SubmissionKey: "stale-submission", RequestDigest: testDigest("9"),
			Members: []ExecutionMemberIntent{{ExecutionItemID: "stale-member", ItemID: "item-two", BatchOrdinal: 0, ItemAttempt: 1, Task: itemTwoTask, Inputs: []ExactArtifact{}}},
		})
		return mutationErr
	})
	if newClaim.Epoch <= claim.Epoch {
		t.Fatalf("superseding claim epoch = %d, want > %d", newClaim.Epoch, claim.Epoch)
	}
	intentCandidates := []CreateExecutionIntentParams{
		{
			Claim: newClaim, ExecutionID: "execution-three-a", RoundID: &roundID, Role: ExecutionCheck,
			WorkflowRole:  "check",
			Manifest:      testExact("audits", "execution-three-a", "manifest-r1"),
			SubmissionKey: "audit-one-round-one-item-two-attempt-one-a", RequestDigest: testDigest("e"),
			Members: []ExecutionMemberIntent{{ExecutionItemID: "execution-item-three-a", ItemID: "item-two", BatchOrdinal: 0, ItemAttempt: 1, Task: itemTwoTask, Inputs: []ExactArtifact{}}},
		},
		{
			Claim: newClaim, ExecutionID: "execution-three-b", RoundID: &roundID, Role: ExecutionCheck,
			WorkflowRole:  "check",
			Manifest:      testExact("audits", "execution-three-b", "manifest-r1"),
			SubmissionKey: "audit-one-round-one-item-two-attempt-one-b", RequestDigest: testDigest("f"),
			Members: []ExecutionMemberIntent{{ExecutionItemID: "execution-item-three-b", ItemID: "item-two", BatchOrdinal: 0, ItemAttempt: 1, Task: itemTwoTask, Inputs: []ExactArtifact{}}},
		},
	}
	type intentResult struct {
		execution Execution
		inserted  bool
		err       error
	}
	intentResults := make([]intentResult, len(intentCandidates))
	for index := range intentResults {
		wait.Add(1)
		go func(index int) {
			defer wait.Done()
			intentResults[index].execution, intentResults[index].inserted, intentResults[index].err = store.CreateExecutionIntent(ctx, intentCandidates[index])
		}(index)
	}
	wait.Wait()
	intentInsertions := 0
	var itemTwoIntent CreateExecutionIntentParams
	var executionThree Execution
	for index, result := range intentResults {
		if result.inserted {
			intentInsertions++
			itemTwoIntent = intentCandidates[index]
			executionThree = result.execution
			if result.err != nil {
				t.Fatalf("winning concurrent execution intent: %v", result.err)
			}
		} else if !errors.Is(result.err, ErrPrecondition) && !errors.Is(result.err, ErrConflict) {
			t.Fatalf("losing concurrent execution intent error = %v", result.err)
		}
	}
	if intentInsertions != 1 {
		t.Fatalf("execution intent insertions = %d, want 1", intentInsertions)
	}
	executionThree, inserted, err = store.CreateExecutionIntent(ctx, itemTwoIntent)
	if err != nil || inserted || executionThree.ExecutionID != itemTwoIntent.ExecutionID {
		t.Fatalf("execution intent replay = (%+v, %t, %v)", executionThree, inserted, err)
	}
	driftedIntent := itemTwoIntent
	driftedIntent.RequestDigest = testDigest("0")
	if _, inserted, err = store.CreateExecutionIntent(ctx, driftedIntent); !errors.Is(err, ErrConflict) || inserted {
		t.Fatalf("execution intent idempotency drift = (%t, %v)", inserted, err)
	}
	currentAudit, err := store.Get(ctx, create.OwnerID, create.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	pausedWithIntent, _, err := store.Transition(ctx, TransitionParams{
		OwnerID: create.OwnerID, AuditID: create.AuditID,
		ExpectedRevision: currentAudit.Revision, ExpectedState: AuditActive, TargetState: AuditPaused,
		Reason:         &StopReason{Code: "owner_paused", Message: "pause before bind"},
		IdempotencyKey: "audit-pause-before-bind", RequestDigest: testDigest("0"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := insertTestRun(ctx, pool, "run-three", create.OwnerID, project.ProjectID, nil); err != nil {
		t.Fatal(err)
	}
	if _, err = store.BindRun(ctx, BindRunParams{Claim: newClaim, ExecutionID: executionThree.ExecutionID, RunID: "run-three"}); !errors.Is(err, ErrPrecondition) {
		t.Fatalf("bind while Audit is paused error = %v", err)
	}
	if _, err := pool.Exec(ctx, `DELETE FROM workflow_runs WHERE run_id = 'run-three'`); err != nil {
		t.Fatal(err)
	}
	if _, _, err = store.Transition(ctx, TransitionParams{
		OwnerID: create.OwnerID, AuditID: create.AuditID,
		ExpectedRevision: pausedWithIntent.Revision, ExpectedState: AuditPaused, TargetState: AuditActive,
		IdempotencyKey: "audit-resume-before-bind", RequestDigest: testDigest("1"),
	}); err != nil {
		t.Fatal(err)
	}
	if _, err = insertAndBindTestRun(t, ctx, pool, newClaim, executionThree, "run-three", create.OwnerID, project.ProjectID); err != nil {
		t.Fatal(err)
	}
	generation, sequence = terminateTestRun(t, ctx, pool, "run-three", "succeeded")
	if _, err = store.ObserveTerminal(ctx, ObserveTerminalParams{Claim: newClaim, ExecutionID: executionThree.ExecutionID, RunID: "run-three", Generation: generation, Sequence: sequence}); err != nil {
		t.Fatal(err)
	}
	proposal := testExact("audit-findings", "candidate-one", "proposal-r1")
	proposal.MediaType = "application/json"
	proposal.SizeBytes = 128
	proposalJSON, err := json.Marshal(proposal)
	if err != nil {
		t.Fatal(err)
	}
	proposalRefJSON, err := json.Marshal(proposal.Ref)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = pool.Exec(ctx, `
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
    'finding-receipt-one', 'finding-proposal-one', 'allocation-finding-one',
    'runtime-finding-one', 'instance-finding-one', 'stage-finding-one', 'worker',
    'invocation-finding-one', 'submission-finding-one', 'candidate-one', $1,
    'run-three', $2, $3, $4, $5, 'check',
    'audit-check', '1', 'contractor/v1alpha1',
    '{"name":"audit-check","version":"1"}'::jsonb, $6,
    $7::jsonb, $8, 'application/json', 128, '[]'::jsonb
)`,
		testDigest("d"), create.OwnerID, project.ProjectID,
		executionThree.ExecutionID, create.AuditID, testDigest("e"),
		proposalRefJSON, proposal.Digest,
	); err != nil {
		t.Fatal(err)
	}
	if _, err = pool.Exec(ctx, `
INSERT INTO finding_proposal_retention (receipt_id)
VALUES ('finding-receipt-one')`); err != nil {
		t.Fatal(err)
	}
	if _, err = pool.Exec(ctx, `
INSERT INTO finding_proposal_audit_holds (
    receipt_id, audit_id, project_id, proposal_ref, evidence
) VALUES ('finding-receipt-one', $1, $2, $3::jsonb, '[]'::jsonb)`,
		create.AuditID, project.ProjectID, proposalJSON); err != nil {
		t.Fatal(err)
	}
	acceptedResult := testExact("outputs", "result", "result-r1")
	acceptedReceipt, inserted, err := store.Collect(ctx, CollectParams{
		Claim: newClaim, ReceiptID: "receipt-three", ExecutionID: executionThree.ExecutionID,
		Disposition: CollectionAccepted, SourceOutput: &acceptedResult, RequestDigest: testDigest("f"),
		Retained: []ArtifactLink{{
			LogicalKey: "rounds/1/items/check-two/result", Artifact: acceptedResult,
			SourceProvenance: json.RawMessage(`{"runId":"run-three"}`), DisplayRef: "Check two result",
		}},
		Items: []CollectionItem{{
			ExecutionItemID: itemTwoIntent.Members[0].ExecutionItemID, Disposition: CollectionAccepted,
			FinalDisposition: FinalAccepted,
			Result:           &acceptedResult,
			Coverage:         Coverage{Status: CoverageSatisfied, Requested: []string{}, Completed: []string{"check"}, Gaps: []string{}},
			FindingAssociations: []FindingAssociation{{
				AssessmentID: "finding-assessment-one", ReceiptID: "finding-receipt-one",
				Proposal: proposal, SemanticAssessment: "supported",
			}},
		}},
	})
	if err != nil || !inserted || len(acceptedReceipt.Retained) != 1 {
		t.Fatalf("accepted collection = (%+v, %t, %v)", acceptedReceipt, inserted, err)
	}
	var findingState string
	var findingRevision, assessmentCount int
	if err := pool.QueryRow(ctx, `
SELECT finding.state, finding.revision,
       (SELECT count(*) FROM audit_finding_assessments AS assessment
         WHERE assessment.finding_id = finding.finding_id)
  FROM audit_findings AS finding
 WHERE finding.audit_id = $1 AND finding.first_receipt_id = 'finding-receipt-one'`,
		create.AuditID).Scan(&findingState, &findingRevision, &assessmentCount); err != nil {
		t.Fatal(err)
	}
	if findingState != "proposed" || findingRevision != 2 || assessmentCount != 1 {
		t.Fatalf("accepted finding assessment = state %q revision %d count %d",
			findingState, findingRevision, assessmentCount)
	}
	if _, inserted, err = store.Collect(ctx, CollectParams{
		Claim: newClaim, ReceiptID: "receipt-three-drift", ExecutionID: executionThree.ExecutionID,
		Disposition: CollectionAccepted, SourceOutput: &acceptedResult, RequestDigest: testDigest("0"),
		Items: []CollectionItem{{
			ExecutionItemID: itemTwoIntent.Members[0].ExecutionItemID, Disposition: CollectionAccepted,
			FinalDisposition: FinalAccepted, Result: &acceptedResult, Coverage: emptyCoverage(),
		}},
	}); !errors.Is(err, ErrConflict) || inserted {
		t.Fatalf("collection idempotency drift = (%t, %v)", inserted, err)
	}
	currentAudit, err = store.Get(ctx, create.OwnerID, create.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	waiting, err := store.TransitionClaimed(ctx, ClaimedTransitionParams{
		Claim: newClaim, ExpectedRevision: currentAudit.Revision,
		ExpectedState: AuditActive, TargetState: AuditWaitingReview,
	})
	if err != nil || waiting.State != AuditWaitingReview {
		t.Fatalf("claimed transition to waiting review = (%+v, %v)", waiting, err)
	}
	active, err := store.TransitionClaimed(ctx, ClaimedTransitionParams{
		Claim: newClaim, ExpectedRevision: waiting.Revision,
		ExpectedState: AuditWaitingReview, TargetState: AuditActive,
	})
	if err != nil || active.State != AuditActive {
		t.Fatalf("claimed transition to active = (%+v, %v)", active, err)
	}
	fencedIntent, _, err := store.CreateExecutionIntent(ctx, CreateExecutionIntentParams{
		Claim: newClaim, ExecutionID: "execution-project-fence", Role: ExecutionDiscovery,
		WorkflowRole: "discovery",
		RoleAttempt:  intPointer(1), Manifest: testExact("audits", "fenced-intent", "manifest-r1"),
		SubmissionKey: "fenced-intent-submission", RequestDigest: testDigest("1"),
		Members: []ExecutionMemberIntent{},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := insertTestRun(ctx, pool, "run-project-fence", create.OwnerID, project.ProjectID, nil); err != nil {
		t.Fatal(err)
	}
	if _, _, err := projects.BeginDeletion(ctx, projectstore.BeginDeletionParams{ProjectID: project.ProjectID, OwnerID: create.OwnerID, ExpectedRevision: project.Revision}); err != nil {
		t.Fatal(err)
	}
	if _, err = store.BindRun(ctx, BindRunParams{Claim: newClaim, ExecutionID: fencedIntent.ExecutionID, RunID: "run-project-fence"}); !errors.Is(err, ErrProjectDeleting) {
		t.Fatalf("bind after Project fence error = %v", err)
	}
	if _, err = store.ObserveSubmissionFailure(ctx, ObserveSubmissionFailureParams{Claim: newClaim, ExecutionID: fencedIntent.ExecutionID}); err != nil {
		t.Fatalf("observe submission failure after Project fence: %v", err)
	}
	if _, inserted, err = store.Collect(ctx, CollectParams{
		Claim: newClaim, ReceiptID: "receipt-project-fence", ExecutionID: fencedIntent.ExecutionID,
		Disposition: CollectionExecutionFailed, ErrorCode: stringPointer("project_deleting"),
		RequestDigest: testDigest("2"), Items: []CollectionItem{},
	}); err != nil || !inserted {
		t.Fatalf("collect fenced submission = (%t, %v)", inserted, err)
	}
	submissionReplay, err := store.ObserveSubmissionFailure(ctx, ObserveSubmissionFailureParams{
		Claim: newClaim, ExecutionID: fencedIntent.ExecutionID,
	})
	if err != nil || submissionReplay.State != ExecutionCollected {
		t.Fatalf("submission failure replay after collection = (%+v, %v)", submissionReplay, err)
	}
	if replay, inserted, replayErr := store.CreateDraft(ctx, create); replayErr != nil || inserted || replay.AuditID != create.AuditID {
		t.Fatalf("create replay after Project fence = (%+v, %t, %v)", replay, inserted, replayErr)
	}
	if replay, inserted, replayErr := store.MaterializeRound(ctx, startParams); replayErr != nil || inserted || replay.AuditID != create.AuditID {
		t.Fatalf("start replay after Project fence = (%+v, %t, %v)", replay, inserted, replayErr)
	}
	_, _, err = store.CreateExecutionIntent(ctx, CreateExecutionIntentParams{
		Claim: newClaim, ExecutionID: "fenced-execution", Role: ExecutionDiscovery, RoleAttempt: intPointer(1),
		WorkflowRole: "discovery",
		Manifest:     testExact("audits", "fenced", "manifest-r1"), SubmissionKey: "fenced-submission", RequestDigest: testDigest("b"),
		Members: []ExecutionMemberIntent{},
	})
	if !errors.Is(err, ErrProjectDeleting) {
		t.Fatalf("Project fence error = %v", err)
	}
	recoveredStore := NewPostgresStore(pool)
	snapshot, err := recoveredStore.GetReconcileSnapshot(ctx, newClaim)
	if err != nil || len(snapshot.Executions) != 0 || len(snapshot.Items) != 0 || len(snapshot.Receipts) != 4 {
		t.Fatalf("reconcile snapshot = (%+v, %v)", snapshot, err)
	}
	if snapshot.Audit.Revision != snapshot.Audit.EventSequence {
		t.Fatalf("Audit projection revision = %d, event sequence = %d", snapshot.Audit.Revision, snapshot.Audit.EventSequence)
	}
	events, err := recoveredStore.ListEvents(ctx, create.AuditID, 0, MaxPageSize)
	if err != nil || len(events) < 13 {
		t.Fatalf("Audit event stream = (%+v, %v)", events, err)
	}
	for index := range events {
		if events[index].Sequence != uint64(index+1) {
			t.Fatalf("Audit event sequence at %d = %d", index, events[index].Sequence)
		}
	}
	coverage, err := recoveredStore.ListCoverage(ctx, create.AuditID, roundID, -1, 10)
	if err != nil || len(coverage) != 2 || coverage[0].Coverage.Status != CoverageBlocked || coverage[1].Coverage.Status != CoverageSatisfied {
		t.Fatalf("Audit coverage = (%+v, %v)", coverage, err)
	}
}

func insertAndBindTestRun(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	claim ControllerClaim,
	execution Execution,
	runID string,
	ownerID string,
	projectID string,
) (Execution, error) {
	t.Helper()
	var bound Execution
	err := persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := insertTestRun(ctx, tx, runID, ownerID, projectID, &execution); err != nil {
			return err
		}
		var err error
		bound, err = NewPostgresStore(tx).BindRun(ctx, BindRunParams{
			Claim: claim, ExecutionID: execution.ExecutionID, RunID: runID,
		})
		return err
	})
	return bound, err
}

func insertTestRun(
	ctx context.Context,
	db persistencepostgres.DBTX,
	runID string,
	ownerID string,
	projectID string,
	execution *Execution,
) error {
	publicationMode := "ordinary"
	var executionID, submissionKey *string
	if execution != nil {
		publicationMode = "audit-managed"
		executionID = &execution.ExecutionID
		submissionKey = &execution.SubmissionKey
	}
	_, err := db.Exec(ctx, `
INSERT INTO workflow_runs (
    run_id, owner_id, project_id, workflow_name, workflow_version,
    workflow_schema_version, workflow_snapshot, parameters,
    runtime_labels, runtime_config_snapshot, publication_mode,
    audit_execution_id, audit_submission_key, state, state_reason_code
) VALUES ($1, $2, $3, 'audit-check', '1', 'contractor/v1alpha1',
          '{}'::jsonb, '{}'::jsonb, ARRAY[]::text[], $4::jsonb,
          $5, $6, $7, 'initializing', 'created')`,
		runID, ownerID, projectID, auditTestRuntimeSnapshot,
		publicationMode, executionID, submissionKey,
	)
	return err
}

func terminateTestRun(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, runID, outcome string,
) (string, uint64) {
	t.Helper()
	_, err := pool.Exec(ctx, `
UPDATE workflow_runs
   SET state = $2, state_reason_code = 'test-terminal',
       started_at = COALESCE(started_at, clock_timestamp()),
       finished_at = clock_timestamp(), updated_at = clock_timestamp()
 WHERE run_id = $1`, runID, outcome)
	if err != nil {
		t.Fatalf("terminate test WorkflowRun: %v", err)
	}
	var generation string
	var sequence int64
	if err := pool.QueryRow(ctx, `
SELECT run_event_generation, next_run_event_sequence - 1
  FROM workflow_runs WHERE run_id = $1`, runID).Scan(&generation, &sequence); err != nil {
		t.Fatal(err)
	}
	return generation, uint64(sequence)
}

func isolatedAuditPool(t *testing.T, ctx context.Context, databaseURL string) *pgxpool.Pool {
	t.Helper()
	adminConfig, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	admin, err := pgxpool.NewWithConfig(ctx, adminConfig)
	if err != nil {
		t.Fatal(err)
	}
	if err := admin.Ping(ctx); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	random := make([]byte, 8)
	if _, err := rand.Read(random); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	schema := "contractor_audit_test_" + hex.EncodeToString(random)
	identifier := pgx.Identifier{schema}.Sanitize()
	if _, err := admin.Exec(ctx, `CREATE SCHEMA `+identifier); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		admin.Close()
		t.Fatal(err)
	}
	config.ConnConfig.RuntimeParams["search_path"] = schema
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		admin.Close()
		t.Fatal(err)
	}
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		pool.Close()
		admin.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		pool.Close()
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cleanupCancel()
		_, _ = admin.Exec(cleanupCtx, `DROP SCHEMA `+identifier+` CASCADE`)
		admin.Close()
	})
	return pool
}

// supersedeBlockedClaim proves that a mutation which began with an old MVCC
// snapshot cannot commit after another holder advances the claim epoch. The
// blocker owns both rows before the mutation starts; production code must wait
// on the claim row, then re-evaluate its exact holder/epoch predicate.
func supersedeBlockedClaim(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	claim ControllerClaim,
	mutate func(*PostgresStore) error,
) ControllerClaim {
	t.Helper()
	blocker, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = blocker.Rollback(context.Background()) }()
	if _, err := blocker.Exec(ctx, `SELECT audit_id FROM audits WHERE audit_id = $1 FOR UPDATE`, claim.AuditID); err != nil {
		t.Fatal(err)
	}
	if _, err := blocker.Exec(ctx, `SELECT audit_id FROM audit_controller_claims WHERE audit_id = $1 FOR UPDATE`, claim.AuditID); err != nil {
		t.Fatal(err)
	}
	connection, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer connection.Release()
	var backendPID int
	if err := connection.QueryRow(ctx, `SELECT pg_backend_pid()`).Scan(&backendPID); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- mutate(NewPostgresStore(connection)) }()
	waitForPostgresLock(t, ctx, pool, backendPID, done)

	newClaim, err := scanClaim(blocker.QueryRow(ctx, `
UPDATE audit_controller_claims
   SET epoch = epoch + 1, holder_id = 'controller-successor',
       claimed_at = clock_timestamp(),
       expires_at = clock_timestamp() + interval '30 seconds'
 WHERE audit_id = $1
RETURNING audit_id, holder_id, epoch, claimed_at, expires_at`, claim.AuditID))
	if err != nil {
		t.Fatal(err)
	}
	if err := blocker.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	select {
	case mutationErr := <-done:
		if !errors.Is(mutationErr, ErrClaimLost) {
			t.Fatalf("mutation using superseded claim error = %v", mutationErr)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("mutation using superseded claim did not finish")
	}
	return newClaim
}

func waitForPostgresLock(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	backendPID int,
	done <-chan error,
) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		select {
		case err := <-done:
			t.Fatalf("mutation completed before the claim fence moved: %v", err)
		default:
		}
		var waitEventType *string
		if err := pool.QueryRow(ctx, `
SELECT wait_event_type FROM pg_stat_activity WHERE pid = $1`, backendPID).Scan(&waitEventType); err != nil {
			t.Fatal(err)
		}
		if waitEventType != nil && *waitEventType == "Lock" {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("mutation did not reach the PostgreSQL claim fence")
}

func stringPointer(value string) *string { return &value }
func intPointer(value int) *int          { return &value }
