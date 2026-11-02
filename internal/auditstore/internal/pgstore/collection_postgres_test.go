package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"strconv"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresCollectionReportsEvidenceBudgetExhaustion(t *testing.T) {
	f := newCollectingAuditFixture(t, "budget", 64)
	result := testExact("audit-budget", "result", "result-r1")
	result.SizeBytes = 48
	evidence := testExact("audit-budget", "evidence", "evidence-r1")
	evidence.SizeBytes = 32
	provenance := json.RawMessage(`{"runId":"run-budget"}`)
	_, inserted, err := f.store.Collect(f.ctx, auditstore.CollectParams{
		Claim: f.claim, ReceiptID: "receipt-budget", ExecutionID: f.execution.ExecutionID,
		Disposition: auditstore.CollectionAccepted, SourceOutput: &result, RequestDigest: testDigest("5"),
		Retained: []auditstore.ArtifactLink{
			{Scope: "project", LogicalKey: "result/" + f.memberID, Artifact: result, SourceProvenance: provenance},
			{Scope: "project", LogicalKey: "evidence/budget/one", Artifact: evidence, SourceProvenance: provenance},
			{Scope: "project", LogicalKey: "evidence/budget/two", Artifact: evidence, SourceProvenance: provenance},
		},
		Items: []auditstore.CollectionItem{{
			ExecutionItemID: f.memberID, Disposition: auditstore.CollectionAccepted,
			FinalDisposition: auditstore.FinalAccepted, Result: &result,
			Coverage: auditstore.Coverage{Status: auditstore.CoverageSatisfied, Requested: []string{}, Completed: []string{}, Gaps: []string{}},
		}},
	})
	if !errors.Is(err, auditstore.ErrEvidenceBudgetExhausted) || !errors.Is(err, auditstore.ErrPrecondition) || inserted {
		t.Fatalf("over-budget collection = (%t, %v), want evidence budget precondition", inserted, err)
	}
	code := "evidence-budget-exhausted"
	receipt, inserted, err := f.store.Collect(f.ctx, auditstore.CollectParams{
		Claim: f.claim, ReceiptID: "receipt-budget", ExecutionID: f.execution.ExecutionID,
		Disposition: auditstore.CollectionInvalidResult, SourceOutput: &result, ErrorCode: &code,
		RequestDigest: testDigest("6"), Retained: []auditstore.ArtifactLink{},
		Items: []auditstore.CollectionItem{{
			ExecutionItemID: f.memberID, Disposition: auditstore.CollectionInvalidResult,
			FinalDisposition: auditstore.FinalInvalidResult,
			Coverage:         auditstore.Coverage{Status: auditstore.CoverageInconclusive, Requested: []string{}, Completed: []string{}, Gaps: []string{code}},
		}},
	})
	if err != nil || !inserted || receipt.Disposition != auditstore.CollectionInvalidResult {
		t.Fatalf("terminal budget receipt = (%+v, %t, %v)", receipt, inserted, err)
	}
	audit, err := f.store.Get(f.ctx, f.audit.OwnerID, f.audit.AuditID)
	if err != nil || audit.RetainedEvidenceBytes != 0 {
		t.Fatalf("budget receipt charged evidence = (%+v, %v)", audit, err)
	}
}

// A later Round expands one proposal into one item per proposed check, and a
// batch may carry several of them. Each member keeps its own assessment; the
// accepted results never supply a current Finding assessment.
func TestPostgresCollectionAcceptsBatchedChecksOfOneFindingProposal(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedAuditPool(t, ctx, databaseURL)
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "project-batched-checks", OwnerID: "owner-batched-checks", Kind: projectstore.KindProject,
		Name: "Batched checks", IdempotencyKey: "project-create", RequestDigest: testDigest("1"),
	})
	if err != nil {
		t.Fatal(err)
	}
	store := newTestStore(pool)
	audit, _, err := store.CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: "audit-batched-checks", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:         auditstore.ProfileIdentity{Name: "profile", Version: "1", Digest: testDigest("2")},
		ProfileSnapshot: json.RawMessage(`{"name":"profile","workflows":{"verify":{"kind":"check"}}}`),
		InputSelection:  json.RawMessage(`{"inputs":{}}`),
		Limits: auditstore.Limits{
			MaxRounds: 2, BatchSize: 2, MaxItemsPerRound: 2, MaxItemsTotal: 2,
			MaxSubmittedRuns: 2, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1 << 20,
		},
		IdempotencyKey: "audit-create", RequestDigest: testDigest("3"),
	})
	if err != nil {
		t.Fatal(err)
	}
	audit, _, err = store.MaterializeRound(ctx, auditstore.MaterializeRoundParams{
		OwnerID: project.OwnerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision,
		RoundID: "round-one", RoundOrdinal: 1, Manifest: testExact("audits", "round-one", "r1"),
		BaselineSnapshot: json.RawMessage(`{"inputs":{},"skills":[]}`),
		DeadlineAt:       time.Now().Add(time.Hour), Items: []auditstore.MaterializedItem{},
		IdempotencyKey: "audit-start", RequestDigest: testDigest("4"),
	})
	if err != nil {
		t.Fatal(err)
	}
	claims, err := store.Claim(ctx, auditstore.ClaimParams{HolderID: "controller-batched-checks", Lease: 30 * time.Second, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim Audit = (%+v, %v)", claims, err)
	}
	claim := claims[0]
	closed := closeAuditRoundForTest(t, ctx, store, claim, "round-one", 1)

	const receiptID = "receipt-two-checks"
	proposal := testExact("audit-findings", "proposal-two-checks", "proposal-r1")
	proposal.MediaType, proposal.SizeBytes = "application/json", 128
	insertAuditProposalHoldForRoundTest(t, ctx, pool, audit, proposal, receiptID)
	audit, err = store.Get(ctx, project.OwnerID, audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	items := make([]auditstore.MaterializedItem, 2)
	members := make([]auditstore.ExecutionMemberIntent, 2)
	for ordinal := range items {
		suffix := strconv.Itoa(ordinal)
		items[ordinal] = auditstore.MaterializedItem{
			ItemID: "item-check-" + suffix, ItemKey: "verify-check-" + suffix, Ordinal: ordinal,
			Kind: "finding-verification", SubjectKey: "subject-two-checks",
			Task:   writeCollectionFixtureArtifact(t, ctx, pool, project.ProjectID, "audits", "task-check-"+suffix),
			Origin: testOrigin("verify-check-" + suffix), WorkflowRole: "verify",
			InitialState: auditstore.ItemReady, Coverage: emptyCoverage(),
			ProposalSources: []auditstore.ProposalItemSource{{
				ReceiptID: receiptID, ProposedCheckOrdinal: ordinal, Proposal: proposal,
			}},
		}
		members[ordinal] = auditstore.ExecutionMemberIntent{
			ExecutionItemID: "member-check-" + suffix, ItemID: items[ordinal].ItemID,
			BatchOrdinal: ordinal, ItemAttempt: 1, Task: items[ordinal].Task, Inputs: []auditstore.ExactArtifact{},
		}
	}
	round, _, err := store.AcceptNextRound(ctx, auditstore.AcceptRoundParams{
		Claim: claim, ExpectedAuditRevision: audit.Revision,
		PreviousRoundID: closed.RoundID, RoundID: "round-two", RoundOrdinal: 2,
		Manifest: testExact("audits", "round-two", "r1"), Items: items,
	})
	if err != nil {
		t.Fatal(err)
	}
	round, err = store.TransitionRound(ctx, auditstore.RoundTransitionParams{
		Claim: claim, RoundID: round.RoundID, ExpectedRevision: round.Revision,
		ExpectedState: auditstore.RoundAccepted, TargetState: auditstore.RoundExecuting,
	})
	if err != nil {
		t.Fatal(err)
	}
	execution, _, err := store.CreateExecutionIntent(ctx, auditstore.CreateExecutionIntentParams{
		Claim: claim, ExecutionID: "execution-two-checks", RoundID: &round.RoundID, Role: auditstore.ExecutionCheck,
		WorkflowRole: "verify", LaneClass: resolved.LaneClassModel, Manifest: testExact("audits", "execution-two-checks", "manifest-r1"),
		SubmissionKey: "submission-two-checks", RequestDigest: testDigest("5"), Members: members,
	})
	if err != nil {
		t.Fatal(err)
	}
	execution, err = insertAndBindTestRun(t, ctx, pool, claim, execution, "run-two-checks", project.OwnerID, project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	generation, sequence := terminateTestRun(t, ctx, pool, "run-two-checks", "succeeded")
	if _, err := store.ObserveTerminal(ctx, auditstore.ObserveTerminalParams{
		Claim: claim, ExecutionID: execution.ExecutionID, RunID: "run-two-checks",
		Generation: generation, Sequence: sequence,
	}); err != nil {
		t.Fatal(err)
	}

	result := writeCollectionFixtureArtifact(t, ctx, pool, project.ProjectID, "audit-batched-checks", "result")
	collected := make([]auditstore.CollectionItem, len(members))
	for index, member := range members {
		collected[index] = auditstore.CollectionItem{
			ExecutionItemID: member.ExecutionItemID, Disposition: auditstore.CollectionAccepted,
			FinalDisposition: auditstore.FinalAccepted, Result: &result,
			Coverage: auditstore.Coverage{Status: auditstore.CoverageViolated, Requested: []string{}, Completed: []string{}, Gaps: []string{}},
			FindingAssociations: []auditstore.FindingAssociation{{
				AssessmentID: "assessment-" + member.ExecutionItemID, ReceiptID: receiptID,
				Proposal: proposal, SemanticAssessment: []string{"refuted", "supported"}[index],
			}},
		}
	}
	if _, inserted, err := store.Collect(ctx, auditstore.CollectParams{
		Claim: claim, ReceiptID: "receipt-collection-two-checks", ExecutionID: execution.ExecutionID,
		Disposition: auditstore.CollectionAccepted, SourceOutput: &result, RequestDigest: testDigest("6"),
		Retained: []auditstore.ArtifactLink{{Scope: "project", LogicalKey: "result/two-checks", Artifact: result,
			SourceProvenance: json.RawMessage(`{"runId":"run-two-checks"}`),
		}},
		Items: collected,
	}); err != nil || !inserted {
		t.Fatalf("collect batched checks of one proposal = (%t, %v)", inserted, err)
	}
	var findings, histories int
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM project_findings WHERE project_id=$1", project.ProjectID).Scan(&findings); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM artifact_bindings WHERE scope_kind='project' AND scope_id=$1 AND namespace='finding-history' AND name LIKE $2", project.ProjectID, auditdomain.FindingSourceHistoryPrefix(receiptID)+"check-result.%").Scan(&histories); err != nil {
		t.Fatal(err)
	}
	if findings != 0 || histories != 2 {
		t.Fatalf("ordinary checks created a Finding or lost independent history: findings=%d histories=%d", findings, histories)
	}
	settled, err := store.ListItems(ctx, audit.AuditID)
	if err != nil || len(settled) != 2 {
		t.Fatalf("Round items = (%+v, %v)", settled, err)
	}
	for _, item := range settled {
		if item.State != auditstore.ItemSettled || item.FinalDisposition == nil || *item.FinalDisposition != auditstore.FinalAccepted {
			t.Fatalf("batched finding check did not settle: %+v", item)
		}
	}
	for _, states := range [][2]auditstore.RoundState{{auditstore.RoundExecuting, auditstore.RoundAssessing}, {auditstore.RoundAssessing, auditstore.RoundClosed}} {
		if round, err = store.TransitionRound(ctx, auditstore.RoundTransitionParams{
			Claim: claim, RoundID: round.RoundID, ExpectedRevision: round.Revision,
			ExpectedState: states[0], TargetState: states[1],
		}); err != nil {
			t.Fatalf("progress Round to %q: %v", states[1], err)
		}
	}
	audit, err = store.Get(ctx, project.OwnerID, audit.AuditID)
	if err != nil || audit.Revision != audit.EventSequence {
		t.Fatalf("Audit revision/event sequence = (%+v, %v)", audit, err)
	}
}

// insertAuditChildFinding records a proposal receipt from the fixture's child
// Run and its Audit hold, which admits the finding the collection assesses.
func insertAuditChildFinding(
	t *testing.T, f collectingAuditFixture, receiptID string, proposal auditstore.ExactArtifact,
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
    proposal_size_bytes, evidence, publication_requested, project_proposal, project_evidence, retention_state, report_projection
, checks_projection) VALUES (
    $1, $1 || '-proposal', $1 || '-allocation', 'runtime-agent', 'runtime-instance',
    'stage-execution', 'worker', 'invocation', $1 || '-submission', 'candidate', $2,
    $3, $4, $5, $6, $7, 'check',
    'audit-check', '1', 'contractor/v1alpha1',
    '{"name":"audit-check","version":"1"}'::jsonb, $8,
    $9::jsonb, $10, 'application/json', $11, '[]'::jsonb, true, $12::jsonb, jsonb_build_array($12::jsonb), 'project-held',
    '{"description":"Test finding.","limitations":[],"preconditions":[],"severity_suggestion":"","standard_refs":[],"subject":null,"title":"Test finding."}'
, json_build_object('schema','contractor.audit.finding-checks-projection.v1','proposal_digest',($10)::text,'size_bytes',($11)::bigint,'discarded',false,'subject',NULL,'limitations',NULL,'checks','[]'::json))`,
		receiptID, testDigest("d"), *f.execution.RunID, f.audit.OwnerID, f.project.ProjectID,
		f.execution.ExecutionID, f.audit.AuditID, testDigest("e"),
		proposalRefJSON, proposal.Digest, proposal.SizeBytes, proposalJSON,
	); err != nil {
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
	if err := persistencepostgres.InTx(f.ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		var err error
		findingID, err = NewFindingEventsTx(tx).AdmitProjectFinding(f.ctx, f.project.ProjectID, receiptID, "finding-"+receiptID)
		return err
	}); err != nil {
		t.Fatal(err)
	}

	return findingID
}

type collectingAuditFixture struct {
	ctx       context.Context
	pool      *pgxpool.Pool
	store     *testStore
	project   projectstore.Project
	audit     auditstore.Audit
	claim     auditstore.ControllerClaim
	execution auditstore.Execution
	memberID  string
	itemID    string
}

type collectingAuditFixtureOptions struct {
	extraPending    bool
	terminalOutcome string
	// ownDatabase builds the fixture in a database of its own
	// (budgetAuditPool), for a test that budgets its reads.
	ownDatabase bool
}

// newCollectingAuditFixture leaves one execution collecting a terminal Run
// under a live Controller claim. An optional second item remains unattempted.
func newCollectingAuditFixture(t *testing.T, name string, maxEvidenceBytes int64, options ...collectingAuditFixtureOptions) collectingAuditFixture {
	t.Helper()
	var option collectingAuditFixtureOptions
	if len(options) != 0 {
		option = options[0]
	}
	if option.terminalOutcome == "" {
		option.terminalOutcome = "succeeded"
	}
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	var pool *pgxpool.Pool
	if option.ownDatabase {
		pool = budgetAuditPool(t, ctx, databaseURL)
	} else {
		pool = isolatedAuditPool(t, ctx, databaseURL)
	}
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "project-" + name, OwnerID: "owner-" + name, Kind: projectstore.KindProject,
		Name: "Audit " + name, IdempotencyKey: "project-create", RequestDigest: testDigest("1"),
	})
	if err != nil {
		t.Fatal(err)
	}
	store := newTestStore(pool)
	maxItems := 1
	if option.extraPending {
		maxItems = 2
	}
	audit, _, err := store.CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: "audit-" + name, OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:         auditstore.ProfileIdentity{Name: "checklist", Version: "1", Digest: testDigest("2")},
		ProfileSnapshot: json.RawMessage(`{"name":"checklist","workflows":{"check":{"kind":"check"}}}`),
		InputSelection:  json.RawMessage(`{}`),
		Limits: auditstore.Limits{
			MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: maxItems, MaxItemsTotal: maxItems,
			MaxSubmittedRuns: 2, MaxItemRunAttempts: 2, MaxEvidenceBytes: maxEvidenceBytes,
		},
		IdempotencyKey: "audit-create", RequestDigest: testDigest("3"),
	})
	if err != nil {
		t.Fatal(err)
	}
	roundID := "round-" + name
	task := writeCollectionFixtureArtifact(t, ctx, pool, project.ProjectID, "audits", name+"-task")
	item := auditstore.MaterializedItem{
		ItemID: "item-" + name, ItemKey: "check-" + name, Ordinal: 0, Kind: "checklist",
		SubjectKey: "subject-" + name, Task: task, Origin: testOrigin("check-" + name),
		WorkflowRole: "check", InitialState: auditstore.ItemReady, Coverage: emptyCoverage(),
	}
	items := []auditstore.MaterializedItem{item}
	if option.extraPending {
		items = append(items, auditstore.MaterializedItem{
			ItemID: "pending-item-" + name, ItemKey: "pending-check-" + name,
			Ordinal: 1, Kind: "checklist", SubjectKey: "pending-subject-" + name,
			Task:   testExact("audits", name+"-pending-task", "task-r1"),
			Origin: testOrigin("pending-check-" + name), WorkflowRole: "check",
			InitialState: auditstore.ItemReady, Coverage: emptyCoverage(),
		})
	}
	audit, _, err = store.MaterializeRound(ctx, auditstore.MaterializeRoundParams{
		OwnerID: audit.OwnerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision,
		RoundID: roundID, RoundOrdinal: 1, Manifest: testExact("audits", name+"-worklist", "worklist-r1"),
		BaselineSnapshot: json.RawMessage(`{"inputs":[],"skills":[]}`),
		DeadlineAt:       time.Now().Add(time.Hour), IdempotencyKey: "audit-start",
		RequestDigest: testDigest("4"), Items: items,
	})
	if err != nil {
		t.Fatal(err)
	}
	claims, err := store.Claim(ctx, auditstore.ClaimParams{HolderID: name + "-controller", Lease: 30 * time.Second, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim Audit = (%+v, %v)", claims, err)
	}
	claim := claims[0]
	if _, err := store.TransitionRound(ctx, auditstore.RoundTransitionParams{
		Claim: claim, RoundID: roundID, ExpectedRevision: 1,
		ExpectedState: auditstore.RoundAccepted, TargetState: auditstore.RoundExecuting,
	}); err != nil {
		t.Fatal(err)
	}
	memberID := "member-" + name
	execution, _, err := store.CreateExecutionIntent(ctx, auditstore.CreateExecutionIntentParams{
		Claim: claim, ExecutionID: "execution-" + name, RoundID: &roundID, Role: auditstore.ExecutionCheck,
		WorkflowRole: "check", LaneClass: resolved.LaneClassModel, Manifest: testExact("audits", name+"-execution", "manifest-r1"),
		SubmissionKey: "submission-" + name, RequestDigest: testDigest("7"),
		Members: []auditstore.ExecutionMemberIntent{{
			ExecutionItemID: memberID, ItemID: item.ItemID, BatchOrdinal: 0, ItemAttempt: 1,
			Task: task, Inputs: []auditstore.ExactArtifact{},
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
	generation, sequence := terminateTestRun(t, ctx, pool, runID, option.terminalOutcome)
	execution, err = store.ObserveTerminal(ctx, auditstore.ObserveTerminalParams{
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

// TestListItemAttemptsAnswersEveryRequestedItem pins the store's answer that
// the public item list and its handler-test double rely on: every requested
// item has an entry, an empty non-nil list for one without attempts.
func TestListItemAttemptsAnswersEveryRequestedItem(t *testing.T) {
	f := newCollectingAuditFixture(t, "attempt-entries", 1<<20, collectingAuditFixtureOptions{extraPending: true})
	waiting := "pending-item-attempt-entries"
	attempts, err := f.store.ListItemAttempts(f.ctx, f.audit.OwnerID, f.audit.AuditID, []string{f.itemID, waiting})
	if err != nil {
		t.Fatal(err)
	}
	if len(attempts) != 2 || len(attempts[f.itemID]) != 1 || attempts[f.itemID][0].ExecutionItemID != f.memberID ||
		attempts[waiting] == nil || len(attempts[waiting]) != 0 {
		t.Fatalf("attempts = %#v", attempts)
	}
}

// Accepted result retention exercises actual immutable versions and blobs;
// metadata-only descriptors cannot demonstrate a readable retained result.
func writeCollectionFixtureArtifact(t *testing.T, ctx context.Context, pool *pgxpool.Pool, projectID, namespace, name string) auditstore.ExactArtifact {
	t.Helper()
	data := []byte("Exact fixture material " + name)
	written, err := artifactstest.WriteAuditArtifact(ctx, artifacts.NewService(artifactpostgres.NewRepository(pool)), projectID, contracts.ArtifactRef{Namespace: namespace, Name: name}, artifacts.Payload{MediaType: "application/zip", Data: data})
	if err != nil {
		t.Fatal(err)
	}
	return auditstore.ExactArtifact{Ref: written.Ref, Digest: auditdomain.DigestBytes(data), MediaType: written.MediaType, SizeBytes: written.Size}
}

// A real accepted negative recheck retains its exact independent result while
// leaving the sole explicit analyst assessment unchanged, including on replay.
func TestPostgresNegativeRecheckRetainsHistoryWithoutChangingCanonicalFinding(t *testing.T) {
	f := newCollectingAuditFixture(t, "negative-recheck", 1<<20)
	proposal := writeCollectionFixtureArtifact(t, f.ctx, f.pool, f.project.ProjectID, artifactpolicy.FindingMaterialsNamespace, "negative-source")
	receiptID := "negative-source-receipt"
	findingID := insertAuditChildFinding(t, f, receiptID, proposal)
	var before auditstore.ProjectFinding
	err := persistencepostgres.InTx(f.ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		high := auditstore.SeverityHigh
		var err error
		before, err = NewOwnerTxStore(tx).AssessProjectFinding(f.ctx, auditstore.ProjectFindingEdit{OwnerID: f.audit.OwnerID, ProjectID: f.project.ProjectID, FindingID: findingID, ExpectedRevision: 1, Verdict: auditstore.VerdictTruePositive, Severity: &high, Rationale: "Explicit human assessment before recheck"})
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	result := writeCollectionFixtureArtifact(t, f.ctx, f.pool, f.project.ProjectID, auditdomain.ArtifactNamespace(f.audit.AuditID), "negative-result")
	params := auditstore.CollectParams{Claim: f.claim, ReceiptID: "negative-check-collection", ExecutionID: f.execution.ExecutionID, Disposition: auditstore.CollectionAccepted, SourceOutput: &result, RequestDigest: testDigest("6"), Retained: []auditstore.ArtifactLink{{Scope: "project", LogicalKey: "negative-result", Artifact: result, SourceProvenance: json.RawMessage(`{}`)}}, Items: []auditstore.CollectionItem{{ExecutionItemID: f.memberID, Disposition: auditstore.CollectionAccepted, FinalDisposition: auditstore.FinalAccepted, Result: &result, Coverage: auditstore.Coverage{Status: auditstore.CoverageInconclusive, Requested: []string{}, Completed: []string{}, Gaps: []string{}}, FindingAssociations: []auditstore.FindingAssociation{{AssessmentID: "negative-check-result", ReceiptID: receiptID, Proposal: proposal, SemanticAssessment: "refuted"}}}}}
	for attempt := 0; attempt < 2; attempt++ {
		_, inserted, err := f.store.Collect(f.ctx, params)
		if err != nil || inserted != (attempt == 0) {
			t.Fatalf("negative collection replay=%t err=%v", inserted, err)
		}
		current, err := NewQueryStore(f.pool).GetAuditProjectFinding(f.ctx, f.audit.OwnerID, f.audit.AuditID, findingID)
		if err != nil || !reflect.DeepEqual(before, current) {
			t.Fatalf("negative result changed current analyst state: %+v %v", current, err)
		}
	}
	project, err := artifacts.NewService(artifactpostgres.NewRepository(f.pool)).Project(f.project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	ref := contracts.ArtifactRef{Namespace: artifactpolicy.FindingHistoryNamespace, Name: auditdomain.FindingSourceHistoryName(receiptID, "check-result", "negative-check-result")}
	record, err := project.Read(f.ctx, ref)
	if err != nil {
		t.Fatal(err)
	}
	var fact struct {
		Original struct {
			Semantic string `json:"semanticAssessment"`
		} `json:"original"`
		Materials struct {
			Result contracts.ExactArtifact `json:"result"`
		} `json:"materials"`
	}
	if err := json.Unmarshal(record.Payload.Data, &fact); err != nil || fact.Original.Semantic != "refuted" || fact.Materials.Result.Ref.Namespace != artifactpolicy.FindingMaterialsNamespace {
		t.Fatalf("negative history lost its independent result: %+v %v", fact, err)
	}
	if _, err := project.ReadExact(f.ctx, fact.Materials.Result); err != nil {
		t.Fatal(err)
	}
}
