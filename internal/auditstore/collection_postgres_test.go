package auditstore

import (
	"context"
	"encoding/json"
	"errors"
	"os"
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
