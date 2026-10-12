package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresAcceptNextRoundIsAtomicReplaySafeAndConsumeOnce(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool := isolatedAuditPool(t, ctx, databaseURL)
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "project-next-round", OwnerID: "owner-next-round", Kind: projectstore.KindProject,
		Name: "Next Round", IdempotencyKey: "project-next-round", RequestDigest: testDigest("1"),
	})
	if err != nil {
		t.Fatal(err)
	}
	store := newTestStore(pool)
	audit, _, err := store.CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: "audit-next-round", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:         auditstore.ProfileIdentity{Name: "profile", Version: "1", Digest: testDigest("2")},
		ProfileSnapshot: json.RawMessage(`{"name":"profile"}`),
		InputSelection:  json.RawMessage(`{"inputs":{}}`),
		Limits: auditstore.Limits{
			MaxRounds: 3, BatchSize: 1, MaxItemsPerRound: 2, MaxItemsTotal: 3,
			MaxSubmittedRuns: 3, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1024,
		},
		IdempotencyKey: "audit-next-round", RequestDigest: testDigest("3"),
	})
	if err != nil {
		t.Fatal(err)
	}
	audit, _, err = store.MaterializeRound(ctx, auditstore.MaterializeRoundParams{
		OwnerID: project.OwnerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision,
		RoundID: "round-one", RoundOrdinal: 1, Manifest: testExact("audits", "round-one", "r1"),
		BaselineSnapshot: json.RawMessage(`{"inputs":{},"skills":[]}`),
		DeadlineAt:       time.Now().Add(time.Hour), Items: []auditstore.MaterializedItem{},
		IdempotencyKey: "start-next-round", RequestDigest: testDigest("4"),
	})
	if err != nil {
		t.Fatal(err)
	}
	claims, err := store.Claim(ctx, auditstore.ClaimParams{HolderID: "controller-next-round", Lease: 20 * time.Second, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim Audit = (%+v, %v)", claims, err)
	}
	claim := claims[0]
	closed := closeAuditRoundForTest(t, ctx, store, claim, "round-one", 1)

	proposal := testExact("audit-findings", "proposal-one", "proposal-r1")
	proposal.MediaType, proposal.SizeBytes = "application/json", 128
	insertAuditProposalHoldForRoundTest(t, ctx, pool, audit, proposal, "proposal-receipt-one")
	audit, err = store.Get(ctx, project.OwnerID, audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	params := auditstore.AcceptRoundParams{
		Claim: claim, ExpectedAuditRevision: audit.Revision,
		PreviousRoundID: closed.RoundID, RoundID: "round-two", RoundOrdinal: 2,
		Manifest: testExact("audits", "round-two", "r1"),
		Items: []auditstore.MaterializedItem{{
			ItemID: "item-round-two", ItemKey: "verify-proposal-one", Ordinal: 0,
			Kind: "finding-verification", SubjectKey: "subject-one",
			Task: testExact("audits", "round-two-task", "r1"), Origin: testOrigin("verify-proposal-one"),
			WorkflowRole: "verify", InitialState: auditstore.ItemAwaitingReview, Coverage: emptyCoverage(),
			ApprovalKind: auditstore.ItemApprovalActiveCheck, ApprovalDigest: testDigest("7"),
			ProposalSources: []auditstore.ProposalItemSource{{
				ReceiptID: "proposal-receipt-one", ProposedCheckOrdinal: 0, Proposal: proposal,
			}},
		}},
	}
	type result struct {
		round    auditstore.Round
		inserted bool
		err      error
	}
	results := make([]result, 2)
	var wait sync.WaitGroup
	for index := range results {
		wait.Add(1)
		go func(index int) {
			defer wait.Done()
			results[index].round, results[index].inserted, results[index].err = store.AcceptNextRound(ctx, params)
		}(index)
	}
	wait.Wait()
	insertions := 0
	for _, result := range results {
		if result.err != nil || result.round.RoundID != params.RoundID {
			t.Fatalf("concurrent next Round acceptance = (%+v, %t, %v)", result.round, result.inserted, result.err)
		}
		if result.inserted {
			insertions++
		}
	}
	if insertions != 1 {
		t.Fatalf("next Round insertions = %d, want 1", insertions)
	}

	drifted := params
	drifted.Items = append([]auditstore.MaterializedItem(nil), params.Items...)
	drifted.Items[0].SubjectKey = "different-subject"
	if _, inserted, err := store.AcceptNextRound(ctx, drifted); !errors.Is(err, auditstore.ErrConflict) || inserted {
		t.Fatalf("drifted acceptance replay = (%t, %v), want conflict", inserted, err)
	}
	var sourceCount int
	if err := pool.QueryRow(ctx, `
SELECT count(*) FROM audit_items
 WHERE audit_id = $1 AND proposal_receipt_id = 'proposal-receipt-one'
   AND proposal_check_ordinal = 0`, audit.AuditID).Scan(&sourceCount); err != nil || sourceCount != 1 {
		t.Fatalf("durable proposal consume fence = (%d, %v)", sourceCount, err)
	}
	var roundReviews, openedReviews int
	if err := pool.QueryRow(ctx, `
SELECT (event.summary->>'reviews')::int,
       (SELECT count(*) FROM audit_review_requests WHERE audit_id = event.audit_id)
  FROM audit_events AS event
 WHERE event.audit_id = $1 AND event.kind = 'round.accepted' AND event.entity_id = 'round-two'`,
		audit.AuditID).Scan(&roundReviews, &openedReviews); err != nil || roundReviews != 1 || openedReviews != 1 {
		t.Fatalf("next round.accepted reviews = (%d, %d opened, %v), want 1", roundReviews, openedReviews, err)
	}

	closeAuditRoundForTest(t, ctx, store, claim, "round-two", 1)
	rounds, err := store.ListRounds(ctx, audit.AuditID)
	if err != nil || len(rounds) != 2 || rounds[0].RoundID != "round-one" || rounds[1].RoundID != "round-two" ||
		rounds[1].ExpectedItemCount != 1 || rounds[1].State != auditstore.RoundClosed {
		t.Fatalf("list Audit rounds = (%+v, %v)", rounds, err)
	}
	audit, err = store.Get(ctx, project.OwnerID, audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	reused := params
	reused.ExpectedAuditRevision = audit.Revision
	reused.PreviousRoundID, reused.RoundID, reused.RoundOrdinal = "round-two", "round-three", 3
	reused.Manifest = testExact("audits", "round-three", "r1")
	reused.Items = append([]auditstore.MaterializedItem(nil), params.Items...)
	reused.Items[0].ItemID, reused.Items[0].ItemKey = "item-round-three", "verify-proposal-again"
	reused.Items[0].Origin = testOrigin("verify-proposal-again")
	if _, inserted, err := store.AcceptNextRound(ctx, reused); !errors.Is(err, auditstore.ErrPrecondition) || inserted {
		t.Fatalf("reused proposal check = (%t, %v), want precondition", inserted, err)
	}
}

func closeAuditRoundForTest(
	t *testing.T,
	ctx context.Context,
	store *testStore,
	claim auditstore.ControllerClaim,
	roundID string,
	revision uint64,
) auditstore.Round {
	t.Helper()
	states := [][2]auditstore.RoundState{
		{auditstore.RoundAccepted, auditstore.RoundExecuting},
		{auditstore.RoundExecuting, auditstore.RoundAssessing},
		{auditstore.RoundAssessing, auditstore.RoundClosed},
	}
	var round auditstore.Round
	var err error
	for _, states := range states {
		round, err = store.TransitionRound(ctx, auditstore.RoundTransitionParams{
			Claim: claim, RoundID: roundID, ExpectedRevision: revision,
			ExpectedState: states[0], TargetState: states[1],
		})
		if err != nil {
			t.Fatalf("transition Round %q to %q: %v", roundID, states[1], err)
		}
		revision = round.Revision
	}
	return round
}

func insertAuditProposalHoldForRoundTest(
	t *testing.T,
	ctx context.Context,
	db *pgxpool.Pool,
	audit auditstore.Audit,
	proposal auditstore.ExactArtifact,
	receiptID string,
) {
	t.Helper()
	proposalRef, err := json.Marshal(proposal.Ref)
	if err != nil {
		t.Fatal(err)
	}
	proposalDescriptor, err := json.Marshal(proposal)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(ctx, `
INSERT INTO finding_proposal_receipts (
    receipt_id, proposal_id, allocation_id, runtime_agent_id,
    runtime_instance_id, stage_execution_id, logical_agent_name,
    invocation_id, submission_id, client_key, request_digest,
    run_id, owner_id, project_id,
    workflow_name, workflow_version, workflow_schema_version,
    workflow_configuration_ref, workflow_closure_digest,
    proposal_ref, proposal_digest, proposal_media_type,
    proposal_size_bytes, evidence, report_projection
) VALUES (
    $1, 'proposal-one', 'allocation-one', 'runtime-one', 'instance-one',
    'stage-one', 'worker', 'invocation-one', 'submission-one', 'candidate-one', $2,
    'run-source-one', $3, $4,
    'source-workflow', '1', 'contractor/v1alpha1',
    '{"name":"source-workflow","version":"1"}'::jsonb, $5,
    $6::jsonb, $7, 'application/json', $8, '[]'::jsonb,
    '{"description":"Test finding.","limitations":[],"severity_suggestion":"","standard_refs":[],"subject":null,"title":"Test finding."}'
)`, receiptID, testDigest("5"), audit.OwnerID, audit.ProjectID,
		testDigest("6"), proposalRef, proposal.Digest, proposal.SizeBytes); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(ctx, `
INSERT INTO finding_proposal_audit_holds (
    receipt_id, audit_id, project_id, proposal_ref, evidence
) VALUES ($1, $2, $3, $4::jsonb, '[]'::jsonb)`,
		receiptID, audit.AuditID, audit.ProjectID, proposalDescriptor); err != nil {
		t.Fatal(err)
	}
}
