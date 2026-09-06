package auditservice

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	managedcredentials "github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestAuditDraftStartReplayAndAtomicUnsupportedRollback(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, databaseURL)
	profileSnapshot := loadAuditServiceProfiles(t)
	profiles := &switchableProfileCatalog{snapshot: profileSnapshot, available: true}
	gateway, err := profileSnapshot.LLMGateway("test-gateway@1")
	if err != nil {
		t.Fatal(err)
	}
	credentials := &switchableCredentialLookup{available: true, gateway: gateway.Ref}
	guard := &countingCredentialGuard{}
	service, err := New(Options{
		Pool: pool, Profiles: profiles,
		TransactionLLMCredentials: runtimeconfig.TransactionLLMCredentialLookupFactoryFunc(
			func(pgx.Tx) (config.CredentialLookup, error) { return credentials, nil },
		),
		CredentialGuard: guard,
		Now:             time.Now,
	})
	if err != nil {
		t.Fatal(err)
	}

	projects := projectstore.NewPostgresStore(pool)
	project, _, err := projects.Create(ctx, projectstore.CreateParams{
		ProjectID: "project-audit-api", OwnerID: "owner-audit-api", Kind: projectstore.KindProject,
		Name: "Audit API project", IdempotencyKey: "create-project",
		RequestDigest: serviceTestDigest("project"),
	})
	if err != nil {
		t.Fatal(err)
	}
	projectArtifacts, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	automatic := writeChecklist(t, ctx, projectArtifacts, "automatic", "automatic")
	manual := writeChecklist(t, ctx, projectArtifacts, "manual", "manual")

	draftParams := CreateDraftParams{
		AuditID: "audit-api-one", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:       ProfileSelector{Name: "test-checklist", Version: "1"},
		Inputs:        map[string]contracts.ArtifactRef{"checklist": automatic.Ref},
		RuntimeLabels: []string{}, Scope: Scope{Objective: "Review the service"},
		IdempotencyKey: "create-audit", RequestDigest: serviceTestDigest("create-audit"),
	}
	draft, created, err := service.CreateDraft(ctx, draftParams)
	if err != nil || !created || draft.State != auditstore.AuditDraft || draft.Revision != 1 {
		t.Fatalf("create draft = (%+v, %t, %v)", draft, created, err)
	}
	profiles.setAvailable(false)
	replayedDraft, created, err := service.CreateDraft(ctx, draftParams)
	if err != nil || created || replayedDraft.AuditID != draft.AuditID {
		t.Fatalf("create replay after catalog removal = (%+v, %t, %v)", replayedDraft, created, err)
	}

	started, err := service.Start(ctx, StartParams{
		OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision,
		IdempotencyKey: "start-audit", RequestDigest: serviceTestDigest("start-audit"),
	})
	if err != nil || started.Replayed || started.Audit.State != auditstore.AuditActive ||
		started.Audit.Hold != auditstore.HoldHeld || started.Audit.Revision != 2 ||
		len(started.Items) != 1 || started.Round.ExpectedItemCount != 1 {
		t.Fatalf("start = (%+v, %v)", started, err)
	}
	baseline, err := DecodeBaseline(started.Audit.BaselineSnapshot)
	if err != nil || len(baseline.LLMCredentialIDs) != 1 || baseline.LLMCredentialIDs[0] != "development-worker" ||
		baseline.Inventory.Worklist.Ref.Revision == nil || len(baseline.Skills) != 0 {
		t.Fatalf("baseline = (%+v, %v)", baseline, err)
	}
	holds, err := auditstore.NewPostgresStore(pool).ListHeldAuditIDsByLLMCredential(ctx, "development-worker", 10)
	if err != nil || len(holds) != 1 || holds[0] != draft.AuditID {
		t.Fatalf("credential holds = (%v, %v)", holds, err)
	}
	pauseParams := MutationParams{
		OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: started.Audit.Revision,
		IdempotencyKey: "pause-audit", RequestDigest: serviceTestDigest("pause-audit"),
	}
	paused, err := service.Pause(ctx, pauseParams)
	if err != nil || paused.Replayed || paused.Audit.State != auditstore.AuditPaused || paused.Audit.Revision != 3 {
		t.Fatalf("pause = (%+v, %v)", paused, err)
	}
	replayedPause, err := service.Pause(ctx, pauseParams)
	if err != nil || !replayedPause.Replayed || replayedPause.Audit.State != auditstore.AuditPaused {
		t.Fatalf("pause replay = (%+v, %v)", replayedPause, err)
	}
	resumed, err := service.Resume(ctx, MutationParams{
		OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: paused.Audit.Revision,
		IdempotencyKey: "resume-audit", RequestDigest: serviceTestDigest("resume-audit"),
	})
	if err != nil || resumed.Audit.State != auditstore.AuditActive || resumed.Audit.Revision != 4 {
		t.Fatalf("resume = (%+v, %v)", resumed, err)
	}
	cancelled, err := service.Cancel(ctx, MutationParams{
		OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: resumed.Audit.Revision,
		IdempotencyKey: "cancel-audit", RequestDigest: serviceTestDigest("cancel-audit"),
	})
	if err != nil || cancelled.Audit.State != auditstore.AuditCancelling || cancelled.Audit.Dispatch != auditstore.DispatchClosed {
		t.Fatalf("cancel = (%+v, %v)", cancelled, err)
	}
	deleting, err := service.Delete(ctx, MutationParams{
		OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: cancelled.Audit.Revision,
		IdempotencyKey: "delete-audit", RequestDigest: serviceTestDigest("delete-audit"),
	})
	if err != nil || deleting.Audit.State != auditstore.AuditCancelling || deleting.Audit.DeletionRequestedAt == nil {
		t.Fatalf("delete intent = (%+v, %v)", deleting, err)
	}
	coverage, err := service.ListCoverage(ctx, project.OwnerID, draft.AuditID, started.Round.RoundID, -1, 10)
	if err != nil || len(coverage) != 1 || coverage[0].Ordinal != 0 || coverage[0].Coverage.Status != auditstore.CoverageNotTested {
		t.Fatalf("initial coverage = (%+v, %v)", coverage, err)
	}
	credentials.setAvailable(false)
	guardCalls := guard.callsCount()
	replayedStart, err := service.Start(ctx, StartParams{
		OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision,
		IdempotencyKey: "start-audit", RequestDigest: serviceTestDigest("start-audit"),
	})
	if err != nil || !replayedStart.Replayed || replayedStart.Audit.AuditID != draft.AuditID || guard.callsCount() != guardCalls {
		t.Fatalf("start replay after credential removal = (%+v, %v), guard=%d", replayedStart, err, guard.callsCount())
	}
	if _, err := service.Get(ctx, "other-owner", draft.AuditID); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("foreign owner get = %v", err)
	}
	foreignProjectID := project.ProjectID
	if listed, err := service.List(ctx, auditstore.ListParams{
		OwnerID: "other-owner", ProjectID: &foreignProjectID, Limit: 10,
	}); err != nil || len(listed) != 0 {
		t.Fatalf("foreign owner list = (%+v, %v)", listed, err)
	}
	if _, err := service.ListItems(ctx, auditstore.ListItemsParams{
		OwnerID: "other-owner", AuditID: draft.AuditID, Limit: 10,
	}); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("foreign owner item list = %v", err)
	}
	if _, err := service.ListCoverage(
		ctx, "other-owner", draft.AuditID, started.Round.RoundID, -1, 10,
	); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("foreign owner coverage = %v", err)
	}
	if _, err := service.Start(ctx, StartParams{
		OwnerID: "other-owner", AuditID: draft.AuditID, ExpectedRevision: draft.Revision,
		IdempotencyKey: "foreign-start", RequestDigest: serviceTestDigest("foreign-start"),
	}); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("foreign owner start = %v", err)
	}

	profiles.setAvailable(true)
	credentials.setAvailable(true)
	manualDraft, _, err := service.CreateDraft(ctx, CreateDraftParams{
		AuditID: "audit-api-manual", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:       ProfileSelector{Name: "test-checklist", Version: "1"},
		Inputs:        map[string]contracts.ArtifactRef{"checklist": manual.Ref},
		RuntimeLabels: []string{}, Scope: Scope{},
		IdempotencyKey: "create-manual", RequestDigest: serviceTestDigest("create-manual"),
	})
	if err != nil {
		t.Fatal(err)
	}
	manualStarted, err := service.Start(ctx, StartParams{
		OwnerID: project.OwnerID, AuditID: manualDraft.AuditID, ExpectedRevision: manualDraft.Revision,
		IdempotencyKey: "start-manual", RequestDigest: serviceTestDigest("start-manual"),
	})
	if err != nil || len(manualStarted.Items) != 1 ||
		manualStarted.Items[0].State != auditstore.ItemAwaitingReview ||
		manualStarted.Items[0].ApprovalKind != auditstore.ItemApprovalApplicability ||
		manualStarted.Items[0].ApprovalDigest == "" {
		t.Fatalf("manual start = (%+v, %v)", manualStarted, err)
	}
	storedManual, err := service.Get(ctx, project.OwnerID, manualDraft.AuditID)
	if err != nil || storedManual.State != auditstore.AuditActive || storedManual.BaselineSnapshot == nil ||
		storedManual.Hold != auditstore.HoldHeld {
		t.Fatalf("manual Audit state = (%+v, %v)", storedManual, err)
	}
	reviews, err := service.ListReviews(ctx, ReviewListParams{
		OwnerID: project.OwnerID, AuditID: manualDraft.AuditID, Limit: 10,
	})
	if err != nil || len(reviews) != 1 || reviews[0].Kind != ApplicabilityReviewKind ||
		reviews[0].SubjectKind != ReviewSubjectItemAction ||
		reviews[0].SubjectID != manualStarted.Items[0].ItemID {
		t.Fatalf("manual item reviews = (%+v, %v)", reviews, err)
	}
	approved, err := service.DecideActionReview(ctx, DecideActionReviewParams{
		OwnerID: project.OwnerID, AuditID: manualDraft.AuditID,
		RequestID: reviews[0].RequestID, ExpectedRequestRevision: reviews[0].Revision,
		DecisionID: "decision-manual-approve", Action: ReviewApprove,
		Rationale:      "The owner approved this exact checklist action.",
		IdempotencyKey: "decision-manual-approve",
		RequestDigest:  serviceTestDigest("decision-manual-approve"),
	})
	if err != nil || approved.Decision.Action != ReviewApprove ||
		approved.Request.State != ReviewDecided {
		t.Fatalf("approve manual item = (%+v, %v)", approved, err)
	}
	approvedItems, err := service.ListItems(ctx, auditstore.ListItemsParams{
		OwnerID: project.OwnerID, AuditID: manualDraft.AuditID, Limit: 10,
	})
	if err != nil || len(approvedItems) != 1 || approvedItems[0].State != auditstore.ItemReady {
		t.Fatalf("approved manual item state = (%+v, %v)", approvedItems, err)
	}
	claims, err := auditstore.NewPostgresStore(pool).Claim(ctx, auditstore.ClaimParams{
		HolderID: "controller-manual-approval", Lease: 20 * time.Second, Limit: 10,
	})
	if err != nil {
		t.Fatal(err)
	}
	var manualClaim auditstore.ControllerClaim
	for _, claim := range claims {
		if claim.AuditID == manualDraft.AuditID {
			manualClaim = claim
		} else {
			_ = auditstore.NewPostgresStore(pool).ReleaseClaim(ctx, claim)
		}
	}
	if manualClaim.AuditID == "" {
		t.Fatal("manual Audit was not claimable after approval")
	}
	if _, err := auditstore.NewPostgresStore(pool).TransitionRound(ctx, auditstore.RoundTransitionParams{
		Claim: manualClaim, RoundID: manualStarted.Round.RoundID,
		ExpectedRevision: manualStarted.Round.Revision,
		ExpectedState:    auditstore.RoundAccepted, TargetState: auditstore.RoundExecuting,
	}); err != nil {
		currentAudit, auditErr := auditstore.NewPostgresStore(pool).Get(ctx, project.OwnerID, manualDraft.AuditID)
		currentRound, roundErr := auditstore.NewPostgresStore(pool).GetRound(
			ctx, manualDraft.AuditID, manualStarted.Round.RoundID,
		)
		t.Fatalf("transition approved manual Round: %v; audit=(state=%s revision=%d deadline=%v, %v); round=(state=%s revision=%d, %v)",
			err, currentAudit.State, currentAudit.Revision, currentAudit.DeadlineAt, auditErr,
			currentRound.State, currentRound.Revision, roundErr)
	}
	if _, err := pool.Exec(ctx, `
UPDATE audit_review_requests SET expires_at = clock_timestamp() - interval '1 second'
 WHERE request_id = $1`, reviews[0].RequestID); err != nil {
		t.Fatal(err)
	}
	roundID := manualStarted.Round.RoundID
	_, inserted, err := auditstore.NewPostgresStore(pool).CreateExecutionIntent(ctx,
		auditstore.CreateExecutionIntentParams{
			Claim: manualClaim, ExecutionID: "execution-expired-approval",
			RoundID: &roundID, Role: auditstore.ExecutionCheck, WorkflowRole: "check",
			Manifest:      manualStarted.Round.Manifest,
			SubmissionKey: "expired-approval", RequestDigest: serviceTestDigest("expired-approval"),
			Members: []auditstore.ExecutionMemberIntent{{
				ExecutionItemID: "member-expired-approval", ItemID: approvedItems[0].ItemID,
				BatchOrdinal: 0, ItemAttempt: 1, Task: approvedItems[0].Task,
				Inputs: []auditstore.ExactArtifact{},
			}},
		})
	if !errors.Is(err, auditstore.ErrPrecondition) || inserted {
		t.Fatalf("expired exact approval execution = (%t, %v)", inserted, err)
	}
	_ = auditstore.NewPostgresStore(pool).ReleaseClaim(ctx, manualClaim)

	rejectedDraft, _, err := service.CreateDraft(ctx, CreateDraftParams{
		AuditID: "audit-api-manual-rejected", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile: ProfileSelector{Name: "test-checklist", Version: "1"},
		Inputs:  map[string]contracts.ArtifactRef{"checklist": manual.Ref},
		Scope:   Scope{}, IdempotencyKey: "create-manual-rejected",
		RequestDigest: serviceTestDigest("create-manual-rejected"),
	})
	if err != nil {
		t.Fatal(err)
	}
	rejectedStarted, err := service.Start(ctx, StartParams{
		OwnerID: project.OwnerID, AuditID: rejectedDraft.AuditID,
		ExpectedRevision: rejectedDraft.Revision, IdempotencyKey: "start-manual-rejected",
		RequestDigest: serviceTestDigest("start-manual-rejected"),
	})
	if err != nil {
		t.Fatal(err)
	}
	rejectedReviews, err := service.ListReviews(ctx, ReviewListParams{
		OwnerID: project.OwnerID, AuditID: rejectedDraft.AuditID, Limit: 10,
	})
	if err != nil || len(rejectedReviews) != 1 {
		t.Fatalf("rejected item review = (%+v, %v)", rejectedReviews, err)
	}
	if _, err := service.DecideActionReview(ctx, DecideActionReviewParams{
		OwnerID: project.OwnerID, AuditID: rejectedDraft.AuditID,
		RequestID:               rejectedReviews[0].RequestID,
		ExpectedRequestRevision: rejectedReviews[0].Revision,
		DecisionID:              "decision-manual-reject", Action: ReviewReject,
		Rationale:      "The exact manual action is outside this assessment.",
		IdempotencyKey: "decision-manual-reject",
		RequestDigest:  serviceTestDigest("decision-manual-reject"),
	}); err != nil {
		t.Fatal(err)
	}
	rejectedItems, err := service.ListItems(ctx, auditstore.ListItemsParams{
		OwnerID: project.OwnerID, AuditID: rejectedDraft.AuditID, Limit: 10,
	})
	if err != nil || len(rejectedItems) != 1 || rejectedItems[0].State != auditstore.ItemSettled ||
		rejectedItems[0].FinalDisposition == nil ||
		*rejectedItems[0].FinalDisposition != auditstore.FinalExcluded {
		t.Fatalf("rejected exact item = (%+v, %v)", rejectedItems, err)
	}
	rejectedCoverage, err := service.ListCoverage(
		ctx, project.OwnerID, rejectedDraft.AuditID, rejectedStarted.Round.RoundID, -1, 10,
	)
	if err != nil || len(rejectedCoverage) != 1 ||
		rejectedCoverage[0].Coverage.Status != auditstore.CoverageExcluded {
		t.Fatalf("rejected item coverage = (%+v, %v)", rejectedCoverage, err)
	}
	var roundCount, artifactCount int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM audit_rounds WHERE audit_id = $1`, manualDraft.AuditID).Scan(&roundCount); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, `
SELECT count(*)
  FROM artifact_bindings
 WHERE scope_kind = 'project' AND scope_id = $1 AND namespace LIKE 'audit-%'`, project.ProjectID).Scan(&artifactCount); err != nil {
		t.Fatal(err)
	}
	if roundCount != 1 || artifactCount != 6 {
		t.Fatalf("manual materialization rows = rounds %d, Audit bindings %d", roundCount, artifactCount)
	}
}

func TestAuditStartUsesOwningTransactionWithSaturatedPool(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, databaseURL)
	profiles := loadAuditServiceProfiles(t)
	gateway, err := profiles.LLMGateway("test-gateway@1")
	if err != nil {
		t.Fatal(err)
	}
	development, err := managedcredentials.NewStaticProvider([]managedcredentials.StaticEntry{{
		Metadata: config.CredentialMetadata{
			Ref:        contracts.LLMCredentialRef{CredentialID: "development-worker"},
			LLMGateway: gateway.Ref, Unrestricted: true,
		},
		Token: contracts.NewSecretString("development-test-token"),
	}})
	if err != nil {
		t.Fatal(err)
	}
	transactionCredentials, err := managedcredentials.NewTransactionLookupFactory(development)
	if err != nil {
		t.Fatal(err)
	}

	limitedConfig := pool.Config()
	limitedConfig.MaxConns = 2
	limited, err := pgxpool.NewWithConfig(ctx, limitedConfig)
	if err != nil {
		t.Fatal(err)
	}
	defer limited.Close()
	service, err := New(Options{
		Pool: limited, Profiles: profiles,
		TransactionLLMCredentials: transactionCredentials,
		CredentialGuard:           &countingCredentialGuard{},
	})
	if err != nil {
		t.Fatal(err)
	}

	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "project-saturated-audit", OwnerID: "owner-saturated-audit",
		Kind: projectstore.KindProject, Name: "Saturated Audit",
		IdempotencyKey: "project-saturated-audit", RequestDigest: serviceTestDigest("saturated-project"),
	})
	if err != nil {
		t.Fatal(err)
	}
	projectArtifacts, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	checklist := writeChecklist(t, ctx, projectArtifacts, "saturated", "automatic")
	draft, _, err := service.CreateDraft(ctx, CreateDraftParams{
		AuditID: "audit-saturated", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:        ProfileSelector{Name: "test-checklist", Version: "1"},
		Inputs:         map[string]contracts.ArtifactRef{"checklist": checklist.Ref},
		Scope:          Scope{Objective: "Exercise transaction-bound credential validation"},
		IdempotencyKey: "create-saturated-audit", RequestDigest: serviceTestDigest("saturated-create"),
	})
	if err != nil {
		t.Fatal(err)
	}

	listener, err := limited.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Release()
	if _, err := listener.Exec(ctx, `LISTEN contractor_audit_transaction_test`); err != nil {
		t.Fatal(err)
	}
	startCtx, startCancel := context.WithTimeout(ctx, 5*time.Second)
	defer startCancel()
	started, err := service.Start(startCtx, StartParams{
		OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision,
		IdempotencyKey: "start-saturated-audit", RequestDigest: serviceTestDigest("saturated-start"),
	})
	if err != nil {
		t.Fatalf("start Audit with one LISTEN connection and one transaction: %v", err)
	}
	if started.Audit.State != auditstore.AuditActive || len(started.Items) != 1 {
		t.Fatalf("started saturated Audit = %+v", started)
	}
}

func TestAuditReportAcceptanceUsesFrozenCandidate(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, databaseURL)
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "project-report-review", OwnerID: "owner-report-review",
		Kind: projectstore.KindProject, Name: "Report review",
		IdempotencyKey: "project-report-review", RequestDigest: serviceTestDigest("report-project"),
	})
	if err != nil {
		t.Fatal(err)
	}
	store := auditstore.NewPostgresStore(pool)
	audit, _, err := store.CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: "audit-report-review", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:         auditstore.ProfileIdentity{Name: "report-review", Version: "1", Digest: serviceTestDigest("profile")},
		ProfileSnapshot: json.RawMessage(`{"interaction":{"reportAcceptance":"human-required"}}`),
		InputSelection:  json.RawMessage(`{"inputs":{}}`),
		Limits: auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1,
			MaxItemsTotal: 1, MaxSubmittedRuns: 1, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1024},
		IdempotencyKey: "audit-report-review", RequestDigest: serviceTestDigest("report-audit"),
	})
	if err != nil {
		t.Fatal(err)
	}
	findingID := seedAuditFinding(
		t, ctx, pool, project.ProjectID, project.OwnerID, audit.AuditID, "report-freeze",
	)
	intake, err := findingintake.New(pool)
	if err != nil {
		t.Fatal(err)
	}
	reviewService := &Service{pool: pool, findings: intake, now: time.Now}
	audit, err = store.Get(ctx, project.OwnerID, audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	revision := "report-manifest-r1"
	manifest := auditstore.ExactArtifact{Ref: contracts.ArtifactRef{
		Namespace: "audit-report-review", Name: "round", Revision: &revision,
	}, Digest: serviceTestDigest("manifest")}
	audit, _, err = store.MaterializeRound(ctx, auditstore.MaterializeRoundParams{
		OwnerID: project.OwnerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision,
		RoundID: "round-report-review", RoundOrdinal: 1, Manifest: manifest,
		BaselineSnapshot: json.RawMessage(`{"inputs":{},"skills":[]}`),
		DeadlineAt:       time.Now().Add(time.Hour), Items: []auditstore.MaterializedItem{},
		IdempotencyKey: "start-report-review", RequestDigest: serviceTestDigest("report-start"),
	})
	if err != nil {
		t.Fatal(err)
	}
	claims, err := store.Claim(ctx, auditstore.ClaimParams{
		HolderID: "report-review-controller", Lease: 20 * time.Second, Limit: 1,
	})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim report Audit = (%+v, %v)", claims, err)
	}
	claim := claims[0]
	round, err := store.GetRound(ctx, audit.AuditID, "round-report-review")
	if err != nil {
		t.Fatal(err)
	}
	for _, transition := range []struct {
		from, to auditstore.RoundState
	}{
		{auditstore.RoundAccepted, auditstore.RoundExecuting},
		{auditstore.RoundExecuting, auditstore.RoundAssessing},
		{auditstore.RoundAssessing, auditstore.RoundClosed},
	} {
		round, err = store.TransitionRound(ctx, auditstore.RoundTransitionParams{
			Claim: claim, RoundID: round.RoundID, ExpectedRevision: round.Revision,
			ExpectedState: transition.from, TargetState: transition.to,
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	audit, err = store.Get(ctx, project.OwnerID, audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	audit, err = store.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{
		Claim: claim, ExpectedRevision: audit.Revision, ExpectedState: auditstore.AuditActive,
		TargetState: auditstore.AuditFinalizing,
	})
	if err != nil {
		t.Fatal(err)
	}
	artifact := func(name, media string, size int64) auditstore.ExactArtifact {
		revision := name + "-r1"
		return auditstore.ExactArtifact{Ref: contracts.ArtifactRef{
			Namespace: "audit-report-review", Name: name, Revision: &revision,
		}, Digest: serviceTestDigest(name), MediaType: media, SizeBytes: size}
	}
	provenance := json.RawMessage(`{"schema":"contractor.audit.report-provenance.v1"}`)
	params := auditstore.ProposeReportParams{
		Claim: claim, ExpectedAuditRevision: audit.Revision,
		RoundID: round.RoundID, ExpectedRoundRevision: round.Revision,
		Machine: auditstore.ArtifactLink{LogicalKey: auditstore.ReportMachineLogicalKey,
			Artifact: artifact("report.json", "application/json", 32), SourceProvenance: provenance},
		Summary: auditstore.ArtifactLink{LogicalKey: auditstore.ReportSummaryLogicalKey,
			Artifact: artifact("report.txt", "text/plain", 16), SourceProvenance: provenance},
		RequestDigest: serviceTestDigest("report-candidate"),
	}
	waiting, inserted, err := store.ProposeReport(ctx, params)
	if err != nil || !inserted || waiting.State != auditstore.AuditWaitingReview {
		t.Fatalf("propose report = (%+v, %t, %v)", waiting, inserted, err)
	}
	if replay, inserted, err := store.ProposeReport(ctx, params); err != nil || inserted ||
		replay.State != auditstore.AuditWaitingReview {
		t.Fatalf("replay report proposal = (%+v, %t, %v)", replay, inserted, err)
	}
	candidate, err := store.GetReportCandidate(ctx, audit.AuditID)
	if err != nil || candidate.SubjectDigest != params.RequestDigest ||
		candidate.Machine.Artifact.Digest != params.Machine.Artifact.Digest {
		t.Fatalf("report candidate = (%+v, %v)", candidate, err)
	}
	// Exercise the claim-bound expiry transaction without consuming the
	// candidate used by the acceptance assertions below.
	expiryTx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := expiryTx.Exec(ctx, `
UPDATE audit_review_requests
   SET expires_at = clock_timestamp() - interval '1 second'
 WHERE request_id = $1`, candidate.RequestID); err != nil {
		_ = expiryTx.Rollback(ctx)
		t.Fatal(err)
	}
	changed, err := auditstore.NewPostgresStore(expiryTx).ExpireReportReview(
		ctx, claim, waiting.Revision,
	)
	if err != nil || !changed {
		_ = expiryTx.Rollback(ctx)
		t.Fatalf("expire exact report review = (%t, %v)", changed, err)
	}
	expiredAudit, err := auditstore.NewPostgresStore(expiryTx).Get(
		ctx, project.OwnerID, audit.AuditID,
	)
	if err != nil || expiredAudit.State != auditstore.AuditFailed ||
		expiredAudit.StopReason == nil || expiredAudit.StopReason.Code != "report_acceptance_expired" {
		_ = expiryTx.Rollback(ctx)
		t.Fatalf("expired report transaction = (%+v, %v)", expiredAudit, err)
	}
	if err := expiryTx.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	rejectTx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateAndApplyReportDecision(
		ctx, rejectTx, audit.AuditID, candidate.RequestID, audit.AuditID,
		int64(candidate.SubjectRevision), candidate.SubjectDigest, ReviewReject,
	); err != nil {
		_ = rejectTx.Rollback(ctx)
		t.Fatalf("reject exact report candidate: %v", err)
	}
	var rejectedState auditstore.AuditState
	var rejectedReason *string
	if err := rejectTx.QueryRow(ctx, `
SELECT state, stop_reason_code FROM audits WHERE audit_id = $1`, audit.AuditID).Scan(
		&rejectedState, &rejectedReason,
	); err != nil || rejectedState != auditstore.AuditFailed || rejectedReason == nil ||
		*rejectedReason != "report_rejected" {
		_ = rejectTx.Rollback(ctx)
		t.Fatalf("rejected report transaction = (%s, %v, %v)", rejectedState, rejectedReason, err)
	}
	if err := rejectTx.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	if err := store.ReleaseClaim(ctx, claim); err != nil {
		t.Fatal(err)
	}
	claims, err = store.Claim(ctx, auditstore.ClaimParams{
		HolderID: "report-review-recovery", Lease: 20 * time.Second, Limit: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(claims) != 0 {
		t.Fatalf("non-expired report review was claimed for reconciliation: %+v", claims)
	}
	if _, err := reviewService.CreateFindingReview(ctx, CreateFindingReviewParams{
		OwnerID: project.OwnerID, AuditID: audit.AuditID, FindingID: findingID,
		ExpectedRevision: 1, RequestID: "review-during-report-acceptance",
		IdempotencyKey: "review-during-report-acceptance",
		RequestDigest:  serviceTestDigest("review-during-report-acceptance"),
	}); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("finding review during report acceptance error = %v", err)
	}
	if _, err := reviewService.Pause(ctx, MutationParams{
		OwnerID: project.OwnerID, AuditID: audit.AuditID, ExpectedRevision: waiting.Revision,
		IdempotencyKey: "pause-report-review", RequestDigest: serviceTestDigest("pause-report-review"),
	}); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("pause during report acceptance error = %v", err)
	}
	review, err := reviewService.GetReview(
		ctx, project.OwnerID, audit.AuditID, candidate.RequestID,
	)
	if err != nil || review.Kind != ReportAcceptanceReviewKind ||
		review.SubjectKind != ReviewSubjectReport {
		t.Fatalf("report review = (%+v, %v)", review, err)
	}
	decision, err := reviewService.DecideActionReview(ctx, DecideActionReviewParams{
		OwnerID: project.OwnerID, AuditID: audit.AuditID, RequestID: review.RequestID,
		ExpectedRequestRevision: review.Revision, DecisionID: "decision-report-approve",
		Action: ReviewApprove, Rationale: "The owner accepts this exact report.",
		IdempotencyKey: "decision-report-approve", RequestDigest: serviceTestDigest("report-approve"),
	})
	if err != nil || decision.Decision.Action != ReviewApprove {
		t.Fatalf("approve report = (%+v, %v)", decision, err)
	}
	completed, err := store.Get(ctx, project.OwnerID, audit.AuditID)
	if err != nil || completed.State != auditstore.AuditCompleted {
		t.Fatalf("completed report Audit = (%+v, %v)", completed, err)
	}
	for _, key := range []string{auditstore.ReportMachineLogicalKey, auditstore.ReportSummaryLogicalKey} {
		if _, err := store.GetArtifactLink(ctx, audit.AuditID, key); err != nil {
			t.Fatalf("accepted report link %q: %v", key, err)
		}
	}
	if _, err := reviewService.CreateFindingReview(ctx, CreateFindingReviewParams{
		OwnerID: project.OwnerID, AuditID: audit.AuditID, FindingID: findingID,
		ExpectedRevision: 1, RequestID: "review-after-report-acceptance",
		IdempotencyKey: "review-after-report-acceptance",
		RequestDigest:  serviceTestDigest("review-after-report-acceptance"),
	}); err != nil {
		t.Fatalf("finding review after report acceptance: %v", err)
	}
}

func TestAuditFindingReviewHistoryAndDeletedRunProvenance(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, databaseURL)

	const ownerID = "owner-finding-review"
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "project-finding-review", OwnerID: ownerID, Kind: projectstore.KindProject,
		Name: "Finding review", IdempotencyKey: "create-finding-review-project",
		RequestDigest: serviceTestDigest("finding-review-project"),
	})
	if err != nil {
		t.Fatal(err)
	}
	const auditID = "audit-finding-review"
	if _, _, err := auditstore.NewPostgresStore(pool).CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: auditID, OwnerID: ownerID, ProjectID: project.ProjectID,
		Profile: auditstore.ProfileIdentity{
			Name: "finding-review", Version: "1", Digest: serviceTestDigest("finding-profile"),
		},
		ProfileSnapshot: json.RawMessage(`{"interaction":{"findingConfirmation":"human-required"}}`),
		InputSelection:  json.RawMessage(`{}`),
		Limits: auditstore.Limits{
			MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 2, MaxItemsTotal: 2,
			MaxSubmittedRuns: 2, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1 << 20,
		},
		IdempotencyKey: "create-finding-review-audit", RequestDigest: serviceTestDigest("finding-audit"),
	}); err != nil {
		t.Fatal(err)
	}
	firstID := seedAuditFinding(t, ctx, pool, project.ProjectID, ownerID, auditID, "first")
	secondID := seedAuditFinding(t, ctx, pool, project.ProjectID, ownerID, auditID, "second")
	intake, err := findingintake.New(pool)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 9, 6, 3, 0, 0, 0, time.UTC)
	service := &Service{pool: pool, findings: intake, now: func() time.Time { return now }}

	listed, err := service.ListFindings(ctx, FindingListParams{OwnerID: ownerID, AuditID: auditID, Limit: 10})
	if err != nil || len(listed) != 2 || listed[0].AnalystVerdict != nil ||
		listed[0].FirstProposal.Origin.RunDeleted != true {
		t.Fatalf("initial findings = (%+v, %v)", listed, err)
	}
	request, err := service.CreateFindingReview(ctx, CreateFindingReviewParams{
		OwnerID: ownerID, AuditID: auditID, FindingID: firstID, ExpectedRevision: 1,
		RequestID: "review-first-1", IdempotencyKey: "review-first-1",
		RequestDigest: serviceTestDigest("review-first-1"),
	})
	if err != nil || request.Replayed || request.Request.SubjectRevision != 1 ||
		request.Request.State != ReviewPending {
		t.Fatalf("create finding review = (%+v, %v)", request, err)
	}
	severity := SeverityHigh
	decisionParams := DecideFindingParams{
		OwnerID: ownerID, AuditID: auditID, RequestID: request.Request.RequestID,
		ExpectedRequestRevision: request.Request.Revision, DecisionID: "decision-first-1",
		Verdict: VerdictTruePositive, Severity: &severity, Rationale: "Confirmed by exact evidence.",
		IdempotencyKey: "decision-first-1", RequestDigest: serviceTestDigest("decision-first-1"),
	}
	decided, err := service.DecideFinding(ctx, decisionParams)
	if err != nil || decided.Finding.State != FindingConfirmed || decided.Finding.Revision != 2 ||
		decided.Finding.AnalystVerdict == nil || *decided.Finding.AnalystVerdict != VerdictTruePositive ||
		decided.Finding.AnalystSeverity == nil || *decided.Finding.AnalystSeverity != SeverityHigh {
		t.Fatalf("true-positive decision = (%+v, %v)", decided, err)
	}
	reportFindings, err := auditstore.NewPostgresStore(pool).ListReportFindings(ctx, auditID)
	var reportFinding *auditstore.ReportFinding
	for index := range reportFindings {
		if reportFindings[index].FindingID == firstID {
			reportFinding = &reportFindings[index]
		}
	}
	if err != nil || len(reportFindings) != 2 || reportFinding == nil ||
		reportFinding.Decision == nil || reportFinding.Decision.Verdict != string(VerdictTruePositive) ||
		reportFinding.Decision.Severity == nil || *reportFinding.Decision.Severity != string(SeverityHigh) {
		t.Fatalf("report finding snapshot = (%+v, %v)", reportFindings, err)
	}
	decisionParams.DecisionID = "ignored-replay-id"
	replayed, err := service.DecideFinding(ctx, decisionParams)
	if err != nil || !replayed.Replayed || replayed.Decision.DecisionID != "decision-first-1" {
		t.Fatalf("decision replay = (%+v, %v)", replayed, err)
	}
	if _, err := service.CreateFindingReview(ctx, CreateFindingReviewParams{
		OwnerID: ownerID, AuditID: auditID, FindingID: firstID, ExpectedRevision: 1,
		RequestID: "stale-review", IdempotencyKey: "stale-review",
		RequestDigest: serviceTestDigest("stale-review"),
	}); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("stale finding revision error = %v", err)
	}

	corrected := decideFindingForTest(t, ctx, service, ownerID, auditID, firstID, 2,
		"correction", VerdictTruePositive, reviewStringPointer(string(SeverityMedium)), nil)
	if corrected.State != FindingConfirmed || corrected.AnalystSeverity == nil ||
		*corrected.AnalystSeverity != SeverityMedium || corrected.Revision != 3 {
		t.Fatalf("corrected finding = %+v", corrected)
	}
	falsePositive := decideFindingForTest(t, ctx, service, ownerID, auditID, firstID, 3,
		"false-positive", VerdictFalsePositive, nil, nil)
	if falsePositive.State != FindingRejected || falsePositive.AnalystVerdict == nil ||
		*falsePositive.AnalystVerdict != VerdictFalsePositive || falsePositive.RejectionReason == nil {
		t.Fatalf("false-positive finding = %+v", falsePositive)
	}
	reopened := decideFindingForTest(t, ctx, service, ownerID, auditID, firstID, 4,
		"reopen", VerdictReopen, nil, nil)
	if reopened.State != FindingProposed || reopened.AnalystVerdict != nil || reopened.Revision != 5 {
		t.Fatalf("reopened finding = %+v", reopened)
	}

	secondReview, err := service.CreateFindingReview(ctx, CreateFindingReviewParams{
		OwnerID: ownerID, AuditID: auditID, FindingID: secondID, ExpectedRevision: 1,
		RequestID: "review-second-cycle", IdempotencyKey: "review-second-cycle",
		RequestDigest: serviceTestDigest("review-second-cycle"),
	})
	if err != nil {
		t.Fatal(err)
	}
	duplicate := decideFindingForTest(t, ctx, service, ownerID, auditID, firstID, 5,
		"duplicate", VerdictDuplicate, nil, &secondID)
	if duplicate.State != FindingDuplicate || duplicate.DuplicateTargetID == nil ||
		*duplicate.DuplicateTargetID != secondID {
		t.Fatalf("duplicate finding = %+v", duplicate)
	}
	_, err = service.DecideFinding(ctx, DecideFindingParams{
		OwnerID: ownerID, AuditID: auditID, RequestID: secondReview.Request.RequestID,
		ExpectedRequestRevision: secondReview.Request.Revision, DecisionID: "decision-second-cycle",
		Verdict: VerdictDuplicate, DuplicateTargetID: &firstID, Rationale: "Would form a cycle.",
		IdempotencyKey: "decision-second-cycle", RequestDigest: serviceTestDigest("decision-second-cycle"),
	})
	if !errors.Is(err, auditstore.ErrConflict) {
		t.Fatalf("duplicate cycle error = %v", err)
	}
	now = now.Add(defaultReviewTTL + time.Second)
	_, err = service.DecideFinding(ctx, DecideFindingParams{
		OwnerID: ownerID, AuditID: auditID, RequestID: secondReview.Request.RequestID,
		ExpectedRequestRevision: secondReview.Request.Revision, DecisionID: "decision-second-expired",
		Verdict: VerdictNeedsEvidence, Rationale: "This request has expired.",
		IdempotencyKey: "decision-second-expired", RequestDigest: serviceTestDigest("decision-second-expired"),
	})
	if !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("expired review decision error = %v", err)
	}
	expiredReview, err := service.GetReview(ctx, ownerID, auditID, secondReview.Request.RequestID)
	if err != nil || expiredReview.State != ReviewExpired || expiredReview.Revision != 2 {
		t.Fatalf("persisted expired review = (%+v, %v)", expiredReview, err)
	}

	history, err := service.ListReviews(ctx, ReviewListParams{
		OwnerID: ownerID, AuditID: auditID, FindingID: &firstID, Limit: 10,
	})
	if err != nil || len(history) != 5 || history[0].Decision == nil || history[4].Decision == nil {
		t.Fatalf("finding decision history = (%+v, %v)", history, err)
	}
	seedAuditFindingAttemptHistory(t, ctx, pool, auditID, firstID, "receipt-first")
	provenance, err := service.ListFindingProvenance(ctx, ProvenanceListParams{
		OwnerID: ownerID, AuditID: auditID, FindingID: firstID, Limit: 10,
	})
	if err != nil || len(provenance) != 4 || provenance[0].Kind != ProvenanceSourceProposal ||
		!provenance[0].Origin.RunDeleted || provenance[1].Attempt == nil ||
		provenance[1].Attempt.ItemAttempt != 1 ||
		provenance[1].Attempt.CollectionDisposition == nil ||
		*provenance[1].Attempt.CollectionDisposition != auditstore.CollectionExecutionFailed ||
		provenance[1].Attempt.Result != nil || provenance[2].Attempt == nil ||
		provenance[2].Attempt.ItemAttempt != 2 || provenance[2].Assessment == nil ||
		provenance[2].Assessment.SemanticAssessment != "supported" ||
		provenance[2].Attempt.Result == nil || !provenance[2].Attempt.RunDeleted ||
		provenance[2].Attempt.RunProvenance == nil ||
		provenance[2].Attempt.RunProvenance.RunID != "deleted-run-attempt-two" ||
		provenance[3].Kind != ProvenanceDirect || provenance[3].Attempt != nil ||
		provenance[3].Assessment == nil || !provenance[3].Assessment.DirectVerification ||
		provenance[3].Assessment.Contract == nil ||
		provenance[3].Assessment.SemanticAssessment != "supported" ||
		!provenance[3].Origin.RunDeleted {
		t.Fatalf("deleted-Run provenance = (%+v, %v)", provenance, err)
	}
	if _, err := service.GetFinding(ctx, "another-owner", auditID, firstID); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("foreign finding read error = %v", err)
	}

	// Finalization owns an immutable report snapshot. Review remains available
	// after completion, but may not change the Audit revision between report
	// artifact creation and its transactional commit.
	finalizingReview, err := service.CreateFindingReview(ctx, CreateFindingReviewParams{
		OwnerID: ownerID, AuditID: auditID, FindingID: firstID, ExpectedRevision: 6,
		RequestID: "review-finalizing", IdempotencyKey: "review-finalizing",
		RequestDigest: serviceTestDigest("review-finalizing"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
UPDATE audits
   SET state = 'finalizing', dispatch_state = 'closed',
       baseline_snapshot = '{}'::jsonb, started_at = $2::timestamptz,
       deadline_at = $2::timestamptz + interval '1 hour'
 WHERE audit_id = $1`, auditID, now); err != nil {
		t.Fatal(err)
	}
	if _, err := service.CreateFindingReview(ctx, CreateFindingReviewParams{
		OwnerID: ownerID, AuditID: auditID, FindingID: secondID, ExpectedRevision: 1,
		RequestID: "review-during-finalizing", IdempotencyKey: "review-during-finalizing",
		RequestDigest: serviceTestDigest("review-during-finalizing"),
	}); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("create review during finalization error = %v", err)
	}
	if _, err := service.DecideFinding(ctx, DecideFindingParams{
		OwnerID: ownerID, AuditID: auditID, RequestID: finalizingReview.Request.RequestID,
		ExpectedRequestRevision: finalizingReview.Request.Revision,
		DecisionID:              "decision-during-finalizing", Verdict: VerdictReopen,
		Rationale: "Must wait for report commit.", IdempotencyKey: "decision-during-finalizing",
		RequestDigest: serviceTestDigest("decision-during-finalizing"),
	}); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("decide review during finalization error = %v", err)
	}
}

func seedAuditFindingAttemptHistory(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool,
	auditID, findingID, proposalReceiptID string,
) {
	t.Helper()
	digest := func(value string) string { return serviceTestDigest("attempt-" + value) }
	taskRef := `{"namespace":"audit-task-packages","name":"check-one","revision":"task-r1"}`
	resultRef := `{"namespace":"audit-results","name":"check-one","revision":"result-r2"}`
	origin := `{"schema":"contractor.audit.item-origin.v1","entryKey":"check-one","provenanceIncomplete":true}`
	err := postgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `
INSERT INTO audit_rounds (
    round_id, audit_id, ordinal, manifest_ref, manifest_digest, state, expected_item_count
) VALUES (
    'round-finding-attempts', $1, 1,
    '{"namespace":"audit-rounds","name":"round-one","revision":"round-r1"}'::jsonb,
    $2, 'closed', 1
)`, auditID, digest("round")); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `
INSERT INTO audit_items (
    item_id, audit_id, round_id, item_key, ordinal, kind, subject_key,
    task_ref, task_digest, origin, workflow_role, state
) VALUES (
    'item-finding-attempts', $1, 'round-finding-attempts', 'check-one', 0,
    'check', 'component-one', $2::jsonb, $3, $4::jsonb, 'check-role', 'ready'
)`, auditID, taskRef, digest("task"), origin); err != nil {
			return err
		}
		for _, attempt := range []struct {
			executionID, memberID, runID, outcome, disposition string
			ordinal                                            int
			hasResult                                          bool
		}{
			{"execution-attempt-one", "execution-item-attempt-one", "deleted-run-attempt-one", "failed", "execution-failed", 1, false},
			{"execution-attempt-two", "execution-item-attempt-two", "deleted-run-attempt-two", "succeeded", "accepted-result", 2, true},
		} {
			runProvenance, _ := json.Marshal(auditstore.RunProvenance{
				Schema: "contractor.audit.run-provenance.v1", RunID: attempt.runID,
				ProvenanceIncomplete: true,
			})
			if _, err := tx.Exec(ctx, `
INSERT INTO audit_executions (
    execution_id, audit_id, round_id, role, workflow_role, manifest_ref, manifest_digest,
    submission_key, request_digest, run_id, state, terminal_outcome,
    terminal_run_generation, terminal_run_sequence, terminal_observed_at,
    run_provenance, run_deleted_at
) VALUES (
    $1, $2, 'round-finding-attempts', 'check', 'check-role',
    '{"namespace":"audit-executions","name":"check-one","revision":"execution-r1"}'::jsonb,
    $3, $4, $5, $6, 'collected', $7, 'generation-one', $8,
    clock_timestamp(), $9::jsonb, clock_timestamp()
)`, attempt.executionID, auditID, digest(attempt.executionID),
				"submission-"+attempt.executionID, digest("request-"+attempt.executionID),
				attempt.runID, attempt.outcome, attempt.ordinal, runProvenance); err != nil {
				return err
			}
			var storedResultRef any
			var storedResultDigest any
			if attempt.hasResult {
				storedResultRef = resultRef
				storedResultDigest = digest("result")
			}
			if _, err := tx.Exec(ctx, `
INSERT INTO audit_execution_items (
    execution_item_id, execution_id, audit_id, round_id, item_id,
    batch_ordinal, item_attempt, task_ref, task_digest, input_refs,
    state, collection_disposition, result_ref, result_digest, collected_at
) VALUES (
    $1, $2, $3, 'round-finding-attempts', 'item-finding-attempts',
    0, $4, $5::jsonb, $6, '[]'::jsonb, 'settled', $7,
    $8::jsonb, $9, clock_timestamp()
)`, attempt.memberID, attempt.executionID, auditID, attempt.ordinal,
				taskRef, digest("task"), attempt.disposition, storedResultRef,
				storedResultDigest); err != nil {
				return err
			}
		}
		if _, err := tx.Exec(ctx, `
INSERT INTO audit_collection_receipts (
    receipt_id, audit_id, execution_id, run_id, terminal_outcome,
    terminal_run_generation, terminal_run_sequence, disposition,
    source_output_ref, source_output_digest, retained_refs, request_digest
) VALUES (
    'collection-attempt-two', $1, 'execution-attempt-two',
    'deleted-run-attempt-two', 'succeeded', 'generation-one', 2,
    'accepted-result', $2::jsonb, $3, '[]'::jsonb, $4
)`, auditID, resultRef, digest("result"), digest("collection")); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `
INSERT INTO audit_finding_assessments (
    assessment_id, finding_id, audit_id, receipt_id, item_id,
    execution_item_id, collection_receipt_id, semantic_assessment,
    result_ref, result_digest
) VALUES (
    'assessment-attempt-two', $4, $1, $5, 'item-finding-attempts',
    'execution-item-attempt-two', 'collection-attempt-two', 'supported',
    $2::jsonb, $3
)`, auditID, resultRef, digest("result"), findingID, proposalReceiptID); err != nil {
			return err
		}
		directResultRef := `{"namespace":"audit-results","name":"direct","revision":"direct-result-r1"}`
		directContractRef := `{"namespace":"audit-contracts","name":"direct","revision":"direct-contract-r1"}`
		if _, err := tx.Exec(ctx, `
INSERT INTO audit_finding_assessments (
    assessment_id, finding_id, audit_id, receipt_id,
    semantic_assessment, result_ref, result_digest,
    direct_verification, contract_ref, contract_digest
) VALUES (
    'assessment-direct', $4, $1, $5, 'supported', $2::jsonb, $3,
    true, $6::jsonb, $7
)`, auditID, directResultRef, digest("direct-result"), findingID,
			proposalReceiptID, directContractRef, digest("direct-contract")); err != nil {
			return err
		}
		_, err := tx.Exec(ctx, `
UPDATE audit_items
   SET state = 'settled', final_disposition = 'accepted-result',
       accepted_result_ref = $2::jsonb, accepted_result_digest = $3,
       last_execution_item_id = 'execution-item-attempt-two'
 WHERE audit_id = $1 AND item_id = 'item-finding-attempts'`,
			auditID, resultRef, digest("result"))
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
}

func seedAuditFinding(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool,
	projectID, ownerID, auditID, suffix string,
) string {
	t.Helper()
	document := auditdomain.FindingProposal{
		Schema: auditdomain.FindingProposalSchema, ClientKey: "candidate-" + suffix,
		Title: "Candidate " + suffix, Description: "A retained candidate for review.",
		Subject:       auditdomain.FindingSubject{Kind: "component", Key: "component-" + suffix},
		Preconditions: []string{}, StandardRefs: []auditdomain.StandardReference{},
		EvidenceIDs: []string{}, ProposedChecks: []auditdomain.ProposedCheck{},
		SeveritySuggestion: "medium", Limitations: []string{},
	}
	payload, err := auditdomain.EncodeFindingProposal(document)
	if err != nil {
		t.Fatal(err)
	}
	projectArtifacts, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Project(projectID)
	if err != nil {
		t.Fatal(err)
	}
	written, err := projectArtifacts.Write(ctx,
		contracts.ArtifactRef{Namespace: "audit-finding-proposals", Name: "candidate-" + suffix},
		artifacts.Payload{MediaType: "application/json", Data: payload}, nil)
	if err != nil {
		t.Fatal(err)
	}
	proposal := findingintake.ExactArtifact{
		Ref: written.Ref, Digest: digestBytes(payload), MediaType: written.MediaType, SizeBytes: written.Size,
	}
	proposalJSON, _ := json.Marshal(proposal)
	proposalRefJSON, _ := json.Marshal(proposal.Ref)
	receiptID := "receipt-" + suffix
	if _, err := pool.Exec(ctx, `
INSERT INTO finding_proposal_receipts (
    receipt_id, proposal_id, allocation_id, runtime_agent_id,
    runtime_instance_id, stage_execution_id, logical_agent_name,
    invocation_id, submission_id, client_key, request_digest,
    run_id, owner_id, project_id,
    workflow_name, workflow_version, workflow_schema_version,
    workflow_configuration_ref, workflow_closure_digest,
    proposal_ref, proposal_digest, proposal_media_type,
    proposal_size_bytes, evidence
) VALUES ($1, $2, $3, 'runtime-review', 'instance-review', 'stage-review', 'worker',
          $4, $5, $6, $7, $8, $9, $10,
          'finding-source', '1', 'contractor/v1alpha1',
          '{"name":"finding-source","version":"1"}'::jsonb, $11,
          $12::jsonb, $13, 'application/json', $14, '[]'::jsonb)`,
		receiptID, "proposal-"+suffix, "allocation-"+suffix,
		"invocation-"+suffix, "submission-"+suffix, document.ClientKey,
		serviceTestDigest("request-"+suffix), "deleted-run-"+suffix, ownerID, projectID,
		serviceTestDigest("workflow-"+suffix), proposalRefJSON, proposal.Digest, proposal.SizeBytes,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
INSERT INTO finding_proposal_retention (
    receipt_id, state, source_run_deleted_at
) VALUES ($1, 'audit-held', clock_timestamp())`, receiptID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
INSERT INTO finding_proposal_audit_holds (
    receipt_id, audit_id, project_id, proposal_ref, evidence
) VALUES ($1, $2, $3, $4::jsonb, '[]'::jsonb)`,
		receiptID, auditID, projectID, proposalJSON); err != nil {
		t.Fatal(err)
	}
	return "finding-" + receiptID
}

func decideFindingForTest(
	t *testing.T, ctx context.Context, service *Service,
	ownerID, auditID, findingID string, findingRevision uint64, key string,
	verdict AnalystVerdict, severityValue *string, duplicateTarget *string,
) Finding {
	t.Helper()
	request, err := service.CreateFindingReview(ctx, CreateFindingReviewParams{
		OwnerID: ownerID, AuditID: auditID, FindingID: findingID,
		ExpectedRevision: findingRevision, RequestID: "review-" + key,
		IdempotencyKey: "review-" + key, RequestDigest: serviceTestDigest("review-" + key),
	})
	if err != nil {
		t.Fatal(err)
	}
	var severity *FindingSeverity
	if severityValue != nil {
		value := FindingSeverity(*severityValue)
		severity = &value
	}
	result, err := service.DecideFinding(ctx, DecideFindingParams{
		OwnerID: ownerID, AuditID: auditID, RequestID: request.Request.RequestID,
		ExpectedRequestRevision: request.Request.Revision, DecisionID: "decision-" + key,
		Verdict: verdict, Severity: severity, Rationale: "Analyst decision for " + key + ".",
		DuplicateTargetID: duplicateTarget, IdempotencyKey: "decision-" + key,
		RequestDigest: serviceTestDigest("decision-" + key),
	})
	if err != nil {
		t.Fatal(err)
	}
	return result.Finding
}

func reviewStringPointer(value string) *string { return &value }

type switchableProfileCatalog struct {
	mu        sync.Mutex
	snapshot  *config.Snapshot
	available bool
}

func (c *switchableProfileCatalog) AuditProfile(selector string) (config.ResolvedAuditProfile, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.available {
		return config.ResolvedAuditProfile{}, config.ErrConfigurationNotFound
	}
	return c.snapshot.AuditProfile(selector)
}

func (c *switchableProfileCatalog) AuditProfiles() []config.ResolvedAuditProfile {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.available {
		return nil
	}
	return c.snapshot.AuditProfiles()
}

func (c *switchableProfileCatalog) setAvailable(value bool) {
	c.mu.Lock()
	c.available = value
	c.mu.Unlock()
}

type switchableCredentialLookup struct {
	mu        sync.Mutex
	available bool
	gateway   contracts.LLMGatewayConfigRef
}

func (l *switchableCredentialLookup) LookupLLMCredential(
	_ context.Context, id string,
) (config.CredentialMetadata, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if !l.available || id != "development-worker" {
		return config.CredentialMetadata{}, errors.New("credential unavailable")
	}
	return config.CredentialMetadata{
		Ref:          contracts.LLMCredentialRef{CredentialID: id},
		LLMGateway:   l.gateway,
		Unrestricted: true,
	}, nil
}

func (l *switchableCredentialLookup) setAvailable(value bool) {
	l.mu.Lock()
	l.available = value
	l.mu.Unlock()
}

type countingCredentialGuard struct {
	mu    sync.Mutex
	calls int
}

func (g *countingCredentialGuard) WithRunCreation(ctx context.Context, fn func() error) error {
	g.mu.Lock()
	g.calls++
	g.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	return fn()
}

func (g *countingCredentialGuard) callsCount() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.calls
}

func writeChecklist(
	t *testing.T, ctx context.Context, store artifacts.ScopedStore, name, reviewPolicy string,
) artifacts.WriteResult {
	t.Helper()
	payload := []byte(`{"schema":"contractor.audit.checklist.v1","items":[{"key":"check-` + name + `","version":"1","statement":"Verify ` + name + `.","applicability":"always","allowed_methods":["static"],"required_evidence":[],"review_policy":"` + reviewPolicy + `"}]}`)
	result, err := store.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: name}, artifacts.Payload{
		MediaType: "application/json", Data: payload,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func loadAuditServiceProfiles(t *testing.T) *config.Snapshot {
	t.Helper()
	root := t.TempDir()
	files := map[string]string{
		"instructions/planner.md": "Execute the selected checklist item.",
		"instructions/worker.md":  "Read the task package and write a result package.",
		"llm-gateways/test.yaml": `apiVersion: contractor/v1alpha1
kind: LLMGatewayConfig
metadata: {name: test-gateway, version: "1"}
spec:
  protocol: openai-compatible@1
  url: http://127.0.0.1:4000/v1
`,
		"model-policies/worker.yaml": `apiVersion: contractor/v1alpha1
kind: ModelPolicy
metadata: {name: worker, version: "1"}
spec:
  model: worker-model
  maxOutputTokens: 1024
  maxModelCalls: 2
  maxToolCalls: 4
  maxTotalTokens: 4096
  temperature: 0
`,
		"agent-templates/worker.yaml": `apiVersion: contractor/v1alpha1
kind: AgentTemplate
metadata: {name: audit-worker, version: "1"}
spec:
  description: Produces one deterministic test result
  runtime: adk@1
  instructions: {ref: instructions/worker.md}
  modelPolicy: worker@1
  toolsets:
    - ref: run-artifacts@1
      tools: [read_artifact, write_artifact]
  sandboxProfile: local-workdir@1
`,
		"workflows/check.yaml": `apiVersion: contractor/v1alpha1
kind: Workflow
metadata: {name: audit-check, version: "1"}
spec:
  parameters: {}
  inputs:
    task: {required: true, mediaTypes: [application/zip]}
  outputs:
    result: {required: true, mediaTypes: [application/zip]}
  executionConfig:
    workers:
      llmGateway: test-gateway@1
      credential: development-worker
  entryStage: check
  stages:
    check:
      objective: Evaluate one checklist item
      instructions: {ref: instructions/planner.md}
      planner: passthrough@1
      agents:
        worker: {template: audit-worker@1}
      context:
        artifacts:
          task: {namespace: inputs, name: task, required: true}
      result:
        artifacts:
          result: {required: true, mediaTypes: [application/zip], from: {namespace: worker, name: result}}
      workflowOutputs: {result: result}
      on:
        succeeded: {succeed: {}}
        failed: {fail: {}}
        interrupted: {fail: {}}
`,
		"audit-profiles/checklist.yaml": `apiVersion: contractor/v1alpha1
kind: AuditProfile
metadata: {name: test-checklist, version: "1"}
spec:
  mode: custom-checklist
  standards: []
  inputs:
    checklist: {required: true, mediaTypes: [application/json]}
  inventory:
    implementation: checklist@1
    sourceInput: checklist
    itemWorkflowRole: check
  workflows:
    check:
      kind: check
      ref: audit-check@1
      inputs:
        task: {source: item-package}
      parameters: {}
      outputs: {result: result}
  execution:
    roundMode: fixed-barrier
    maxRounds: 1
    batchSize: 1
    maxItemsPerRound: 10
    maxItemsTotal: 10
    maxSubmittedRuns: 20
    maxItemRunAttempts: 2
    deadlineSeconds: 3600
    maxEvidenceBytes: 1048576
    incompleteRound: assess-with-gaps
  interaction:
    activeChecks: prohibited
    findingConfirmation: disabled
    notApplicable: profile-rule
    reportAcceptance: automatic
`,
	}
	for _, directory := range []string{
		"instructions", "llm-gateways", "model-policies", "execution-configs",
		"agent-templates", "workflows", "audit-profiles", "skills",
	} {
		if err := os.MkdirAll(filepath.Join(root, directory), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	for name, contents := range files {
		if err := os.WriteFile(filepath.Join(root, name), []byte(contents), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	snapshot, err := config.Load(root, config.MVPDescriptors())
	if err != nil {
		t.Fatalf("load Audit service config: %v", err)
	}
	return snapshot
}

func isolatedAuditServicePool(
	t *testing.T, ctx context.Context, databaseURL string,
) *pgxpool.Pool {
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
	schema := "contractor_audit_service_test_" + hex.EncodeToString(random)
	identifier := pgx.Identifier{schema}.Sanitize()
	if _, err := admin.Exec(ctx, `CREATE SCHEMA `+identifier); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	configuration, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		admin.Close()
		t.Fatal(err)
	}
	configuration.ConnConfig.RuntimeParams["search_path"] = schema
	pool, err := pgxpool.NewWithConfig(ctx, configuration)
	if err != nil {
		admin.Close()
		t.Fatal(err)
	}
	if _, err := postgres.ApplyMigrations(ctx, pool); err != nil {
		pool.Close()
		admin.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		pool.Close()
		cleanup, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := admin.Exec(cleanup, `DROP SCHEMA `+identifier+` CASCADE`); err != nil {
			t.Logf("drop test schema: %v", err)
		}
		admin.Close()
	})
	return pool
}

func serviceTestDigest(value string) string {
	return "sha256:" + hex.EncodeToString([]byte(strings.Repeat(value, 64))[:32])
}
