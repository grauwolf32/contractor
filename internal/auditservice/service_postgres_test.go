package auditservice

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/auditstore/reviewstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	managedcredentials "github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/persistence/leasetest"
	"github.com/grauwolf32/contractor/internal/persistence/postgres"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/persistence/postgrestest"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
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
		TransactionLLMCredentials: runtimeconfigtest.TransactionLLMCredentialLookupFactoryFunc(
			func(pgx.Tx) (resolved.CredentialLookup, error) { return credentials, nil },
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
		Labels:  map[string]string{"cloud": "aws", "team": "appsec"},
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
	if err != nil || !slices.Equal(baseline.RequiredPlacementLabels, []string{"cloud/aws"}) {
		t.Fatalf("pinned placement requirement = %v, %v", baseline.RequiredPlacementLabels, err)
	}
	for _, source := range []json.RawMessage{draft.InputSelection, started.Audit.BaselineSnapshot} {
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(source, &fields); err != nil {
			t.Fatal(err)
		}
		for _, invalid := range []json.RawMessage{nil, json.RawMessage(`null`), json.RawMessage(`["cloud","local"]`), json.RawMessage(`["cloud/*"]`)} {
			fields["requiredPlacementLabels"] = invalid
			if invalid == nil {
				delete(fields, "requiredPlacementLabels")
			}
			payload, err := json.Marshal(fields)
			if err != nil {
				t.Fatal(err)
			}
			if fields["schema"] != nil && string(fields["schema"]) == `"`+DraftSelectionSchema+`"` {
				if _, err := DecodeDraftSelection(payload); !errors.Is(err, ErrInvalid) {
					t.Fatalf("draft accepted invalid placement %s: %v", invalid, err)
				}
			} else if _, err := DecodeBaseline(payload); !errors.Is(err, ErrInvalid) {
				t.Fatalf("baseline accepted invalid placement %s: %v", invalid, err)
			}
		}
	}
	if err != nil || len(baseline.LLMCredentialIDs) != 1 || baseline.LLMCredentialIDs[0] != "development-worker" ||
		baseline.Inventory.Worklist.Ref.Revision == nil || len(baseline.Skills) != 0 {
		t.Fatalf("baseline = (%+v, %v)", baseline, err)
	}
	if baseline.Standards == nil || ValidateBaseline(baseline) != nil {
		t.Fatalf("decoded baseline lost its required empty-array shape: %+v", baseline)
	}
	owned, err := DecodeBaseline(started.Audit.BaselineSnapshot)
	if err != nil {
		t.Fatal(err)
	}
	owned.Labels["team"] = "changed"
	owned.RequiredPlacementLabels[0] = "local"
	owned.LLMCredentialIDs[0] = "changed"
	item := &owned.Inventory.ExecutionManifest.Items[0]
	*item.TaskRef.Revision = "changed-task"
	*owned.Inputs["checklist"].Ref.Revision = "changed-input"
	stored, err := service.Get(ctx, project.OwnerID, draft.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	again, err := DecodeBaseline(stored.BaselineSnapshot)
	if err != nil || !reflect.DeepEqual(baseline, again) {
		t.Fatalf("decoded baseline mutation changed another read: %v", err)
	}
	// Nil RuntimeConfig collections and empty manifest inputs retain the
	// decoder's existing empty-array/null representation without a deep copy.
	shape := again
	shape.RuntimeConfig.Labels = nil
	shape.RuntimeConfig.LLMCredentialIDs = nil
	shape.RuntimeConfig.RuntimeCredentialIDs = nil
	shape.Inventory.ExecutionManifest.Items[0].Inputs = []auditdomain.ExactInput{}
	shapeBytes, err := EncodeBaseline(shape)
	if err != nil {
		t.Fatal(err)
	}
	normalized, err := DecodeBaseline(shapeBytes)
	if err != nil || normalized.RuntimeConfig.Labels == nil || normalized.RuntimeConfig.LLMCredentialIDs == nil || normalized.RuntimeConfig.RuntimeCredentialIDs == nil || normalized.Inventory.ExecutionManifest.Items[0].Inputs != nil {
		t.Fatalf("baseline collection representation changed: %v", err)
	}
	var baselineFields map[string]json.RawMessage
	if err := json.Unmarshal(started.Audit.BaselineSnapshot, &baselineFields); err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name  string
		value json.RawMessage
	}{
		{name: "missing"},
		{name: "null", value: json.RawMessage(`null`)},
	} {
		t.Run("baseline standards "+test.name, func(t *testing.T) {
			baselineFields["standards"] = test.value
			if test.value == nil {
				delete(baselineFields, "standards")
			}
			payload, err := json.Marshal(baselineFields)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := DecodeBaseline(payload); !errors.Is(err, ErrInvalid) {
				t.Fatalf("DecodeBaseline missing/null standards = %v, want ErrInvalid", err)
			}
		})
	}
	holds, err := auditstoretest.NewStore(pool).ListHeldAuditIDsByLLMCredential(ctx, "development-worker", 10)
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
	coveragePage, err := service.ListCoverage(ctx, project.OwnerID, draft.AuditID, started.Round.RoundID, -1, 10, nil)
	coverage := coveragePage.Rows
	if err != nil || len(coverage) != 1 || coverage[0].Ordinal != 0 || coverage[0].Coverage.Status != auditstore.CoverageNotTested {
		t.Fatalf("initial coverage = (%+v, %v)", coverage, err)
	}
	if current, err := service.Get(ctx, project.OwnerID, draft.AuditID); err != nil || coveragePage.AuditRevision != current.Revision {
		t.Fatalf("coverage page revision = %d, Audit = (%+v, %v)", coveragePage.AuditRevision, current, err)
	}
	pinned := coveragePage.AuditRevision
	if page, err := service.ListCoverage(ctx, project.OwnerID, draft.AuditID, started.Round.RoundID, -1, 10, &pinned); err != nil ||
		page.AuditRevision != pinned || len(page.Rows) != 1 {
		t.Fatalf("pinned coverage = (%+v, %v)", page, err)
	}
	stale := pinned - 1
	if _, err := service.ListCoverage(ctx, project.OwnerID, draft.AuditID, started.Round.RoundID, -1, 10, &stale); !errors.Is(err, auditstore.ErrConflict) {
		t.Fatalf("stale coverage pin = %v", err)
	}
	if coverage[0].Details == nil || coverage[0].Details.Objective == "" || !json.Valid(coverage[0].Details.TaskDocument) {
		t.Fatalf("coverage omitted the retained task: %+v", coverage[0].Details)
	}
	// Every coverage status, with the counts the workspace must report. Each
	// expectation also matches the status's auditdomain bucket, from which the
	// report concludes completed-with-gaps, so workspace and report agree.
	workspaceCases := []struct {
		status                             auditstore.CoverageStatus
		completed, issues, gaps, unchecked int
	}{
		{auditstore.CoverageNotTested, 0, 0, 0, 1}, {auditstore.CoverageInconclusive, 0, 0, 1, 0},
		{auditstore.CoverageSatisfied, 1, 0, 0, 0}, {auditstore.CoverageViolated, 1, 1, 0, 0},
		{auditstore.CoverageNotApplicable, 1, 0, 0, 0}, {auditstore.CoverageBlocked, 0, 0, 1, 0},
		{auditstore.CoverageExcluded, 0, 0, 1, 0}, {auditstore.CoverageTracedComplete, 1, 0, 0, 0},
		{auditstore.CoverageTracedPartial, 0, 0, 1, 0}, {auditstore.CoverageUnmapped, 0, 0, 1, 0},
	}
	coveredStatuses := map[auditstore.CoverageStatus]bool{}
	for _, tc := range workspaceCases {
		if !tc.status.Valid() || coveredStatuses[tc.status] {
			t.Fatalf("workspace case %q is invalid or repeated", tc.status)
		}
		coveredStatuses[tc.status] = true
		bucket := tc.status.Bucket()
		if (bucket == auditdomain.CoverageBucketCompleted) != (tc.completed == 1) ||
			(bucket == auditdomain.CoverageBucketGap) != (tc.gaps == 1) ||
			(bucket == auditdomain.CoverageBucketUnchecked) != (tc.unchecked == 1) {
			t.Fatalf("workspace counts for %s disagree with the report's %q bucket", tc.status, bucket)
		}
	}
	for _, bucket := range []auditdomain.CoverageBucket{
		auditdomain.CoverageBucketCompleted, auditdomain.CoverageBucketGap, auditdomain.CoverageBucketUnchecked,
	} {
		for _, status := range auditdomain.CoverageStatusesIn(bucket) {
			if !coveredStatuses[auditstore.CoverageStatus(status)] {
				t.Fatalf("workspace cases omit coverage status %q", status)
			}
		}
	}
	for _, tc := range workspaceCases {
		if _, err := pool.Exec(ctx, `UPDATE audit_items SET coverage_status=$1 WHERE audit_id=$2`, tc.status, draft.AuditID); err != nil {
			t.Fatal(err)
		}
		summary, err := service.GetWorkspace(ctx, project.OwnerID, draft.AuditID)
		if err != nil || summary.TotalChecks != 1 || summary.CompletedChecks != tc.completed || summary.Issues != tc.issues || summary.Gaps != tc.gaps || summary.Unchecked != tc.unchecked || summary.RoundID == nil || *summary.RoundID != started.Round.RoundID {
			t.Fatalf("coverage summary for %s: %+v err=%v", tc.status, summary, err)
		}
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
	if listed, preparation, err := service.ListPage(ctx, auditstore.ListParams{
		OwnerID: "other-owner", ProjectID: &foreignProjectID, Limit: 10,
	}, func(auditstore.Audit) bool { return true }); err != nil || len(listed) != 0 || len(preparation) != 0 {
		t.Fatalf("foreign owner list = (%+v, %v, %v)", listed, preparation, err)
	}
	if _, err := service.ListItems(ctx, auditstore.ListItemsParams{
		OwnerID: "other-owner", AuditID: draft.AuditID, Limit: 10,
	}); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("foreign owner item list = %v", err)
	}
	if _, err := service.ListCoverage(
		ctx, "other-owner", draft.AuditID, started.Round.RoundID, -1, 10, nil,
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
	reviews, err := service.ListReviews(ctx, auditstore.ReviewListParams{
		OwnerID: project.OwnerID, AuditID: manualDraft.AuditID, Limit: 10,
	})
	if err != nil || len(reviews) != 1 || reviews[0].Kind != string(auditstore.ItemApprovalApplicability) ||
		reviews[0].SubjectKind != auditstore.ReviewSubjectItemAction ||
		reviews[0].SubjectID != manualStarted.Items[0].ItemID {
		t.Fatalf("manual item reviews = (%+v, %v)", reviews, err)
	}
	approved, err := service.DecideActionReview(ctx, DecideActionReviewParams{
		OwnerID: project.OwnerID, AuditID: manualDraft.AuditID,
		RequestID: reviews[0].RequestID, ExpectedRequestRevision: reviews[0].Revision,
		DecisionID: "decision-manual-approve", Action: auditstore.ReviewApprove,
		Rationale:      "The owner approved this exact checklist action.",
		IdempotencyKey: "decision-manual-approve",
		RequestDigest:  serviceTestDigest("decision-manual-approve"),
	})
	if err != nil || approved.Decision.Action != auditstore.ReviewApprove ||
		approved.Request.State != auditstore.ReviewDecided {
		t.Fatalf("approve manual item = (%+v, %v)", approved, err)
	}
	if _, err := service.DecideActionReview(ctx, DecideActionReviewParams{
		OwnerID: project.OwnerID, AuditID: manualDraft.AuditID,
		RequestID: reviews[0].RequestID, ExpectedRequestRevision: reviews[0].Revision,
		DecisionID: "decision-manual-approve", Action: auditstore.ReviewApprove,
		Rationale:      "The owner approved this exact checklist action.",
		IdempotencyKey: "decision-manual-approve",
		RequestDigest:  serviceTestDigest("decision-manual-approve-changed"),
	}); !errors.Is(err, contracts.ErrIdempotencyConflict) || errors.Is(err, auditstore.ErrConflict) {
		t.Fatalf("action decision key reused with another request = %v", err)
	}
	approvedItems, err := service.ListItems(ctx, auditstore.ListItemsParams{
		OwnerID: project.OwnerID, AuditID: manualDraft.AuditID, Limit: 10,
	})
	if err != nil || len(approvedItems) != 1 || approvedItems[0].State != auditstore.ItemReady {
		t.Fatalf("approved manual item state = (%+v, %v)", approvedItems, err)
	}
	claims, err := auditstoretest.NewStore(pool).Claim(ctx, auditstore.ClaimParams{
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
			_ = auditstoretest.NewStore(pool).ReleaseClaim(ctx, claim)
		}
	}
	if manualClaim.AuditID == "" {
		t.Fatal("manual Audit was not claimable after approval")
	}
	if _, err := auditstoretest.NewStore(pool).TransitionRound(ctx, auditstore.RoundTransitionParams{
		Claim: manualClaim, RoundID: manualStarted.Round.RoundID,
		ExpectedRevision: manualStarted.Round.Revision,
		ExpectedState:    auditstore.RoundAccepted, TargetState: auditstore.RoundExecuting,
	}); err != nil {
		currentAudit, auditErr := auditstoretest.NewStore(pool).Get(ctx, project.OwnerID, manualDraft.AuditID)
		currentRound, roundErr := auditstoretest.NewStore(pool).GetRound(
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
	_, inserted, err := auditstoretest.NewStore(pool).CreateExecutionIntent(ctx,
		auditstore.CreateExecutionIntentParams{
			Claim: manualClaim, ExecutionID: "execution-expired-approval",
			RoundID: &roundID, Role: auditstore.ExecutionCheck, WorkflowRole: "check", LaneClass: resolved.LaneClassModel,
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
	_ = auditstoretest.NewStore(pool).ReleaseClaim(ctx, manualClaim)

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
	rejectedReviews, err := service.ListReviews(ctx, auditstore.ReviewListParams{
		OwnerID: project.OwnerID, AuditID: rejectedDraft.AuditID, Limit: 10,
	})
	if err != nil || len(rejectedReviews) != 1 {
		t.Fatalf("rejected item review = (%+v, %v)", rejectedReviews, err)
	}
	// Only requests that can still be decided wait: a pending request past its
	// expiry is not counted, whether or not anything has expired it yet.
	pendingReviews := func(step string, want int) {
		t.Helper()
		summary, err := service.GetWorkspace(ctx, project.OwnerID, rejectedDraft.AuditID)
		if err != nil || summary.PendingReviews != want {
			t.Fatalf("%s: workspace pending reviews = (%+v, %v), want %d", step, summary, err, want)
		}
	}
	setReviewExpiry := func(expiry string) {
		t.Helper()
		if _, err := pool.Exec(ctx, `UPDATE audit_review_requests SET expires_at = `+expiry+` WHERE request_id = $1`,
			rejectedReviews[0].RequestID); err != nil {
			t.Fatal(err)
		}
	}
	var originalExpiry *time.Time
	if err := pool.QueryRow(ctx, `SELECT expires_at FROM audit_review_requests WHERE request_id = $1`,
		rejectedReviews[0].RequestID).Scan(&originalExpiry); err != nil {
		t.Fatal(err)
	}
	pendingReviews("pending request", 1)
	setReviewExpiry("clock_timestamp() - interval '1 second'")
	pendingReviews("pending request past its expiry", 0)
	setReviewExpiry("clock_timestamp() + interval '1 hour'")
	pendingReviews("pending request before its expiry", 1)
	if _, err := pool.Exec(ctx, `UPDATE audit_review_requests SET expires_at = $2 WHERE request_id = $1`,
		rejectedReviews[0].RequestID, originalExpiry); err != nil {
		t.Fatal(err)
	}
	if _, err := service.DecideActionReview(ctx, DecideActionReviewParams{
		OwnerID: project.OwnerID, AuditID: rejectedDraft.AuditID,
		RequestID:               rejectedReviews[0].RequestID,
		ExpectedRequestRevision: rejectedReviews[0].Revision,
		DecisionID:              "decision-manual-reject", Action: auditstore.ReviewReject,
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
	rejectedPage, err := service.ListCoverage(
		ctx, project.OwnerID, rejectedDraft.AuditID, rejectedStarted.Round.RoundID, -1, 10, nil,
	)
	rejectedCoverage := rejectedPage.Rows
	if err != nil || len(rejectedCoverage) != 1 ||
		rejectedCoverage[0].Coverage.Status != auditstore.CoverageExcluded {
		t.Fatalf("rejected item coverage = (%+v, %v)", rejectedCoverage, err)
	}
	// The owner's rejection is a visible coverage gap, not completed work
	// (AUD-R178), and the decided request no longer waits.
	rejectedSummary, err := service.GetWorkspace(ctx, project.OwnerID, rejectedDraft.AuditID)
	if err != nil || rejectedSummary.TotalChecks != 1 || rejectedSummary.CompletedChecks != 0 ||
		rejectedSummary.Issues != 0 || rejectedSummary.Gaps != 1 || rejectedSummary.Unchecked != 0 ||
		rejectedSummary.PendingReviews != 0 {
		t.Fatalf("rejected item workspace = (%+v, %v)", rejectedSummary, err)
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

	notApplicableDraft, _, err := service.CreateDraft(ctx, CreateDraftParams{
		AuditID: "audit-api-manual-not-applicable", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile: ProfileSelector{Name: "test-checklist", Version: "1"},
		Inputs:  map[string]contracts.ArtifactRef{"checklist": manual.Ref},
		Scope:   Scope{}, IdempotencyKey: "create-manual-not-applicable",
		RequestDigest: serviceTestDigest("create-manual-not-applicable"),
	})
	if err != nil {
		t.Fatal(err)
	}
	notApplicableStarted, err := service.Start(ctx, StartParams{
		OwnerID: project.OwnerID, AuditID: notApplicableDraft.AuditID,
		ExpectedRevision: notApplicableDraft.Revision,
		IdempotencyKey:   "start-manual-not-applicable",
		RequestDigest:    serviceTestDigest("start-manual-not-applicable"),
	})
	if err != nil {
		t.Fatal(err)
	}
	notApplicableReviews, err := service.ListReviews(ctx, auditstore.ReviewListParams{
		OwnerID: project.OwnerID, AuditID: notApplicableDraft.AuditID, Limit: 10,
	})
	if err != nil || len(notApplicableReviews) != 1 ||
		!containsRequestedReviewAction(notApplicableReviews[0].RequestedActions, auditstore.ReviewNotApplicable) {
		t.Fatalf("not-applicable review authority = (%+v, %v)", notApplicableReviews, err)
	}
	rationale := "This exact documentation requirement is outside the approved application scope."
	decision, err := service.DecideActionReview(ctx, DecideActionReviewParams{
		OwnerID: project.OwnerID, AuditID: notApplicableDraft.AuditID,
		RequestID:               notApplicableReviews[0].RequestID,
		ExpectedRequestRevision: notApplicableReviews[0].Revision,
		DecisionID:              "decision-manual-not-applicable",
		Action:                  auditstore.ReviewNotApplicable, Rationale: rationale,
		IdempotencyKey: "decision-manual-not-applicable",
		RequestDigest:  serviceTestDigest("decision-manual-not-applicable"),
	})
	if err != nil || decision.Decision.Action != auditstore.ReviewNotApplicable {
		t.Fatalf("not-applicable decision = (%+v, %v)", decision, err)
	}
	notApplicableItems, err := service.ListItems(ctx, auditstore.ListItemsParams{
		OwnerID: project.OwnerID, AuditID: notApplicableDraft.AuditID, Limit: 10,
	})
	if err != nil || len(notApplicableItems) != 1 ||
		notApplicableItems[0].State != auditstore.ItemSettled ||
		notApplicableItems[0].FinalDisposition == nil ||
		*notApplicableItems[0].FinalDisposition != auditstore.FinalNotApplicable {
		t.Fatalf("not-applicable item = (%+v, %v)", notApplicableItems, err)
	}
	notApplicablePage, err := service.ListCoverage(
		ctx, project.OwnerID, notApplicableDraft.AuditID,
		notApplicableStarted.Round.RoundID, -1, 10, nil,
	)
	notApplicableCoverage := notApplicablePage.Rows
	if err != nil || len(notApplicableCoverage) != 1 ||
		notApplicableCoverage[0].Coverage.Status != auditstore.CoverageNotApplicable ||
		notApplicableCoverage[0].Coverage.Rationale != rationale {
		t.Fatalf("not-applicable coverage = (%+v, %v)", notApplicableCoverage, err)
	}
}

func containsRequestedReviewAction(values []auditstore.ReviewRequestedAction, action auditstore.ReviewAction) bool {
	for _, value := range values {
		if value == auditstore.ReviewRequestedAction(action) {
			return true
		}
	}
	return false
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
		Metadata: resolved.CredentialMetadata{
			Ref:        llmgateway.LLMCredentialRef{CredentialID: "development-worker"},
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

func TestAuditStartRechecksProjectHTTPTargetCredentialUse(t *testing.T) {
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
	lookup := &switchableCredentialLookup{available: true, gateway: gateway.Ref}
	service, err := New(Options{
		Pool: pool, Profiles: profiles,
		TransactionLLMCredentials: runtimeconfigtest.TransactionLLMCredentialLookupFactoryFunc(
			func(pgx.Tx) (resolved.CredentialLookup, error) { return lookup, nil },
		),
		CredentialGuard: &countingCredentialGuard{},
	})
	if err != nil {
		t.Fatal(err)
	}
	cipher, err := managedcredentials.NewTokenCipher([]byte(strings.Repeat("k", 32)))
	if err != nil {
		t.Fatal(err)
	}
	runtimeCredentials, err := managedcredentials.NewRuntimeCredentialService(managedcredentials.RuntimeCredentialServiceOptions{
		Pool: pool, Lease: leasetest.Hold(t, pool).Lease, Cipher: cipher,
		Usage:    managedcredentials.NewRuntimeCredentialRepository(pool),
		Projects: projectstore.NewPostgresStore(pool), Audits: hookstore.NewCredentialHolds(pool), Barrier: managedcredentials.NewLifecycleBarrier(),
	})
	if err != nil {
		t.Fatal(err)
	}
	material, err := managedcredentials.NewHTTPOriginBearerCredential("origin-secret")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runtimeCredentials.Create(ctx, managedcredentials.RuntimeCredentialCreateRequest{
		CredentialID: "foreign-origin", Material: material,
		IdempotencyKey: "create-foreign-origin", ActorID: "another-owner",
	}); err != nil {
		t.Fatal(err)
	}

	projects := projectstore.NewPostgresStore(pool)
	project, _, err := projects.Create(ctx, projectstore.CreateParams{
		ProjectID: "project-target-audit", OwnerID: "owner-target-audit", Kind: projectstore.KindProject,
		Name: "Target Audit", IdempotencyKey: "project-target-audit", RequestDigest: serviceTestDigest("target-project"),
	})
	if err != nil {
		t.Fatal(err)
	}
	// Stored before ownership enforcement, or by a principal that has since
	// lost the Operations capability.
	project, err = projects.Update(ctx, projectstore.UpdateParams{
		ProjectID: project.ProjectID, OwnerID: project.OwnerID, ExpectedRevision: project.Revision,
		Name: project.Name, HTTPTarget: &runtimesettings.HTTPOriginTargetRef{
			URL: "https://attacker.example.test",
			Credential: &runtimesettings.RuntimeCredentialRef{
				CredentialID: "foreign-origin", Kind: runtimesettings.RuntimeCredentialOriginBearer,
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	projectArtifacts, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	checklist := writeChecklist(t, ctx, projectArtifacts, "target", "automatic")
	draft, _, err := service.CreateDraft(ctx, CreateDraftParams{
		AuditID: "audit-target", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:        ProfileSelector{Name: "test-checklist", Version: "1"},
		Inputs:         map[string]contracts.ArtifactRef{"checklist": checklist.Ref},
		Scope:          Scope{Objective: "Exercise Project target credential ownership"},
		IdempotencyKey: "create-target-audit", RequestDigest: serviceTestDigest("target-create"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := service.Start(ctx, StartParams{
		OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision,
		IdempotencyKey: "start-target-owner", RequestDigest: serviceTestDigest("target-start-owner"),
	}); !errors.Is(err, managedcredentials.ErrRuntimeCredentialNotFound) {
		t.Fatalf("start with another owner's credential = %v", err)
	}
	started, err := service.Start(ctx, StartParams{
		OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision,
		IdempotencyKey: "start-target-operations", RequestDigest: serviceTestDigest("target-start-operations"),
		OperationsPrincipal: true,
	})
	if err != nil || started.Audit.State != auditstore.AuditActive {
		t.Fatalf("start by an Operations principal = (%+v, %v)", started.Audit, err)
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
	store := auditstoretest.NewStore(pool)
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
	artifact := func(name, media, content string) auditstore.ExactArtifact {
		written, err := artifactstest.WriteAuditArtifact(
			ctx, artifacts.NewService(artifacts.NewPostgresRepository(pool)), project.ProjectID,
			contracts.ArtifactRef{Namespace: "audit-report-review", Name: name},
			artifacts.Payload{MediaType: media, Data: []byte(content)},
		)
		if err != nil {
			t.Fatal(err)
		}
		return auditstore.ExactArtifact{
			Ref: written.Ref, Digest: auditdomain.DigestBytes([]byte(content)), MediaType: written.MediaType, SizeBytes: written.Size,
		}
	}
	provenance := json.RawMessage(`{"schema":"contractor.audit.report-provenance.v1"}`)
	params := auditstore.ProposeReportParams{
		Claim: claim, ExpectedAuditRevision: audit.Revision,
		RoundID: round.RoundID, ExpectedRoundRevision: round.Revision,
		Machine: auditstore.ArtifactLink{LogicalKey: auditstore.ReportMachineLogicalKey,
			Artifact: artifact("report.json", "application/json", `{"schema":"contractor.audit.report.v1"}`), SourceProvenance: provenance},
		Summary: auditstore.ArtifactLink{LogicalKey: auditstore.ReportSummaryLogicalKey,
			Artifact: artifact("report.md", "text/markdown", "# Frozen report\n"), SourceProvenance: provenance},
		RequestDigest: serviceTestDigest("report-candidate"),
	}
	params.ExpectedProjectRevision, err = store.ReportProjectRevision(ctx, audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	testLegacyReportWriteRejection(t, ctx, pool, params)
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
	testLegacyReportCandidateRejection(t, ctx, pool, reviewService, project.OwnerID, params, candidate)
	proposed, err := reviewService.GetReport(ctx, project.OwnerID, audit.AuditID)
	if err != nil || proposed.Status != ReportProposed || proposed.Summary != "# Frozen report\n" {
		t.Fatalf("current proposed Markdown report = %+v, %v", proposed, err)
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
	changed, err := auditstoretest.NewStore(expiryTx).ExpireReportReview(
		ctx, claim, waiting.Revision,
	)
	if err != nil || !changed {
		_ = expiryTx.Rollback(ctx)
		t.Fatalf("expire exact report review = (%t, %v)", changed, err)
	}
	expiredAudit, err := auditstoretest.NewStore(expiryTx).Get(
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
		ctx, reviewstore.NewTx(rejectTx), audit.AuditID, waiting.Revision, candidate.RequestID, audit.AuditID,
		int64(candidate.SubjectRevision), candidate.SubjectDigest, auditstore.ReviewReject,
	); err != nil {
		_ = rejectTx.Rollback(ctx)
		t.Fatalf("reject exact report candidate: %v", err)
	}
	rejected, err := auditstoretest.NewStore(rejectTx).Get(ctx, project.OwnerID, audit.AuditID)
	if err != nil || rejected.State != auditstore.AuditFailed || rejected.StopReason == nil ||
		rejected.StopReason.Code != "report_rejected" || rejected.Revision != waiting.Revision+1 ||
		rejected.Revision != rejected.EventSequence {
		_ = rejectTx.Rollback(ctx)
		t.Fatalf("rejected report transaction = (%+v, %v)", rejected, err)
	}
	rejectEvents, err := auditstoretest.NewStore(rejectTx).EventsAfter(
		ctx, project.OwnerID, audit.AuditID, rejected.EventSequence-1, 1,
	)
	if err != nil || len(rejectEvents) != 1 || rejectEvents[0].Kind != "audit.state_changed" ||
		!strings.Contains(string(rejectEvents[0].Summary), `"to": "failed"`) {
		_ = rejectTx.Rollback(ctx)
		t.Fatalf("rejected report event = (%+v, %v)", rejectEvents, err)
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
	decideFindingForTest(t, ctx, reviewService, project.OwnerID, audit.AuditID, findingID, 1, "during-report", auditstore.VerdictNeedsEvidence, nil, nil)

	if _, err := reviewService.Pause(ctx, MutationParams{
		OwnerID: project.OwnerID, AuditID: audit.AuditID, ExpectedRevision: waiting.Revision,
		IdempotencyKey: "pause-report-review", RequestDigest: serviceTestDigest("pause-report-review"),
	}); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("pause during report acceptance error = %v", err)
	}
	review, err := reviewService.GetReview(
		ctx, project.OwnerID, audit.AuditID, candidate.RequestID,
	)
	if err != nil || review.Kind != auditstore.ReportAcceptanceReviewKind ||
		review.SubjectKind != auditstore.ReviewSubjectReport {
		t.Fatalf("report review = (%+v, %v)", review, err)
	}
	decision, err := reviewService.DecideActionReview(ctx, DecideActionReviewParams{
		OwnerID: project.OwnerID, AuditID: audit.AuditID, RequestID: review.RequestID,
		ExpectedRequestRevision: review.Revision, DecisionID: "decision-report-approve",
		Action: auditstore.ReviewApprove, Rationale: "The owner accepts this exact report.",
		IdempotencyKey: "decision-report-approve", RequestDigest: serviceTestDigest("report-approve"),
	})
	if err != nil || decision.Decision.Action != auditstore.ReviewApprove {
		t.Fatalf("approve report = (%+v, %v)", decision, err)
	}
	completed, err := store.Get(ctx, project.OwnerID, audit.AuditID)
	if err != nil || completed.State != auditstore.AuditCompleted ||
		completed.Revision != completed.EventSequence {
		t.Fatalf("completed report Audit = (%+v, %v)", completed, err)
	}
	acceptEvents, err := store.EventsAfter(ctx, project.OwnerID, audit.AuditID, completed.EventSequence-2, 2)
	if err != nil || len(acceptEvents) != 2 || acceptEvents[0].Kind != "audit.report_committed" ||
		!strings.Contains(string(acceptEvents[0].Summary), candidate.SubjectDigest) ||
		acceptEvents[1].Kind != "review.decided" {
		t.Fatalf("accepted report events = (%+v, %v)", acceptEvents, err)
	}
	for _, key := range []string{auditstore.ReportMachineLogicalKey, auditstore.ReportSummaryLogicalKey} {
		if _, err := store.GetArtifactLink(ctx, audit.AuditID, key); err != nil {
			t.Fatalf("accepted report link %q: %v", key, err)
		}
	}
	testLegacyCommittedReportRejection(t, ctx, pool, reviewService, project.OwnerID, project.ProjectID, params)
	decideFindingForTest(t, ctx, reviewService, project.OwnerID, audit.AuditID, findingID, 2, "after-report", auditstore.VerdictFalsePositive, nil, nil)

}

func seedAuditFinding(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool,
	projectID, ownerID, auditID, suffix string, proposedChecks ...auditdomain.ProposedCheck,
) string {
	t.Helper()
	return seedAuditFindingDocument(t, ctx, pool, projectID, ownerID, auditID, suffix,
		auditFindingDocument(suffix, proposedChecks...))
}

func auditFindingDocument(suffix string, proposedChecks ...auditdomain.ProposedCheck) auditdomain.FindingProposal {
	return auditdomain.FindingProposal{
		Schema: auditdomain.FindingProposalSchema, ClientKey: "candidate-" + suffix,
		Title: "Candidate " + suffix, Description: "A retained candidate for review.",
		Subject:       &auditdomain.FindingSubject{Kind: "component", Key: "component-" + suffix},
		Preconditions: []string{}, StandardRefs: []auditdomain.StandardReference{},
		EvidenceIDs: []string{}, ProposedChecks: append([]auditdomain.ProposedCheck{}, proposedChecks...),
		SeveritySuggestion: "medium", Limitations: []string{},
	}
}

func seedAuditFindingDocument(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool,
	projectID, ownerID, auditID, suffix string, document auditdomain.FindingProposal, fingerprints ...auditdomain.FindingFingerprint,
) string {
	t.Helper()
	fingerprint := auditdomain.UnavailableFindingFingerprint("not_recorded")
	if len(fingerprints) == 1 {
		fingerprint = fingerprints[0]
	}
	return seedAuditFindingProjected(t, ctx, pool, projectID, ownerID, auditID, suffix, document, fingerprint, auditdomain.UnavailableFindingAnchors("not_recorded"))
}

func seedAuditFindingProjected(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool,
	projectID, ownerID, auditID, suffix string, document auditdomain.FindingProposal, fingerprint auditdomain.FindingFingerprint, anchors auditdomain.FindingAnchorProjection,
) string {
	t.Helper()
	if len(document.EvidenceIDs) == 0 {
		document.EvidenceIDs = []string{"evidence-1"}
	}
	retainedEvidence := make([]findingintake.ExactArtifact, 0, len(document.EvidenceIDs))
	for _, id := range document.EvidenceIDs {
		body := []byte("Retained evidence " + suffix + " " + id)
		file, err := artifactstest.WriteAuditArtifact(ctx, artifacts.NewService(artifacts.NewPostgresRepository(pool)), projectID, contracts.ArtifactRef{Namespace: artifactpolicy.FindingMaterialsNamespace, Name: suffix + "-" + id}, artifacts.Payload{MediaType: "text/plain", Data: body})
		if err != nil {
			t.Fatal(err)
		}
		retainedEvidence = append(retainedEvidence, findingintake.ExactArtifact{Ref: file.Ref, Digest: auditdomain.DigestBytes(body), MediaType: file.MediaType, SizeBytes: file.Size})
	}
	evidenceJSON, _ := json.Marshal(retainedEvidence)
	payload, err := auditdomain.EncodeFindingProposal(document)
	if err != nil {
		t.Fatal(err)
	}
	projection, err := auditdomain.EncodeFindingReportProjection(auditdomain.ProjectFindingReport(document))
	if err != nil {
		t.Fatal(err)
	}
	namespace := artifactpolicy.FindingMaterialsNamespace
	written, err := artifactstest.WriteAuditArtifact(ctx, artifacts.NewService(artifacts.NewPostgresRepository(pool)), projectID,
		contracts.ArtifactRef{Namespace: namespace, Name: "candidate-" + suffix},
		artifacts.Payload{MediaType: "application/json", Data: payload})
	if err != nil {
		t.Fatal(err)
	}
	proposal := findingintake.ExactArtifact{
		Ref: written.Ref, Digest: auditdomain.DigestBytes(payload), MediaType: written.MediaType, SizeBytes: written.Size,
	}
	proposalJSON, _ := json.Marshal(proposal)
	proposalRefJSON, _ := json.Marshal(proposal.Ref)
	fingerprintJSON, err := auditdomain.EncodeFindingFingerprint(fingerprint)
	if err != nil {
		t.Fatal(err)
	}
	anchorJSON, err := auditdomain.EncodeFindingAnchors(anchors)
	if err != nil {
		t.Fatal(err)
	}
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
    proposal_size_bytes, evidence, report_projection, fingerprint_projection, anchor_projection,
 retention_state,source_run_deleted_at,project_proposal,project_evidence,publication_requested
, checks_projection) VALUES ($1, $2, $3, 'runtime-review', 'instance-review', 'stage-review', 'worker',
          $4, $5, $6, $7, $8, $9, $10,
          'finding-source', '1', 'contractor/v1alpha1',
          '{"name":"finding-source","version":"1"}'::jsonb, $11,
          $12::jsonb, $13, 'application/json', $14, $18::jsonb, $15, $16,$17,'project-held',clock_timestamp(),$19::jsonb,$18::jsonb,true, json_build_object('schema','contractor.audit.finding-checks-projection.v1','proposal_digest',($13)::text,'size_bytes',($14)::bigint,'discarded',false,'subject',NULL,'limitations',NULL,'checks','[]'::json))`,
		receiptID, "proposal-"+suffix, "allocation-"+suffix,
		"invocation-"+suffix, "submission-"+suffix, document.ClientKey,
		serviceTestDigest("request-"+suffix), "deleted-run-"+suffix, ownerID, projectID,
		serviceTestDigest("workflow-"+suffix), proposalRefJSON, proposal.Digest, proposal.SizeBytes,
		string(projection), string(fingerprintJSON), string(anchorJSON), evidenceJSON, proposalJSON,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO finding_proposal_audit_holds(receipt_id,audit_id,project_id,proposal_ref,evidence) VALUES($1,$2,$3,$4::jsonb,$5::jsonb)`, receiptID, auditID, projectID, proposalJSON, evidenceJSON); err != nil {
		t.Fatal(err)
	}
	var findingID string
	err = persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		events := hookstore.NewFindingEventsTx(tx)
		var err error
		findingID, err = events.AdmitProjectFinding(ctx, projectID, receiptID, "finding-"+suffix)
		if err != nil {
			return err
		}
		return events.RecordFindingSourceAdded(ctx, auditID, receiptID, "deleted-run-"+suffix)
	})
	if err != nil {
		t.Fatal(err)
	}
	return findingID
}

func decideFindingForTest(
	t *testing.T, ctx context.Context, service *Service,
	ownerID, auditID, findingID string, findingRevision uint64, key string,
	verdict auditstore.AnalystVerdict, severityValue *string, duplicateTarget *string,
) auditstore.ProjectFinding {
	t.Helper()

	var severity *auditstore.FindingSeverity
	if severityValue != nil {
		value := auditstore.FindingSeverity(*severityValue)
		severity = &value
	}
	current, err := service.reads().GetAuditProjectFinding(ctx, ownerID, auditID, findingID)
	if err != nil {
		t.Fatal(err)
	}
	result, err := service.EditProjectFinding(ctx, ProjectFindingEditParams{
		ProjectFindingEdit: auditstore.ProjectFindingEdit{OwnerID: ownerID, ProjectID: current.ProjectID, FindingID: findingID, ExpectedRevision: findingRevision, Verdict: verdict, Severity: severity, Rationale: "Analyst decision for " + key + ".", DuplicateTargetID: duplicateTarget},
		IdempotencyKey:     "decision-" + key, RequestDigest: serviceTestDigest("decision-" + key),
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

func (c *switchableProfileCatalog) AuditProfile(selector string) (auditprofile.ResolvedAuditProfile, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.available {
		return auditprofile.ResolvedAuditProfile{}, config.ErrConfigurationNotFound
	}
	return c.snapshot.AuditProfile(selector)
}

func (c *switchableProfileCatalog) ResolveAuditProfile(_ context.Context, selector string) (auditprofile.ResolvedAuditProfile, error) {
	return c.AuditProfile(selector)
}

func (c *switchableProfileCatalog) ResolvePinnedAuditProfile(ctx context.Context, pinned auditprofile.ResolvedAuditProfile) (auditprofile.ResolvedAuditProfile, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.available {
		return auditprofile.ResolvedAuditProfile{}, config.ErrConfigurationNotFound
	}
	return c.snapshot.ResolvePinnedAuditProfile(ctx, pinned)
}

func (c *switchableProfileCatalog) AuditProfiles() []auditprofile.ResolvedAuditProfile {
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
	gateway   llmgateway.LLMGatewayConfigRef
}

func (l *switchableCredentialLookup) LookupLLMCredential(
	_ context.Context, id string,
) (resolved.CredentialMetadata, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if !l.available || id != "development-worker" {
		return resolved.CredentialMetadata{}, errors.New("credential unavailable")
	}
	return resolved.CredentialMetadata{
		Ref:          llmgateway.LLMCredentialRef{CredentialID: id},
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
	return loadAuditServiceProfilesWithProfile(t, nil)
}

func loadAuditServiceProfilesWithProfile(t *testing.T, edit func(string) string) *config.Snapshot {
	return loadAuditServiceProfilesWithEdits(t, edit, nil)
}

func loadAuditServiceProfilesWithEdits(
	t *testing.T, editProfile, editWorker func(string) string, customize ...func(map[string]string),
) *config.Snapshot {
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
      modelPolicy: worker@1
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
    source: {source: audit-input, name: checklist}
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
    notApplicable: profile-rule
    reportAcceptance: automatic
`,
	}
	if editProfile != nil {
		files["audit-profiles/checklist.yaml"] = editProfile(files["audit-profiles/checklist.yaml"])
	}
	if editWorker != nil {
		files["agent-templates/worker.yaml"] = editWorker(files["agent-templates/worker.yaml"])
	}
	for _, apply := range customize {
		apply(files)
	}
	for _, directory := range []string{
		"instructions", "llm-gateways", "model-policies",
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
	snapshot, _, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatalf("load Audit service config: %v", err)
	}
	return snapshot
}

func isolatedAuditServicePool(
	t *testing.T, ctx context.Context, databaseURL string,
) *pgxpool.Pool {
	t.Helper()
	pool := postgrestest.NewPool(t, ctx, databaseURL, postgrestest.Options{})
	if _, err := postgres.ApplyMigrations(ctx, pool, postgres.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	return pool
}

func serviceTestDigest(value string) string {
	return "sha256:" + hex.EncodeToString([]byte(strings.Repeat(value, 64))[:32])
}
