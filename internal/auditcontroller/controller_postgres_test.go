//go:build integration

package auditcontroller

import (
	"context"
	cryptorand "crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditimport"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/grauwolf32/contractor/internal/settingsstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresControllerDispatchesWithoutAuditDeadline(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	harness := newPostgresControllerHarness(t, ctx, 1)
	if _, err := harness.pool.Exec(ctx, `UPDATE audits SET deadline_at=NULL WHERE audit_id=$1`, harness.started.Audit.AuditID); err != nil {
		t.Fatal(err)
	}
	controller := harness.controller(t)
	for step := 0; step < 2; step++ {
		if worked, err := controller.RunOnce(ctx); err != nil || !worked {
			t.Fatalf("unlimited admission step=%d worked=%t error=%v", step, worked, err)
		}
	}
	executions, err := harness.audits.ListExecutions(ctx, harness.started.Audit.AuditID)
	if err != nil || len(executions) != 1 || executions[0].RunID == nil {
		t.Fatalf("unlimited Audit did not submit a Run: %v", err)
	}
}

func TestPostgresControllerDispatchesOrdinaryRunsThroughDerivedWindow(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	harness := newPostgresControllerHarness(t, ctx, 3)
	settings := settingsstore.NewPostgresStore(harness.pool)
	if _, err := settings.UpdateSchedulerSettings(ctx, settingsstore.UpdateSchedulerSettingsParams{
		MaxConcurrentRuns: 2, ExpectedRevision: 1,
	}); err != nil {
		t.Fatal(err)
	}

	controller := harness.controller(t)
	for operation, wantWorked := range []bool{true, true, true, false} {
		worked, err := controller.RunOnce(ctx)
		if err != nil || worked != wantWorked {
			t.Fatalf("controller operation %d = (%t, %v), want worked=%t", operation, worked, err, wantWorked)
		}
	}
	executions, err := harness.audits.ListExecutions(ctx, harness.started.Audit.AuditID)
	if err != nil || len(executions) != 2 {
		t.Fatalf("initial executions = (%+v, %v)", executions, err)
	}
	for _, execution := range executions {
		if execution.RunID == nil || execution.State != auditstore.ExecutionSubmitted {
			t.Fatalf("initial execution is not submitted: %+v", execution)
		}
		run, getErr := harness.runs.GetRun(ctx, *execution.RunID)
		if getErr != nil || run.PublicationMode != runstore.PublicationAuditManaged ||
			run.AuditExecutionID == nil || *run.AuditExecutionID != execution.ExecutionID {
			t.Fatalf("trusted child Run = (%+v, %v)", run, getErr)
		}
	}
	// The controller queues children; this fixture supplies Scheduler admission.
	if _, err := harness.runs.TransitionRun(ctx, *executions[0].RunID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "test_admitted"}); err != nil {
		t.Fatal(err)
	}
	if _, err := harness.runs.TransitionRun(
		ctx, *executions[0].RunID, runstore.RunRunning, runstore.RunSucceeded,
		runstore.Reason{Code: "test_succeeded"},
	); err != nil {
		t.Fatal(err)
	}
	runID := *executions[0].RunID
	blocker, err := harness.runs.RunDeletionBlocker(ctx, harness.started.Audit.OwnerID, runID)
	if err != nil || blocker == nil || *blocker != runstore.RunAuditCollectionPending {
		t.Fatalf("uncollected Audit Run deletion blocker = (%v, %v)", blocker, err)
	}
	if err := harness.runs.DeleteReleasedTerminalRun(ctx, harness.started.Audit.OwnerID, runID); err == nil {
		t.Fatal("uncollected Audit Run deletion unexpectedly succeeded")
	} else {
		var blocked *runstore.RunNotDeletableError
		if !errors.As(err, &blocked) || blocked.Reason != runstore.RunAuditCollectionPending {
			t.Fatalf("uncollected Audit Run deletion error = %v", err)
		}
	}
	if worked, err := controller.RunOnce(ctx); err != nil || !worked {
		t.Fatalf("observe terminal child = (%t, %v)", worked, err)
	}
	if worked, err := controller.RunOnce(ctx); err != nil || !worked {
		t.Fatalf("refill derived dispatch window = (%t, %v)", worked, err)
	}
	executions, err = harness.audits.ListExecutions(ctx, harness.started.Audit.AuditID)
	if err != nil || len(executions) != 3 {
		t.Fatalf("refilled executions = (%+v, %v)", executions, err)
	}
	items, err := harness.audits.ListItems(ctx, harness.started.Audit.AuditID)
	if err != nil || items[0].State != auditstore.ItemCollecting || items[0].FinalDisposition != nil {
		t.Fatalf("terminal observation prematurely settled item = (%+v, %v)", items, err)
	}
}

func TestPostgresControllerBatchBuildsOnePinnedRunForTwoItems(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	harness := newPostgresControllerHarness(t, ctx, 3, 2)
	settings := settingsstore.NewPostgresStore(harness.pool)
	if _, err := settings.UpdateSchedulerSettings(ctx, settingsstore.UpdateSchedulerSettingsParams{
		MaxConcurrentRuns: 2, ExpectedRevision: 1,
	}); err != nil {
		t.Fatal(err)
	}
	controller := harness.controller(t)

	if worked, err := controller.RunOnce(ctx); err != nil || !worked {
		t.Fatalf("activate batch round = (%t, %v)", worked, err)
	}
	if worked, err := controller.RunOnce(ctx); err != nil || !worked {
		t.Fatalf("dispatch first batch = (%t, %v)", worked, err)
	}
	executions, err := harness.audits.ListExecutions(ctx, harness.started.Audit.AuditID)
	if err != nil || len(executions) != 1 || executions[0].RunID == nil {
		t.Fatalf("batch executions = (%+v, %v)", executions, err)
	}
	members, err := harness.audits.ListExecutionItems(ctx, executions[0].ExecutionID)
	if err != nil || len(members) != 2 || members[0].BatchOrdinal != 0 || members[1].BatchOrdinal != 1 {
		t.Fatalf("batch members = (%+v, %v)", members, err)
	}
	runArtifacts, err := harness.artifacts.Run(*executions[0].RunID)
	if err != nil {
		t.Fatal(err)
	}
	taskInput, err := runArtifacts.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "task"})
	if err != nil {
		t.Fatal(err)
	}
	pkg, err := auditdomain.ValidatePackage(taskInput.Payload.Data)
	if err != nil || pkg.Manifest.Kind != auditdomain.PackageKindTaskSet || len(pkg.Members()) != 2 {
		t.Fatalf("forked task set = (%+v, %v)", pkg, err)
	}

	if worked, err := controller.RunOnce(ctx); err != nil || !worked {
		t.Fatalf("dispatch final partial batch = (%t, %v)", worked, err)
	}
	executions, err = harness.audits.ListExecutions(ctx, harness.started.Audit.AuditID)
	if err != nil || len(executions) != 2 {
		t.Fatalf("three items used %d Runs, want 2: %v", len(executions), err)
	}
	lastMembers, err := harness.audits.ListExecutionItems(ctx, executions[1].ExecutionID)
	if err != nil || len(lastMembers) != 1 {
		t.Fatalf("partial batch members = (%+v, %v)", lastMembers, err)
	}
}

func TestPostgresControllerDispatchesReadyItemsBehindReviewWindow(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	harness := newPostgresControllerReviewHarness(t, ctx, auditstore.MaxReconcileRows+5, 2)
	controller := harness.controller(t)
	for step := 0; step < 2; step++ {
		if worked, err := controller.RunOnce(ctx); err != nil || !worked {
			t.Fatalf("reconcile step=%d worked=%t error=%v", step, worked, err)
		}
	}
	executions, err := harness.audits.ListExecutions(ctx, harness.started.Audit.AuditID)
	if err != nil || len(executions) == 0 || executions[0].RunID == nil {
		t.Fatalf("ready items behind the review window were not dispatched: (%+v, %v)", executions, err)
	}
	members, err := harness.audits.ListExecutionItems(ctx, executions[0].ExecutionID)
	if err != nil || len(members) != 1 {
		t.Fatalf("dispatched members = (%+v, %v)", members, err)
	}
	items, err := harness.audits.ListItems(ctx, harness.started.Audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	for _, item := range items {
		if item.ItemID == members[0].ItemID && item.ItemKey != "check-0" {
			t.Fatalf("dispatched item %q, want the first automatic item", item.ItemKey)
		}
	}
	claims, err := harness.audits.Claim(ctx, auditstore.ClaimParams{
		HolderID: "window-inspector", Lease: 5 * time.Second, Limit: 1,
	})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim for snapshot inspection = (%+v, %v)", claims, err)
	}
	snapshot, err := harness.audits.GetReconcileSnapshot(ctx, claims[0])
	if err != nil {
		t.Fatal(err)
	}
	if err := harness.audits.ReleaseClaim(ctx, claims[0]); err != nil {
		t.Fatal(err)
	}
	if snapshot.Audit.State != auditstore.AuditActive || snapshot.OnlyAwaitingReview || !snapshot.MoreItems {
		t.Fatalf("snapshot state=%q onlyAwaitingReview=%t moreItems=%t",
			snapshot.Audit.State, snapshot.OnlyAwaitingReview, snapshot.MoreItems)
	}
	found := false
	for _, item := range snapshot.Items {
		found = found || item.ItemID == members[0].ItemID
	}
	if !found {
		t.Fatalf("in-flight item %q is outside the bounded reconcile window", members[0].ItemID)
	}
}

func TestPostgresControllerWaitsForReviewBeyondReconcileWindow(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	harness := newPostgresControllerReviewHarness(t, ctx, auditstore.MaxReconcileRows+5, 0)
	controller := harness.controller(t)
	for step := 0; step < 4; step++ {
		if _, err := controller.RunOnce(ctx); err != nil {
			t.Fatalf("reconcile step=%d error=%v", step, err)
		}
	}
	audit, err := harness.audits.Get(ctx, harness.started.Audit.OwnerID, harness.started.Audit.AuditID)
	if err != nil || audit.State != auditstore.AuditWaitingReview {
		t.Fatalf("Audit with only reviews beyond the window = (%q, %v), want waiting_review", audit.State, err)
	}
}

func TestPostgresControllerDispatchesApprovedItemWhileOtherReviewsWait(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	harness := newPostgresControllerReviewHarness(t, ctx, 3, 0)
	auditID, ownerID := harness.started.Audit.AuditID, harness.started.Audit.OwnerID
	controller := harness.controller(t)
	for step := 0; step < 3; step++ {
		if _, err := controller.RunOnce(ctx); err != nil {
			t.Fatalf("reconcile step=%d error=%v", step, err)
		}
	}
	audit, err := harness.audits.Get(ctx, ownerID, auditID)
	if err != nil || audit.State != auditstore.AuditWaitingReview {
		t.Fatalf("Audit awaiting every review = (%q, %v), want waiting_review", audit.State, err)
	}
	reviews, err := harness.service.ListReviews(ctx, auditservice.ReviewListParams{
		OwnerID: ownerID, AuditID: auditID, Limit: 10,
	})
	if err != nil || len(reviews) != 3 {
		t.Fatalf("pending item reviews = (%+v, %v)", reviews, err)
	}
	decide := func(index int, action auditservice.ReviewAction) {
		t.Helper()
		key := fmt.Sprintf("decision-%d", index)
		if _, err := harness.service.DecideActionReview(ctx, auditservice.DecideActionReviewParams{
			OwnerID: ownerID, AuditID: auditID, RequestID: reviews[index].RequestID,
			ExpectedRequestRevision: reviews[index].Revision, DecisionID: key, Action: action,
			Rationale: "The owner decided this exact checklist action.", IdempotencyKey: key,
			RequestDigest: postgresDigest(key),
		}); err != nil {
			t.Fatalf("decide review %d with %q: %v", index, action, err)
		}
	}

	decide(0, auditservice.ReviewReject)
	audit, err = harness.audits.Get(ctx, ownerID, auditID)
	if err != nil || audit.State != auditstore.AuditWaitingReview {
		t.Fatalf("Audit after rejection with reviews left = (%q, %v), want waiting_review", audit.State, err)
	}
	decide(1, auditservice.ReviewApprove)
	audit, err = harness.audits.Get(ctx, ownerID, auditID)
	if err != nil || audit.State != auditstore.AuditActive {
		t.Fatalf("Audit after approval with a review left = (%q, %v), want active", audit.State, err)
	}
	var transitions int
	if err := harness.pool.QueryRow(ctx, `
SELECT count(*) FROM audit_events
 WHERE audit_id = $1 AND kind = 'audit.state_changed'
   AND summary = '{"from": "waiting_review", "to": "active"}'::jsonb`, auditID).Scan(&transitions); err != nil ||
		transitions != 1 {
		t.Fatalf("review activation events = (%d, %v)", transitions, err)
	}
	if worked, err := controller.RunOnce(ctx); err != nil || !worked {
		t.Fatalf("dispatch approved item = (%t, %v)", worked, err)
	}
	executions, err := harness.audits.ListExecutions(ctx, auditID)
	if err != nil || len(executions) != 1 || executions[0].RunID == nil {
		t.Fatalf("approved item executions = (%+v, %v)", executions, err)
	}
	members, err := harness.audits.ListExecutionItems(ctx, executions[0].ExecutionID)
	if err != nil || len(members) != 1 || members[0].ItemID != reviews[1].SubjectID {
		t.Fatalf("approved item members = (%+v, %v)", members, err)
	}
}

func TestPostgresDispatchReservationOrdersWithSettingsDecrease(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	harness := newPostgresControllerHarness(t, ctx, 3)
	settings := settingsstore.NewPostgresStore(harness.pool)
	current, err := settings.UpdateSchedulerSettings(ctx, settingsstore.UpdateSchedulerSettingsParams{
		MaxConcurrentRuns: 2, ExpectedRevision: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	claims, err := harness.audits.Claim(ctx, auditstore.ClaimParams{
		HolderID: "postgres-race-controller", Lease: time.Minute, Limit: 1,
	})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim Audit = (%+v, %v)", claims, err)
	}
	claim := claims[0]
	if _, err := harness.audits.TransitionRound(ctx, auditstore.RoundTransitionParams{
		Claim: claim, RoundID: harness.started.Round.RoundID,
		ExpectedRevision: harness.started.Round.Revision,
		ExpectedState:    auditstore.RoundAccepted, TargetState: auditstore.RoundExecuting,
	}); err != nil {
		t.Fatal(err)
	}
	snapshot, err := harness.audits.GetReconcileSnapshot(ctx, claim)
	if err != nil {
		t.Fatal(err)
	}
	builder := harness.builder(t)
	first, err := builder.Prepare(ctx, snapshot, snapshot.Items[0], 1)
	if err != nil {
		t.Fatal(err)
	}
	first.Intent.Claim = claim
	if _, inserted, err := harness.audits.CreateExecutionIntent(ctx, first.Intent); err != nil || !inserted {
		t.Fatalf("first reservation = (%t, %v)", inserted, err)
	}
	second, err := builder.Prepare(ctx, snapshot, snapshot.Items[1], 1)
	if err != nil {
		t.Fatal(err)
	}
	second.Intent.Claim = claim

	start := make(chan struct{})
	type reserveResult struct {
		inserted bool
		err      error
	}
	reserved := make(chan reserveResult, 1)
	updated := make(chan error, 1)
	go func() {
		<-start
		_, inserted, reserveErr := harness.audits.CreateExecutionIntent(ctx, second.Intent)
		reserved <- reserveResult{inserted: inserted, err: reserveErr}
	}()
	go func() {
		<-start
		_, updateErr := settings.UpdateSchedulerSettings(ctx, settingsstore.UpdateSchedulerSettingsParams{
			MaxConcurrentRuns: 1, ExpectedRevision: current.Revision,
		})
		updated <- updateErr
	}()
	close(start)
	reservation := <-reserved
	if reservation.err != nil && !errors.Is(reservation.err, auditstore.ErrPrecondition) {
		t.Fatalf("racing reservation error = %v", reservation.err)
	}
	if err := <-updated; err != nil {
		t.Fatalf("lower Scheduler setting: %v", err)
	}
	stored, err := harness.audits.GetReconcileSnapshot(ctx, claim)
	if err != nil {
		t.Fatal(err)
	}
	if reservation.inserted {
		if stored.Audit.OutstandingRunCount != 2 {
			t.Fatalf("old reservation won but outstanding = %d, want 2", stored.Audit.OutstandingRunCount)
		}
	} else if stored.Audit.OutstandingRunCount != 1 {
		t.Fatalf("lower setting won but outstanding = %d, want 1", stored.Audit.OutstandingRunCount)
	}
	third, err := builder.Prepare(ctx, stored, stored.Items[2], 1)
	if err != nil {
		t.Fatal(err)
	}
	third.Intent.Claim = claim
	if _, inserted, err := harness.audits.CreateExecutionIntent(ctx, third.Intent); !errors.Is(err, auditstore.ErrPrecondition) || inserted {
		t.Fatalf("reservation above lowered window = (%t, %v)", inserted, err)
	}
}

func TestPostgresControllersConvergeWithoutDuplicateExecutionAttempts(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	harness := newPostgresControllerHarness(t, ctx, 4)
	settings := settingsstore.NewPostgresStore(harness.pool)
	if _, err := settings.UpdateSchedulerSettings(ctx, settingsstore.UpdateSchedulerSettingsParams{
		MaxConcurrentRuns: 4, ExpectedRevision: 1,
	}); err != nil {
		t.Fatal(err)
	}

	controllers := []*Controller{
		harness.controllerForHolder(t, "controller-a"),
		harness.controllerForHolder(t, "controller-b"),
	}
	if worked, err := controllers[0].RunOnce(ctx); err != nil || !worked {
		t.Fatalf("activate immutable round = (%t, %v)", worked, err)
	}
	preflightClaims, err := harness.audits.Claim(ctx, auditstore.ClaimParams{
		HolderID: "controller-preflight", Lease: 5 * time.Second, Limit: 1,
	})
	if err != nil || len(preflightClaims) != 1 {
		t.Fatalf("preflight claim = (%+v, %v)", preflightClaims, err)
	}
	preflightSnapshot, err := harness.audits.GetReconcileSnapshot(ctx, preflightClaims[0])
	if err != nil {
		t.Fatal(err)
	}
	if _, err := harness.builder(t).Prepare(ctx, preflightSnapshot, preflightSnapshot.Items[0], 1); err != nil {
		t.Fatalf("preflight pinned submission: %v", err)
	}
	if err := harness.audits.ReleaseClaim(ctx, preflightClaims[0]); err != nil {
		t.Fatal(err)
	}
	for cycle := 0; cycle < 12; cycle++ {
		start := make(chan struct{})
		type result struct {
			worked bool
			err    error
		}
		results := make(chan result, len(controllers))
		for _, controller := range controllers {
			go func(controller *Controller) {
				<-start
				worked, err := controller.RunOnce(ctx)
				results <- result{worked: worked, err: err}
			}(controller)
		}
		close(start)
		for range controllers {
			if result := <-results; result.err != nil {
				t.Fatalf("concurrent controller cycle %d: %v", cycle, result.err)
			}
		}

		executions, err := harness.audits.ListExecutions(ctx, harness.started.Audit.AuditID)
		if err != nil {
			t.Fatal(err)
		}
		if len(executions) == 4 {
			break
		}
	}

	executions, err := harness.audits.ListExecutions(ctx, harness.started.Audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	if len(executions) != 4 {
		current, currentErr := harness.audits.Get(ctx, harness.started.Audit.OwnerID, harness.started.Audit.AuditID)
		round, roundErr := harness.audits.GetRound(ctx, harness.started.Audit.AuditID, harness.started.Round.RoundID)
		reason := "<nil>"
		if current.StopReason != nil {
			reason = current.StopReason.Code + ": " + current.StopReason.Message
		}
		t.Fatalf("execution count after two-controller reconciliation = %d, want 4: %+v; Audit state/reason=(%s, %s, %v); round=(%+v, %v)", len(executions), executions, current.State, reason, currentErr, round, roundErr)
	}
	seenItems := make(map[string]bool, len(executions))
	seenRuns := make(map[string]bool, len(executions))
	for _, execution := range executions {
		if execution.State != auditstore.ExecutionSubmitted || execution.RunID == nil {
			t.Fatalf("execution is not durably submitted: %+v", execution)
		}
		if seenRuns[*execution.RunID] {
			t.Fatalf("duplicate child Run %q", *execution.RunID)
		}
		seenRuns[*execution.RunID] = true
		members, err := harness.audits.ListExecutionItems(ctx, execution.ExecutionID)
		if err != nil || len(members) != 1 {
			t.Fatalf("execution %s members = (%+v, %v)", execution.ExecutionID, members, err)
		}
		if members[0].ItemAttempt != 1 || seenItems[members[0].ItemID] {
			t.Fatalf("duplicate or non-first logical attempt: %+v", members[0])
		}
		seenItems[members[0].ItemID] = true
	}
	if len(seenItems) != 4 {
		t.Fatalf("distinct reconciled items = %d, want 4", len(seenItems))
	}
	audit, err := harness.audits.Get(ctx, harness.started.Audit.OwnerID, harness.started.Audit.AuditID)
	if err != nil || audit.SubmittedRunCount != 4 || audit.OutstandingRunCount != 4 {
		t.Fatalf("Audit counters after reconciliation = (%+v, %v)", audit, err)
	}

	// A fresh process-local Controller has no wakeup history. It must still
	// reconcile PostgreSQL and observe the full dispatch window without
	// creating a fifth execution or a second attempt for any item.
	restarted := harness.controllerForHolder(t, "controller-after-restart")
	if worked, err := restarted.RunOnce(ctx); err != nil || worked {
		t.Fatalf("restarted controller at full window = (%t, %v), want no work", worked, err)
	}
	replayed, err := harness.audits.ListExecutions(ctx, harness.started.Audit.AuditID)
	if err != nil || len(replayed) != 4 {
		t.Fatalf("executions after restart reconciliation = (%+v, %v)", replayed, err)
	}
}

func TestPostgresControllerCollectsAndPublishesExactAuditReport(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	harness := newPostgresControllerHarness(t, ctx, 1)
	controller := harness.controllerWithCollector(t)

	for operation := 0; operation < 8; operation++ {
		executions, err := harness.audits.ListExecutions(ctx, harness.started.Audit.AuditID)
		if err != nil {
			t.Fatal(err)
		}
		if len(executions) == 1 && executions[0].State == auditstore.ExecutionSubmitted {
			break
		}
		worked, err := controller.RunOnce(ctx)
		if err != nil || !worked {
			t.Fatalf("dispatch operation %d = (%t, %v)", operation, worked, err)
		}
	}
	executions, err := harness.audits.ListExecutions(ctx, harness.started.Audit.AuditID)
	if err != nil || len(executions) != 1 || executions[0].RunID == nil {
		t.Fatalf("submitted Audit execution = (%+v, %v)", executions, err)
	}
	execution := executions[0]
	members, err := harness.audits.ListExecutionItems(ctx, execution.ExecutionID)
	if err != nil || len(members) != 1 {
		t.Fatalf("Audit execution members = (%+v, %v)", members, err)
	}
	projectArtifacts, err := harness.artifacts.Project(harness.started.Audit.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	taskPackage, err := projectArtifacts.Read(ctx, members[0].Task.Ref)
	if err != nil {
		t.Fatal(err)
	}
	validatedTask, err := auditdomain.ValidatePackage(taskPackage.Payload.Data)
	if err != nil {
		t.Fatal(err)
	}
	taskMember, exists := validatedTask.MemberByID("task-document")
	if !exists {
		t.Fatal("task package has no task-document member")
	}
	task, err := auditdomain.DecodeItemTask(taskMember.Data())
	if err != nil {
		t.Fatal(err)
	}
	requested := []string{}
	if task.Checklist != nil {
		requested = append(requested, task.Checklist.RequiredEvidence...)
	}
	resultDocument, err := auditdomain.EncodeCheckResultSet(auditdomain.CheckResultSet{
		Schema: auditdomain.CheckResultsSchema, ExecutionManifestDigest: execution.Manifest.Digest,
		Results: []auditdomain.CheckResult{{
			ItemKey: task.ItemKey, SubjectKey: task.SubjectKey, Assessment: "satisfied",
			Summary: "The bounded checklist item was satisfied.", EvidenceIDs: []string{},
			Coverage: auditdomain.ResultCoverage{
				Requested: requested, Completed: append([]string{}, requested...), Gaps: []string{},
			},
			Proposals: []auditdomain.ProposalSelection{},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	resultPayload, _, err := auditdomain.BuildPackage(
		"result-package", auditdomain.PackageKindCheckResults, "",
		[]auditdomain.PackageInput{{
			ID: auditdomain.CheckResultsMemberID, Path: "check-results.json",
			MediaType: auditdomain.JSONMediaType, Data: resultDocument,
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	runID := *execution.RunID
	runArtifacts, err := harness.artifacts.Run(runID)
	if err != nil {
		t.Fatal(err)
	}
	workerResult, err := runArtifacts.Write(
		ctx, contracts.ArtifactRef{Namespace: "worker", Name: "result"},
		artifacts.Payload{MediaType: auditdomain.PackageMediaType, Data: resultPayload}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	bound, err := harness.artifacts.BindOutputExact(ctx, runID, "result", workerResult.Ref, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := harness.artifacts.FreezeRunOutputs(ctx, runID); err != nil {
		t.Fatal(err)
	}
	run, err := harness.runs.GetRun(ctx, runID)
	if err != nil {
		t.Fatal(err)
	}
	if run.State == runstore.RunPending {
		if _, err := harness.runs.TransitionRun(
			ctx, runID, runstore.RunPending, runstore.RunRunning,
			runstore.Reason{Code: "test_started"},
		); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := harness.runs.TransitionRun(
		ctx, runID, runstore.RunRunning, runstore.RunSucceeded,
		runstore.Reason{Code: "test_succeeded"},
	); err != nil {
		t.Fatal(err)
	}

	for operation := 0; operation < 12; operation++ {
		audit, err := harness.audits.Get(ctx, harness.started.Audit.OwnerID, harness.started.Audit.AuditID)
		if err != nil {
			t.Fatal(err)
		}
		if audit.State == auditstore.AuditCompleted {
			break
		}
		worked, err := controller.RunOnce(ctx)
		if err != nil || !worked {
			t.Fatalf("collection operation %d = (%t, %v), Audit state %s", operation, worked, err, audit.State)
		}
	}
	completed, err := harness.audits.Get(ctx, harness.started.Audit.OwnerID, harness.started.Audit.AuditID)
	if err != nil || completed.State != auditstore.AuditCompleted {
		t.Fatalf("completed Audit = (%+v, %v)", completed, err)
	}
	receipts, err := harness.audits.CollectionDispositionCounts(ctx, completed.AuditID)
	if err != nil || receipts.AcceptedResult != 1 {
		t.Fatalf("collection disposition counts = (%+v, %v)", receipts, err)
	}
	resultLink, err := harness.audits.GetArtifactLink(ctx, completed.AuditID, "result/"+members[0].ExecutionItemID)
	if err != nil {
		t.Fatal(err)
	}
	resultMetadata, err := projectArtifacts.Metadata(ctx, resultLink.Artifact.Ref)
	if err != nil || !resultMetadata.Frozen || resultMetadata.Digest != resultLink.Artifact.Digest {
		t.Fatalf("retained result metadata = (%+v, %v)", resultMetadata, err)
	}
	lineage, err := projectArtifacts.ListLineage(
		ctx, resultLink.Artifact.Ref, artifacts.LineagePageQuery{Limit: 2},
	)
	if err != nil || len(lineage) != 1 || lineage[0].Kind != artifacts.LineageAuditImport ||
		lineage[0].SourceScopeID != runID || lineage[0].TargetScopeID != completed.ProjectID {
		t.Fatalf("retained result lineage = (%+v, %v)", lineage, err)
	}
	outputMetadata, err := runArtifacts.Metadata(ctx, bound.TargetRef)
	if err != nil {
		t.Fatal(err)
	}
	access, err := auditimport.NewArtifactAccess(harness.artifacts)
	if err != nil {
		t.Fatal(err)
	}
	replayed, err := access.RetainRunExact(
		ctx, runID,
		auditstore.ExactArtifact{
			Ref: outputMetadata.Ref, Digest: outputMetadata.Digest,
			MediaType: outputMetadata.MediaType, SizeBytes: outputMetadata.Size,
		},
		completed.ProjectID,
		contracts.ArtifactRef{Namespace: resultLink.Artifact.Ref.Namespace, Name: resultLink.Artifact.Ref.Name},
	)
	if err != nil || replayed.Digest != resultLink.Artifact.Digest ||
		replayed.Ref.Revision == nil || *replayed.Ref.Revision != *resultLink.Artifact.Ref.Revision {
		t.Fatalf("retained result replay = (%+v, %v)", replayed, err)
	}

	auditService, err := auditservice.New(auditservice.Options{
		Pool: harness.pool, Profiles: harness.snapshot,
		TransactionLLMCredentials: controllerTransactionCredentialLookup(),
		CredentialGuard:           controllerCredentialGuard{},
	})
	if err != nil {
		t.Fatal(err)
	}
	report, err := auditService.GetReport(ctx, completed.OwnerID, completed.AuditID)
	if err != nil || report.Status != auditservice.ReportReady ||
		!json.Valid(report.Machine) || !strings.Contains(report.Summary, "not a security or compliance certification") {
		t.Fatalf("Audit report = (%+v, %v)", report, err)
	}
	if report.MachineArtifact == nil || report.MachineArtifact.Ref.Revision == nil {
		t.Fatalf("Audit report has no exact machine artifact: %+v", report)
	}
	if _, err := projectArtifacts.Write(
		ctx,
		contracts.ArtifactRef{
			Namespace: report.MachineArtifact.Ref.Namespace,
			Name:      report.MachineArtifact.Ref.Name,
		},
		artifacts.Payload{MediaType: "application/json", Data: []byte(`{"tampered":true}`)},
		report.MachineArtifact.Ref.Revision,
	); !errors.Is(err, artifacts.ErrArtifactFrozen) {
		t.Fatalf("mutate frozen Audit report error = %v", err)
	}

	if err := harness.runs.DeleteReleasedTerminalRun(ctx, completed.OwnerID, runID); err != nil {
		t.Fatal(err)
	}
	attempts, err := harness.audits.ListItemAttempts(
		ctx, completed.OwnerID, completed.AuditID, []string{members[0].ItemID},
	)
	if err != nil || len(attempts[members[0].ItemID]) != 1 {
		t.Fatalf("Audit attempt tombstone = (%+v, %v)", attempts, err)
	}
	attempt := attempts[members[0].ItemID][0]
	if !attempt.RunDeleted || attempt.RunID == nil || *attempt.RunID != runID ||
		attempt.RunProvenance == nil || attempt.RunProvenance.Workflow == nil ||
		attempt.RunProvenance.Workflow.Name != "audit-check" {
		t.Fatalf("Audit attempt Run provenance = %+v", attempt)
	}
	retainedItems, err := harness.audits.ListItems(ctx, completed.AuditID)
	if err != nil || len(retainedItems) != 1 {
		t.Fatalf("retained Audit items = (%+v, %v)", retainedItems, err)
	}
	origin := retainedItems[0].Origin
	if origin.SourceRef == nil || origin.SourceRef.Revision == nil ||
		origin.EntryKey != "check-0" || origin.EntryVersion != "1" ||
		origin.SourceContentDigest == "" || origin.CanonicalInventoryDigest == "" {
		t.Fatalf("retained exact checklist origin = %+v", origin)
	}
	if _, err := projectArtifacts.Read(ctx, resultLink.Artifact.Ref); err != nil {
		t.Fatalf("retained result after source Run deletion: %v", err)
	}
	report, err = auditService.GetReport(ctx, completed.OwnerID, completed.AuditID)
	if err != nil || report.Status != auditservice.ReportReady {
		t.Fatalf("Audit report after source Run deletion = (%+v, %v)", report, err)
	}
	deleteParams := auditservice.MutationParams{
		OwnerID: completed.OwnerID, AuditID: completed.AuditID, ExpectedRevision: completed.Revision,
		IdempotencyKey: "delete-completed-audit", RequestDigest: postgresDigest("delete-completed-audit"),
	}
	if _, err := auditService.Delete(ctx, deleteParams); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("delete Audit with pre-Run-deletion revision = %v", err)
	}
	refreshed, err := harness.audits.Get(ctx, completed.OwnerID, completed.AuditID)
	if err != nil || refreshed.State != completed.State || refreshed.Revision != completed.Revision+1 {
		t.Fatalf("Audit revision after source Run deletion = (%+v, %v)", refreshed, err)
	}
	deleteParams.ExpectedRevision = refreshed.Revision
	deletion, err := auditService.Delete(ctx, deleteParams)
	if err != nil || deletion.Audit.State != auditstore.AuditDeleting || deletion.Audit.DeletionRequestedAt == nil {
		t.Fatalf("begin completed Audit deletion = (%+v, %v)", deletion, err)
	}
	if worked, err := controller.RunOnce(ctx); err != nil || !worked {
		t.Fatalf("purge completed Audit = (%t, %v)", worked, err)
	}
	if _, err := harness.audits.Get(ctx, completed.OwnerID, completed.AuditID); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("deleted Audit lookup = %v", err)
	}
	if _, err := projectArtifacts.Read(ctx, resultLink.Artifact.Ref); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("deleted Audit retained result = %v", err)
	}
}

func TestPostgresControllerDeletesActiveAuditWhileOwnerQueuePaused(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	harness := newPostgresControllerHarness(t, ctx, 2)
	controller := harness.controllerWithCollector(t)
	if worked, err := controller.RunOnce(ctx); err != nil || !worked {
		t.Fatalf("activate round = (%t, %v)", worked, err)
	}
	if worked, err := controller.RunOnce(ctx); err != nil || !worked {
		t.Fatalf("dispatch child = (%t, %v)", worked, err)
	}
	executions, err := harness.audits.ListExecutions(ctx, harness.started.Audit.AuditID)
	if err != nil || len(executions) != 1 || executions[0].RunID == nil {
		t.Fatalf("submitted child = (%+v, %v)", executions, err)
	}
	runID := *executions[0].RunID
	if _, err := harness.runs.UpdateOwnerQueueControl(
		ctx, runstore.UpdateOwnerQueueControlParams{
			OwnerID: harness.started.Audit.OwnerID, ExpectedRevision: 0, Paused: true,
		},
	); err != nil {
		t.Fatal(err)
	}
	service, err := auditservice.New(auditservice.Options{
		Pool: harness.pool, Profiles: harness.snapshot,
		TransactionLLMCredentials: controllerTransactionCredentialLookup(),
		CredentialGuard:           controllerCredentialGuard{},
	})
	if err != nil {
		t.Fatal(err)
	}
	current, err := harness.audits.Get(ctx, harness.started.Audit.OwnerID, harness.started.Audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	deletion, err := service.Delete(ctx, auditservice.MutationParams{
		OwnerID: current.OwnerID, AuditID: current.AuditID, ExpectedRevision: current.Revision,
		IdempotencyKey: "delete-active-audit", RequestDigest: postgresDigest("delete-active-audit"),
	})
	if err != nil || deletion.Audit.State != auditstore.AuditCancelling || deletion.Audit.DeletionRequestedAt == nil {
		t.Fatalf("active Audit deletion = (%+v, %v)", deletion, err)
	}
	if worked, err := controller.RunOnce(ctx); err != nil || !worked {
		t.Fatalf("cancel child while queue paused = (%t, %v)", worked, err)
	}
	run, err := harness.runs.GetRun(ctx, runID)
	if err != nil || run.State != runstore.RunCancelling {
		t.Fatalf("cancelled child state = (%+v, %v)", run, err)
	}
	if _, err := harness.runs.TransitionRun(
		ctx, runID, runstore.RunCancelling, runstore.RunCancelled,
		runstore.Reason{Code: "test_cancelled"},
	); err != nil {
		t.Fatal(err)
	}
	for operation := 0; operation < 12; operation++ {
		if _, err := harness.audits.Get(ctx, current.OwnerID, current.AuditID); errors.Is(err, auditstore.ErrNotFound) {
			break
		} else if err != nil {
			t.Fatal(err)
		}
		worked, err := controller.RunOnce(ctx)
		if err != nil || !worked {
			t.Fatalf("deletion operation %d = (%t, %v)", operation, worked, err)
		}
	}
	if _, err := harness.audits.Get(ctx, current.OwnerID, current.AuditID); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("deleted active Audit lookup = %v", err)
	}
	if _, err := harness.runs.GetRun(ctx, runID); !errors.Is(err, runstore.ErrNotFound) {
		t.Fatalf("deleted child Run lookup = %v", err)
	}
	queue, err := harness.runs.GetOwnerQueueControl(ctx, current.OwnerID)
	if err != nil || !queue.Paused {
		t.Fatalf("owner queue state after Audit cleanup = (%+v, %v)", queue, err)
	}
}

type postgresControllerHarness struct {
	pool       *pgxpool.Pool
	snapshot   *config.Snapshot
	artifacts  *artifacts.Service
	audits     *auditstore.PostgresStore
	runs       *runstore.PostgresStore
	runService *runservice.Service
	service    *auditservice.Service
	started    auditservice.StartedAudit
}

func newPostgresControllerHarness(
	t *testing.T, ctx context.Context, itemCount int, batchSizes ...int,
) *postgresControllerHarness {
	t.Helper()
	return newPostgresControllerHarnessWithConfig(t, ctx, 0, itemCount, loadControllerConfig(t, batchSizes...))
}

// newPostgresControllerReviewHarness materializes manualCount manual-review
// checklist items ordered before itemCount automatic ones.
func newPostgresControllerReviewHarness(
	t *testing.T, ctx context.Context, manualCount, itemCount int, batchSizes ...int,
) *postgresControllerHarness {
	t.Helper()
	return newPostgresControllerHarnessWithConfig(
		t, ctx, manualCount, itemCount,
		loadControllerConfigWithItemLimit(t, max(10, manualCount+itemCount), batchSizes...),
	)
}

func newPostgresControllerHarnessWithConfig(
	t *testing.T, ctx context.Context, manualCount, itemCount int, snapshot *config.Snapshot,
) *postgresControllerHarness {
	t.Helper()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	pool := isolatedControllerPool(t, ctx, databaseURL)
	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	projects := projectstore.NewPostgresStore(pool)
	project, _, err := projects.Create(ctx, projectstore.CreateParams{
		ProjectID: "project-audit-controller", OwnerID: "owner-audit-controller",
		Kind: projectstore.KindProject, Name: "Audit Controller test",
		IdempotencyKey: "create-controller-project", RequestDigest: postgresDigest("project"),
	})
	if err != nil {
		t.Fatal(err)
	}
	projectArtifacts, err := artifactService.Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	checklist := `{"schema":"contractor.audit.checklist.v1","items":[`
	entries := make([]string, 0, manualCount+itemCount)
	for index := range manualCount {
		// Inventory orders items by key, so these precede every automatic check.
		entries = append(entries, fmt.Sprintf(
			`{"key":"approval-%04d","version":"1","statement":"Review check %d.","applicability":"always","allowed_methods":["static"],"required_evidence":[],"review_policy":"manual"}`,
			index, index,
		))
	}
	for index := range itemCount {
		entries = append(entries, fmt.Sprintf(
			`{"key":"check-%d","version":"1","statement":"Verify check %d.","applicability":"always","allowed_methods":["static"],"required_evidence":[],"review_policy":"automatic"}`,
			index, index,
		))
	}
	checklist += strings.Join(entries, ",")
	checklist += `]}`
	input, err := projectArtifacts.Write(
		ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "checklist"},
		artifacts.Payload{MediaType: "application/json", Data: []byte(checklist)}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	guard := controllerCredentialGuard{}
	auditService, err := auditservice.New(auditservice.Options{
		Pool: pool, Profiles: snapshot,
		TransactionLLMCredentials: controllerTransactionCredentialLookup(),
		CredentialGuard:           guard,
	})
	if err != nil {
		t.Fatal(err)
	}
	draft, _, err := auditService.CreateDraft(ctx, auditservice.CreateDraftParams{
		AuditID: "audit-controller", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:       auditservice.ProfileSelector{Name: "test-checklist", Version: "1"},
		Inputs:        map[string]contracts.ArtifactRef{"checklist": input.Ref},
		RuntimeLabels: []string{}, Scope: auditservice.Scope{Objective: "Verify the test checklist"},
		IdempotencyKey: "create-controller-audit", RequestDigest: postgresDigest("audit-create"),
	})
	if err != nil {
		t.Fatal(err)
	}
	started, err := auditService.Start(ctx, auditservice.StartParams{
		OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision,
		IdempotencyKey: "start-controller-audit", RequestDigest: postgresDigest("audit-start"),
	})
	if err != nil {
		t.Fatal(err)
	}
	runs := runstore.NewPostgresStore(pool)
	audits := auditstore.NewPostgresStore(pool)
	runCreation, err := runservice.New(runservice.Options{
		Runs: runs, Workflows: snapshot, LLMCredentials: controllerCredentialLookup{},
		CredentialGuard: guard, RuntimeCredentials: controllerRuntimeCredentials{}, Projects: projects,
		SkillInitializationAvailable: true,
		PublicTransaction: func(ctx context.Context, fn func(runservice.PublicRunWriter, *artifacts.Service) error) error {
			return persistencepostgres.InTx(ctx, pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead}, func(tx pgx.Tx) error {
				lookup, bindErr := runtimeconfig.BindTransactionLLMCredentialLookup(
					tx, controllerTransactionCredentialLookup(),
				)
				if bindErr != nil {
					return bindErr
				}
				return fn(
					runstore.NewRunCreationPostgresStore(tx, lookup),
					artifacts.NewService(artifacts.NewPostgresRepository(tx)),
				)
			})
		},
		AuditTransaction: func(ctx context.Context, fn func(runservice.AuditRunWriter, *artifacts.Service, runservice.AuditExecutionWriter) error) error {
			return persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
				return fn(
					runstore.NewPostgresStore(tx), artifacts.NewService(artifacts.NewPostgresRepository(tx)),
					auditstore.NewPostgresStore(tx),
				)
			})
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return &postgresControllerHarness{
		pool: pool, snapshot: snapshot, artifacts: artifactService,
		audits: audits, runs: runs, runService: runCreation, service: auditService, started: started,
	}
}

func (h *postgresControllerHarness) builder(t *testing.T) *PinnedSubmissionBuilder {
	t.Helper()
	access, err := NewProjectArtifactAccess(h.artifacts)
	if err != nil {
		t.Fatal(err)
	}
	builder, err := NewPinnedSubmissionBuilder(access, h.audits)
	if err != nil {
		t.Fatal(err)
	}
	return builder
}

func (h *postgresControllerHarness) controller(t *testing.T) *Controller {
	t.Helper()
	var ids int
	controller, err := New(
		h.audits, h.runs, h.runService, h.builder(t), &postgresNotifier{},
		Options{
			HolderID: "postgres-audit-controller", ClaimLease: 5 * time.Second,
			OperationTimeout: time.Second, ClaimBatch: 1,
			NewID: func(prefix string) (string, error) {
				ids++
				return fmt.Sprintf("%s%d", prefix, ids), nil
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return controller
}

func (h *postgresControllerHarness) controllerForHolder(t *testing.T, holderID string) *Controller {
	t.Helper()
	var ids int
	controller, err := New(
		h.audits, h.runs, h.runService, h.builder(t), &postgresNotifier{},
		Options{
			HolderID: holderID, ClaimLease: 5 * time.Second,
			OperationTimeout: time.Second, ClaimBatch: 1,
			NewID: func(prefix string) (string, error) {
				ids++
				return fmt.Sprintf("%s%s_%d", prefix, holderID, ids), nil
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return controller
}

func (h *postgresControllerHarness) controllerWithCollector(
	t *testing.T, findings ...auditimport.FindingRetention,
) *Controller {
	t.Helper()
	access, err := auditimport.NewArtifactAccess(h.artifacts)
	if err != nil {
		t.Fatal(err)
	}
	collector, err := auditimport.New(h.audits, h.runs, access, findings...)
	if err != nil {
		t.Fatal(err)
	}
	var ids int
	controller, err := New(
		h.audits, h.runs, h.runService, h.builder(t), &postgresNotifier{},
		Options{
			HolderID: "postgres-audit-collection-controller", ClaimLease: 5 * time.Second,
			OperationTimeout: 2 * time.Second, ClaimBatch: 1, Collector: collector,
			NewID: func(prefix string) (string, error) {
				ids++
				return fmt.Sprintf("%s%d", prefix, ids), nil
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return controller
}

type postgresNotifier struct{}

func (*postgresNotifier) Wake()         {}
func (*postgresNotifier) Cancel(string) {}

type controllerCredentialLookup struct{}

func (controllerCredentialLookup) LookupLLMCredential(context.Context, string) (config.CredentialMetadata, error) {
	return config.CredentialMetadata{}, fmt.Errorf("credential is unavailable")
}

func controllerTransactionCredentialLookup() runtimeconfig.TransactionLLMCredentialLookupFactory {
	return runtimeconfig.TransactionLLMCredentialLookupFactoryFunc(
		func(pgx.Tx) (config.CredentialLookup, error) { return controllerCredentialLookup{}, nil },
	)
}

type controllerCredentialGuard struct{}

func (controllerCredentialGuard) WithRunCreation(ctx context.Context, fn func() error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return fn()
}

type controllerRuntimeCredentials struct{}

func (controllerRuntimeCredentials) ValidateRuntimeCredentialUse(
	context.Context, credentials.RuntimeCredentialUser, string, ...string,
) error {
	return nil
}

func loadControllerConfig(t *testing.T, batchSizes ...int) *config.Snapshot {
	t.Helper()
	return loadControllerConfigWithItemLimit(t, 10, batchSizes...)
}

func loadControllerConfigWithItemLimit(t *testing.T, maxItems int, batchSizes ...int) *config.Snapshot {
	t.Helper()
	batchSize := 1
	if len(batchSizes) > 0 {
		batchSize = batchSizes[0]
	}
	if len(batchSizes) > 1 || batchSize < 1 || batchSize > config.MaxAuditBatchSize {
		t.Fatalf("invalid test batch size %v", batchSizes)
	}
	return loadControllerConfigWithFindings(t, batchSize, maxItems, false)
}

// loadControllerConfigWithFindings optionally lets the worker propose
// findings and requires human confirmation for them.
func loadControllerConfigWithFindings(t *testing.T, batchSize, maxItems int, findings bool) *config.Snapshot {
	t.Helper()
	findingTools, findingConfirmation := "", "disabled"
	if findings {
		findingTools = `
    - ref: security-findings@1
      tools: [finding]`
		findingConfirmation = "human-required"
	}
	root := t.TempDir()
	files := map[string]string{
		"instructions/planner.md": "Execute the selected checklist item.",
		"instructions/worker.md":  "Read the task package and write a result package.",
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
      tools: [read_artifact, write_artifact]` + findingTools + `
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
		"audit-profiles/checklist.yaml": fmt.Sprintf(`apiVersion: contractor/v1alpha1
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
    batchSize: %d
    maxItemsPerRound: %d
    maxItemsTotal: %d
    maxSubmittedRuns: %d
    maxItemRunAttempts: 2
    deadlineSeconds: 3600
    maxEvidenceBytes: 1048576
    incompleteRound: assess-with-gaps
  interaction:
    activeChecks: prohibited
    findingConfirmation: %s
    notApplicable: profile-rule
    reportAcceptance: automatic
`, batchSize, maxItems, maxItems, 2*maxItems, findingConfirmation),
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
		t.Fatalf("load Audit Controller test configuration: %v", err)
	}
	return snapshot
}

func isolatedControllerPool(
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
		t.Skipf("PostgreSQL is unavailable: %v", err)
	}
	random := make([]byte, 8)
	if _, err := cryptorand.Read(random); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	schema := "contractor_audit_controller_test_" + hex.EncodeToString(random)
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
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		pool.Close()
		admin.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		pool.Close()
		cleanupContext, cleanupCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cleanupCancel()
		_, _ = admin.Exec(cleanupContext, `DROP SCHEMA `+identifier+` CASCADE`)
		admin.Close()
	})
	return pool
}

func postgresDigest(value string) string {
	result := ""
	for len(result) < 64 {
		result += hex.EncodeToString([]byte(value))
		if value == "" {
			result += "0"
		}
	}
	return "sha256:" + result[:64]
}
