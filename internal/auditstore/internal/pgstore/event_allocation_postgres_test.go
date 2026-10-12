package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"slices"
	"sort"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

// auditEventDensityViolationsSQL lists the Audits whose events are not
// numbered 1..N with N = next_event_sequence - 1.
const auditEventDensityViolationsSQL = `
SELECT audit.audit_id, events.count, events.first_sequence, events.last_sequence,
       audit.next_event_sequence - 1
  FROM audits AS audit
  CROSS JOIN LATERAL (
      SELECT count(*) AS count,
             COALESCE(min(event.sequence_number), 1) AS first_sequence,
             COALESCE(max(event.sequence_number), 0) AS last_sequence
        FROM audit_events AS event
       WHERE event.audit_id = audit.audit_id
  ) AS events
 WHERE events.count <> audit.next_event_sequence - 1
    OR events.last_sequence <> audit.next_event_sequence - 1
    OR events.first_sequence <> 1
 ORDER BY audit.audit_id`

// requireDenseAuditEvents fails the test for every Audit whose event
// sequence has a gap, a duplicate count or a reservation without events.
func requireDenseAuditEvents(t *testing.T, ctx context.Context, db persistencepostgres.DBTX) {
	t.Helper()
	rows, err := db.Query(ctx, auditEventDensityViolationsSQL)
	if err != nil {
		t.Errorf("check Audit event density: %v", err)
		return
	}
	defer rows.Close()
	for rows.Next() {
		var auditID string
		var count, first, last, reserved int64
		if err := rows.Scan(&auditID, &count, &first, &last, &reserved); err != nil {
			t.Errorf("scan Audit event density: %v", err)
			return
		}
		t.Errorf("Audit %s has %d events numbered %d..%d, want 1..%d", auditID, count, first, last, reserved)
	}
	if err := rows.Err(); err != nil {
		t.Errorf("iterate Audit event density: %v", err)
	}
}

func TestPostgresAuditEventAllocatorRejectsBrokenReservations(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedAuditPool(t, ctx, databaseURL)
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "project-allocator", OwnerID: "owner-allocator", Kind: projectstore.KindProject,
		Name: "Allocator", IdempotencyKey: "project-create", RequestDigest: testDigest("1"),
	})
	if err != nil {
		t.Fatal(err)
	}
	audit, _, err := newTestStore(pool).CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: "audit-allocator", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:         auditstore.ProfileIdentity{Name: "checklist", Version: "1", Digest: testDigest("2")},
		ProfileSnapshot: json.RawMessage(`{"name":"checklist"}`), InputSelection: json.RawMessage(`{}`),
		Limits: auditstore.Limits{
			MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1,
			MaxSubmittedRuns: 1, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1,
		},
		IdempotencyKey: "audit-create", RequestDigest: testDigest("3"),
	})
	if err != nil || audit.EventSequence != 1 {
		t.Fatalf("create Audit = (%+v, %v), want event sequence 1", audit, err)
	}
	const appendSQL = `
WITH changed AS (
    UPDATE audits AS audit
       SET next_event_sequence = audit.next_event_sequence + $2
     WHERE audit.audit_id = $1
    RETURNING contractor_append_audit_events(audit.audit_id, $3::jsonb) AS last_event_sequence
)
SELECT last_event_sequence FROM changed`
	event := `{"kind":"review.requested","entityId":"review-allocator","entityRevision":1,"summary":{}}`
	for _, broken := range []struct {
		name     string
		reserved int
		events   string
	}{
		{"reserved more than appended", 2, "[" + event + "]"},
		{"reserved fewer than appended", 1, "[" + event + "," + event + "]"},
		{"appended without a reservation", 0, "[" + event + "]"},
		{"not an array", 1, event},
	} {
		var last int64
		err := pool.QueryRow(ctx, appendSQL, audit.AuditID, broken.reserved, broken.events).Scan(&last)
		var postgresError *pgconn.PgError
		if !errors.As(err, &postgresError) || postgresError.Code != persistencepostgres.SQLStateAssertFailure {
			t.Errorf("%s = (%d, %v), want SQLSTATE P0004", broken.name, last, err)
		}
	}
	var direct int64
	err = pool.QueryRow(ctx, `SELECT contractor_append_audit_events($1, $2::jsonb)`,
		audit.AuditID, "["+event+"]").Scan(&direct)
	var postgresError *pgconn.PgError
	if !errors.As(err, &postgresError) || postgresError.Code != persistencepostgres.SQLStateAssertFailure {
		t.Errorf("unreserved direct append = (%d, %v), want SQLSTATE P0004", direct, err)
	}
	var last int64
	if err := pool.QueryRow(ctx, appendSQL, audit.AuditID, 2, "["+event+","+event+"]").Scan(&last); err != nil || last != 3 {
		t.Fatalf("reserved append = (%d, %v), want last sequence 3", last, err)
	}
	stored, err := newTestStore(pool).Get(ctx, audit.OwnerID, audit.AuditID)
	if err != nil || stored.EventSequence != 3 {
		t.Fatalf("Audit after appends = (%+v, %v), want event sequence 3", stored, err)
	}
	requireDenseAuditEvents(t, ctx, pool)
}

// TestPostgresAuditEventAllocatorIsTheOnlyEventWriter checks the migrated
// schema: one routine inserts Audit events, and the only other routine that
// advances an Audit's event counter hands its events to it.
func TestPostgresAuditEventAllocatorIsTheOnlyEventWriter(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedAuditPool(t, ctx, databaseURL)
	routines := func(pattern string) []string {
		rows, err := pool.Query(ctx, `
SELECT routine.proname
  FROM pg_proc AS routine
  JOIN pg_namespace AS namespace ON namespace.oid = routine.pronamespace
 WHERE namespace.nspname = current_schema() AND routine.prosrc ~* $1
 ORDER BY routine.proname`, pattern)
		if err != nil {
			t.Fatal(err)
		}
		names, err := pgx.CollectRows(rows, pgx.RowTo[string])
		if err != nil {
			t.Fatal(err)
		}
		return names
	}
	if writers := routines(`insert\s+into\s+audit_events\M`); !slices.Equal(writers, []string{"contractor_append_audit_events"}) {
		t.Errorf("routines inserting Audit events = %q, want only the allocator", writers)
	}
	reservations := routines(`next_event_sequence\s*=`)
	if !slices.Equal(reservations, []string{"contractor_admit_audit_finding"}) {
		t.Errorf("routines advancing Audit event counters = %q", reservations)
	}
	for _, name := range reservations {
		var callsAllocator bool
		if err := pool.QueryRow(ctx, `
SELECT prosrc ~ 'contractor_append_audit_events\(' FROM pg_proc
 WHERE proname = $1 AND pronamespace = current_schema()::regnamespace`, name).Scan(&callsAllocator); err != nil || !callsAllocator {
			t.Errorf("%s reserves Audit events without the allocator (%t, %v)", name, callsAllocator, err)
		}
	}
}

// TestPostgresAuditEventSequencesStayDense drives the Audit mutations that
// record events and checks after every step that each Audit's events stay
// numbered 1..N with N = EventSequence. isolatedAuditPool repeats the check
// after every other test of this package, which adds the next Round
// (round_postgres_test.go) and settlement with expired item reviews
// (review_expiry_postgres_test.go). Preparation and initial inventory
// acceptance need a resolved profile catalog; the artifacts preparation and
// auditcontroller initial inventory tests check them.
func TestPostgresAuditEventSequencesStayDense(t *testing.T) {
	if os.Getenv("CONTRACTOR_TEST_DATABASE_URL") == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	recorded := map[string]bool{}
	step := func(t *testing.T, f collectingAuditFixture, name string) auditstore.Audit {
		t.Helper()
		requireDenseAuditEvents(t, f.ctx, f.pool)
		if t.Failed() {
			t.Fatalf("Audit events are not dense after %s", name)
		}
		rows, err := f.pool.Query(f.ctx, `SELECT DISTINCT kind FROM audit_events WHERE audit_id = $1`, f.audit.AuditID)
		if err != nil {
			t.Fatal(err)
		}
		kinds, err := pgx.CollectRows(rows, pgx.RowTo[string])
		if err != nil {
			t.Fatal(err)
		}
		for _, kind := range kinds {
			recorded[kind] = true
		}
		audit, err := f.store.Get(f.ctx, f.audit.OwnerID, f.audit.AuditID)
		if err != nil {
			t.Fatal(err)
		}
		return audit
	}

	t.Run("collection review and accepted report", func(t *testing.T) {
		f := newCollectingAuditFixture(t, "dense-accept", 1<<20, collectingAuditFixtureOptions{extraPending: true})
		step(t, f, "dispatch")
		proposal := testExact("audit-findings", "candidate-dense", "proposal-r1")
		proposal.MediaType, proposal.SizeBytes = "application/json", 128
		findingID := insertAuditChildFinding(t, f, "receipt-dense", proposal)
		if _, err := f.pool.Exec(f.ctx, `
INSERT INTO audit_review_requests (
    request_id, audit_id, finding_id, subject_kind, subject_id, kind,
    subject_revision, subject_digest, requested_actions, idempotency_key, request_digest
) VALUES ('review-dense', $1, $2, 'finding', $2, 'finding-triage', 1, $3,
          '["true_positive","false_positive"]'::jsonb, 'review-dense', $3)`,
			f.audit.AuditID, findingID, testDigest("8")); err != nil {
			t.Fatal(err)
		}
		step(t, f, "finding admission")
		revision := uint64(1)
		if err := f.store.AppendReviewEvents(f.ctx, []auditstore.ReviewEventParams{
			{AuditID: f.audit.AuditID, Kind: "review.requested", EntityID: "review-dense",
				EntityRevision: &revision, Summary: map[string]any{"subjectKind": "finding", "findingId": findingID}},
			{AuditID: f.audit.AuditID, Kind: "review.decided", EntityID: "decision-dense",
				Summary: map[string]any{"verdict": "true_positive"}},
		}); err != nil {
			t.Fatal(err)
		}
		step(t, f, "review events")
		if err := f.store.RecordRejectedFindingProposal(f.ctx, auditstore.RejectedFindingProposalParams{
			AuditID: f.audit.AuditID, ReceiptID: "receipt-rejected-dense",
			RunID: *f.execution.RunID, Reason: "invalid-proposal",
		}); err != nil {
			t.Fatal(err)
		}
		if err := f.store.RecordDirectFindingAssessment(f.ctx, auditstore.DirectFindingAssessmentParams{
			AuditID: f.audit.AuditID, FindingID: findingID, FindingRevision: 1,
			AssessmentID: "assessment-direct-dense", RetainedBytes: 1,
		}); err != nil {
			t.Fatal(err)
		}
		step(t, f, "finding events")
		result := testExact("audit-dense", "result", "result-r1")
		if _, inserted, err := f.store.Collect(f.ctx, auditstore.CollectParams{
			Claim: f.claim, ReceiptID: "receipt-collection-dense", ExecutionID: f.execution.ExecutionID,
			Disposition: auditstore.CollectionAccepted, SourceOutput: &result, RequestDigest: testDigest("9"),
			Retained: []auditstore.ArtifactLink{{
				LogicalKey: "result/" + f.memberID, Artifact: result,
				SourceProvenance: json.RawMessage(`{"runId":"run-dense"}`),
			}},
			Items: []auditstore.CollectionItem{{
				ExecutionItemID: f.memberID, Disposition: auditstore.CollectionAccepted,
				FinalDisposition: auditstore.FinalAccepted, Result: &result,
				Coverage: auditstore.Coverage{Status: auditstore.CoverageSatisfied, Requested: []string{}, Completed: []string{}, Gaps: []string{}},
				FindingAssociations: []auditstore.FindingAssociation{{
					AssessmentID: "assessment-dense", ReceiptID: "receipt-dense",
					Proposal: proposal, SemanticAssessment: "supported",
				}},
			}},
		}); err != nil || !inserted {
			t.Fatalf("accepted collection = (%t, %v)", inserted, err)
		}
		step(t, f, "collection expiring a triage review")
		roundID := "round-dense-accept"
		failedTask := testExact("audits", "dense-accept-pending-task", "task-r1")
		failed, _, err := f.store.CreateExecutionIntent(f.ctx, auditstore.CreateExecutionIntentParams{
			Claim: f.claim, ExecutionID: "execution-dense-failed", RoundID: &roundID, Role: auditstore.ExecutionCheck,
			WorkflowRole: "check", Manifest: testExact("audits", "dense-failed-execution", "manifest-r1"),
			SubmissionKey: "submission-dense-failed", RequestDigest: testDigest("a"),
			Members: []auditstore.ExecutionMemberIntent{{
				ExecutionItemID: "member-dense-failed", ItemID: "pending-item-dense-accept", BatchOrdinal: 0,
				ItemAttempt: 1, Task: failedTask, Inputs: []auditstore.ExactArtifact{},
			}},
		})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := f.store.ObserveSubmissionFailure(f.ctx, auditstore.ObserveSubmissionFailureParams{
			Claim: f.claim, ExecutionID: failed.ExecutionID,
		}); err != nil {
			t.Fatal(err)
		}
		collectFinalFailure(t, f, failed.ExecutionID, "member-dense-failed", "receipt-dense-failed")
		step(t, f, "submission failure")
		finalizeDenseFixture(t, f, roundID)
		step(t, f, "finalizing")
		if _, released, err := f.store.ReleaseDispatchHold(f.ctx, f.claim); err != nil || !released {
			t.Fatalf("release dispatch hold = (%t, %v)", released, err)
		}
		waiting := proposeDenseReport(t, f, roundID, step(t, f, "hold release"))
		candidate, err := f.store.GetReportCandidate(f.ctx, f.audit.AuditID)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := f.store.AcceptReportCandidate(f.ctx, auditstore.ReportDecisionParams{
			AuditID: f.audit.AuditID, ExpectedAuditRevision: waiting.Revision,
			RequestID: candidate.RequestID, SubjectRevision: candidate.SubjectRevision,
			SubjectDigest: candidate.SubjectDigest,
		}); err != nil {
			t.Fatal(err)
		}
		if completed := step(t, f, "accepted report"); completed.State != auditstore.AuditCompleted {
			t.Fatalf("accepted report left Audit %s", completed.State)
		}
	})

	t.Run("rejected report", func(t *testing.T) {
		f, roundID := newFinalizingDenseFixture(t, "dense-reject")
		waiting := proposeDenseReport(t, f, roundID, step(t, f, "finalizing"))
		candidate, err := f.store.GetReportCandidate(f.ctx, f.audit.AuditID)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := f.store.RejectReportCandidate(f.ctx, auditstore.ReportDecisionParams{
			AuditID: f.audit.AuditID, ExpectedAuditRevision: waiting.Revision,
			RequestID: candidate.RequestID, SubjectRevision: candidate.SubjectRevision,
			SubjectDigest: candidate.SubjectDigest,
		}); err != nil {
			t.Fatal(err)
		}
		if failed := step(t, f, "rejected report"); failed.State != auditstore.AuditFailed {
			t.Fatalf("rejected report left Audit %s", failed.State)
		}
	})

	t.Run("expired report review", func(t *testing.T) {
		f, roundID := newFinalizingDenseFixture(t, "dense-expire")
		waiting := proposeDenseReport(t, f, roundID, step(t, f, "finalizing"))
		if _, err := f.pool.Exec(f.ctx, `
UPDATE audit_review_requests SET expires_at = clock_timestamp() - interval '1 second'
 WHERE audit_id = $1 AND subject_kind = 'audit-report'`, f.audit.AuditID); err != nil {
			t.Fatal(err)
		}
		if changed, err := f.store.ExpireReportReview(f.ctx, f.claim, waiting.Revision); err != nil || !changed {
			t.Fatalf("expire report review = (%t, %v)", changed, err)
		}
		step(t, f, "expired report review")
	})

	t.Run("owner pause closes report review", func(t *testing.T) {
		f, roundID := newFinalizingDenseFixture(t, "dense-pause")
		waiting := proposeDenseReport(t, f, roundID, step(t, f, "finalizing"))
		paused, changed, err := f.store.Transition(f.ctx, auditstore.TransitionParams{
			OwnerID: f.audit.OwnerID, AuditID: f.audit.AuditID, ExpectedRevision: waiting.Revision,
			ExpectedState: auditstore.AuditWaitingReview, TargetState: auditstore.AuditPaused,
			IdempotencyKey: "pause-dense", RequestDigest: testDigest("b"),
		})
		if err != nil || !changed || paused.Revision != waiting.Revision+2 || paused.EventSequence != waiting.EventSequence+2 {
			t.Fatalf("pause from waiting_review = (%+v, %t, %v), want one transition and one review expiry", paused, changed, err)
		}
		step(t, f, "owner pause")
		if _, changed, err := f.store.Resume(f.ctx, auditstore.ResumeParams{
			OwnerID: f.audit.OwnerID, AuditID: f.audit.AuditID, ExpectedRevision: paused.Revision,
			IdempotencyKey: "resume-dense", RequestDigest: testDigest("c"),
		}); err != nil || !changed {
			t.Fatalf("resume = (%t, %v)", changed, err)
		}
		step(t, f, "resume")
	})

	t.Run("owner deletion closes report review", func(t *testing.T) {
		f, roundID := newFinalizingDenseFixture(t, "dense-delete")
		waiting := proposeDenseReport(t, f, roundID, step(t, f, "finalizing"))
		deleting, changed, err := f.store.RequestDelete(f.ctx, auditstore.DeleteParams{
			OwnerID: f.audit.OwnerID, AuditID: f.audit.AuditID, ExpectedRevision: waiting.Revision,
			IdempotencyKey: "delete-dense", RequestDigest: testDigest("d"),
		})
		if err != nil || !changed || deleting.EventSequence != waiting.EventSequence+2 {
			t.Fatalf("delete from waiting_review = (%+v, %t, %v), want one request and one review expiry", deleting, changed, err)
		}
		step(t, f, "owner deletion")
	})

	t.Run("settled and committed report", func(t *testing.T) {
		f := newCollectingAuditFixture(t, "dense-commit", 1<<20,
			collectingAuditFixtureOptions{extraPending: true, terminalOutcome: "failed"})
		roundID := "round-dense-commit"
		collectFinalFailure(t, f, f.execution.ExecutionID, f.memberID, "receipt-dense-commit")
		finalizeDenseFixture(t, f, roundID)
		step(t, f, "finalizing")
		if settled, err := f.store.SettleUndispatched(f.ctx, f.claim, auditstore.MaxReconcileRows); err != nil || settled != 1 {
			t.Fatalf("settle undispatched = (%d, %v), want 1", settled, err)
		}
		finalizing := step(t, f, "settlement")
		round, err := f.store.GetRound(f.ctx, f.audit.AuditID, roundID)
		if err != nil {
			t.Fatal(err)
		}
		machine, summary := denseReportLinks()
		if _, err := f.store.CommitReport(f.ctx, auditstore.CommitReportParams{
			Claim: f.claim, ExpectedAuditRevision: finalizing.Revision,
			RoundID: roundID, ExpectedRoundRevision: round.Revision,
			Machine: machine, Summary: summary, RequestDigest: testDigest("e"),
		}); err != nil {
			t.Fatal(err)
		}
		step(t, f, "committed report")
	})

	t.Run("trusted activation and Project deletion", func(t *testing.T) {
		f := newCollectingAuditFixture(t, "dense-project", 1<<20)
		waiting, err := f.store.TransitionClaimed(f.ctx, auditstore.ClaimedTransitionParams{
			Claim: f.claim, ExpectedRevision: f.audit.Revision,
			ExpectedState: auditstore.AuditActive, TargetState: auditstore.AuditWaitingReview,
		})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := f.store.transitionTrusted(f.ctx, auditstore.TrustedTransitionParams{
			AuditID: f.audit.AuditID, ExpectedRevision: waiting.Revision,
			ExpectedState: auditstore.AuditWaitingReview, TargetState: auditstore.AuditActive,
		}); err != nil {
			t.Fatal(err)
		}
		step(t, f, "trusted activation")
		if _, _, err := projectstore.NewPostgresStore(f.pool).BeginDeletion(f.ctx, projectstore.BeginDeletionParams{
			ProjectID: f.project.ProjectID, OwnerID: f.project.OwnerID, ExpectedRevision: f.project.Revision,
		}); err != nil {
			t.Fatal(err)
		}
		var phase string
		if err := f.pool.QueryRow(f.ctx, `
UPDATE projects
   SET deletion_claim_id = 'claim-dense', deletion_claimed_at = clock_timestamp(),
       deletion_claim_expires_at = clock_timestamp() + interval '1 minute'
 WHERE project_id = $1
RETURNING deletion_phase`, f.project.ProjectID).Scan(&phase); err != nil {
			t.Fatal(err)
		}
		if changed, err := f.store.RequestProjectOwnedDeletion(f.ctx, auditstore.ProjectDeletionClaim{
			ProjectID: f.project.ProjectID, OwnerID: f.project.OwnerID, ClaimID: "claim-dense", Phase: phase,
		}); err != nil || !changed {
			t.Fatalf("Project-owned Audit deletion = (%t, %v)", changed, err)
		}
		step(t, f, "Project deletion")
	})

	want := []string{
		"audit.created", "audit.delete_requested", "audit.dispatch_hold_released",
		"audit.report_committed", "audit.resumed", "audit.state_changed",
		"execution.collected", "execution.intent_created", "execution.run_bound",
		"execution.terminal_observed", "finding.assessed", "finding.proposal_rejected",
		"finding.proposed", "items.dispatch_closed", "review.decided", "review.expired",
		"review.requested", "round.accepted", "round.state_changed",
	}
	var got []string
	for kind := range recorded {
		got = append(got, kind)
	}
	sort.Strings(got)
	if !t.Failed() && !slices.Equal(got, want) {
		t.Errorf("recorded event kinds = %q, want %q", got, want)
	}
}

// newFinalizingDenseFixture settles the fixture's only item with a final
// execution failure, closes its Round and enters finalizing.
func newFinalizingDenseFixture(t *testing.T, name string) (collectingAuditFixture, string) {
	t.Helper()
	f := newCollectingAuditFixture(t, name, 1<<20, collectingAuditFixtureOptions{terminalOutcome: "failed"})
	collectFinalFailure(t, f, f.execution.ExecutionID, f.memberID, "receipt-"+name)
	roundID := "round-" + name
	finalizeDenseFixture(t, f, roundID)
	return f, roundID
}

// collectFinalFailure settles one execution member with a final failure.
func collectFinalFailure(t *testing.T, f collectingAuditFixture, executionID, memberID, receiptID string) {
	t.Helper()
	if _, inserted, err := f.store.Collect(f.ctx, auditstore.CollectParams{
		Claim: f.claim, ReceiptID: receiptID, ExecutionID: executionID,
		Disposition: auditstore.CollectionExecutionFailed, ErrorCode: stringPointer("run_failed"),
		RequestDigest: testDigest("5"),
		Items: []auditstore.CollectionItem{{
			ExecutionItemID: memberID, Disposition: auditstore.CollectionExecutionFailed,
			FinalDisposition: auditstore.FinalExecutionFailed,
			Coverage: auditstore.Coverage{
				Status: auditstore.CoverageBlocked, Requested: []string{}, Completed: []string{},
				Gaps: []string{"execution-failed"}, Rationale: "The child Run failed.",
			},
		}},
	}); err != nil || !inserted {
		t.Fatalf("collect final failure = (%t, %v)", inserted, err)
	}
}

// finalizeDenseFixture closes the executing Round and moves the Audit from
// active to finalizing.
func finalizeDenseFixture(t *testing.T, f collectingAuditFixture, roundID string) {
	t.Helper()
	round, err := f.store.GetRound(f.ctx, f.audit.AuditID, roundID)
	if err != nil {
		t.Fatal(err)
	}
	for _, states := range [][2]auditstore.RoundState{{auditstore.RoundExecuting, auditstore.RoundAssessing}, {auditstore.RoundAssessing, auditstore.RoundClosed}} {
		round, err = f.store.TransitionRound(f.ctx, auditstore.RoundTransitionParams{
			Claim: f.claim, RoundID: roundID, ExpectedRevision: round.Revision,
			ExpectedState: states[0], TargetState: states[1],
		})
		if err != nil {
			t.Fatalf("transition Round to %q: %v", states[1], err)
		}
	}
	audit, err := f.store.Get(f.ctx, f.audit.OwnerID, f.audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.store.TransitionClaimed(f.ctx, auditstore.ClaimedTransitionParams{
		Claim: f.claim, ExpectedRevision: audit.Revision,
		ExpectedState: auditstore.AuditActive, TargetState: auditstore.AuditFinalizing,
	}); err != nil {
		t.Fatal(err)
	}
}

// proposeDenseReport proposes a report for the closed Round of a finalizing
// Audit and returns the Audit waiting for its review.
func proposeDenseReport(t *testing.T, f collectingAuditFixture, roundID string, finalizing auditstore.Audit) auditstore.Audit {
	t.Helper()
	round, err := f.store.GetRound(f.ctx, f.audit.AuditID, roundID)
	if err != nil {
		t.Fatal(err)
	}
	machine, summary := denseReportLinks()
	waiting, proposed, err := f.store.ProposeReport(f.ctx, auditstore.ProposeReportParams{
		Claim: f.claim, ExpectedAuditRevision: finalizing.Revision,
		RoundID: roundID, ExpectedRoundRevision: round.Revision,
		Machine: machine, Summary: summary, RequestDigest: testDigest("f"),
	})
	if err != nil || !proposed || waiting.State != auditstore.AuditWaitingReview {
		t.Fatalf("propose report = (%+v, %t, %v)", waiting, proposed, err)
	}
	return waiting
}

func denseReportLinks() (auditstore.ArtifactLink, auditstore.ArtifactLink) {
	revision := "report-r1"
	link := func(key, name, media string) auditstore.ArtifactLink {
		return auditstore.ArtifactLink{LogicalKey: key, Artifact: auditstore.ExactArtifact{
			Ref:    contracts.ArtifactRef{Namespace: "audits", Name: name, Revision: &revision},
			Digest: testDigest("1"), MediaType: media, SizeBytes: 16,
		}, SourceProvenance: json.RawMessage(`{}`)}
	}
	return link(auditstore.ReportMachineLogicalKey, "report.json", "application/json"),
		link(auditstore.ReportSummaryLogicalKey, "report.md", "text/markdown")
}
