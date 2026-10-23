package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
)

// injectUniqueViolationSQL makes every statement that writes audits fail with
// unique_violation in a session that set contractor_test.inject, so a test can
// drive a store method into the recovery path behind its failed statement.
const injectUniqueViolationSQL = `
CREATE FUNCTION contractor_test_inject_unique_violation() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF current_setting('contractor_test.inject', true) = 'unique_violation' THEN
        RAISE EXCEPTION 'injected unique violation' USING ERRCODE = 'unique_violation';
    END IF;
    RETURN NULL;
END $$;
CREATE TRIGGER contractor_test_inject_unique_violation BEFORE INSERT OR UPDATE ON audits
FOR EACH STATEMENT EXECUTE FUNCTION contractor_test_inject_unique_violation()`

// TestPostgresRecoveryPathsRunInsideACallerTransaction drives every store
// method that recovers from a failed statement with further statements into
// that recovery on a store backed by the caller's transaction. A failed
// statement aborts a PostgreSQL transaction, so without a savepoint the
// recovery would fail with 25P02 instead of returning the method's own
// outcome, and the caller could not continue its transaction (V368-029).
func TestPostgresRecoveryPathsRunInsideACallerTransaction(t *testing.T) {
	f := newCollectingAuditFixture(t, "txrecovery", 1<<20, collectingAuditFixtureOptions{extraPending: true})
	if _, err := f.pool.Exec(f.ctx, injectUniqueViolationSQL); err != nil {
		t.Fatal(err)
	}
	if f.audit.CurrentRoundID == nil {
		t.Fatal("fixture Audit has no current Round")
	}
	roundID := *f.audit.CurrentRoundID
	round, err := f.store.GetRound(f.ctx, f.audit.AuditID, roundID)
	if err != nil {
		t.Fatal(err)
	}
	item := func(name string) auditstore.MaterializedItem {
		return auditstore.MaterializedItem{
			ItemID: "item-" + name, ItemKey: "check-" + name, Ordinal: 0, Kind: "checklist",
			SubjectKey: "subject-" + name, Task: testExact("audits", name+"-task", "task-r1"),
			Origin: testOrigin("check-" + name), WorkflowRole: "check", InitialState: auditstore.ItemReady,
			Coverage: emptyCoverage(),
		}
	}
	result := testExact("audit-txrecovery", "result", "result-r1")
	machine, summary := denseReportLinks()
	next := item("txrecovery-next")
	next.ProposalSources = []auditstore.ProposalItemSource{{
		ReceiptID: "receipt-txrecovery-proposal", ProposedCheckOrdinal: 0,
		Proposal: testExact("audits", "txrecovery-proposal", "proposal-r1"),
	}}
	deadline := time.Now().Add(time.Hour).UTC()
	for _, path := range []struct {
		name string
		want error
		call func(context.Context, *testStore) error
	}{
		{"CreateDraft", auditstore.ErrConflict, func(ctx context.Context, s *testStore) error {
			_, _, err := s.CreateDraft(ctx, auditstore.CreateDraftParams{
				AuditID: "audit-txrecovery-draft", OwnerID: f.audit.OwnerID, ProjectID: f.project.ProjectID,
				Profile: f.audit.Profile, ProfileSnapshot: f.audit.ProfileSnapshot,
				InputSelection: json.RawMessage(`{}`), Limits: f.audit.Limits,
				IdempotencyKey: "txrecovery-create", RequestDigest: testDigest("1"),
			})
			return err
		}},
		{"MaterializeRound", auditstore.ErrConflict, func(ctx context.Context, s *testStore) error {
			_, _, err := s.MaterializeRound(ctx, auditstore.MaterializeRoundParams{
				OwnerID: f.audit.OwnerID, AuditID: f.audit.AuditID, ExpectedRevision: f.audit.Revision,
				RoundID: "round-txrecovery-start", RoundOrdinal: 1,
				Manifest:         testExact("audits", "txrecovery-start-worklist", "worklist-r1"),
				BaselineSnapshot: json.RawMessage(`{"inputs":[],"skills":[]}`),
				DeadlineAt:       deadline, Items: []auditstore.MaterializedItem{item("txrecovery-start")},
				IdempotencyKey: "txrecovery-start", RequestDigest: testDigest("2"),
			})
			return err
		}},
		{"Transition", auditstore.ErrConflict, func(ctx context.Context, s *testStore) error {
			_, _, err := s.Transition(ctx, auditstore.TransitionParams{
				OwnerID: f.audit.OwnerID, AuditID: f.audit.AuditID, ExpectedRevision: f.audit.Revision,
				ExpectedState: auditstore.AuditActive, TargetState: auditstore.AuditPaused,
				IdempotencyKey: "txrecovery-pause", RequestDigest: testDigest("3"),
			})
			return err
		}},
		{"Resume", auditstore.ErrConflict, func(ctx context.Context, s *testStore) error {
			_, _, err := s.Resume(ctx, auditstore.ResumeParams{
				OwnerID: f.audit.OwnerID, AuditID: f.audit.AuditID, ExpectedRevision: f.audit.Revision,
				DeadlineAt: &deadline, IdempotencyKey: "txrecovery-resume", RequestDigest: testDigest("4"),
			})
			return err
		}},
		{"RequestDelete", auditstore.ErrConflict, func(ctx context.Context, s *testStore) error {
			_, _, err := s.RequestDelete(ctx, auditstore.DeleteParams{
				OwnerID: f.audit.OwnerID, AuditID: f.audit.AuditID, ExpectedRevision: f.audit.Revision,
				IdempotencyKey: "txrecovery-delete", RequestDigest: testDigest("5"),
			})
			return err
		}},
		{"CreateExecutionIntent", auditstore.ErrConflict, func(ctx context.Context, s *testStore) error {
			_, _, err := s.CreateExecutionIntent(ctx, auditstore.CreateExecutionIntentParams{
				Claim: f.claim, ExecutionID: "execution-txrecovery-pending", RoundID: &roundID,
				Role: auditstore.ExecutionCheck, WorkflowRole: "check", LaneClass: resolved.LaneClassModel,
				Manifest:      testExact("audits", "txrecovery-pending-execution", "manifest-r1"),
				SubmissionKey: "submission-txrecovery-pending", RequestDigest: testDigest("6"),
				Members: []auditstore.ExecutionMemberIntent{{
					ExecutionItemID: "member-txrecovery-pending", ItemID: "pending-item-txrecovery",
					BatchOrdinal: 0, ItemAttempt: 1, Inputs: []auditstore.ExactArtifact{},
					Task: testExact("audits", "txrecovery-pending-task", "task-r1"),
				}},
			})
			return err
		}},
		{"Collect", auditstore.ErrConflict, func(ctx context.Context, s *testStore) error {
			_, _, err := s.Collect(ctx, auditstore.CollectParams{
				Claim: f.claim, ReceiptID: "receipt-txrecovery", ExecutionID: f.execution.ExecutionID,
				Disposition: auditstore.CollectionAccepted, SourceOutput: &result, RequestDigest: testDigest("7"),
				Retained: []auditstore.ArtifactLink{{
					LogicalKey: "result/" + f.memberID, Artifact: result,
					SourceProvenance: json.RawMessage(`{"runId":"run-txrecovery"}`),
				}},
				Items: []auditstore.CollectionItem{{
					ExecutionItemID: f.memberID, Disposition: auditstore.CollectionAccepted,
					FinalDisposition: auditstore.FinalAccepted, Result: &result,
					Coverage: auditstore.Coverage{Status: auditstore.CoverageSatisfied, Requested: []string{}, Completed: []string{}, Gaps: []string{}},
				}},
			})
			return err
		}},
		{"AcceptInitialRound", auditstore.ErrPrecondition, func(ctx context.Context, s *testStore) error {
			_, _, err := s.AcceptInitialRound(ctx, auditstore.AcceptInitialRoundParams{
				Claim: f.claim, ExpectedAuditRevision: f.audit.Revision, RoundID: "round-txrecovery-initial",
				Manifest: testExact("audits", "txrecovery-initial-worklist", "worklist-r1"),
				Items:    []auditstore.MaterializedItem{item("txrecovery-initial")},
				Inventory: auditstore.ArtifactLink{
					LogicalKey: auditstore.InitialInventoryLogicalKey, SourceProvenance: json.RawMessage(`{}`),
					Artifact: testExact("audits", "txrecovery-inventory", "inventory-r1"),
				},
			})
			return err
		}},
		{"AcceptNextRound", auditstore.ErrConflict, func(ctx context.Context, s *testStore) error {
			_, _, err := s.AcceptNextRound(ctx, auditstore.AcceptRoundParams{
				Claim: f.claim, ExpectedAuditRevision: f.audit.Revision, PreviousRoundID: roundID,
				RoundID: "round-txrecovery-next", RoundOrdinal: 2,
				Manifest: testExact("audits", "txrecovery-next-worklist", "worklist-r1"),
				Items:    []auditstore.MaterializedItem{next},
			})
			return err
		}},
		{"ProposeReport", auditstore.ErrPrecondition, func(ctx context.Context, s *testStore) error {
			_, _, err := s.ProposeReport(ctx, auditstore.ProposeReportParams{
				Claim: f.claim, ExpectedAuditRevision: f.audit.Revision,
				RoundID: roundID, ExpectedRoundRevision: round.Revision,
				Machine: machine, Summary: summary, RequestDigest: testDigest("8"),
			})
			return err
		}},
	} {
		t.Run(path.name, func(t *testing.T) {
			tx, err := f.pool.Begin(f.ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = tx.Rollback(context.WithoutCancel(f.ctx)) }()
			if _, err := tx.Exec(f.ctx, `SET LOCAL contractor_test.inject = 'unique_violation'`); err != nil {
				t.Fatal(err)
			}
			if err := path.call(f.ctx, newTestStore(tx)); !errors.Is(err, path.want) {
				t.Fatalf("%s after a failed statement in a transaction = %v, want %v", path.name, err, path.want)
			}
			var usable int
			if err := tx.QueryRow(f.ctx, `SELECT 1`).Scan(&usable); err != nil || usable != 1 {
				t.Fatalf("%s left the caller's transaction unusable: %v", path.name, err)
			}
		})
	}
}

// TestPostgresCreateDraftReplaysAConcurrentWinnerInsideATransaction races two
// transactions on the same create request. Each one generated its own Audit
// ID, so the loser's idempotency record waits for the winner's, fails with a
// unique violation when the winner commits, and the loser must still read and
// replay the winner's Audit in its own transaction, without its own Audit row.
func TestPostgresCreateDraftReplaysAConcurrentWinnerInsideATransaction(t *testing.T) {
	f := newCollectingAuditFixture(t, "txreplay", 1<<20)
	params := auditstore.CreateDraftParams{
		AuditID: "audit-txreplay-draft", OwnerID: f.audit.OwnerID, ProjectID: f.project.ProjectID,
		Profile: f.audit.Profile, ProfileSnapshot: f.audit.ProfileSnapshot,
		InputSelection: json.RawMessage(`{}`), Limits: f.audit.Limits,
		IdempotencyKey: "txreplay-create", RequestDigest: testDigest("9"),
	}
	winner, err := f.pool.Begin(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = winner.Rollback(context.WithoutCancel(f.ctx)) }()
	won, inserted, err := newTestStore(winner).CreateDraft(f.ctx, params)
	if err != nil || !inserted {
		t.Fatalf("winning create = (%t, %v)", inserted, err)
	}
	loser, err := f.pool.Begin(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = loser.Rollback(context.WithoutCancel(f.ctx)) }()
	var loserPID int
	if err := loser.QueryRow(f.ctx, `SELECT pg_backend_pid()`).Scan(&loserPID); err != nil {
		t.Fatal(err)
	}
	losing := params
	losing.AuditID = "audit-txreplay-draft-loser"
	var replayed auditstore.Audit
	var loserInserted bool
	done := make(chan error, 1)
	go func() {
		audit, inserted, err := newTestStore(loser).CreateDraft(f.ctx, losing)
		replayed, loserInserted = audit, inserted
		done <- err
	}()
	waitForPostgresLock(t, f.ctx, f.pool, loserPID, done)
	if err := winner.Commit(f.ctx); err != nil {
		t.Fatal(err)
	}
	if err := <-done; err != nil || loserInserted || replayed.AuditID != won.AuditID || replayed.Revision != won.Revision {
		t.Fatalf("losing create in a transaction = (%+v, %t, %v), want the winner's replay", replayed, loserInserted, err)
	}
	if err := loser.Commit(f.ctx); err != nil {
		t.Fatalf("commit after the replay: %v", err)
	}
	if _, err := f.store.Get(f.ctx, f.audit.OwnerID, losing.AuditID); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("losing Audit after the replay = %v, want it rolled back", err)
	}
}
