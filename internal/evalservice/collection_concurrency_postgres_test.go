package evalservice

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

func collectingEvalAudit(t *testing.T) (*serviceHarness, evalstore.Experiment, evalstore.Claim, string, auditstore.CollectParams) {
	t.Helper()
	h := newHarness(t)
	e := h.prepared(t, "audit")
	h.command(t, e, "start")
	coordinator := h.coordinator(t, "projection-race-setup")
	var auditID, memberID string
	for pass := range 20 {
		tick(t, coordinator)
		err := h.pool.QueryRow(t.Context(), `SELECT a.audit_id, s.member_id FROM eval_submissions s
JOIN audits a ON s.execution_kind = 'audit' AND s.execution_id = a.audit_id
WHERE a.state = 'active' LIMIT 1`).Scan(&auditID, &memberID)
		if err == nil {
			break
		}
		if !errors.Is(err, pgx.ErrNoRows) || pass == 19 {
			t.Fatalf("Audit fixture did not start: %v", err)
		}
	}
	e = h.get(t, e.ID)
	claims, err := evalstore.NewPostgresStore(h.pool).Claim(t.Context(), "projection-race-collector", 5*time.Minute, 1)
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim Eval: %v %v", claims, err)
	}
	audits := auditstoretest.NewStore(h.pool)
	auditClaims, err := audits.Claim(t.Context(), auditstore.ClaimParams{HolderID: "projection-race-audit", Lease: 5 * time.Minute, Limit: 1})
	if err != nil || len(auditClaims) != 1 || auditClaims[0].AuditID != auditID {
		t.Fatalf("claim Audit: %v %v", auditClaims, err)
	}
	claim := auditClaims[0]
	snapshot, err := audits.GetReconcileSnapshot(t.Context(), claim)
	if err != nil || snapshot.Round == nil || len(snapshot.Items) != 1 {
		t.Fatalf("Audit snapshot: %v %v", snapshot, err)
	}
	if _, err := audits.TransitionRound(t.Context(), auditstore.RoundTransitionParams{
		Claim: claim, RoundID: snapshot.Round.RoundID, ExpectedRevision: snapshot.Round.Revision,
		ExpectedState: auditstore.RoundAccepted, TargetState: auditstore.RoundExecuting,
	}); err != nil {
		t.Fatal(err)
	}
	item := snapshot.Items[0]
	execution, _, err := audits.CreateExecutionIntent(t.Context(), auditstore.CreateExecutionIntentParams{
		Claim: claim, ExecutionID: "projection-race-execution", RoundID: &snapshot.Round.RoundID,
		Role: auditstore.ExecutionCheck, WorkflowRole: item.WorkflowRole, Manifest: snapshot.Round.Manifest,
		SubmissionKey: "projection-race-submission", RequestDigest: evaldomain.Digest([]byte("execution")),
		Members: []auditstore.ExecutionMemberIntent{{ExecutionItemID: "projection-race-item", ItemID: item.ItemID,
			ItemAttempt: 1, Task: item.Task, Inputs: []auditstore.ExactArtifact{}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := audits.ObserveSubmissionFailure(t.Context(), auditstore.ObserveSubmissionFailureParams{
		Claim: claim, ExecutionID: execution.ExecutionID,
	}); err != nil {
		t.Fatal(err)
	}
	code := "fixture-submission-failed"
	return h, e, claims[0], memberID, auditstore.CollectParams{
		Claim: claim, ExecutionID: execution.ExecutionID, ReceiptID: "projection-race-receipt",
		RequestDigest: evaldomain.Digest([]byte("collection")), Disposition: auditstore.CollectionExecutionFailed,
		ErrorCode: &code, Retained: []auditstore.ArtifactLink{},
		Items: []auditstore.CollectionItem{{ExecutionItemID: "projection-race-item",
			Disposition: auditstore.CollectionExecutionFailed, FinalDisposition: auditstore.FinalExecutionFailed,
			Coverage: auditstore.Coverage{Status: auditstore.CoverageBlocked,
				Requested: []string{}, Completed: []string{}, Gaps: []string{"fixture-submission-failed"}}}},
	}
}

func TestSharedPostgresAuditCollectionAndEvalProjectionUseOneLockOrder(t *testing.T) {
	h, e, claim, member, collection := collectingEvalAudit(t)
	ctx, cancel := context.WithTimeout(t.Context(), 3*time.Minute)
	defer cancel()
	for pass := range 100 {
		if err := raceAuditCollectionAndProjection(ctx, h, e, claim, member, collection); err != nil {
			t.Fatalf("collection race %d: %v", pass, err)
		}
	}
}

func raceAuditCollectionAndProjection(ctx context.Context, h *serviceHarness, e evalstore.Experiment, claim evalstore.Claim, member string, collection auditstore.CollectParams) error {
	projection, err := h.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer func() { _ = projection.Rollback(context.WithoutCancel(ctx)) }()
	st := evalstore.NewTxStore(projection)
	if _, err := st.LockCollection(ctx, scope(e), e.ID, claim); err != nil {
		return err
	}
	row, err := st.ExecutionObservation(ctx, e.OwnerID, e.ID, member)
	if err != nil {
		return err
	}
	revision, err := st.ProjectionRevision(ctx, e.OwnerID, e.ID, member)
	if err != nil {
		return err
	}
	// Hold the same queue boundary as PublishView, then start the production
	// Audit Collect statement. Its trigger must wait before taking a member
	// lock, leaving ProjectMember free to finish and release the queue.
	if _, err := projection.Exec(ctx, `SELECT experiment_id FROM eval_projection_queue
WHERE experiment_id = $1 FOR UPDATE`, e.ID); err != nil {
		return err
	}
	started := make(chan uint32, 1)
	finished := make(chan error, 1)
	go func() {
		tx, err := h.pool.Begin(ctx)
		if err != nil {
			started <- 0
			finished <- err
			return
		}
		started <- tx.Conn().PgConn().PID()
		_, changed, err := auditstoretest.NewTxStore(tx).Collect(ctx, collection)
		if err == nil && !changed {
			err = errors.New("collection did not change its execution")
		}
		err = errors.Join(err, tx.Rollback(context.WithoutCancel(ctx)))
		finished <- err
	}()
	joined := false
	defer func() {
		_ = projection.Rollback(context.WithoutCancel(ctx))
		if !joined {
			<-finished
		}
	}()
	pid := <-started
	for {
		var blocked bool
		if err := h.pool.QueryRow(ctx, `SELECT $2::int = ANY(pg_blocking_pids($1::int))`, pid, projection.Conn().PgConn().PID()).Scan(&blocked); err != nil {
			return err
		}
		if blocked {
			break
		}
		select {
		case err := <-finished:
			joined = true
			return fmt.Errorf("Audit collection finished before reaching the queue: %w", err)
		default:
		}
	}
	view := executionMember(row)
	view.Assessment = "unscored"
	if err := st.ProjectMember(ctx, scope(e), e.ID, member, claim, revision, view, false, nil); err != nil {
		return err
	}
	if err := projection.Commit(ctx); err != nil {
		return err
	}
	err = <-finished
	joined = true
	return err
}

func TestSharedPostgresMemberCollectionRetriesConflictsWithoutDamageRotation(t *testing.T) {
	h, e, claim, member, _ := collectingEvalAudit(t)
	ctx := t.Context()
	for _, sql := range []string{
		`CREATE SEQUENCE collection_conflict_attempts`,
		`CREATE FUNCTION inject_collection_conflict() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF nextval('collection_conflict_attempts') <= 2 THEN
    RAISE EXCEPTION 'injected collection conflict' USING ERRCODE = 'serialization_failure';
  END IF;
  RETURN NEW;
END $$`,
		`CREATE TRIGGER collection_conflict BEFORE UPDATE OF projected_revision ON eval_member_projections
FOR EACH ROW EXECUTE FUNCTION inject_collection_conflict()`,
	} {
		if _, err := h.pool.Exec(ctx, sql); err != nil {
			t.Fatal(err)
		}
	}
	if err := h.service.collectMember(ctx, e, member, claim); err != nil {
		t.Fatalf("collection did not retry: %v", err)
	}
	var attempts int
	if err := h.pool.QueryRow(ctx, `SELECT last_value FROM collection_conflict_attempts`).Scan(&attempts); err != nil || attempts != 3 {
		t.Fatalf("collection attempts: %d %v", attempts, err)
	}
	if _, err := h.pool.Exec(ctx, `CREATE OR REPLACE FUNCTION inject_collection_conflict() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'injected collection conflict' USING ERRCODE = 'serialization_failure';
END $$`); err != nil {
		t.Fatal(err)
	}
	if _, err := h.pool.Exec(ctx, `SELECT contractor_eval_dirty($1,$2)`, e.ID, member); err != nil {
		t.Fatal(err)
	}
	if _, err := h.pool.Exec(ctx, `UPDATE eval_member_projections SET checked_at = '2000-01-01T00:00:00Z'
WHERE experiment_id = $1`, e.ID); err != nil {
		t.Fatal(err)
	}
	if err := h.service.CollectView(ctx, e, claim); !pg.IsTransactionConflict(err) {
		t.Fatalf("exhausted conflict lost its class: %v", err)
	}
	var rotated int
	if err := h.pool.QueryRow(ctx, `SELECT count(*) FROM eval_member_projections
WHERE experiment_id = $1 AND checked_at <> '2000-01-01T00:00:00Z'`, e.ID).Scan(&rotated); err != nil || rotated != 0 {
		t.Fatalf("transaction conflicts rotated %d members: %v", rotated, err)
	}
}
