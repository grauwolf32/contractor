package scheduler

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

// A renewal failure must still stop authoritative writes after cleanup has
// detached ordinary cancellation. Pause a real report write, move the SQL claim
// to another lane, then let the original lane reach its terminal commit.
func TestPostgresClaimLossDuringFinalizationPreventsReportAndProgression(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
	defer cancel()
	pool := isolatedSchedulerPool(t, ctx)
	fixture := createFinalizingFixture(t, ctx, pool)
	store := &claimLossReportStore{
		PostgresStore: fixture.store,
		started:       make(chan context.Context, 1),
		resume:        make(chan struct{}),
	}
	s := newLaneTestScheduler(t, newLaneTestStore(), newMutableSchedulerSettings(2, time.Second), 10*time.Millisecond)
	s.store = store
	s.persistence = fixture.persistence
	done := make(chan error, 1)
	go func() {
		defer close(done)
		_, err := s.progressOneClaim(ctx)
		done <- err
	}()
	defer func() {
		cancel()
		store.unblock()
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Error("claim-loss lane did not stop")
		}
	}()
	var executionContext context.Context
	select {
	case executionContext = <-store.started:
	case <-ctx.Done():
		t.Fatal("finalization report was not reached", ctx.Err())
	}
	stealRunClaim(t, ctx, pool, "run-1")
	select {
	case <-executionContext.Done():
	case <-ctx.Done():
		t.Fatal("lost claim did not interrupt execution", ctx.Err())
	}
	store.unblock()
	if err := receiveError(t, done, time.Second, "lost claim finalization"); !errors.Is(err, ErrClaimLost) {
		t.Fatalf("claim loss = %v", err)
	}
	run, err := fixture.store.GetRun(ctx, "run-1")
	if err != nil || run.State != runstore.RunRunning || run.SchedulerClaim == nil ||
		run.SchedulerClaim.ClaimID != "competing-lane" {
		t.Fatalf("old lane changed the Run or its replacement claim: state=%s claim=%+v err=%v", run.State, run.SchedulerClaim, err)
	}
	execution, err := fixture.store.GetStageExecution(ctx, fixture.executionID)
	if err != nil || execution.State != runstore.StageFinalizing {
		t.Fatalf("old lane committed Stage progression: state=%s err=%v", execution.State, err)
	}
	store.mu.Lock()
	reportCommitted := store.reportCommitted
	store.mu.Unlock()
	if reportCommitted {
		t.Fatal("old lane committed a report after losing its claim")
	}
}

func stealRunClaim(t *testing.T, ctx context.Context, pool *pgxpool.Pool, runID string) {
	t.Helper()
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback(ctx)
	// Keep expiry and replacement under one row lock so the renewing lane
	// cannot accidentally renew the expired claim between these operations.
	if _, err := tx.Exec(ctx, `UPDATE workflow_runs
SET scheduler_claimed_at = clock_timestamp() - interval '2 minutes',
    scheduler_claim_expires_at = clock_timestamp() - interval '1 minute'
WHERE run_id = $1`, runID); err != nil {
		t.Fatal(err)
	}
	claimed, err := runstore.NewPostgresStore(tx).ClaimRunnableRun(ctx, "competing-lane", time.Minute)
	if err != nil || claimed.RunID != runID {
		t.Fatalf("replacement claim = %+v, %v", claimed, err)
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
}

type claimLossReportStore struct {
	*runstore.PostgresStore
	mu               sync.Mutex
	executionContext context.Context
	reportCommitted  bool
	started          chan context.Context
	resume           chan struct{}
	once             sync.Once
}

func (s *claimLossReportStore) unblock() { s.once.Do(func() { close(s.resume) }) }

func (s *claimLossReportStore) ListStageExecutions(ctx context.Context, runID string) ([]runstore.StageExecution, error) {
	s.mu.Lock()
	s.executionContext = ctx
	s.mu.Unlock()
	return s.PostgresStore.ListStageExecutions(ctx, runID)
}

func (s *claimLossReportStore) RecordPlannerExecutionReport(ctx context.Context, params runstore.RecordPlannerExecutionReportParams) error {
	s.mu.Lock()
	executionContext := s.executionContext
	s.mu.Unlock()
	s.started <- executionContext
	<-s.resume
	err := s.PostgresStore.RecordPlannerExecutionReport(ctx, params)
	s.mu.Lock()
	s.reportCommitted = err == nil
	s.mu.Unlock()
	return err
}
