package scheduler

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config/resolved"

	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"github.com/jackc/pgx/v5/pgxpool"
)

// A renewal failure must still stop authoritative writes after cleanup has
// detached ordinary cancellation. Pause a real report write, move the SQL claim
// to another lane, then let the original lane reach its terminal commit.
func TestPostgresClaimLossDuringFinalizationPreventsReportAndProgression(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedSchedulerPool(t, ctx)
	fixture := createFinalizingFixture(t, ctx, pool)
	store := &claimLossReportStore{
		PostgresStore: fixture.store,
		reports:       telemetry.NewRepository(pool),
		started:       make(chan context.Context, 1),
		resume:        make(chan struct{}),
	}
	s := newLaneTestScheduler(t, newLaneTestStore(), newMutableSchedulerSettings(2, time.Second), 10*time.Millisecond)
	s.store = store
	s.options.Reports = store
	s.persistence = fixture.persistence
	// The test's deadline bounds each store operation, and the lane renews
	// its claim only when the test releases the renewal that finds the claim
	// stolen.
	s.options.OperationTimeout = time.Minute
	clock := newRenewalClock(s.options.ClaimDuration)
	s.options.Clock = clock
	laneContext, stopLane := context.WithCancel(ctx)
	done := make(chan error, 1)
	go func() {
		defer close(done)
		_, err := s.progressOneClaim(laneContext, resolved.LaneClassModel, nil)
		done <- err
	}()
	defer func() {
		stopLane()
		store.unblock()
		select {
		case <-done:
		case <-ctx.Done():
			t.Error("claim-loss lane did not stop before the test deadline")
		}
	}()
	executionContext := awaitValue(t, ctx, store.started, "finalization report")
	stealRunClaim(t, ctx, pool, "run-1")
	clock.renew(t, ctx)
	select {
	case <-executionContext.Done():
	case <-ctx.Done():
		t.Fatal("lost claim did not interrupt execution", ctx.Err())
	}
	store.unblock()
	if err := awaitValue(t, ctx, done, "lost claim finalization"); !errors.Is(err, ErrClaimLost) {
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

// renewalClock is the wall clock except for a lane's claim renewal wait:
// After of the renewal interval (SCH-R166) returns a tick that renew sends, so
// the lane renews its claim exactly when the test says.
type renewalClock struct {
	interval time.Duration
	armed    chan chan time.Time
}

func newRenewalClock(claimDuration time.Duration) *renewalClock {
	// A lane arms one renewal wait at a time. The buffer keeps After, which
	// runs under the Scheduler's clock lock, from blocking.
	return &renewalClock{interval: claimDuration / 3, armed: make(chan chan time.Time, 8)}
}

func (c *renewalClock) Now() time.Time { return time.Now() }

func (c *renewalClock) After(duration time.Duration) <-chan time.Time {
	if duration != c.interval {
		return time.After(duration)
	}
	tick := make(chan time.Time, 1)
	c.armed <- tick
	return tick
}

// renew ends the lane's renewal wait once the lane has armed it.
func (c *renewalClock) renew(t *testing.T, ctx context.Context) {
	t.Helper()
	awaitValue(t, ctx, c.armed, "armed claim renewal") <- time.Now()
}

// storedRunClaim returns the ID of the claim that holds runID.
func storedRunClaim(t *testing.T, ctx context.Context, pool *pgxpool.Pool, runID string) string {
	t.Helper()
	var claimID *string
	if err := pool.QueryRow(ctx, `SELECT scheduler_claim_id FROM workflow_runs WHERE run_id = $1`, runID).
		Scan(&claimID); err != nil {
		t.Fatal(err)
	}
	if claimID == nil {
		t.Fatalf("Run %s holds no claim", runID)
	}
	return *claimID
}

// expireRunClaim ends the lease of claimID on runID a minute ago by the
// database clock, as if the lane had held the Run past its lease, and keeps
// the claim's owner.
func expireRunClaim(t *testing.T, ctx context.Context, pool *pgxpool.Pool, runID, claimID string) {
	t.Helper()
	tag, err := pool.Exec(ctx, `UPDATE workflow_runs
SET scheduler_claimed_at = clock_timestamp() - interval '2 minutes',
    scheduler_claim_expires_at = clock_timestamp() - interval '1 minute'
WHERE run_id = $1 AND scheduler_claim_id = $2`, runID, claimID)
	if err != nil {
		t.Fatal(err)
	}
	if tag.RowsAffected() != 1 {
		t.Fatalf("claim %s no longer holds Run %s", claimID, runID)
	}
}

// requireUnexpiredRunClaim fails unless claimID holds runID with an expiry
// ahead of the database clock.
func requireUnexpiredRunClaim(t *testing.T, ctx context.Context, pool *pgxpool.Pool, runID, claimID string) {
	t.Helper()
	var held bool
	if err := pool.QueryRow(ctx, `SELECT COALESCE(
    scheduler_claim_id = $2 AND scheduler_claim_expires_at > clock_timestamp(), false)
FROM workflow_runs WHERE run_id = $1`, runID, claimID).Scan(&held); err != nil {
		t.Fatal(err)
	}
	if !held {
		t.Fatalf("claim %s no longer holds Run %s ahead of the database clock", claimID, runID)
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
	claimed, err := runstore.NewPostgresStore(tx, artifactpgtest.NewStorage()).ClaimRunnableRun(ctx, "competing-lane", time.Minute, nil, resolved.LaneClassModel)
	if err != nil || claimed.RunID != runID {
		t.Fatalf("replacement claim = %+v, %v", claimed, err)
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
}

type claimLossReportStore struct {
	*runstore.PostgresStore
	reports          *telemetry.Repository
	mu               sync.Mutex
	executionContext context.Context
	reportCommitted  bool
	started          chan context.Context
	resume           chan struct{}
	once             sync.Once
}

func (s *claimLossReportStore) unblock() { s.once.Do(func() { close(s.resume) }) }

// GetActiveStageExecution is the progression attempt's Stage read, so it
// sees the execution context the lost claim must cancel.
func (s *claimLossReportStore) GetActiveStageExecution(ctx context.Context, runID string) (runstore.StageExecution, error) {
	s.mu.Lock()
	s.executionContext = ctx
	s.mu.Unlock()
	return s.PostgresStore.GetActiveStageExecution(ctx, runID)
}

func (s *claimLossReportStore) RecordAllocationReport(ctx context.Context, envelope telemetry.AllocationReportEnvelope) error {
	return s.reports.RecordAllocationReport(ctx, envelope)
}

func (s *claimLossReportStore) RecordPlannerReport(ctx context.Context, envelope telemetry.PlannerReportEnvelope) error {
	s.mu.Lock()
	executionContext := s.executionContext
	s.mu.Unlock()
	s.started <- executionContext
	<-s.resume
	err := s.reports.RecordPlannerReport(ctx, envelope)
	s.mu.Lock()
	s.reportCommitted = err == nil
	s.mu.Unlock()
	return err
}

func (s *claimLossReportStore) RebuildStageMetrics(
	ctx context.Context, stageExecutionID, schemaVersion string,
) (telemetry.StageMetricsRecord, error) {
	return s.reports.RebuildStageMetrics(ctx, stageExecutionID, schemaVersion)
}
