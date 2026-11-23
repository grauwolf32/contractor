package scheduler

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/config/resolved"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
)

// Hold cancellation cleanup beyond the original lease. Another lane must not
// acquire the Run while the first lane is still responsible for its cleanup.
// The lease runs out by the database clock, and the lane renews its claim only
// when the test releases a renewal, so no local timer decides the outcome.
func TestPostgresCancellationCleanupRetainsRunClaim(t *testing.T) {
	// Prove the goroutine ordering at quiescence before involving network I/O.
	// A stopped renewal fails here immediately, rather than at a SQL test's
	// watchdog while waiting for a renewal that can never arrive.
	if !t.Run("renewal lifetime", func(t *testing.T) {
		synctest.Test(t, cancellationRenewalLifetime)
	}) {
		return
	}
	for _, notify := range []bool{true, false} {
		name := "polled cancellation"
		if notify {
			name = "immediate cancellation"
		}
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			pool := isolatedSchedulerPool(t, ctx)
			postgresStore := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
			workflow := loadSchedulerWorkflow(t)
			encoded, err := json.Marshal(workflow)
			if err != nil {
				t.Fatal(err)
			}
			const runID = "run-cancel-claim"
			if _, err := runstoretest.CreateRun(ctx, postgresStore, runstore.CreateRunParams{
				LaneClass: resolved.LaneClassModel,
				RunID:     runID, OwnerID: "owner", WorkflowName: workflow.Ref.Name,
				WorkflowVersion: workflow.Ref.Version, WorkflowSchemaVersion: contracts.APIVersion,
				WorkflowSnapshot: encoded, Parameters: map[string]string{"objective": "cancel"},
				RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
			}); err != nil {
				t.Fatal(err)
			}
			if _, err := postgresStore.TransitionRun(
				ctx, runID, runstore.RunInitializing, runstore.RunPending, runstore.Reason{Code: "initialized"},
			); err != nil {
				t.Fatal(err)
			}
			if _, err := postgresStore.TransitionRun(
				ctx, runID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"},
			); err != nil {
				t.Fatal(err)
			}
			store := &cancellationClaimStore{
				PostgresStore: postgresStore, started: make(chan string, 1),
				cleaning: make(chan string, 1), finish: make(chan struct{}), renewed: make(chan error, 8),
			}
			s := newLaneTestScheduler(t, newLaneTestStore(), newMutableSchedulerSettings(2, time.Second), 20*time.Millisecond)
			s.store = store
			// The Run has no Stage, so the cleanup ends with the commit that
			// finishes a cancelled Run.
			persistence, err := NewPostgresPersistence(pool, artifactpgtest.NewStorage())
			if err != nil {
				t.Fatal(err)
			}
			s.persistence = persistence
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
				select {
				case <-done:
				case <-ctx.Done():
					t.Error("cancellation lane did not stop before the test deadline")
				}
			}()
			awaitValue(t, ctx, store.started, "initial Run work")
			claimID := storedRunClaim(t, ctx, pool, runID)
			if _, err := postgresStore.RequestRunCancellation(ctx, runID, runstore.WorkflowRunCancellation{
				Code: runstore.CancellationUserRequested, RequestedAt: time.Now(),
			}); err != nil {
				t.Fatal(err)
			}
			if notify {
				s.Cancel(runID)
			} else {
				// The renewal reports the cancelling Run to the lane.
				clock.renew(t, ctx)
				if err := awaitValue(t, ctx, store.renewed, "renewal that reports the cancellation"); err != nil {
					t.Fatalf("renewal before cleanup = %v", err)
				}
			}
			awaitValue(t, ctx, store.cleaning, "cancellation cleanup")
			// The original lease runs out by the database clock while the
			// cleanup holds the Run; only the lane's renewal keeps the claim.
			expireRunClaim(t, ctx, pool, runID, claimID)
			clock.renew(t, ctx)
			if err := awaitValue(t, ctx, store.renewed, "renewal during cleanup"); err != nil {
				t.Fatalf("cleanup lost the claim: renewal = %v", err)
			}
			requireUnexpiredRunClaim(t, ctx, pool, runID, claimID)
			if run, err := postgresStore.ClaimRunnableRun(ctx, "competing-lane", time.Minute, nil, resolved.LaneClassModel); !errors.Is(err, runstore.ErrNoWork) {
				t.Fatalf("cleanup lost exclusive ownership: competing claim = %s, %v", run.RunID, err)
			}
			close(store.finish)
			if err := awaitValue(t, ctx, done, "completed cancellation"); err != nil {
				t.Fatal(err)
			}
			run, err := postgresStore.GetRun(ctx, runID)
			if err != nil || run.State != runstore.RunCancelled || run.SchedulerClaim != nil {
				t.Fatalf("cancelled Run = %+v, %v", run, err)
			}
		})
	}
}

type cancellationClaimStore struct {
	*runstore.PostgresStore
	first    atomic.Bool
	started  chan string
	cleaning chan string
	finish   chan struct{}
	renewed  chan error
}

func (s *cancellationClaimStore) HasStageExecutions(ctx context.Context, runID string) (bool, error) {
	// Renewal reads the Run too, but never probes its StageExecutions. Block
	// only the initial execution path, independent of goroutine scheduling.
	if s.first.CompareAndSwap(false, true) {
		s.started <- runID
		<-ctx.Done()
		return false, ctx.Err()
	}
	return s.PostgresStore.HasStageExecutions(ctx, runID)
}

// RenewRunClaim reports the outcome of each renewal, which the renewal clock
// lets happen only when the test releases it.
func (s *cancellationClaimStore) RenewRunClaim(
	ctx context.Context, runID, claimID string, duration time.Duration,
) (runstore.WorkflowRunState, error) {
	state, err := s.PostgresStore.RenewRunClaim(ctx, runID, claimID, duration)
	select {
	case s.renewed <- err:
	default:
	}
	return state, err
}

// ListStageExecutionsWithUnreleasedAllocations is the first read of the
// cleanup of a cancelling Run without an active Stage (SCH-R179).
func (s *cancellationClaimStore) ListStageExecutionsWithUnreleasedAllocations(
	ctx context.Context, runID string,
) ([]runstore.StageExecution, error) {
	s.cleaning <- runID
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-s.finish:
		return s.PostgresStore.ListStageExecutionsWithUnreleasedAllocations(ctx, runID)
	}
}

func TestPostgresFailActiveRunEndsCancellingRunAsCancelled(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
	defer cancel()
	pool := isolatedSchedulerPool(t, ctx)
	store := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	createSchedulerRun(t, ctx, store, artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)), loadSchedulerWorkflow(t))
	if _, err := store.RequestRunCancellation(ctx, "run-1", runstore.WorkflowRunCancellation{
		Code: runstore.CancellationUserRequested, RequestedAt: time.Now(),
	}); err != nil {
		t.Fatal(err)
	}
	persistence, err := NewPostgresPersistence(pool, artifactpgtest.NewStorage())
	if err != nil {
		t.Fatal(err)
	}
	s := &Scheduler{store: store, persistence: persistence, options: Options{
		OperationTimeout: 5 * time.Second, Clock: staticClock{now: time.Now().UTC()},
		NewID:  func(prefix string) (string, error) { return prefix + "1", nil },
		Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
	}}
	if err := s.failInvalidRunState(fixtureClaim(t, ctx, pool, "run-1"), "run-1", nil); err != nil {
		t.Fatalf("fail cancelling Run: %v", err)
	}
	run, err := store.GetRun(ctx, "run-1")
	if err != nil || run.State != runstore.RunCancelled {
		t.Fatalf("Run = (%+v, %v), want cancelled", run, err)
	}
}

// cancellationRenewalLifetime drives the production claim/cancellation path
// with the existing store double. synctest.Wait is the synchronization
// boundary; neither missing renewal nor premature shutdown consumes a timer.
func cancellationRenewalLifetime(t *testing.T) {
	h := newSchedulerHarness(t)
	store := &cancellationLifetimeStore{
		memorySchedulerStore: h.store, started: make(chan struct{}),
		cleaning: make(chan struct{}), finish: make(chan struct{}),
	}
	h.scheduler.store = store
	clock := &manualRenewalClock{now: h.clock.now, waits: make(chan chan time.Time, 4)}
	h.scheduler.options.Clock = clock
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	done := make(chan error, 1)
	go func() {
		_, err := h.scheduler.RunOnce(ctx)
		done <- err
	}()
	<-store.started
	synctest.Wait()
	h.requestCancellation("cancel while executing")
	h.scheduler.Cancel(h.store.run.RunID)
	synctest.Wait()
	select {
	case <-store.cleaning:
	default:
		t.Fatal("execution cancellation did not enter cleanup")
	}
	select {
	case tick := <-clock.waits:
		tick <- clock.now
	default:
		t.Fatal("claim renewal never armed")
	}
	synctest.Wait()
	if store.renewals != 1 {
		t.Fatalf("cleanup stopped claim renewal: calls=%d", store.renewals)
	}
	select {
	case <-clock.waits:
	default:
		t.Fatal("cleanup renewal did not rearm")
	}
	close(store.finish)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if h.store.run.State != runstore.RunCancelled {
		t.Fatalf("cleanup state=%s", h.store.run.State)
	}
}

type cancellationLifetimeStore struct {
	*memorySchedulerStore
	first                     bool
	started, cleaning, finish chan struct{}
	renewals                  int
}

func (s *cancellationLifetimeStore) HasStageExecutions(ctx context.Context, _ string) (bool, error) {
	if !s.first {
		s.first = true
		close(s.started)
		<-ctx.Done()
		return false, ctx.Err()
	}
	return false, nil
}

func (s *cancellationLifetimeStore) ListStageExecutionsWithUnreleasedAllocations(ctx context.Context, _ string) ([]runstore.StageExecution, error) {
	close(s.cleaning)
	select {
	case <-s.finish:
		return nil, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (s *cancellationLifetimeStore) RenewRunClaim(ctx context.Context, runID, claimID string, duration time.Duration) (runstore.WorkflowRunState, error) {
	s.renewals++
	return s.memorySchedulerStore.RenewRunClaim(ctx, runID, claimID, duration)
}
