package scheduler

import (
	"context"
	"encoding/json"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

// Hold cancellation cleanup beyond the original lease. Another lane must not
// acquire the Run while the first lane is still responsible for its cleanup.
func TestPostgresCancellationCleanupRetainsRunClaim(t *testing.T) {
	for _, notify := range []bool{true, false} {
		name := "polled cancellation"
		if notify {
			name = "immediate cancellation"
		}
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
			defer cancel()
			pool := isolatedSchedulerPool(t, ctx)
			postgresStore := runstore.NewPostgresStore(pool)
			workflow := loadSchedulerWorkflow(t)
			encoded, err := json.Marshal(workflow)
			if err != nil {
				t.Fatal(err)
			}
			const runID = "run-cancel-claim"
			if _, err := postgresStore.CreateRun(ctx, runstore.CreateRunParams{
				RunID: runID, OwnerID: "owner", WorkflowName: workflow.Ref.Name,
				WorkflowVersion: workflow.Ref.Version, WorkflowSchemaVersion: contracts.APIVersion,
				WorkflowSnapshot: encoded, Parameters: map[string]string{"objective": "cancel"},
				RuntimeConfig: runtimeconfig.BuiltInRunSnapshot(),
			}); err != nil {
				t.Fatal(err)
			}
			if _, err := postgresStore.TransitionRun(ctx, runID, runstore.RunInitializing,
				runstore.RunRunning, runstore.Reason{Code: "initialized"}); err != nil {
				t.Fatal(err)
			}
			store := &cancellationClaimStore{
				PostgresStore: postgresStore, started: make(chan string, 1),
				cleaning: make(chan string, 1), finish: make(chan struct{}),
			}
			s := newLaneTestScheduler(t, newLaneTestStore(), newMutableSchedulerSettings(2, time.Second), 20*time.Millisecond)
			s.store = store
			s.options.ClaimDuration = 200 * time.Millisecond
			done := make(chan error, 1)
			go func() {
				defer close(done)
				_, err := s.progressOneClaim(ctx)
				done <- err
			}()
			defer func() {
				cancel()
				select {
				case <-done:
				case <-time.After(time.Second):
					t.Error("cancellation lane did not stop")
				}
			}()
			receiveString(t, store.started, time.Second, "initial Run work")
			if _, err := postgresStore.RequestRunCancellation(ctx, runID, runstore.WorkflowRunCancellation{
				Code: runstore.CancellationUserRequested, RequestedAt: time.Now(),
			}); err != nil {
				t.Fatal(err)
			}
			if notify {
				s.Cancel(runID)
			}
			receiveString(t, store.cleaning, time.Second, "cancellation cleanup")
			timer := time.NewTimer(3 * s.options.ClaimDuration)
			defer timer.Stop()
			select {
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			case <-timer.C:
			}
			if run, err := postgresStore.ClaimRunnableRun(ctx, "competing-lane", time.Minute); !errors.Is(err, runstore.ErrNoWork) {
				t.Fatalf("cleanup lost exclusive ownership: competing claim = %s, %v", run.RunID, err)
			}
			close(store.finish)
			if err := receiveError(t, done, time.Second, "completed cancellation"); err != nil {
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
}

func (s *cancellationClaimStore) ListStageExecutions(ctx context.Context, runID string) ([]runstore.StageExecution, error) {
	// Renewal reads the Run too, but never lists its StageExecutions. Block
	// only the initial execution path, independent of goroutine scheduling.
	if s.first.CompareAndSwap(false, true) {
		s.started <- runID
		<-ctx.Done()
		return nil, ctx.Err()
	}
	s.cleaning <- runID
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-s.finish:
		return s.PostgresStore.ListStageExecutions(ctx, runID)
	}
}
