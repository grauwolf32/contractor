package scheduler

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"testing"
	"time"
)

type retentionResult struct {
	deleted int64
	err     error
}

type retentionStore struct {
	Store
	results chan retentionResult
	calls   chan struct{}
}

func (s *retentionStore) CleanupExpiredTelemetry(ctx context.Context, _ time.Time, _ int) (int64, error) {
	select {
	case result := <-s.results:
		s.calls <- struct{}{}
		return result.deleted, result.err
	case <-ctx.Done():
		return 0, ctx.Err()
	}
}

type retentionWait struct {
	duration time.Duration
	tick     chan time.Time
}

type retentionClock struct {
	waits chan retentionWait
}

func (c *retentionClock) Now() time.Time { return time.Now().UTC() }

func (c *retentionClock) After(duration time.Duration) <-chan time.Time {
	wait := retentionWait{duration: duration, tick: make(chan time.Time, 1)}
	c.waits <- wait
	return wait.tick
}

func TestMetricsRetentionDrainsFullBatchesBeforeLongWait(t *testing.T) {
	store := &retentionStore{
		results: make(chan retentionResult, 3), calls: make(chan struct{}, 3),
	}
	store.results <- retentionResult{deleted: 500}
	store.results <- retentionResult{deleted: 500}
	store.results <- retentionResult{deleted: 17}
	clock := &retentionClock{waits: make(chan retentionWait, 3)}
	scheduler := &Scheduler{store: store, options: Options{
		OperationTimeout: time.Second, MetricsCleanupBatch: 500,
		MetricsCleanupInterval: time.Hour, Clock: clock,
		Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
	}}
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan struct{})
	go func() { scheduler.monitorMetricsRetention(ctx); close(done) }()
	for index := 0; index < 3; index++ {
		receiveRetentionCall(t, store.calls)
		wait := receiveRetentionWait(t, clock.waits)
		want := metricsCleanupDrainPause
		if index == 2 {
			want = time.Hour
		}
		if wait.duration != want {
			t.Fatalf("wait %d = %s, want %s", index, wait.duration, want)
		}
		if index < 2 {
			wait.tick <- time.Now()
		}
	}
	cancel()
	receiveRetentionDone(t, done)
}

func TestMetricsRetentionWaitsAfterErrorAndStopsDuringDrain(t *testing.T) {
	for _, test := range []struct {
		name   string
		result retentionResult
		want   time.Duration
	}{
		{name: "error", result: retentionResult{err: errors.New("database unavailable")}, want: time.Hour},
		{name: "full batch", result: retentionResult{deleted: 500}, want: metricsCleanupDrainPause},
	} {
		t.Run(test.name, func(t *testing.T) {
			store := &retentionStore{
				results: make(chan retentionResult, 1), calls: make(chan struct{}, 1),
			}
			store.results <- test.result
			clock := &retentionClock{waits: make(chan retentionWait, 1)}
			scheduler := &Scheduler{store: store, options: Options{
				OperationTimeout: time.Second, MetricsCleanupBatch: 500,
				MetricsCleanupInterval: time.Hour, Clock: clock,
				Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
			}}
			ctx, cancel := context.WithCancel(t.Context())
			done := make(chan struct{})
			go func() { scheduler.monitorMetricsRetention(ctx); close(done) }()
			receiveRetentionCall(t, store.calls)
			wait := receiveRetentionWait(t, clock.waits)
			if wait.duration != test.want {
				t.Fatalf("wait = %s, want %s", wait.duration, test.want)
			}
			cancel()
			receiveRetentionDone(t, done)
			select {
			case <-store.calls:
				t.Fatal("retention made another cleanup call after cancellation")
			default:
			}
		})
	}
}

func receiveRetentionCall(t *testing.T, calls <-chan struct{}) {
	t.Helper()
	select {
	case <-calls:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for cleanup call")
	}
}

func receiveRetentionWait(t *testing.T, waits <-chan retentionWait) retentionWait {
	t.Helper()
	select {
	case wait := <-waits:
		return wait
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for retention interval")
		return retentionWait{}
	}
}

func receiveRetentionDone(t *testing.T, done <-chan struct{}) {
	t.Helper()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("retention loop did not stop after cancellation")
	}
}
