package telemetry

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"testing"
	"time"
)

// hangGuard only bounds a wait for the loop under test; no assertion depends
// on how much time passes.
const hangGuard = 10 * time.Second

type retentionResult struct {
	deleted int64
	err     error
}

type scriptedCleaner struct {
	results chan retentionResult
	calls   chan int
}

func (c *scriptedCleaner) CleanupExpired(ctx context.Context, _ time.Time, batch int) (int64, error) {
	select {
	case result := <-c.results:
		c.calls <- batch
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

func (c *retentionClock) After(duration time.Duration) <-chan time.Time {
	wait := retentionWait{duration: duration, tick: make(chan time.Time, 1)}
	c.waits <- wait
	return wait.tick
}

func newScriptedRetention(cleaner *scriptedCleaner, clock *retentionClock) *Retention {
	return &Retention{
		cleaner: cleaner, passTimeout: time.Second,
		logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
		now:    func() time.Time { return time.Date(2026, 10, 12, 0, 0, 0, 0, time.UTC) },
		after:  clock.After,
	}
}

func TestRetentionDrainsFullBatchesBeforeLongWait(t *testing.T) {
	cleaner := &scriptedCleaner{results: make(chan retentionResult, 3), calls: make(chan int, 3)}
	cleaner.results <- retentionResult{deleted: retentionBatch}
	cleaner.results <- retentionResult{deleted: retentionBatch}
	cleaner.results <- retentionResult{deleted: 17}
	clock := &retentionClock{waits: make(chan retentionWait, 3)}
	retention := newScriptedRetention(cleaner, clock)
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan error, 1)
	go func() { done <- retention.Run(ctx) }()
	for index := 0; index < 3; index++ {
		if batch := receiveCleanupCall(t, cleaner.calls); batch != retentionBatch {
			t.Fatalf("cleanup %d batch = %d, want %d", index, batch, retentionBatch)
		}
		wait := receiveRetentionWait(t, clock.waits)
		want := retentionDrainPause
		if index == 2 {
			want = retentionInterval
		}
		if wait.duration != want {
			t.Fatalf("wait %d = %s, want %s", index, wait.duration, want)
		}
		if index < 2 {
			wait.tick <- time.Time{}
		}
	}
	cancel()
	receiveRetentionDone(t, done)
}

func TestRetentionWaitsAfterErrorAndStopsDuringDrain(t *testing.T) {
	for _, test := range []struct {
		name   string
		result retentionResult
		want   time.Duration
	}{
		{name: "error", result: retentionResult{err: errors.New("database unavailable")}, want: retentionInterval},
		{name: "full batch", result: retentionResult{deleted: retentionBatch}, want: retentionDrainPause},
	} {
		t.Run(test.name, func(t *testing.T) {
			cleaner := &scriptedCleaner{results: make(chan retentionResult, 1), calls: make(chan int, 1)}
			cleaner.results <- test.result
			clock := &retentionClock{waits: make(chan retentionWait, 1)}
			retention := newScriptedRetention(cleaner, clock)
			ctx, cancel := context.WithCancel(t.Context())
			done := make(chan error, 1)
			go func() { done <- retention.Run(ctx) }()
			receiveCleanupCall(t, cleaner.calls)
			wait := receiveRetentionWait(t, clock.waits)
			if wait.duration != test.want {
				t.Fatalf("wait = %s, want %s", wait.duration, test.want)
			}
			cancel()
			receiveRetentionDone(t, done)
			select {
			case <-cleaner.calls:
				t.Fatal("retention made another cleanup call after cancellation")
			default:
			}
		})
	}
}

func TestNewRetentionRequiresItsDependencies(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	repository := NewRepository(nil)
	for name, build := range map[string]func() (*Retention, error){
		"repository": func() (*Retention, error) { return NewRetention(nil, time.Second, logger) },
		"logger":     func() (*Retention, error) { return NewRetention(repository, time.Second, nil) },
		"timeout":    func() (*Retention, error) { return NewRetention(repository, 0, logger) },
	} {
		if retention, err := build(); err == nil || retention != nil {
			t.Fatalf("missing %s = (%v, %v), want an error", name, retention, err)
		}
	}
	if _, err := NewRetention(repository, time.Second, logger); err != nil {
		t.Fatalf("complete retention: %v", err)
	}
}

func receiveCleanupCall(t *testing.T, calls <-chan int) int {
	t.Helper()
	select {
	case batch := <-calls:
		return batch
	case <-time.After(hangGuard):
		t.Fatal("cleanup call did not arrive")
		return 0
	}
}

func receiveRetentionWait(t *testing.T, waits <-chan retentionWait) retentionWait {
	t.Helper()
	select {
	case wait := <-waits:
		return wait
	case <-time.After(hangGuard):
		t.Fatal("retention interval wait did not arrive")
		return retentionWait{}
	}
}

func receiveRetentionDone(t *testing.T, done <-chan error) {
	t.Helper()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("retention stopped with %v, want nil after cancellation", err)
		}
	case <-time.After(hangGuard):
		t.Fatal("retention loop did not stop after cancellation")
	}
}
