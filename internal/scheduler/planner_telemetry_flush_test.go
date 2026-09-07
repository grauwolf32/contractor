package scheduler

import (
	"context"
	"io"
	"log/slog"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/telemetry"
)

type flushContextProbe struct {
	timeout time.Duration
	flush   func(context.Context) telemetry.PlannerExportResult
}

func (p flushContextProbe) Instrumentation() telemetry.PlannerInstrumentation {
	return telemetry.NoopPlannerInstrumentation()
}
func (p flushContextProbe) FlushTimeout() time.Duration { return p.timeout }
func (p flushContextProbe) Flush(ctx context.Context) telemetry.PlannerExportResult {
	return p.flush(ctx)
}
func (p flushContextProbe) Close() {}

func TestPlannerTelemetryFlushAfterCancellationRetainsBounds(t *testing.T) {
	for _, test := range []struct {
		name         string
		flush        time.Duration
		finalization time.Duration
		remaining    time.Duration
		wantBound    time.Duration
	}{
		{"flush", time.Second, 3 * time.Second, 5 * time.Second, time.Second},
		{"finalization", 3 * time.Second, time.Second, 5 * time.Second, time.Second},
		{"stage", 3 * time.Second, 5 * time.Second, time.Second, time.Second},
	} {
		t.Run(test.name, func(t *testing.T) {
			now := time.Now()
			s := &Scheduler{options: Options{
				Clock: staticClock{now: now}, FinalizationTimeout: test.finalization,
				Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
			}}
			type contextKey struct{}
			ctx, cancel := context.WithCancel(context.WithValue(context.Background(), contextKey{}, "correlation"))
			cancel()
			called := false
			probe := flushContextProbe{timeout: test.flush, flush: func(flushCtx context.Context) telemetry.PlannerExportResult {
				called = true
				if err := flushCtx.Err(); err != nil {
					t.Fatalf("flush inherited cancellation: %v", err)
				}
				if flushCtx.Value(contextKey{}) != "correlation" {
					t.Fatal("flush lost context values")
				}
				deadline, ok := flushCtx.Deadline()
				if !ok || time.Until(deadline) > test.wantBound || time.Until(deadline) <= 0 {
					t.Fatalf("flush deadline is not bounded by %v: %v", test.wantBound, deadline)
				}
				return telemetry.PlannerExportResult{Attempted: true, Succeeded: true}
			}}
			result := s.flushPlannerTelemetry(ctx, probe, now.Add(test.remaining), "stage-test")
			if !called || result == nil || !result.Succeeded {
				t.Fatalf("cancelled-run telemetry was not flushed: %+v", result)
			}
		})
	}
}

func TestPlannerTelemetryFlushSkipsExpiredStage(t *testing.T) {
	now := time.Now()
	s := &Scheduler{options: Options{
		Clock: staticClock{now: now}, FinalizationTimeout: time.Second,
		Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
	}}
	probe := flushContextProbe{timeout: time.Second, flush: func(context.Context) telemetry.PlannerExportResult {
		t.Fatal("expired stage must not start an export")
		return telemetry.PlannerExportResult{}
	}}
	result := s.flushPlannerTelemetry(context.Background(), probe, now, "stage-test")
	if result == nil || result.Succeeded || result.ErrorCode != "flush_timeout" {
		t.Fatalf("unexpected expired-stage export result: %+v", result)
	}
}
