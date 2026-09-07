package performance

import (
	"context"
	"math"
	"net/http"
	"net/http/httptest"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

type fakeClock struct {
	mu     sync.Mutex
	at     time.Time
	timers map[*fakeTimer]bool
}
type fakeTimer struct {
	clock *fakeClock
	at    time.Time
	ch    chan time.Time
}

func newFakeClock() *fakeClock {
	return &fakeClock{at: time.Date(2026, 9, 6, 10, 0, 0, 0, time.UTC), timers: map[*fakeTimer]bool{}}
}
func (c *fakeClock) Now() time.Time { c.mu.Lock(); defer c.mu.Unlock(); return c.at }
func (c *fakeClock) NewTimer(d time.Duration) Timer {
	c.mu.Lock()
	defer c.mu.Unlock()
	t := &fakeTimer{clock: c, at: c.at.Add(d), ch: make(chan time.Time, 1)}
	c.timers[t] = true
	return t
}
func (t *fakeTimer) C() <-chan time.Time { return t.ch }
func (t *fakeTimer) Stop() bool {
	t.clock.mu.Lock()
	defer t.clock.mu.Unlock()
	_, ok := t.clock.timers[t]
	delete(t.clock.timers, t)
	return ok
}
func (c *fakeClock) advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.at = c.at.Add(d)
	for t := range c.timers {
		if !t.at.After(c.at) {
			delete(c.timers, t)
			t.ch <- c.at
		}
	}
}
func (c *fakeClock) timerCount() int { c.mu.Lock(); defer c.mu.Unlock(); return len(c.timers) }
func ptr[T any](v T) *T              { return &v }
func testProcess(cpu float64) Process {
	return Process{CPUUserSeconds: ptr(cpu), CPUSystemSeconds: ptr(cpu / 2), RSSBytes: ptr(uint64(2048)), HeapLiveBytes: ptr(uint64(1024)), Goroutines: ptr(uint64(3)), GCCycles: ptr(uint64(1))}
}
func testPool() Pool {
	return Pool{TotalConnections: ptr(uint32(2)), MaxConnections: ptr(uint32(4)), IdleConnections: ptr(uint32(2)), AcquiredConnections: ptr(uint32(0)), AcquireCount: ptr(uint64(5)), AcquireDurationSeconds: ptr(.125), EmptyAcquireCount: ptr(uint64(2)), CanceledAcquireCount: ptr(uint64(1))}
}
func await(t *testing.T, condition func() bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for !condition() {
		if time.Now().After(deadline) {
			t.Fatal("condition timed out")
		}
		time.Sleep(time.Millisecond)
	}
}

func TestPerformanceScheduleSkipsMissedTicksAndStops(t *testing.T) {
	clock := newFakeClock()
	var reads atomic.Int64
	c := New(Options{Clock: clock, ReadProcess: func() (Process, Reason) { return testProcess(float64(reads.Add(1))), "" }, ReadPool: func() (Pool, Reason) { return testPool(), "" }})
	if reads.Load() != 0 || clock.timerCount() != 0 || c.Snapshot().Current != nil {
		t.Fatal("construction started collection")
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- c.Run(ctx) }()
	await(t, func() bool { return clock.timerCount() == 1 && c.Snapshot().RetainedFrames == 1 })
	clock.advance(90 * time.Second)
	await(t, func() bool { return clock.timerCount() == 1 && c.Snapshot().RetainedFrames == 2 })
	view := c.Snapshot()
	if reads.Load() != 2 || view.SkippedSamples != 5 || view.Current.Process.Freshness.Status != Partial {
		t.Fatalf("catchup/coverage mismatch: %+v reads=%d", view, reads.Load())
	}
	cancel()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if clock.timerCount() != 0 {
		t.Fatal("timer leaked")
	}
	clock.advance(time.Hour)
	if reads.Load() != 2 {
		t.Fatal("reads after stop")
	}
}

func TestPerformanceHistoryBoundsDetachedReadsAndOutage(t *testing.T) {
	clock := newFakeClock()
	start := clock.Now()
	c := New(Options{Clock: clock, ReadProcess: func() (Process, Reason) { return testProcess(clock.Now().Sub(start).Seconds()), "" }, ReadPool: func() (Pool, Reason) { return Pool{}, DatabaseUnavailable }})
	for i := 0; i < 500; i++ {
		c.Collect()
		clock.advance(15 * time.Second)
	}
	view := c.Snapshot()
	if view.Current == nil || view.RejectedSamples != 0 || view.RetainedFrames > 240 || view.RetainedBytes > MaxLiveBytes {
		t.Fatalf("bounds: %+v", view)
	}
	if view.Current.Pool.Freshness.Status != Unavailable || view.Current.Process.CPUCores == nil || *view.Current.Process.CPUCores != 1.5 {
		t.Fatal("DB outage poisoned process data")
	}
	*view.Current.Process.RSSBytes = 0
	if *c.Snapshot().Current.Process.RSSBytes != 2048 {
		t.Fatal("snapshot aliases storage")
	}
	history := c.History(start, clock.Now())
	if len(history) > 240 || len(history) == 0 {
		t.Fatal("history bounds")
	}
	history[0].HTTP.Surfaces[0].Counts[0][0] = 123
	if c.History(start, clock.Now())[0].HTTP.Surfaces[0].Counts[0][0] != 0 {
		t.Fatal("history aliases storage")
	}
	clock.advance(time.Hour)
	if len(c.History(start, clock.Now())) != 0 {
		t.Fatal("expired fine history returned after collection stopped")
	}
}

func TestPerformanceUnknownBaselineResetAndMissingMeasurements(t *testing.T) {
	clock := newFakeClock()
	cpu := 10.0
	unavailable := false
	c := New(Options{Clock: clock, ReadProcess: func() (Process, Reason) {
		if unavailable {
			return Process{}, UnsupportedPlatform
		}
		return testProcess(cpu), ""
	}})
	c.Collect()
	if c.Snapshot().Current.Process.CPUCores != nil || *c.Snapshot().Current.Process.Freshness.Reason != MissingBaseline {
		t.Fatal("fabricated initial CPU rate")
	}
	clock.advance(15 * time.Second)
	cpu = 1
	c.Collect()
	if c.Snapshot().Current.Process.CPUCores != nil || *c.Snapshot().Current.Process.Freshness.Reason != CounterReset {
		t.Fatal("counter reset fabricated rate")
	}
	clock.advance(15 * time.Second)
	unavailable = true
	c.Collect()
	p := c.Snapshot().Current.Process
	if p.RSSBytes != nil || p.CPUCores != nil || p.Freshness.ObservedAt != nil || p.Freshness.Status != Unavailable {
		t.Fatal("unknown values became zeros or stale values")
	}
	other := New(Options{Clock: clock})
	if other.Snapshot().Generation == c.Snapshot().Generation {
		t.Fatal("restart reused generation")
	}
}

func TestPerformanceSamplingIsSerialized(t *testing.T) {
	entered, release := make(chan struct{}), make(chan struct{})
	var calls, active, maximum atomic.Int64
	c := New(Options{ReadProcess: func() (Process, Reason) {
		n := active.Add(1)
		maximum.Store(n)
		if calls.Add(1) == 1 {
			close(entered)
			<-release
		}
		active.Add(-1)
		return testProcess(1), ""
	}})
	done := make(chan struct{}, 2)
	go func() { c.Collect(); done <- struct{}{} }()
	<-entered
	go func() { c.Collect(); done <- struct{}{} }()
	close(release)
	<-done
	<-done
	if maximum.Load() != 1 || calls.Load() != 2 {
		t.Fatal("overlapping readers")
	}
}

func TestPerformanceBadGroupsDoNotDropHTTP(t *testing.T) {
	c := New(Options{ReadProcess: func() (Process, Reason) { p := testProcess(math.NaN()); return p, "" }, ReadPool: func() (Pool, Reason) { p := testPool(); p.AcquireCount = ptr(uint64(1 << 63)); return p, "" }})
	c.Wrap(Public, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(204) })).ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("GET", "/v1/work", nil))
	c.Collect()
	view := c.Snapshot()
	if view.Current == nil || view.Current.HTTP.Surfaces[0].Duration.Count != 1 || view.Current.Process.Freshness.Status != Unavailable || view.Current.Pool.Freshness.Status != Unavailable {
		t.Fatal("invalid group poisoned independent metrics")
	}
}

func TestPerformanceFirstHTTPWindowIncludesStartupRequests(t *testing.T) {
	clock := newFakeClock()
	c := New(Options{Clock: clock, ReadProcess: func() (Process, Reason) { return testProcess(1), "" }})
	clock.advance(3 * time.Second)
	c.Wrap(Public, http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})).ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("GET", "/v1/work", nil))
	c.Collect()
	current := c.Snapshot().Current
	if current.HTTP.Freshness.Coverage.DurationSeconds != 3 || current.HTTP.Surfaces[0].Duration.Count != 1 {
		t.Fatal("startup requests lost their observation window")
	}
}

func TestPerformanceRealProcessAndWorkingPoolDoNotQuery(t *testing.T) {
	reader := NewProcessReader()
	p, reason := reader()
	if runtime.GOOS == "linux" && (reason != "" || p.RSSBytes == nil || *p.RSSBytes == 0 || p.CPUUserSeconds == nil) {
		t.Fatalf("Linux process readings unavailable: %s", reason)
	}
	if p.GCPauses == nil || len(p.GCPauses.BoundsSeconds) > MaxGCPauseBounds || len(p.GCPauses.Counts) != len(p.GCPauses.BoundsSeconds)+1 {
		t.Fatal("GC histogram missing/bounds")
	}
	p.GCPauses.Counts[0] = 123456789
	next, _ := reader()
	if next.GCPauses.Counts[0] == 123456789 {
		t.Fatal("runtime histogram aliases previous sample")
	}
	config, err := pgxpool.ParseConfig("postgres://test@127.0.0.1:1/test?sslmode=disable")
	if err != nil {
		t.Fatal(err)
	}
	config.MaxConns = 2
	config.MinConns = 0
	pool, err := pgxpool.NewWithConfig(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	stats, poolReason := WorkingPoolReader(pool)()
	if poolReason != "" || *stats.MaxConnections != 2 || *stats.TotalConnections != 0 || pool.Stat().AcquireCount() != 0 {
		t.Fatal("in-memory stats attempted DB acquisition")
	}
	c := New(Options{ReadPool: WorkingPoolReader(pool)})
	c.Collect()
	if c.Snapshot().Current == nil || c.Snapshot().RejectedSamples != 0 {
		t.Fatalf("real process sample rejected: %+v", c.Snapshot())
	}
}

func BenchmarkPerformanceCollect(b *testing.B) {
	clock := newFakeClock()
	c := New(Options{Clock: clock, ReadProcess: func() (Process, Reason) { return testProcess(1), "" }, ReadPool: func() (Pool, Reason) { return testPool(), "" }})
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		clock.advance(15 * time.Second)
		c.Collect()
	}
}
func BenchmarkHTTPInstrumentation(b *testing.B) {
	for _, enabled := range []bool{false, true} {
		b.Run(map[bool]string{false: "disabled", true: "enabled"}[enabled], func(b *testing.B) {
			h := http.Handler(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(204) }))
			if enabled {
				h = newHTTPRecorder(time.Now).Wrap(Public, h)
			}
			r := httptest.NewRequest("GET", "/v1/runs", nil)
			w := &discardWriter{header: http.Header{}}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				h.ServeHTTP(w, r)
			}
		})
	}
}

type discardWriter struct{ header http.Header }

func (w *discardWriter) Header() http.Header         { return w.header }
func (w *discardWriter) WriteHeader(int)             {}
func (w *discardWriter) Write(p []byte) (int, error) { return len(p), nil }
