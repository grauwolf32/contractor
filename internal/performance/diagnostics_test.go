package performance

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
)

type diagnosticFake struct {
	reads, sizes, writes, closes atomic.Int64
	database                     func(context.Context) (Database, Reason)
	size                         func(context.Context) (DatabaseSize, Reason)
	flush                        func(context.Context, []Minute, time.Time) error
}

func counterDatabase(n uint64) Database {
	return Database{Commits: ptr(n), Rollbacks: ptr(n), Deadlocks: ptr(n), TempFiles: ptr(n), TempBytes: ptr(n), BlocksRead: ptr(n), BlocksHit: ptr(3 * n)}
}
func (f *diagnosticFake) ReadDatabase(ctx context.Context) (Database, Reason) {
	n := f.reads.Add(1)
	if f.database != nil {
		return f.database(ctx)
	}
	return counterDatabase(uint64(n) * 60), ""
}
func (f *diagnosticFake) ReadSize(ctx context.Context) (DatabaseSize, Reason) {
	f.sizes.Add(1)
	if f.size != nil {
		return f.size(ctx)
	}
	return DatabaseSize{SizeBytes: ptr(uint64(1024))}, ""
}
func (f *diagnosticFake) Flush(ctx context.Context, m []Minute, now time.Time) error {
	f.writes.Add(1)
	if f.flush != nil {
		return f.flush(ctx, m, now)
	}
	return nil
}
func (f *diagnosticFake) Close() { f.closes.Add(1) }

func TestDiagnosticsCadenceNoCatchupAndShutdown(t *testing.T) {
	clock, backend := newFakeClock(), &diagnosticFake{}
	var opens atomic.Int64
	d := NewDiagnostics(DiagnosticOptions{Clock: clock, Open: func(context.Context) (DiagnosticBackend, error) { opens.Add(1); return backend, nil }})
	if opens.Load() != 0 || clock.timerCount() != 0 {
		t.Fatal("construction did work")
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- d.Run(ctx) }()
	await(t, func() bool { return clock.timerCount() == 1 && backend.writes.Load() == 1 })
	first := d.Snapshot()
	if first.Database.Freshness.Status != Partial || *first.Database.Freshness.Reason != MissingBaseline {
		t.Fatal("first rates fabricated")
	}
	clock.advance(time.Minute)
	await(t, func() bool { return clock.timerCount() == 1 && backend.writes.Load() == 2 })
	second := d.Snapshot()
	if backend.sizes.Load() != 1 || !second.DatabaseSize.Freshness.ObservedAt.Equal(*first.DatabaseSize.Freshness.ObservedAt) || second.Database.Rates.Commits != 1 {
		t.Fatal("sparse size timestamp or rate")
	}
	clock.advance(9 * time.Minute)
	await(t, func() bool { return clock.timerCount() == 1 && backend.writes.Load() == 3 })
	view := d.Snapshot()
	if backend.reads.Load() != 3 || backend.sizes.Load() != 2 || opens.Load() != 1 || view.Database.Rates.IntervalSeconds != 540 || view.Database.Freshness.Status != Partial {
		t.Fatal("missed ticks caught up or asserted complete")
	}
	cancel()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if clock.timerCount() != 0 || backend.closes.Load() != 1 {
		t.Fatal("shutdown leak")
	}
}

func TestDatabaseRatesResetsAndRealIntervals(t *testing.T) {
	before, after := counterDatabase(10), counterDatabase(40)
	r, reason := databaseRates(before, after, 90*time.Second)
	if reason != "" || r.Commits != 1.0/3 || *r.BufferHitRatio != .75 {
		t.Fatal("rate denominator/cache ratio")
	}
	for _, kind := range []string{"decrease", "reset", "unknown-reset", "missing", "elapsed"} {
		t.Run(kind, func(t *testing.T) {
			a, b, elapsed := counterDatabase(10), counterDatabase(40), time.Minute
			want := CounterReset
			switch kind {
			case "decrease":
				b.TempBytes = ptr(uint64(9))
			case "reset":
				a.StatsReset = ptr(time.Unix(1, 0))
				b.StatsReset = ptr(time.Unix(2, 0))
			case "unknown-reset":
				b.StatsReset = ptr(time.Unix(2, 0))
			case "missing":
				b.Rollbacks = nil
				want = MissingBaseline
			case "elapsed":
				elapsed = 0
				want = MissingBaseline
			}
			if r, reason := databaseRates(a, b, elapsed); r != nil || reason != want {
				t.Fatalf("%+v %s", r, reason)
			}
		})
	}
}

func TestDiagnosticsFailureKeepsOriginalObservationsAndRecovers(t *testing.T) {
	clock := newFakeClock()
	failed, sizeFailed := false, false
	backend := &diagnosticFake{database: func(context.Context) (Database, Reason) {
		if failed {
			return Database{}, DatabaseUnavailable
		}
		return counterDatabase(uint64(clock.Now().Unix())), ""
	}, size: func(context.Context) (DatabaseSize, Reason) {
		if sizeFailed {
			return DatabaseSize{}, BudgetExceeded
		}
		return DatabaseSize{SizeBytes: ptr(uint64(42))}, ""
	}}
	d := NewDiagnostics(DiagnosticOptions{Clock: clock, Open: func(context.Context) (DiagnosticBackend, error) { return backend, nil }})
	d.Cycle(context.Background())
	first := d.Snapshot()
	failed, sizeFailed = true, true
	clock.advance(5 * time.Minute)
	d.Cycle(context.Background())
	v := d.Snapshot()
	if v.Database.Freshness.Status != Unavailable || v.Database.Rates != nil || !v.Database.Freshness.ObservedAt.Equal(*first.Database.Freshness.ObservedAt) || !v.DatabaseSize.Freshness.ObservedAt.Equal(*first.DatabaseSize.Freshness.ObservedAt) || *v.DatabaseSize.Freshness.Reason != BudgetExceeded {
		t.Fatal("failure fabricated fresh observation")
	}
	*v.Database.Commits = 0
	if *d.Snapshot().Database.Commits == 0 {
		t.Fatal("snapshot aliases cache")
	}
	failed = false
	clock.advance(time.Minute)
	d.Cycle(context.Background())
	v = d.Snapshot()
	if v.Database.Rates.IntervalSeconds != 360 || v.Database.Rates.Commits != 1 || backend.sizes.Load() != 2 {
		t.Fatal("recovery retried size immediately or used tick denominator")
	}
}

func enqueueMinutes(d *Diagnostics, start time.Time, count int) {
	for i := 0; i <= count*4; i++ {
		d.Observe(Sample{Version: 1, Generation: "generation-a", ObservedAt: start.Add(time.Duration(i) * 15 * time.Second)})
	}
}

func TestDiagnosticsBoundedQueueAndNonBlockingWriter(t *testing.T) {
	clock := newFakeClock()
	entered, release := make(chan []Minute, 1), make(chan struct{})
	backend := &diagnosticFake{flush: func(ctx context.Context, m []Minute, _ time.Time) error {
		entered <- m
		select {
		case <-release:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}}
	d := NewDiagnostics(DiagnosticOptions{Clock: clock, Open: func(context.Context) (DiagnosticBackend, error) { return backend, nil }})
	enqueueMinutes(d, clock.Now(), 12)
	if v := d.Snapshot(); v.PendingMinutes != 10 || v.DroppedMinutes != 2 {
		t.Fatalf("queue: %+v", v)
	}
	done := make(chan struct{})
	go func() { d.Cycle(context.Background()); close(done) }()
	batch := <-entered
	if len(batch) != 10 || !batch[0].MinuteStart.Equal(clock.Now().Add(2*time.Minute)) {
		t.Fatal("did not drop oldest")
	}
	d.Cycle(context.Background()) // overlap is skipped, never queued
	if backend.reads.Load() != 1 {
		t.Fatal("overlapping cycle")
	}
	finished := make(chan struct{})
	go func() {
		enqueueMinutes(d, clock.Now().Add(12*time.Minute), 3)
		c := New(Options{Clock: clock, Diagnostics: d, ReadProcess: func() (Process, Reason) { return testProcess(1), "" }})
		h := c.Wrap(Public, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(204) }))
		h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("GET", "/api/test", nil))
		c.Collect()
		if c.Snapshot().Current.Process == nil {
			t.Error("process unavailable during write")
		}
		close(finished)
	}()
	select {
	case <-finished:
	case <-time.After(time.Second):
		t.Fatal("sampler or HTTP waited for writer")
	}
	close(release)
	<-done
	v := d.Snapshot()
	if v.PendingMinutes != 3 || v.DroppedMinutes != 5 {
		t.Fatalf("ack removed new records: %+v", v)
	}
	for _, p := range d.queue {
		if len(p.raw) > MaxRecordBytes {
			t.Fatal("oversize queue entry")
		}
	}
}

func TestDiagnosticsFailureRetainsQueueAndSkipsMissingMinutes(t *testing.T) {
	clock := newFakeClock()
	d := NewDiagnostics(DiagnosticOptions{Clock: clock, Open: func(context.Context) (DiagnosticBackend, error) { return nil, errors.New("offline") }})
	enqueueMinutes(d, clock.Now(), 1)
	d.Cycle(context.Background())
	if d.Snapshot().PendingMinutes != 1 || *d.Snapshot().WriteReason != DatabaseUnavailable {
		t.Fatal("failed write lost pending")
	}
	d.Observe(Sample{Version: 1, Generation: "generation-a", ObservedAt: clock.Now().Add(time.Hour)})
	if d.Snapshot().SkippedMinutes != 58 || d.Snapshot().PendingMinutes > 2 {
		t.Fatal("generated missed-minute backlog")
	}
}

func TestDiagnosticsEarlierCycleDeadlineAndCancellation(t *testing.T) {
	backend := &diagnosticFake{database: func(ctx context.Context) (Database, Reason) {
		deadline, ok := ctx.Deadline()
		if !ok || time.Until(deadline) > 5*time.Second {
			t.Error("unbounded cycle")
		}
		<-ctx.Done()
		return Database{}, BudgetExceeded
	}, flush: func(ctx context.Context, _ []Minute, _ time.Time) error { return ctx.Err() }}
	d := NewDiagnostics(DiagnosticOptions{Open: func(context.Context) (DiagnosticBackend, error) { return backend, nil }})
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	start := time.Now()
	d.Cycle(ctx)
	if time.Since(start) > 250*time.Millisecond || backend.sizes.Load() != 0 || d.Snapshot().Database.Freshness.Status != Unavailable {
		t.Fatal("deadline not respected")
	}
}

func TestDiagnosticPoolLazyAndConnectBounded(t *testing.T) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	accepted := make(chan net.Conn, 1)
	go func() {
		if c, e := l.Accept(); e == nil {
			accepted <- c
		}
	}()
	p, err := NewDiagnosticPool(context.Background(), "postgres://test@"+l.Addr().String()+"/test?sslmode=disable&pool_min_conns=4&pool_min_idle_conns=3&application_name=not-performance")
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	cfg := p.Config()
	if p.Stat().TotalConns() != 0 || cfg.MaxConns != 1 || cfg.MinConns != 0 || cfg.MinIdleConns != 0 || cfg.ConnConfig.RuntimeParams["application_name"] != DiagnosticApplicationName {
		t.Fatal("diagnostic pool not isolated/lazy")
	}
	start := time.Now()
	_, reason := NewDatabaseStore(p).ReadDatabase(context.Background())
	if reason == "" || time.Since(start) > 600*time.Millisecond {
		t.Fatal("unbounded connect")
	}
	select {
	case c := <-accepted:
		c.Close()
	case <-time.After(time.Second):
		t.Fatal("no fixture connection")
	}
}

func TestDiagnosticConnectBudgetIncludesDNSAndEarlierDeadline(t *testing.T) {
	p, err := NewDiagnosticPool(context.Background(), "postgres://test@example.invalid/test?sslmode=disable")
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	for _, timeout := range []time.Duration{time.Second, 20 * time.Millisecond} {
		cfg := p.Config().ConnConfig
		cfg.LookupFunc = func(ctx context.Context, _ string) ([]string, error) { <-ctx.Done(); return nil, ctx.Err() }
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		start := time.Now()
		_, err := pgx.ConnectConfig(ctx, cfg)
		cancel()
		if err == nil || time.Since(start) > min(timeout, 250*time.Millisecond)+200*time.Millisecond {
			t.Fatal("DNS escaped connection budget")
		}
	}
}

func TestMinuteValidationAndGenerationAwareHistory(t *testing.T) {
	start := newFakeClock().Now()
	b := newHistoryBuilder(5 * time.Minute)
	for i := 0; i < 2; i++ {
		m := Minute{Version: 1, Generation: "a", MinuteStart: start.Add(time.Duration(i) * time.Minute), Status: Partial, CoverageSeconds: 30, CPU: &CPUAggregate{UserSeconds: float64(i+1) * 10, DurationSeconds: float64(i+1) * 10, Cores: 1}, HTTP: &[2]HTTPSurface{{Surface: Public}, {Surface: Private}}, Process: ProcessGauges{RSSBytes: &GaugeSummary{Last: float64(i + 1), Min: float64(i + 1), Max: float64(i + 1), Samples: 1, ObservedAt: start.Add(time.Duration(i) * time.Minute)}}}
		m.HTTP[0].Counts[0][0] = 1
		m.HTTP[0].Duration.Observe(float64(i + 1))
		m.DatabaseSize = &DatabaseSize{Freshness: databaseFreshness(start, start, 300, 1, "", true), SizeBytes: ptr(uint64(100))}
		if err := m.Validate(); err != nil {
			t.Fatal(err)
		}
		if err := b.add(m); err != nil {
			t.Fatal(err)
		}
	}
	if err := b.add(Minute{Version: 1, Generation: "b", MinuteStart: start, Status: Partial}); err != nil {
		t.Fatal(err)
	}
	points := b.finish()
	if len(points) != 2 || points[0].HTTP[0].Duration.Count != 2 || points[0].HTTP[0].Duration.SumSeconds != 3 || points[0].CPU.Cores != 1 || points[0].CPU.DurationSeconds != 30 || points[0].Process.RSSBytes.Samples != 2 || points[0].Status != Partial || !points[0].DatabaseSize.Freshness.ObservedAt.Equal(start) {
		t.Fatalf("aggregation: %+v", points)
	}
	bad := Minute{Version: 1, Generation: "a", MinuteStart: start, Status: OK, CoverageSeconds: 61}
	if bad.Validate() == nil {
		t.Fatal("accepted invalid coverage")
	}
	bad.CoverageSeconds = 0
	bad.GCPausesLast = &GCPauseHistogram{BoundsSeconds: make([]float64, MaxGCPauseBounds+1)}
	if bad.Validate() == nil {
		t.Fatal("accepted unbounded GC histogram")
	}
	for i := 0; i < MaxHistoryPoints; i++ {
		if err := b.add(Minute{Generation: time.Unix(int64(i), 0).String(), MinuteStart: start}); err == ErrHistoryRange {
			return
		}
	}
	t.Fatal("accepted more than 1000 points")
}

func TestObserveOwnsFrames(t *testing.T) {
	d := NewDiagnostics(DiagnosticOptions{})
	start := newFakeClock().Now()
	p := testProcess(1)
	p.Freshness = freshness(start, start, 1, "", true)
	d.Observe(Sample{Version: 1, Generation: "a", ObservedAt: start, Process: &p})
	*p.RSSBytes = 9
	raw, _ := json.Marshal(d.samples[0])
	var s Sample
	json.Unmarshal(raw, &s)
	if *s.Process.RSSBytes != 2048 {
		t.Fatal("retained borrowed pointer")
	}
}
