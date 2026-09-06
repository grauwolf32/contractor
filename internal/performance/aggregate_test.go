package performance

import (
	"math"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestPerformanceHistogramQuantilesAndMerging(t *testing.T) {
	var empty Histogram
	if empty.Quantile(.95) != nil {
		t.Fatal("empty window fabricated quantile")
	}
	a, b := Histogram{}, Histogram{}
	a.Observe(.01)
	b.Observe(.1)
	b.Observe(120)
	a.Merge(b)
	if a.Count != 3 || a.Buckets[13] != 3 || a.Buckets[12] != 2 || math.Abs(a.SumSeconds-120.11) > 1e-9 {
		t.Fatal("histogram totals")
	}
	if q := a.Quantile(.99); q == nil || !q.GreaterThan || q.Seconds != 60 {
		t.Fatal("overflow percentile fabricated finite latency")
	}
	if q := a.Quantile(.5); q == nil || q.GreaterThan || q.Seconds <= .01 || q.Seconds > .1 {
		t.Fatal("histogram interpolation")
	}
	for _, q := range []float64{-1, 2, math.NaN()} {
		if a.Quantile(q) != nil {
			t.Fatal("invalid quantile accepted")
		}
	}
}

func minuteFixture(t *testing.T) ([]Sample, time.Time) {
	t.Helper()
	clock := newFakeClock()
	start := clock.Now()
	c := New(Options{Clock: clock, ReadProcess: func() (Process, Reason) {
		p := testProcess(clock.Now().Sub(start).Seconds())
		p.RSSBytes = ptr(uint64(100 + clock.Now().Sub(start).Seconds()))
		return p, ""
	}, ReadPool: func() (Pool, Reason) { return testPool(), "" }})
	c.Collect()
	h := c.Wrap(Public, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(200) }))
	for range 4 {
		clock.advance(15 * time.Second)
		h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("GET", "/v1/work", nil))
		c.Collect()
	}
	return c.History(start, clock.Now().Add(time.Nanosecond)), start
}

func TestPerformanceMinuteUsesCountsCPUTimeAndGaugeExtrema(t *testing.T) {
	samples, start := minuteFixture(t)
	value, err := AggregateMinute(start, samples)
	if err != nil {
		t.Fatal(err)
	}
	if value.Status != OK || value.CoverageSeconds != 60 || value.HTTP[0].Counts[0][1] != 4 || value.HTTP[0].Duration.Count != 4 {
		t.Fatalf("minute count/coverage: %+v", value)
	}
	if value.CPU == nil || value.CPU.Cores != 1.5 || value.CPU.DurationSeconds != 60 || value.CPU.UserSeconds != 60 {
		t.Fatal("CPU aggregate is not time-weighted delta")
	}
	if value.Process.RSSBytes.Min != 115 || value.Process.RSSBytes.Max != 160 || value.Process.RSSBytes.Last != 160 || value.Process.RSSBytes.Samples != 4 {
		t.Fatal("gauge extrema/coverage")
	}
	if value.PoolLast == nil || *value.PoolLast.AcquireCount != 5 || *value.PoolLast.AcquireDurationSeconds != .125 {
		t.Fatal("pool counters replaced by misleading latency")
	}
	*samples[4].Pool.AcquireCount = 999
	if *value.PoolLast.AcquireCount != 5 {
		t.Fatal("aggregate aliases input")
	}
}

func TestPerformanceMinuteNeverSplitsAmbiguousOrMixedGenerationCounters(t *testing.T) {
	samples, start := minuteFixture(t)
	samples[2].Generation = "another-generation"
	if _, err := AggregateMinute(start, samples); err == nil {
		t.Fatal("merged across restart")
	}
	samples, start = minuteFixture(t)
	partial, err := AggregateMinute(start, samples[:3])
	if err != nil {
		t.Fatal(err)
	}
	if partial.Status != Partial || partial.CoverageSeconds != 30 || partial.HTTP[0].Duration.Count != 2 {
		t.Fatal("partial minute presented as complete")
	}
	samples[1].HTTP.Freshness.Coverage.StartedAt = start.Add(-time.Second)
	partial, err = AggregateMinute(start, samples[:2])
	if err != nil {
		t.Fatal(err)
	}
	if partial.Status != Partial || partial.OmittedWindows != 1 || partial.HTTP != nil {
		t.Fatal("ambiguous count apportioned or fabricated zero")
	}
	empty, err := AggregateMinute(start, nil)
	if err != nil || empty.Status != Unavailable || empty.HTTP != nil || empty.CPU != nil {
		t.Fatal("missing minute became zero")
	}
}

func TestPerformanceMinuteCachedGaugeIsNotAnotherObservation(t *testing.T) {
	samples, start := minuteFixture(t)
	samples[2].Process.Freshness.ObservedAt = samples[1].Process.Freshness.ObservedAt
	value, err := AggregateMinute(start, samples)
	if err != nil {
		t.Fatal(err)
	}
	if value.Process.RSSBytes.Samples != 3 {
		t.Fatal("cached gauge repeated as fresh")
	}
}

func BenchmarkPerformanceMinute(b *testing.B) {
	clock := newFakeClock()
	start := clock.Now()
	c := New(Options{Clock: clock, ReadProcess: func() (Process, Reason) { return testProcess(clock.Now().Sub(start).Seconds()), "" }, ReadPool: func() (Pool, Reason) { return testPool(), "" }})
	for range 5 {
		c.Collect()
		clock.advance(15 * time.Second)
	}
	samples := c.History(start, clock.Now())
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := AggregateMinute(start, samples); err != nil {
			b.Fatal(err)
		}
	}
}
