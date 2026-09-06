package performance

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

type Timer interface {
	C() <-chan time.Time
	Stop() bool
}
type Clock interface {
	Now() time.Time
	NewTimer(time.Duration) Timer
}
type realClock struct{}
type realTimer struct{ *time.Timer }

func (realClock) Now() time.Time                 { return time.Now() }
func (realClock) NewTimer(d time.Duration) Timer { return realTimer{time.NewTimer(d)} }
func (t realTimer) C() <-chan time.Time          { return t.Timer.C }

type Options struct {
	Clock       Clock
	ReadProcess ProcessReader
	ReadPool    PoolReader
}

type storedFrame struct {
	at   time.Time
	data []byte
}

type Collector struct {
	clock              Clock
	process            ProcessReader
	pool               PoolReader
	generation         string
	http               *HTTPRecorder
	running            atomic.Bool
	sampleMu           sync.Mutex
	last               time.Time // retains Go monotonic time; UTC conversion is only for wire fields
	sampled            bool
	previousProcess    Process
	previousPool       Pool
	mu                 sync.RWMutex
	frames             [LiveFrames]storedFrame
	head, count, bytes int
	skipped, rejected  uint64
}

func New(options Options) *Collector {
	if options.Clock == nil {
		options.Clock = realClock{}
	}
	if options.ReadProcess == nil {
		options.ReadProcess = NewProcessReader()
	}
	if options.ReadPool == nil {
		options.ReadPool = WorkingPoolReader(nil)
	}
	return &Collector{clock: options.Clock, process: options.ReadProcess, pool: options.ReadPool, generation: rand.Text(), http: newHTTPRecorder(options.Clock.Now), last: options.Clock.Now()}
}

func (c *Collector) Wrap(surface Surface, next http.Handler) http.Handler {
	return c.http.Wrap(surface, next)
}

// Run is owned by the Server lifecycle. Initial collection happens here, not
// during startup parsing/construction, and scheduled ticks never catch up.
func (c *Collector) Run(ctx context.Context) error {
	if !c.running.CompareAndSwap(false, true) {
		return errors.New("performance collector already running")
	}
	defer c.running.Store(false)
	for {
		if ctx.Err() != nil {
			return nil
		}
		c.Collect()
		now := c.clock.Now()
		next := now.Truncate(SampleInterval).Add(SampleInterval)
		timer := c.clock.NewTimer(next.Sub(now))
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil
		case <-timer.C():
			timer.Stop()
		}
	}
}

func freshness(start, end time.Time, expected uint64, reason Reason, known bool) Freshness {
	f := Freshness{Status: OK, LastAttemptAt: end.UTC(), IntervalSeconds: 15, Coverage: Coverage{StartedAt: start.UTC(), EndedAt: end.UTC(), DurationSeconds: math.Max(0, end.Sub(start).Seconds()), ExpectedSamples: expected, ObservedSamples: 1}}
	if known {
		at := end.UTC()
		f.ObservedAt = &at
	} else {
		f.Status = Unavailable
		f.Coverage.ObservedSamples = 0
	}
	if reason != "" {
		f.Reason = &reason
		if known {
			f.Status = Partial
		}
	}
	if expected > 1 && f.Status == OK {
		reason = SamplingGap
		f.Reason = &reason
		f.Status = Partial
	}
	return f
}

// Collect serializes reader access; it is also the deterministic sampling seam
// used by tests. Reader failures cannot escape as scheduler/lifecycle failures.
func (c *Collector) Collect() {
	c.sampleMu.Lock()
	defer c.sampleMu.Unlock()
	p, processReason := c.process()
	pool, poolReason := c.pool()
	at := c.clock.Now()
	start, expected := c.last, uint64(1)
	if start.IsZero() {
		start = at
	}
	elapsed := at.Sub(start).Seconds()
	if elapsed < 0 {
		start = at
		elapsed = 0
		processReason = CounterReset
	}
	if elapsed >= SampleInterval.Seconds()*1.5 {
		expected = uint64(elapsed/SampleInterval.Seconds() + .5)
	}
	if processReason == "" && (p.CPUUserSeconds == nil || p.CPUSystemSeconds == nil || p.RSSBytes == nil) {
		processReason = ReadFailed
	}
	if p.CPUUserSeconds != nil && p.CPUSystemSeconds != nil {
		previous := c.previousProcess
		if previous.CPUUserSeconds == nil || previous.CPUSystemSeconds == nil || elapsed <= 0 {
			if processReason == "" {
				processReason = MissingBaseline
			}
		} else if *p.CPUUserSeconds < *previous.CPUUserSeconds || *p.CPUSystemSeconds < *previous.CPUSystemSeconds {
			processReason = CounterReset
		} else {
			cores := ((*p.CPUUserSeconds - *previous.CPUUserSeconds) + (*p.CPUSystemSeconds - *previous.CPUSystemSeconds)) / elapsed
			if !math.IsNaN(cores) && !math.IsInf(cores, 0) {
				p.CPUCores = &cores
			}
		}
	}
	if p.GCCycles != nil && c.previousProcess.GCCycles != nil && *p.GCCycles < *c.previousProcess.GCCycles {
		processReason = CounterReset
	}
	if decreased(pool.AcquireCount, c.previousPool.AcquireCount) || decreased(pool.EmptyAcquireCount, c.previousPool.EmptyAcquireCount) || decreased(pool.CanceledAcquireCount, c.previousPool.CanceledAcquireCount) {
		poolReason = CounterReset
	}
	if decreasedFloat(pool.AcquireDurationSeconds, c.previousPool.AcquireDurationSeconds) || decreasedFloat(pool.EmptyAcquireWaitSeconds, c.previousPool.EmptyAcquireWaitSeconds) {
		poolReason = CounterReset
	}
	p.Freshness = freshness(start, at, expected, processReason, p.CPUUserSeconds != nil || p.RSSBytes != nil || p.HeapLiveBytes != nil)
	pool.Freshness = freshness(start, at, expected, poolReason, pool.TotalConnections != nil)
	// Reject faulty groups independently. A bad process reading must not discard
	// valid HTTP counters, nor may an invalid pool statistic hide process data.
	if (Sample{Version: 1, Generation: c.generation, ObservedAt: at.UTC(), Process: &p}).Validate() != nil {
		p = Process{Freshness: freshness(start, at, expected, ReadFailed, false)}
	}
	if (Sample{Version: 1, Generation: c.generation, ObservedAt: at.UTC(), Pool: &pool}).Validate() != nil {
		pool = Pool{Freshness: freshness(start, at, expected, ReadFailed, false)}
	}
	httpReason := Reason("")
	if !c.sampled {
		httpReason = MissingBaseline
	}
	h := HTTP{Freshness: freshness(start, at, expected, httpReason, true), Surfaces: c.http.drain()}
	sample := Sample{Version: 1, Generation: c.generation, ObservedAt: at.UTC(), HTTP: &h, Process: &p, Pool: &pool}
	c.last, c.previousProcess, c.previousPool = at, p, pool
	c.sampled = true
	c.mu.Lock()
	defer c.mu.Unlock()
	c.skipped += expected - 1
	if err := sample.Validate(); err != nil {
		c.rejected++
		return
	}
	raw, err := json.Marshal(sample)
	if err != nil || len(raw) > MaxRecordBytes {
		c.rejected++
		return
	}
	for c.count > 0 && (c.count == LiveFrames || c.bytes+len(raw) > MaxLiveBytes || !c.frames[c.head].at.After(at.Add(-time.Hour))) {
		c.evict()
	}
	index := (c.head + c.count) % LiveFrames
	c.frames[index] = storedFrame{at: at.UTC(), data: raw}
	c.count++
	c.bytes += len(raw)
}

func decreased(current, previous *uint64) bool {
	return current != nil && previous != nil && *current < *previous
}

func decreasedFloat(current, previous *float64) bool {
	return current != nil && previous != nil && *current < *previous
}

func (c *Collector) evict() {
	c.bytes -= len(c.frames[c.head].data)
	c.frames[c.head] = storedFrame{}
	c.head = (c.head + 1) % LiveFrames
	c.count--
}

type View struct {
	Generation      string
	Current         *Sample
	SkippedSamples  uint64
	RejectedSamples uint64
	RetainedFrames  int
	RetainedBytes   int
}

// Snapshot and History detach decoded copies; callers cannot mutate collection
// state. Reads never trigger collection, SQL, or background work.
func (c *Collector) Snapshot() View {
	c.mu.RLock()
	view := View{Generation: c.generation, SkippedSamples: c.skipped, RejectedSamples: c.rejected, RetainedFrames: c.count, RetainedBytes: c.bytes}
	var raw []byte
	if c.count > 0 {
		raw = c.frames[(c.head+c.count-1)%LiveFrames].data
	}
	c.mu.RUnlock()
	if len(raw) != 0 {
		var sample Sample
		if json.Unmarshal(raw, &sample) == nil {
			view.Current = &sample
		}
	}
	return view
}

func (c *Collector) History(from, to time.Time) []Sample {
	c.mu.RLock()
	var frames [LiveFrames][]byte
	n := 0
	cutoff := c.clock.Now().Add(-time.Hour)
	for i := 0; i < c.count; i++ {
		frame := c.frames[(c.head+i)%LiveFrames]
		if !frame.at.Before(from) && frame.at.Before(to) && frame.at.After(cutoff) {
			frames[n] = frame.data
			n++
		}
	}
	c.mu.RUnlock()
	result := make([]Sample, 0, n)
	for _, raw := range frames[:n] {
		var value Sample
		if json.Unmarshal(raw, &value) == nil {
			result = append(result, value)
		}
	}
	return result
}
