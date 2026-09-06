package performance

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"sync/atomic"
	"time"
)

type DiagnosticBackend interface {
	ReadDatabase(context.Context) (Database, Reason)
	ReadSize(context.Context) (DatabaseSize, Reason)
	Flush(context.Context, []Minute, time.Time) error
	Close()
}

type DiagnosticOptions struct {
	Clock Clock
	Open  func(context.Context) (DiagnosticBackend, error)
}

type DiagnosticView struct {
	Database           *Database     `json:"database,omitempty"`
	DatabaseSize       *DatabaseSize `json:"databaseSize,omitempty"`
	PendingMinutes     int           `json:"pendingMinutes"`
	DroppedMinutes     uint64        `json:"droppedMinutes"`
	SkippedMinutes     uint64        `json:"skippedMinutes"`
	LastWriteAttemptAt *time.Time    `json:"lastWriteAttemptAt,omitempty"`
	WriteReason        *Reason       `json:"writeReason,omitempty"`
}

type pendingMinute struct {
	generation string
	start      time.Time
	raw        []byte
}

// Diagnostics owns all optional SQL work in one serial worker. The independent
// 15-second sampler only reads a cache and enqueues bounded in-memory records.
type Diagnostics struct {
	clock            Clock
	open             func(context.Context) (DiagnosticBackend, error)
	backend          DiagnosticBackend
	running, cycling atomic.Bool
	mu               sync.Mutex
	view             DiagnosticView
	queue            []pendingMinute // at most ten immutable records, at most 32KiB each
	minuteStart      time.Time
	samples          []Sample // at most six frames, including the previous CPU baseline
	lastSample       time.Time
	previousDatabase Database
	previousAt       time.Time // retains monotonic component for rates
	lastSizeAttempt  time.Time
}

func NewDiagnostics(options DiagnosticOptions) *Diagnostics {
	if options.Clock == nil {
		options.Clock = realClock{}
	}
	return &Diagnostics{clock: options.Clock, open: options.Open}
}

func (d *Diagnostics) Run(ctx context.Context) error {
	if !d.running.CompareAndSwap(false, true) {
		return errors.New("performance diagnostics already running")
	}
	defer d.running.Store(false)
	defer func() {
		if d.backend != nil {
			d.backend.Close()
		}
	}()
	for {
		if ctx.Err() != nil {
			return nil
		}
		d.Cycle(ctx)
		now := d.clock.Now()
		timer := d.clock.NewTimer(now.Truncate(DatabaseInterval).Add(DatabaseInterval).Sub(now))
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil
		case <-timer.C():
			timer.Stop()
		}
	}
}

// Cycle is a deterministic test seam. Concurrent attempts are skipped, not
// queued. Every call shares one five-second deadline across all SQL work.
func (d *Diagnostics) Cycle(ctx context.Context) {
	if !d.cycling.CompareAndSwap(false, true) {
		return
	}
	defer d.cycling.Store(false)
	if ctx.Err() != nil {
		return
	}
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if d.backend == nil && d.open != nil {
		backend, err := d.open(ctx)
		if err == nil {
			d.backend = backend
		}
	}
	db, reason := Database{}, Reason(DatabaseUnavailable)
	if d.backend != nil {
		db, reason = d.backend.ReadDatabase(ctx)
	}
	at := d.clock.Now()
	known := db.Commits != nil || db.ClientConnections != nil
	start := d.previousAt
	if start.IsZero() || start.After(at) {
		start = at
	}
	expected := max(uint64(1), uint64(at.Sub(start)/DatabaseInterval))
	if known {
		if db.Commits != nil {
			rates, rateReason := databaseRates(d.previousDatabase, db, at.Sub(d.previousAt))
			db.Rates = rates
			if reason == "" {
				reason = rateReason
			}
		}
		db.Freshness = databaseFreshness(start, at, 60, expected, reason, true)
		if (Sample{Version: 1, Generation: "diagnostic", ObservedAt: at.UTC(), Database: &db}).Validate() != nil {
			known = false
			reason = ReadFailed
		} else {
			d.previousDatabase, d.previousAt = db, at
		}
	}
	d.mu.Lock()
	if !known {
		// Retain known values and their timestamps, but never assert freshness
		// or reuse a rate as if it were measured during this failed attempt.
		if d.view.Database != nil {
			db = *d.view.Database
			db.Rates = nil
		}
		db.Freshness.Status = Unavailable
		db.Freshness.Reason = &reason
		db.Freshness.LastAttemptAt = at.UTC()
		db.Freshness.IntervalSeconds = 60
		if db.Freshness.ObservedAt == nil {
			db.Freshness = databaseFreshness(start, at, 60, expected, reason, false)
		}
	}
	d.view.Database = &db
	d.mu.Unlock()
	if d.lastSizeAttempt.IsZero() || at.Sub(d.lastSizeAttempt) >= DatabaseSizeInterval {
		previous := d.lastSizeAttempt
		d.lastSizeAttempt = at
		if previous.IsZero() || previous.After(at) {
			previous = at
		}
		size, sizeReason := DatabaseSize{}, Reason(DatabaseUnavailable)
		if d.backend != nil && ctx.Err() == nil {
			size, sizeReason = d.backend.ReadSize(ctx)
		}
		finished := d.clock.Now()
		known := size.SizeBytes != nil
		size.Freshness = databaseFreshness(previous, finished, 300, 1, sizeReason, known)
		if (Sample{Version: 1, Generation: "diagnostic", ObservedAt: finished.UTC(), DatabaseSize: &size}).Validate() != nil {
			known = false
			sizeReason = ReadFailed
		}
		d.mu.Lock()
		if !known && d.view.DatabaseSize != nil {
			size = *d.view.DatabaseSize
			size.Freshness.Status = Unavailable
			size.Freshness.Reason = &sizeReason
			size.Freshness.LastAttemptAt = finished.UTC()
		} else if !known {
			size = DatabaseSize{Freshness: databaseFreshness(previous, finished, 300, 1, sizeReason, false)}
		}
		d.view.DatabaseSize = &size
		d.mu.Unlock()
	}
	d.flush(ctx)
}

func databaseFreshness(start, end time.Time, interval uint32, expected uint64, reason Reason, known bool) Freshness {
	f := freshness(start, end, expected, reason, known)
	f.IntervalSeconds = interval
	return f
}

func databaseRates(previous, current Database, elapsed time.Duration) (*DatabaseRates, Reason) {
	if previous.Commits == nil || elapsed <= 0 {
		return nil, MissingBaseline
	}
	if (previous.StatsReset == nil) != (current.StatsReset == nil) || (previous.StatsReset != nil && !previous.StatsReset.Equal(*current.StatsReset)) {
		return nil, CounterReset
	}
	pairs := [][2]*uint64{{previous.Commits, current.Commits}, {previous.Rollbacks, current.Rollbacks}, {previous.Deadlocks, current.Deadlocks}, {previous.TempFiles, current.TempFiles}, {previous.TempBytes, current.TempBytes}, {previous.BlocksRead, current.BlocksRead}, {previous.BlocksHit, current.BlocksHit}}
	var delta [7]float64
	for i, pair := range pairs {
		if pair[0] == nil || pair[1] == nil {
			return nil, MissingBaseline
		}
		if *pair[1] < *pair[0] {
			return nil, CounterReset
		}
		delta[i] = float64(*pair[1]-*pair[0]) / elapsed.Seconds()
	}
	r := &DatabaseRates{IntervalSeconds: elapsed.Seconds(), Commits: delta[0], Rollbacks: delta[1], Deadlocks: delta[2], TempFiles: delta[3], TempBytes: delta[4], BlocksRead: delta[5], BlocksHit: delta[6]}
	if total := delta[5] + delta[6]; total > 0 {
		ratio := delta[6] / total
		r.BufferHitRatio = &ratio
	}
	return r, ""
}

func (d *Diagnostics) Snapshot() DiagnosticView {
	d.mu.Lock()
	view := d.view
	view.PendingMinutes = len(d.queue)
	raw, _ := json.Marshal(view)
	d.mu.Unlock()
	var detached DiagnosticView
	_ = json.Unmarshal(raw, &detached)
	return detached
}

// Observe has no I/O, channel send or dependency on the writer. The collector
// owns its cadence; a delayed frame emits at most one old partial minute and
// records skipped minutes, never a backlog of synthetic samples.
func (d *Diagnostics) Observe(sample Sample) {
	// Own the retained frame: callers may reuse their readers' pointer fields.
	if sample.Validate() != nil {
		return
	}
	raw, _ := json.Marshal(sample)
	var detached Sample
	_ = json.Unmarshal(raw, &detached)
	sample = detached
	d.mu.Lock()
	defer d.mu.Unlock()
	if !sample.ObservedAt.After(d.lastSample) {
		return
	}
	d.lastSample = sample.ObservedAt
	if d.minuteStart.IsZero() {
		d.minuteStart = sample.ObservedAt.Truncate(time.Minute)
	}
	end := d.minuteStart.Add(time.Minute)
	if sample.ObservedAt.After(end) {
		d.emitMinute()
		next := sample.ObservedAt.Truncate(time.Minute)
		if skipped := int64(next.Sub(d.minuteStart)/time.Minute) - 1; skipped > 0 {
			d.view.SkippedMinutes += uint64(skipped)
		}
		var preceding []Sample
		if len(d.samples) > 0 {
			preceding = append(preceding, d.samples[len(d.samples)-1])
		}
		d.samples = preceding
		d.minuteStart = next
	}
	if len(d.samples) >= 6 {
		d.samples = d.samples[1:]
		d.view.SkippedMinutes++
	}
	d.samples = append(d.samples, sample)
	if sample.ObservedAt.Equal(d.minuteStart.Add(time.Minute)) {
		d.emitMinute()
		d.minuteStart = d.minuteStart.Add(time.Minute)
		d.samples = []Sample{sample}
	}
}

func (d *Diagnostics) emitMinute() {
	m, err := AggregateMinute(d.minuteStart, d.samples)
	if err != nil || m.Generation == "" {
		d.view.DroppedMinutes++
		return
	}
	m.DroppedMinutes = d.view.DroppedMinutes
	if m.Validate() != nil {
		d.view.DroppedMinutes++
		return
	}
	if len(d.queue) == MaxPendingMinutes {
		copy(d.queue, d.queue[1:])
		d.queue = d.queue[:len(d.queue)-1]
		d.view.DroppedMinutes++
	}
	m.DroppedMinutes = d.view.DroppedMinutes
	raw, _ := json.Marshal(m)
	d.queue = append(d.queue, pendingMinute{m.Generation, m.MinuteStart, raw})
}

func (d *Diagnostics) flush(ctx context.Context) {
	d.mu.Lock()
	batch := append([]pendingMinute(nil), d.queue...)
	d.mu.Unlock()
	minutes := make([]Minute, 0, len(batch))
	for _, entry := range batch {
		var m Minute
		_ = json.Unmarshal(entry.raw, &m)
		minutes = append(minutes, m)
	}
	var err error
	if d.backend == nil {
		err = errors.New("performance database unavailable")
	} else {
		err = d.backend.Flush(ctx, minutes, d.clock.Now())
	}
	at := d.clock.Now().UTC()
	d.mu.Lock()
	defer d.mu.Unlock()
	d.view.LastWriteAttemptAt = &at
	if err != nil {
		reason := databaseReason(err)
		d.view.WriteReason = &reason
		return
	}
	d.view.WriteReason = nil
	// Enqueues and overflow may happen while SQL is in flight. Acknowledge only
	// the exact batch keys, never remove newly queued entries by an old count.
	retained := d.queue[:0]
	for _, pending := range d.queue {
		written := false
		for _, entry := range batch {
			if pending.generation == entry.generation && pending.start.Equal(entry.start) {
				written = true
				break
			}
		}
		if !written {
			retained = append(retained, pending)
		}
	}
	clear(d.queue[len(retained):])
	d.queue = retained
}
