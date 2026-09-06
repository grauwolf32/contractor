package performance

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

var ErrHistoryRange = errors.New("performance history range exceeds its bounds")

func (m Minute) Validate() error {
	if m.Version != 1 || strings.TrimSpace(m.Generation) == "" || len(m.Generation) > 128 || m.MinuteStart.IsZero() || !m.MinuteStart.Equal(m.MinuteStart.Truncate(time.Minute)) || m.CoverageSeconds < 0 || m.CoverageSeconds > 60 || math.IsNaN(m.CoverageSeconds) || math.IsInf(m.CoverageSeconds, 0) {
		return errInvalidRecord
	}
	if m.Status != OK && m.Status != Partial && m.Status != Unavailable {
		return errInvalidRecord
	}
	if m.CPU != nil && (m.CPU.DurationSeconds <= 0 || m.CPU.DurationSeconds > 60) {
		return errInvalidRecord
	}
	for _, g := range []*GaugeSummary{m.Process.RSSBytes, m.Process.HeapLiveBytes, m.Process.Goroutines, m.Pool.Acquired, m.Pool.Idle, m.Pool.Total, m.Pool.Max} {
		if g != nil && (g.Samples == 0 || g.ObservedAt.IsZero() || g.ObservedAt.After(m.MinuteStart.Add(time.Minute)) || g.Min > g.Last || g.Last > g.Max) {
			return errInvalidRecord
		}
	}
	sample := Sample{Version: 1, Generation: m.Generation, ObservedAt: m.MinuteStart.Add(time.Minute), Database: m.Database, DatabaseSize: m.DatabaseSize, Pool: m.PoolLast}
	if m.GCPausesLast != nil {
		sample.Process = &Process{Freshness: freshness(m.MinuteStart, sample.ObservedAt, 1, "", true), GCPauses: m.GCPausesLast}
	}
	if m.HTTP != nil {
		sample.HTTP = &HTTP{Freshness: freshness(m.MinuteStart, m.MinuteStart.Add(time.Minute), 1, "", true), Surfaces: *m.HTTP}
	}
	if sample.Validate() != nil {
		return errInvalidRecord
	}
	raw, err := json.Marshal(m)
	if err != nil || len(raw) > MaxRecordBytes {
		return errInvalidRecord
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	var value any
	if d.Decode(&value) != nil || !validNumbers(value) {
		return errInvalidRecord
	}
	return nil
}

// Flush performs one idempotent bounded batch and one bounded expiry pass in a
// single maintenance transaction. It is called only by the diagnostic worker.
func (s *DatabaseStore) Flush(ctx context.Context, minutes []Minute, now time.Time) error {
	if len(minutes) > MaxPendingMinutes {
		return errInvalidRecord
	}
	var generations []string
	var starts []time.Time
	var coverage []float64
	var payloads [][]byte
	for _, minute := range minutes {
		if minute.Validate() != nil {
			return errInvalidRecord
		}
		if !minute.MinuteStart.Add(HistoryRetention).After(now) {
			continue
		}
		if minute.MinuteStart.After(now) {
			return errInvalidRecord
		}
		raw, _ := json.Marshal(minute)
		generations = append(generations, minute.Generation)
		starts = append(starts, minute.MinuteStart)
		coverage = append(coverage, minute.CoverageSeconds)
		payloads = append(payloads, raw)
	}
	return s.transaction(ctx, true, false, func(ctx context.Context, tx pgx.Tx) error {
		if len(starts) > 0 {
			_, err := tx.Exec(ctx, `INSERT INTO performance_minutes
(server_generation,minute_start,schema_version,coverage_seconds,payload,expires_at)
SELECT generation,started,1,coverage,payload,started+interval '168 hours'
FROM unnest($1::text[],$2::timestamptz[],$3::double precision[],$4::bytea[]) AS batch(generation,started,coverage,payload)
ON CONFLICT (server_generation,minute_start) DO NOTHING`, generations, starts, coverage, payloads)
			if err != nil {
				return err
			}
		}
		_, err := tx.Exec(ctx, cleanupPerformanceSQL, now)
		return err
	})
}

const cleanupPerformanceSQL = `WITH expired AS (
 SELECT server_generation,minute_start FROM performance_minutes WHERE expires_at <= $1
 ORDER BY expires_at,server_generation,minute_start LIMIT 1000 FOR UPDATE SKIP LOCKED
) DELETE FROM performance_minutes p USING expired e
WHERE p.server_generation=e.server_generation AND p.minute_start=e.minute_start`

type HistoryPoint struct {
	Minute
	StepSeconds     uint32 `json:"stepSeconds"`
	ObservedMinutes uint64 `json:"observedMinutes"`
	ExpectedMinutes uint64 `json:"expectedMinutes"`
}

// HistoryRepository deliberately accepts the normal request-budget DB handle.
// Reading history never starts a diagnostic pool, collector, writer or cleanup.
type HistoryRepository struct {
	db  postgres.DBTX
	now func() time.Time
}

func NewHistoryRepository(db postgres.DBTX, now func() time.Time) *HistoryRepository {
	if now == nil {
		now = time.Now
	}
	return &HistoryRepository{db: db, now: now}
}

const historyReadSQL = `SELECT server_generation,minute_start,payload FROM performance_minutes
WHERE minute_start >= $1 AND minute_start < $2 AND expires_at > $3
ORDER BY minute_start,server_generation LIMIT $4`

func (r *HistoryRepository) Read(ctx context.Context, from, to time.Time, step time.Duration) ([]HistoryPoint, error) {
	now := r.now()
	if from.IsZero() || !to.After(from) || to.Sub(from) > HistoryRetention || (step != time.Minute && step != 5*time.Minute && step != time.Hour) {
		return nil, ErrHistoryRange
	}
	if from.Before(now.Add(-HistoryRetention)) {
		from = now.Add(-HistoryRetention)
	}
	if !to.After(from) {
		return []HistoryPoint{}, nil
	}
	if math.Ceil(to.Sub(from).Seconds()/step.Seconds()) > MaxHistoryPoints {
		return nil, ErrHistoryRange
	}
	// Rows stream into at most 1000 bounded accumulators, not a payload slice.
	// More generations can exceed this even when the time range alone fits.
	limit := MaxHistoryPoints * int(step/time.Minute)
	rows, err := r.db.Query(ctx, historyReadSQL, from, to, now, limit+1)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	b := newHistoryBuilder(step)
	count := 0
	for rows.Next() {
		count++
		if count > limit {
			return nil, ErrHistoryRange
		}
		var generation string
		var start time.Time
		var raw []byte
		if err = rows.Scan(&generation, &start, &raw); err != nil {
			return nil, err
		}
		var m Minute
		if len(raw) > MaxRecordBytes || json.Unmarshal(raw, &m) != nil || m.Generation != generation || !m.MinuteStart.Equal(start) || m.Validate() != nil {
			return nil, errInvalidRecord
		}
		if err = b.add(m); err != nil {
			return nil, err
		}
	}
	if err = rows.Err(); err != nil {
		return nil, err
	}
	points := b.finish()
	for _, p := range points {
		// A bounded input can still overflow the JSON-safe aggregate domain.
		raw, err := json.Marshal(p)
		if err != nil || len(raw) > MaxRecordBytes {
			return nil, errInvalidRecord
		}
		d := json.NewDecoder(bytes.NewReader(raw))
		d.UseNumber()
		var value any
		if d.Decode(&value) != nil || !validNumbers(value) {
			return nil, errInvalidRecord
		}
	}
	return points, nil
}

type historyKey struct {
	generation string
	start      time.Time
}
type historyBuilder struct {
	step   time.Duration
	points map[historyKey]*HistoryPoint
	last   map[historyKey]time.Time
}

func newHistoryBuilder(step time.Duration) *historyBuilder {
	return &historyBuilder{step: step, points: map[historyKey]*HistoryPoint{}, last: map[historyKey]time.Time{}}
}

func (b *historyBuilder) add(m Minute) error {
	k := historyKey{m.Generation, m.MinuteStart.UTC().Truncate(b.step)}
	p := b.points[k]
	if p == nil {
		if len(b.points) >= MaxHistoryPoints {
			return ErrHistoryRange
		}
		p = &HistoryPoint{Minute: Minute{Version: 1, Generation: k.generation, MinuteStart: k.start, Status: OK}, StepSeconds: uint32(b.step / time.Second), ExpectedMinutes: uint64(b.step / time.Minute)}
		b.points[k] = p
	}
	if last := b.last[k]; !last.IsZero() && !m.MinuteStart.After(last) {
		return errInvalidRecord
	}
	b.last[k] = m.MinuteStart
	p.ObservedMinutes++
	p.CoverageSeconds += m.CoverageSeconds
	p.OmittedWindows += m.OmittedWindows
	p.DroppedMinutes = max(p.DroppedMinutes, m.DroppedMinutes)
	if m.Status != OK {
		p.Status = Partial
	}
	if m.HTTP != nil {
		if p.HTTP == nil {
			p.HTTP = &[2]HTTPSurface{{Surface: Public}, {Surface: Private}}
		}
		for i, source := range m.HTTP {
			target := &p.HTTP[i]
			target.InFlight = source.InFlight
			for method, classes := range source.Counts {
				for class, n := range classes {
					target.Counts[method][class] += n
				}
			}
			target.Duration.Merge(source.Duration)
		}
	}
	if m.CPU != nil {
		if p.CPU == nil {
			p.CPU = &CPUAggregate{}
		}
		p.CPU.UserSeconds += m.CPU.UserSeconds
		p.CPU.SystemSeconds += m.CPU.SystemSeconds
		p.CPU.DurationSeconds += m.CPU.DurationSeconds
		p.CPU.Cores = (p.CPU.UserSeconds + p.CPU.SystemSeconds) / p.CPU.DurationSeconds
	}
	for _, pair := range []struct {
		target **GaugeSummary
		source *GaugeSummary
	}{
		{&p.Process.RSSBytes, m.Process.RSSBytes}, {&p.Process.HeapLiveBytes, m.Process.HeapLiveBytes}, {&p.Process.Goroutines, m.Process.Goroutines},
		{&p.Pool.Acquired, m.Pool.Acquired}, {&p.Pool.Idle, m.Pool.Idle}, {&p.Pool.Total, m.Pool.Total}, {&p.Pool.Max, m.Pool.Max},
	} {
		mergeGauge(pair.target, pair.source)
	}
	p.PoolLast = m.PoolLast
	p.GCPausesLast = m.GCPausesLast
	// DB values/rates remain the latest *observation*, not five fake identical
	// samples or an average of rates. Its original interval/timestamps survive.
	if m.Database != nil {
		p.Database = m.Database
	}
	if m.DatabaseSize != nil {
		p.DatabaseSize = m.DatabaseSize
	}
	return nil
}

func mergeGauge(target **GaugeSummary, source *GaugeSummary) {
	if source == nil {
		return
	}
	if *target == nil {
		copied := *source
		*target = &copied
		return
	}
	g := *target
	if !source.ObservedAt.After(g.ObservedAt) {
		return
	}
	g.Last, g.ObservedAt = source.Last, source.ObservedAt
	g.Min, g.Max = math.Min(g.Min, source.Min), math.Max(g.Max, source.Max)
	g.Samples += source.Samples
}

func (b *historyBuilder) finish() []HistoryPoint {
	result := make([]HistoryPoint, 0, len(b.points))
	for _, p := range b.points {
		if p.ObservedMinutes < p.ExpectedMinutes || p.CoverageSeconds < float64(p.StepSeconds) {
			p.Status = Partial
		}
		result = append(result, *p)
	}
	sort.Slice(result, func(i, j int) bool {
		if result[i].MinuteStart.Equal(result[j].MinuteStart) {
			return result[i].Generation < result[j].Generation
		}
		return result[i].MinuteStart.Before(result[j].MinuteStart)
	})
	return result
}
