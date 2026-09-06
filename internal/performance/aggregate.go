package performance

import (
	"encoding/json"
	"errors"
	"math"
	"sort"
	"time"
)

type GaugeSummary struct {
	Last       float64   `json:"last"`
	Min        float64   `json:"min"`
	Max        float64   `json:"max"`
	Samples    uint64    `json:"samples"`
	ObservedAt time.Time `json:"observedAt"`
}

func addGauge(target **GaugeSummary, value float64, at time.Time) {
	if *target == nil {
		*target = &GaugeSummary{Last: value, Min: value, Max: value, Samples: 1, ObservedAt: at}
		return
	}
	g := *target
	if !at.After(g.ObservedAt) {
		return
	} // cached observations are not new samples
	g.Last, g.ObservedAt = value, at
	g.Min, g.Max = math.Min(g.Min, value), math.Max(g.Max, value)
	g.Samples++
}

type ProcessGauges struct {
	RSSBytes      *GaugeSummary `json:"rssBytes,omitempty"`
	HeapLiveBytes *GaugeSummary `json:"heapLiveBytes,omitempty"`
	Goroutines    *GaugeSummary `json:"goroutines,omitempty"`
}

type PoolGauges struct {
	Acquired *GaugeSummary `json:"acquired,omitempty"`
	Idle     *GaugeSummary `json:"idle,omitempty"`
	Total    *GaugeSummary `json:"total,omitempty"`
	Max      *GaugeSummary `json:"max,omitempty"`
}

type CPUAggregate struct {
	UserSeconds     float64 `json:"userSeconds"`
	SystemSeconds   float64 `json:"systemSeconds"`
	DurationSeconds float64 `json:"durationSeconds"`
	Cores           float64 `json:"cores"`
}

type Minute struct {
	Version         int               `json:"version"`
	Generation      string            `json:"generation"`
	MinuteStart     time.Time         `json:"minuteStart"`
	Status          Status            `json:"status"`
	OmittedWindows  uint64            `json:"omittedWindows"`
	CoverageSeconds float64           `json:"coverageSeconds"`
	HTTP            *[2]HTTPSurface   `json:"http,omitempty"`
	CPU             *CPUAggregate     `json:"cpu,omitempty"`
	Process         ProcessGauges     `json:"process"`
	Pool            PoolGauges        `json:"pool"`
	PoolLast        *Pool             `json:"poolLast,omitempty"`
	GCPausesLast    *GCPauseHistogram `json:"gcPausesLast,omitempty"`
}

// AggregateMinute is conservative about ambiguous windows. It never prorates
// request counts or CPU across a minute boundary: straddling windows remain in
// fine history and make this aggregate explicitly partial. Entirely missing
// HTTP/CPU stays absent, not zero. The history writer in V32-003 owns scheduling.
// An optional preceding sample supplies the CPU baseline, not another gauge.
func AggregateMinute(start time.Time, samples []Sample) (Minute, error) {
	if start.IsZero() || !start.Equal(start.Truncate(time.Minute)) || len(samples) > LiveFrames {
		return Minute{}, errInvalidRecord
	}
	end := start.Add(time.Minute)
	ordered := append([]Sample(nil), samples...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].ObservedAt.Before(ordered[j].ObservedAt) })
	result := Minute{Version: 1, MinuteStart: start.UTC(), Status: OK}
	var previous *Sample
	var lastWindowEnd time.Time
	for i := range ordered {
		sample := &ordered[i]
		if err := sample.Validate(); err != nil {
			return Minute{}, err
		}
		if sample.ObservedAt.After(end) {
			continue
		}
		if previous != nil && sample.ObservedAt.Equal(previous.ObservedAt) {
			return Minute{}, errors.New("duplicate performance observation")
		}
		if !sample.ObservedAt.After(start) {
			previous = sample
			continue
		}
		if result.Generation == "" {
			result.Generation = sample.Generation
		}
		if sample.Generation != result.Generation || (previous != nil && previous.Generation != result.Generation) {
			return Minute{}, errors.New("cannot aggregate performance generations")
		}
		if sample.HTTP != nil {
			coverage := sample.HTTP.Freshness.Coverage
			if coverage.StartedAt.Before(start) || coverage.EndedAt.After(end) || coverage.StartedAt.Before(lastWindowEnd) {
				result.OmittedWindows++
				result.Status = Partial
			} else {
				if result.HTTP == nil {
					result.HTTP = &[2]HTTPSurface{{Surface: Public}, {Surface: Private}}
				}
				for surface, value := range sample.HTTP.Surfaces {
					target := &result.HTTP[surface]
					target.InFlight = value.InFlight
					for method, classes := range value.Counts {
						for class, count := range classes {
							target.Counts[method][class] += count
						}
					}
					target.Duration.Merge(value.Duration)
				}
				result.CoverageSeconds += coverage.EndedAt.Sub(coverage.StartedAt).Seconds()
				lastWindowEnd = coverage.EndedAt
				if sample.HTTP.Freshness.Status != OK {
					result.Status = Partial
				}
			}
		}
		if p := sample.Process; p != nil {
			if p.Freshness.Status != OK {
				result.Status = Partial
			}
			if at := p.Freshness.ObservedAt; at != nil {
				for _, g := range []struct {
					target **GaugeSummary
					value  *uint64
				}{{&result.Process.RSSBytes, p.RSSBytes}, {&result.Process.HeapLiveBytes, p.HeapLiveBytes}, {&result.Process.Goroutines, p.Goroutines}} {
					if g.value != nil {
						addGauge(g.target, float64(*g.value), *at)
					}
				}
				result.GCPausesLast = p.GCPauses
			}
			if previous != nil && previous.Process != nil && !previous.ObservedAt.Before(start) && p.Freshness.Coverage.StartedAt.Equal(previous.ObservedAt) {
				before := previous.Process
				seconds := p.Freshness.Coverage.DurationSeconds
				if p.CPUCores != nil && seconds > 0 && p.CPUUserSeconds != nil && p.CPUSystemSeconds != nil && before.CPUUserSeconds != nil && before.CPUSystemSeconds != nil && *p.CPUUserSeconds >= *before.CPUUserSeconds && *p.CPUSystemSeconds >= *before.CPUSystemSeconds {
					if result.CPU == nil {
						result.CPU = &CPUAggregate{}
					}
					result.CPU.UserSeconds += *p.CPUUserSeconds - *before.CPUUserSeconds
					result.CPU.SystemSeconds += *p.CPUSystemSeconds - *before.CPUSystemSeconds
					result.CPU.DurationSeconds += seconds
				} else {
					result.Status = Partial
				}
			} else {
				result.Status = Partial
			}
		}
		if pool := sample.Pool; pool != nil {
			if pool.Freshness.Status != OK {
				result.Status = Partial
			}
			if at := pool.Freshness.ObservedAt; at != nil {
				for _, g := range []struct {
					target **GaugeSummary
					value  *uint32
				}{{&result.Pool.Acquired, pool.AcquiredConnections}, {&result.Pool.Idle, pool.IdleConnections}, {&result.Pool.Total, pool.TotalConnections}, {&result.Pool.Max, pool.MaxConnections}} {
					if g.value != nil {
						addGauge(g.target, float64(*g.value), *at)
					}
				}
				result.PoolLast = pool // cumulative counters, not misleading latency quantiles
			}
		}
		previous = sample
	}
	if result.Generation == "" {
		result.Status = Unavailable
	}
	if result.CoverageSeconds < 60 && result.Status == OK {
		result.Status = Partial
	}
	if result.CPU != nil {
		result.CPU.Cores = (result.CPU.UserSeconds + result.CPU.SystemSeconds) / result.CPU.DurationSeconds
	}
	raw, err := json.Marshal(result)
	if err != nil || len(raw) > MaxRecordBytes {
		return Minute{}, errInvalidRecord
	}
	var detached Minute
	if err := json.Unmarshal(raw, &detached); err != nil {
		return Minute{}, errInvalidRecord
	}
	return detached, nil
}
