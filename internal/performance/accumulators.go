package performance

import (
	"time"
)

type httpWindowAccumulator struct{ start, end, lastEnd time.Time }

func (window *httpWindowAccumulator) add(result *Minute, sample *Sample) {
	if sample.HTTP != nil {
		coverage := sample.HTTP.Freshness.Coverage
		if coverage.StartedAt.Before(window.start) || coverage.EndedAt.After(window.end) || coverage.StartedAt.Before(window.lastEnd) {
			result.OmittedWindows++
			result.Status = Partial
		} else {
			mergeHTTPSurfaces(&result.HTTP, sample.HTTP.Surfaces)
			result.CoverageSeconds += coverage.EndedAt.Sub(coverage.StartedAt).Seconds()
			window.lastEnd = coverage.EndedAt
			if sample.HTTP.Freshness.Status != OK {
				result.Status = Partial
			}
		}
	}
}

func (result *Minute) accumulateDatabase(sample *Sample) {
	// Cached DB observations keep their original freshness. They are not
	// converted into a new gauge sample or a new interval rate here.
	if sample.Database != nil {
		result.Database = sample.Database
		if sample.Database.Freshness.Status != OK {
			result.Status = Partial
		}
	}
	if sample.DatabaseSize != nil {
		result.DatabaseSize = sample.DatabaseSize
		if sample.DatabaseSize.Freshness.Status != OK {
			result.Status = Partial
		}
	}
}

func (result *Minute) accumulateProcessGauges(sample *Sample) {
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
	}
}

func (result *Minute) accumulateCPUDelta(start time.Time, previous, sample *Sample) {
	if p := sample.Process; p != nil {
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
}

func (result *Minute) accumulatePoolGauges(sample *Sample) {
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
}

// mergeHTTPSurfaces merges the fixed counters and histogram by value. In-flight
// is the latest gauge, not a count to sum across windows.
func mergeHTTPSurfaces(target **[2]HTTPSurface, source [2]HTTPSurface) {
	if *target == nil {
		*target = &[2]HTTPSurface{{Surface: Public}, {Surface: Private}}
	}
	for surface, value := range source {
		destination := &(*target)[surface]
		destination.InFlight = value.InFlight
		for method, classes := range value.Counts {
			for class, count := range classes {
				destination.Counts[method][class] += count
			}
		}
		destination.Duration.Merge(value.Duration)
	}
}
