package performance

import (
	"math"
	"runtime/metrics"
)

type ProcessReader func() (Process, Reason)

// NewProcessReader owns a fixed selection of inexpensive runtime counters.
// The collector serializes calls. No ReadMemStats, profile, or forced GC occurs.
func NewProcessReader() ProcessReader {
	samples := []metrics.Sample{
		{Name: "/gc/heap/live:bytes"},
		{Name: "/sched/goroutines:goroutines"},
		{Name: "/gc/cycles/total:gc-cycles"},
		{Name: "/gc/pauses:seconds"},
	}
	return func() (Process, Reason) {
		value, reason := readOSProcess()
		metrics.Read(samples)
		for i, target := range []**uint64{&value.HeapLiveBytes, &value.Goroutines, &value.GCCycles} {
			if samples[i].Value.Kind() == metrics.KindUint64 {
				n := samples[i].Value.Uint64()
				*target = &n
			} else {
				reason = ReadFailed
			}
		}
		if samples[3].Value.Kind() == metrics.KindFloat64Histogram {
			h := samples[3].Value.Float64Histogram()
			if len(h.Buckets) >= 2 && len(h.Buckets)-2 <= MaxGCPauseBounds && len(h.Counts)+1 == len(h.Buckets) {
				// Runtime histograms use [-Inf, finite edges..., +Inf]. Clone the
				// reusable metrics.Read memory before publishing a sample.
				if math.IsInf(h.Buckets[0], -1) && math.IsInf(h.Buckets[len(h.Buckets)-1], 1) {
					value.GCPauses = &GCPauseHistogram{BoundsSeconds: append([]float64(nil), h.Buckets[1:len(h.Buckets)-1]...), Counts: append([]uint64(nil), h.Counts...)}
				} else {
					reason = ReadFailed
				}
			} else {
				reason = RecordLimit
			}
		} else {
			reason = ReadFailed
		}
		return value, reason
	}
}
