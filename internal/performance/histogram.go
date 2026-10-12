package performance

import "math"

func (h *Histogram) Observe(seconds float64) {
	if seconds < 0 || math.IsNaN(seconds) || math.IsInf(seconds, 0) {
		return
	}
	h.Count++
	h.SumSeconds += seconds
	for i, bound := range HTTPBucketBoundsSeconds() {
		if seconds <= bound {
			h.Buckets[i]++
		}
	}
	h.Buckets[13]++
}

func (h *Histogram) Merge(other Histogram) {
	h.Count += other.Count
	h.SumSeconds += other.SumSeconds
	for i := range h.Buckets {
		h.Buckets[i] += other.Buckets[i]
	}
}
