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

type Quantile struct {
	Seconds     float64
	GreaterThan bool // true means >60s, not an invented finite overflow estimate
}

func (h Histogram) Quantile(q float64) *Quantile {
	if h.Count == 0 || q < 0 || q > 1 || math.IsNaN(q) {
		return nil
	}
	rank := q * float64(h.Count)
	if rank < 1 {
		rank = 1
	}
	lower, previous := 0.0, uint64(0)
	for i, upper := range HTTPBucketBoundsSeconds() {
		count := h.Buckets[i]
		if float64(count) >= rank && count > previous {
			return &Quantile{Seconds: lower + (upper-lower)*(rank-float64(previous))/float64(count-previous)}
		}
		lower, previous = upper, count
	}
	return &Quantile{Seconds: 60, GreaterThan: true}
}
