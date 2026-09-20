package evaldomain

import (
	"math"
	"sort"
	"time"
)

type Coverage struct {
	Expected int            `json:"expectedPairs"`
	Included int            `json:"includedPairs"`
	Excluded int            `json:"excludedPairs"`
	Reasons  map[string]int `json:"reasons"`
}
type Distribution struct {
	Total *float64 `json:"total"`
	P50   *float64 `json:"p50"`
	P90   *float64 `json:"p90"`
	Count int      `json:"count"`
}
type Bin struct {
	Lower          float64        `json:"lower"`
	Upper          float64        `json:"upper"`
	UpperInclusive bool           `json:"upperInclusive"`
	Counts         map[string]int `json:"counts"`
	FilterToken    string         `json:"filterToken"`
}
type Delta struct {
	PairID     string  `json:"pairId"`
	SuiteID    string  `json:"suiteId"`
	CaseID     string  `json:"caseId"`
	Sample     int     `json:"sample"`
	A          float64 `json:"a"`
	B          float64 `json:"b"`
	Difference float64 `json:"difference"`
	Regression bool    `json:"regression"`
}
type Cohort struct {
	Coverage      Coverage
	Bins          []Bin
	Distributions map[string]Distribution
	Deltas        []Delta
}

func distribution(values []float64) Distribution {
	d := Distribution{Count: len(values)}
	if len(values) == 0 {
		return d
	}
	values = append([]float64{}, values...)
	sort.Float64s(values)
	var total float64
	for _, v := range values {
		total += v
	}
	p50, p90 := values[int(math.Ceil(float64(len(values))*0.5))-1], values[int(math.Ceil(float64(len(values))*0.9))-1]
	d.Total, d.P50, d.P90 = &total, &p50, &p90
	return d
}
func InBin(value float64, b Bin) bool {
	return value >= b.Lower && (value < b.Upper || b.UpperInclusive && value == b.Upper)
}

// MeasurementCohort retains raw samples for exact nearest-rank percentiles.
// Filter tokens are assigned by the authenticated HTTP adapter, not this reducer.
func MeasurementCohort(pairs []Pair, suite, metric, scope string, pinsVerified bool) Cohort {
	c := Cohort{Coverage: Coverage{Reasons: map[string]int{}}, Bins: []Bin{}, Deltas: []Delta{}}
	a, b := []float64{}, []float64{}
	for _, p := range pairs {
		if suite != "" && p.SuiteID != suite {
			continue
		}
		c.Coverage.Expected++
		x, y, why := ComparableMeasure(p.A, p.B, metric, pinsVerified)
		if why == "" && scope != "" && x.Scope.Kind != scope {
			why = "scope-mismatch"
		}
		if why != "" {
			c.Coverage.Excluded++
			c.Coverage.Reasons[why]++
			continue
		}
		c.Coverage.Included++
		a = append(a, *x.Value)
		b = append(b, *y.Value)
		c.Deltas = append(c.Deltas, Delta{p.ID, p.SuiteID, p.CaseID, p.Sample, *x.Value, *y.Value, *y.Value - *x.Value, p.Regression})
	}
	c.Distributions = map[string]Distribution{"a": distribution(a), "b": distribution(b)}
	if len(a) == 0 {
		return c
	}
	lo, hi := a[0], a[0]
	for _, values := range [][]float64{a, b} {
		for _, v := range values {
			lo = math.Min(lo, v)
			hi = math.Max(hi, v)
		}
	}
	n := int(math.Ceil(math.Sqrt(float64(2 * len(a)))))
	if n > MaxChartBins {
		n = MaxChartBins
	}
	if lo == hi {
		n = 1
	}
	for i := 0; i < n; i++ {
		lower, upper := lo+(hi-lo)*float64(i)/float64(n), lo+(hi-lo)*float64(i+1)/float64(n)
		if i == n-1 {
			upper = hi
		}
		c.Bins = append(c.Bins, Bin{Lower: lower, Upper: upper, UpperInclusive: i == n-1, Counts: map[string]int{"a": 0, "b": 0}})
	}
	for arm, values := range map[string][]float64{"a": a, "b": b} {
		for _, v := range values {
			i := 0
			if hi > lo {
				i = min(int((v-lo)/(hi-lo)*float64(n)), n-1)
			}
			// Correct boundary rounding using the exact published edges.
			if i > 0 && v < c.Bins[i].Lower {
				i--
			}
			if i+1 < n && v >= c.Bins[i].Upper {
				i++
			}
			c.Bins[i].Counts[arm]++
		}
	}
	return c
}
func SortDeltas(d []Delta, absolute bool) {
	if absolute {
		sort.SliceStable(d, func(i, j int) bool { return math.Abs(d[i].Difference) > math.Abs(d[j].Difference) })
	}
}

type ProgressObservation struct {
	ObservedAt time.Time
	A          int
	B          int
}
type ProgressPoint struct {
	ElapsedMS  int64  `json:"elapsedMs"`
	ObservedAt string `json:"observedAt"`
	A          *int   `json:"a"`
	B          *int   `json:"b"`
	GapBefore  bool   `json:"gapBefore"`
}

// ProgressBuckets keeps only real server observations. A bucket is never filled
// with an invented zero/interpolated point; long unobserved intervals stay gaps.
func ProgressBuckets(start time.Time, observations []ProgressObservation) ([]ProgressPoint, int64) {
	out := []ProgressPoint{}
	if len(observations) == 0 {
		return out, 1
	}
	rows := append([]ProgressObservation{}, observations...)
	sort.SliceStable(rows, func(i, j int) bool { return rows[i].ObservedAt.Before(rows[j].ObservedAt) })
	maxElapsed := rows[len(rows)-1].ObservedAt.Sub(start).Milliseconds()
	if maxElapsed < 0 {
		return out, 1
	}
	width := maxElapsed/MaxProgressBuckets + 1
	bucket := int64(-1)
	for _, o := range rows {
		elapsed := o.ObservedAt.Sub(start).Milliseconds()
		if elapsed < 0 {
			continue
		}
		p := ProgressPoint{ElapsedMS: elapsed, ObservedAt: o.ObservedAt.UTC().Format(time.RFC3339Nano), A: &o.A, B: &o.B}
		i := elapsed / width
		if i == bucket {
			p.GapBefore = out[len(out)-1].GapBefore
			out[len(out)-1] = p
			continue
		}
		p.GapBefore = len(out) == 0 && elapsed > 0 || len(out) > 0 && elapsed-out[len(out)-1].ElapsedMS > max(width, ProgressGap.Milliseconds())
		out = append(out, p)
		bucket = i
	}
	return out, width
}
