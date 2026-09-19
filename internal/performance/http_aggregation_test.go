package performance

import (
	"reflect"
	"testing"
	"time"
)

func TestMinuteAndHistoryHTTPAggregationAgree(t *testing.T) {
	samples, start := minuteFixture(t)
	// Cover every fixed dimension with distinct values on both surfaces.
	for i := 1; i < len(samples); i++ {
		for surface := range samples[i].HTTP.Surfaces {
			h := &samples[i].HTTP.Surfaces[surface]
			h.Duration = Histogram{}
			for method := range h.Counts {
				for class := range h.Counts[method] {
					count := uint64(i + surface + method + class)
					h.Counts[method][class] = count
					for range count {
						h.Duration.Observe(float64(class+1) / 100)
					}
				}
			}
		}
	}
	minute, err := AggregateMinute(start, samples)
	if err != nil {
		t.Fatal(err)
	}
	builder := newHistoryBuilder(time.Minute)
	for i := 1; i < len(samples); i++ {
		if err := builder.add(Minute{Generation: minute.Generation, MinuteStart: start.Add(time.Duration(i) * time.Second), Status: OK, HTTP: &samples[i].HTTP.Surfaces}); err != nil {
			t.Fatal(err)
		}
	}
	points := builder.finish()
	if len(points) != 1 || !reflect.DeepEqual(points[0].HTTP, minute.HTTP) {
		t.Fatal("minute and history HTTP results differ")
	}
	original := samples[1].HTTP.Surfaces[0].Counts[0][0]
	points[0].HTTP[0].Counts[0][0] = 9999
	minute.HTTP[0].Duration.Buckets[0] = 9999
	if samples[1].HTTP.Surfaces[0].Counts[0][0] != original || samples[1].HTTP.Surfaces[0].Duration.Buckets[0] == 9999 {
		t.Fatal("aggregate aliases input HTTP dimensions")
	}
}
