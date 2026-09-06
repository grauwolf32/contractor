package performance

import (
	"encoding/json"
	"math"
	"testing"
	"time"
)

func TestRecordBoundsAndHTTPDimensions(t *testing.T) {
	bounds := HTTPBucketBoundsSeconds()
	for i, value := range bounds {
		if math.IsNaN(value) || math.IsInf(value, 0) || value <= 0 || (i > 0 && value <= bounds[i-1]) {
			t.Fatal("histogram bounds must be finite and increasing")
		}
	}
	bounds[0] = 999
	if HTTPBucketBoundsSeconds()[0] != .005 {
		t.Fatal("bounds alias mutable global state")
	}
	if len(HTTPMethods()) != 10 || len(HTTPStatusClasses()) != 6 {
		t.Fatal("HTTP dimensions changed")
	}
	raw, err := json.Marshal(Sample{Version: 1, HTTP: &HTTP{}})
	if err != nil || len(raw) >= MaxRecordBytes || LiveFrames*MaxRecordBytes > MaxLiveBytes {
		t.Fatalf("record bounds: %v", err)
	}
}

func TestSampleRejectsUnsafeOrUnboundedRecords(t *testing.T) {
	at := time.Date(2026, 9, 6, 10, 0, 0, 0, time.UTC)
	freshness := Freshness{Status: OK, ObservedAt: &at, LastAttemptAt: at, IntervalSeconds: 15, Coverage: Coverage{StartedAt: at.Add(-15 * time.Second), EndedAt: at, DurationSeconds: 15, ExpectedSamples: 1, ObservedSamples: 1}}
	base := Sample{Version: 1, Generation: "generation", ObservedAt: at, Process: &Process{Freshness: freshness}}
	if err := base.Validate(); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*Sample){
		func(s *Sample) { s.Generation = string(make([]byte, 129)) },
		func(s *Sample) { s.Process.Freshness.Status = "secret-canary" },
		func(s *Sample) { reason := Reason("secret-canary"); s.Process.Freshness.Reason = &reason },
		func(s *Sample) { s.Process.Freshness.IntervalSeconds = 60 },
		func(s *Sample) { s.Process.Freshness.Coverage.ObservedSamples = 0 },
		func(s *Sample) { value := -1.0; s.Process.CPUCores = &value },
		func(s *Sample) { value := math.Inf(1); s.Process.CPUCores = &value },
		func(s *Sample) { value := uint64(1 << 53); s.Process.RSSBytes = &value },
	} {
		candidate := base
		process := *base.Process
		candidate.Process = &process
		mutate(&candidate)
		if candidate.Validate() == nil {
			t.Fatal("accepted unsafe performance sample")
		}
	}
}
