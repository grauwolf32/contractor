package main

import (
	"testing"
	"time"
)

func TestBenchmarkOptionsAndSummaryAreBounded(t *testing.T) {
	valid := options{repetitions: 5, httpRequests: 10_000, collectionCycles: 240, profileDuration: time.Second}
	if err := validateOptions(valid); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*options){
		func(value *options) { value.repetitions = 21 },
		func(value *options) { value.httpRequests = 999 },
		func(value *options) { value.collectionCycles = 10_001 },
		func(value *options) { value.profileDuration = 11 * time.Second },
	} {
		candidate := valid
		mutate(&candidate)
		if validateOptions(candidate) == nil {
			t.Fatalf("accepted invalid benchmark options: %+v", candidate)
		}
	}
	result := summarize([]measurement{
		{ThroughputPerSecond: 100, P95Nanoseconds: 10, CPUSeconds: 1, RSSBytes: 1000, AllocationsPerOp: 2},
		{ThroughputPerSecond: 200, P95Nanoseconds: 20, CPUSeconds: 3, RSSBytes: 3000, AllocationsPerOp: 4},
	})
	if result.ThroughputMean != 150 || result.P95Mean != 15 || result.CPUMean != 2 ||
		result.RSSMean != 2000 || result.AllocationsMean != 3 || result.ThroughputCV <= 0 || result.P95CV <= 0 {
		t.Fatalf("summary = %+v", result)
	}
}

func TestCollectionMeasurementReportsFixedStateAndLogicalIO(t *testing.T) {
	result := measureCollection(true, 240)
	if result.Operations != 240 || result.RetainedFrames != 240 || result.RetainedBytes <= 0 ||
		result.RetainedBytes > 8*1024*1024 || result.PendingMinutes > 10 ||
		result.DatabaseReads != 60 || result.DatabaseSizeReads != 12 || result.HistoryWrites != 60 {
		t.Fatalf("collection measurement = %+v", result)
	}
	disabled := measureCollection(false, 240)
	if disabled.DatabaseReads != 0 || disabled.DatabaseSizeReads != 0 || disabled.HistoryWrites != 0 ||
		disabled.RetainedFrames != 0 || disabled.RetainedBytes != 0 || disabled.PendingMinutes != 0 {
		t.Fatalf("disabled collection performed work: %+v", disabled)
	}
}
