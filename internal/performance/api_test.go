package performance

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestReadServiceUsesOnlyRetainedMemoryForCurrentAndFineHistory(t *testing.T) {
	clock := newFakeClock()
	reads := 0.0
	collector := New(Options{
		Clock: clock,
		ReadProcess: func() (Process, Reason) {
			reads++
			return testProcess(reads), ""
		},
		ReadPool: func() (Pool, Reason) { return testPool(), "" },
	})
	collector.Collect()
	clock.advance(SampleInterval)
	collector.Collect()
	service := NewReadService(true, collector, nil, nil, clock.Now)

	snapshot := service.Snapshot()
	if !snapshot.Enabled || snapshot.Current == nil || snapshot.Generation == "" ||
		snapshot.ObservedAt != clock.Now() {
		t.Fatalf("snapshot = %+v", snapshot)
	}
	history, err := service.History(
		context.Background(), clock.Now().Add(-time.Minute), clock.Now().Add(time.Second), "15s",
	)
	if err != nil {
		t.Fatal(err)
	}
	points, ok := history.Points.([]FineHistoryPoint)
	if !ok || len(points) != 2 || points[0].Kind != "sample" || points[1].Generation != snapshot.Generation {
		t.Fatalf("fine points = %#v", history.Points)
	}
	if reads != 2 {
		t.Fatalf("read API triggered collection: process reads=%v", reads)
	}
	if _, err := service.History(
		context.Background(), clock.Now().Add(-time.Hour), clock.Now(), "15s",
	); err != nil {
		t.Fatalf("exact retained-hour request failed: %v", err)
	}
}

func TestReadServiceDisabledAndFineRangeAreExplicit(t *testing.T) {
	clock := newFakeClock()
	service := NewReadService(false, nil, nil, nil, clock.Now)
	if snapshot := service.Snapshot(); snapshot.Enabled || snapshot.Current != nil || snapshot.Generation == "" {
		t.Fatalf("disabled snapshot = %+v", snapshot)
	}
	history, err := service.History(
		context.Background(), clock.Now().Add(-time.Minute), clock.Now(), "15s",
	)
	if err != nil {
		t.Fatal(err)
	}
	points, ok := history.Points.([]FineHistoryPoint)
	if !ok || len(points) != 0 {
		t.Fatalf("disabled fine history = %#v", history.Points)
	}
	_, err = service.History(
		context.Background(), clock.Now().Add(-time.Hour-time.Second), clock.Now(), "15s",
	)
	if !errors.Is(err, ErrHistoryRange) {
		t.Fatalf("excessive fine range error = %v", err)
	}
	_, err = service.History(context.Background(), clock.Now().Add(-time.Minute), clock.Now(), "30s")
	if !errors.Is(err, ErrHistoryStep) {
		t.Fatalf("unknown step error = %v", err)
	}
}
