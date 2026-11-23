package performance

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestNVIDIAGPUParsing(t *testing.T) {
	devices, reason := parseNVIDIAGPUs([]byte("GPU-bb, NVIDIA GPU B, 0, 1024, 32768, 44, 49.84, 450\nGPU-aa, NVIDIA GPU A, 75, [N/A], N/A, [Not Supported], 80, 300\n"))
	if reason != UnsupportedMetric || len(devices) != 2 || devices[0].ID != "GPU-aa" {
		t.Fatalf("identities/partial: %+v %s", devices, reason)
	}
	if devices[0].MemoryUsedBytes != nil || devices[0].MemoryTotalBytes != nil || devices[0].TemperatureCelsius != nil || *devices[0].PowerWatts != 80 {
		t.Fatal("unsupported fields should be omitted independently")
	}
	if *devices[1].UtilizationPercent != 0 || *devices[1].MemoryUsedBytes != 1<<30 || *devices[1].MemoryTotalBytes != 32<<30 || *devices[1].PowerWatts != 49.84 {
		t.Fatal("zero or units changed")
	}
	for _, row := range []string{
		"GPU-aa, GPU, NaN, 0, 1, 1, 1, 1",
		"GPU-aa, GPU, 101, 0, 1, 1, 1, 1",
		"GPU-aa, GPU, 1, 2, 1, 1, 1, 1",
		"GPU-aa, GPU, 1, -1, 1, 1, 1, 1",
		"GPU-aa, GPU, 1, 1, 1e20, 1, 1, 1",
		"GPU-aa, GPU, 1, 0, 1, 1, +Inf, 1",
		"GPU-aa, GPU, 1, 0, 1, 1, 1",
		"unexpected identity, GPU, 1, 0, 1, 1, 1, 1",
		"GPU-aa, GPU\tname, 1, 0, 1, 1, 1, 1",
		strings.Repeat("GPU-aa, GPU, 1, 0, 1, 1, 1, 1\n", 2),
	} {
		if _, reason := parseNVIDIAGPUs([]byte(row)); reason != ReadFailed {
			t.Errorf("invalid row accepted: %q (%s)", row, reason)
		}
	}
	if _, reason := parseNVIDIAGPUs(nil); reason != GPUNotAvailable {
		t.Fatal("empty probe became zero measurements")
	}
	var rows strings.Builder
	for i := 0; i < MaxGPUDevices+1; i++ {
		fmt.Fprintf(&rows, "GPU-%x, GPU %d, 1, 0, 1, 1, 1, 1\n", i, i)
	}
	devices, reason = parseNVIDIAGPUs([]byte(rows.String()))
	if len(devices) != MaxGPUDevices || reason != RecordLimit {
		t.Fatal("device bound not enforced")
	}
	output := &gpuOutput{}
	if _, err := io.Copy(output, strings.NewReader(strings.Repeat("x", gpuProbeBytes+1))); err == nil || !output.exceeded || output.buffer.Len() > gpuProbeBytes {
		t.Fatal("output bound bypassed by io.Copy")
	}
}

func TestMissingNVIDIABinaryDisablesGPUCollection(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	reader := NewNVIDIAGPUReader()
	if reader != nil {
		t.Fatal("missing binary must disable the optional reader")
	}
	c := New(Options{ReadGPU: reader})
	for range 2 {
		c.Collect()
	}
	view := c.Snapshot()
	if view.Current == nil || view.Current.GPU != nil || view.RejectedSamples != 0 || view.Current.Process == nil || view.Current.HTTP == nil {
		t.Fatal("missing nvidia-smi affected ordinary collection")
	}
}

func fakeNVIDIABinary(t *testing.T, script string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("shell fixture")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "nvidia-smi")
	if err := os.WriteFile(path, []byte("#!/bin/sh\n"+script), 0700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir)
	return path
}

func TestNVIDIAGPUProbeFailureAndCancellation(t *testing.T) {
	for _, tc := range []struct {
		name, script string
		reason       Reason
	}{
		{"driver unavailable", "echo 'private driver diagnostic' >&2; exit 9", ReadFailed},
		{"valid", "printf 'GPU-aa, GPU, 0, 1, 2, 3, 4, 5\\n'", ""},
		{"bounded output", "printf '" + strings.Repeat("x", gpuProbeBytes+1) + "'", RecordLimit},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fakeNVIDIABinary(t, tc.script)
			reader := NewNVIDIAGPUReader()
			if reader == nil {
				t.Fatal("fixture binary unavailable")
			}
			_, reason := reader(context.Background())
			if reason != tc.reason {
				t.Fatalf("reason=%s want=%s", reason, tc.reason)
			}
		})
	}
	t.Run("removed binary", func(t *testing.T) {
		path := fakeNVIDIABinary(t, "exit 1")
		reader := NewNVIDIAGPUReader()
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
		if _, reason := reader(context.Background()); reason != GPUNotAvailable {
			t.Fatal(reason)
		}
	})
	t.Run("cancellation", func(t *testing.T) {
		fakeNVIDIABinary(t, "exec /bin/sleep 10")
		reader := NewNVIDIAGPUReader()
		ctx, cancel := context.WithTimeout(context.Background(), 40*time.Millisecond)
		defer cancel()
		start := time.Now()
		if _, reason := reader(ctx); reason != BudgetExceeded || time.Since(start) > time.Second {
			t.Fatal("probe ignored cancellation", reason)
		}
	})
	t.Run("deadline", func(t *testing.T) {
		fakeNVIDIABinary(t, "exec /bin/sleep 10")
		start := time.Now()
		if _, reason := NewNVIDIAGPUReader()(context.Background()); reason != BudgetExceeded || time.Since(start) > 3*time.Second {
			t.Fatal("probe exceeded deadline", reason)
		}
	})
}

func TestGPUFailureIsolationAndReadOnlySnapshots(t *testing.T) {
	reads := 0
	c := New(Options{ReadGPU: func(context.Context) ([]GPUDevice, Reason) {
		reads++
		return []GPUDevice{{ID: "GPU-aa", Name: "GPU", UtilizationPercent: ptr(101.0)}}, ""
	}})
	if reads != 0 {
		t.Fatal("collection during construction")
	}
	c.Collect()
	for range 3 {
		view := c.Snapshot()
		if view.Current.GPU.Freshness.Status != Unavailable || len(view.Current.GPU.Devices) != 0 || view.RejectedSamples != 0 || view.Current.HTTP == nil || view.Current.Process == nil {
			t.Fatal("GPU failure poisoned other observations")
		}
	}
	if reads != 1 {
		t.Fatal("read path probed hardware")
	}
}

func TestGPUSamplingStopsWithServerAndDoesNotBlockReads(t *testing.T) {
	entered := make(chan struct{})
	c := New(Options{ReadGPU: func(ctx context.Context) ([]GPUDevice, Reason) {
		close(entered)
		<-ctx.Done()
		return nil, BudgetExceeded
	}})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- c.Run(ctx) }()
	<-entered
	read := make(chan struct{})
	go func() { c.Snapshot(); close(read) }()
	select {
	case <-read:
	case <-time.After(time.Second):
		t.Fatal("snapshot waited for GPU subprocess")
	}
	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("Server shutdown did not cancel GPU sampling")
	}
}

func TestMaximumGPUDevicesFitBoundedSamplesAndHistory(t *testing.T) {
	clock := newFakeClock()
	start := clock.Now()
	c := New(Options{Clock: clock, ReadGPU: func(context.Context) ([]GPUDevice, Reason) {
		devices := make([]GPUDevice, MaxGPUDevices)
		for i := range devices {
			devices[i] = GPUDevice{ID: fmt.Sprintf("GPU-%064x", i), Name: strings.Repeat("N", 128), UtilizationPercent: ptr(100.0), MemoryUsedBytes: ptr(uint64(16 << 30)), MemoryTotalBytes: ptr(uint64(32 << 30)), TemperatureCelsius: ptr(90.0), PowerWatts: ptr(450.0), PowerLimitWatts: ptr(575.0)}
		}
		return devices, ""
	}})
	for range 5 {
		c.Collect()
		clock.advance(SampleInterval)
	}
	if c.Snapshot().RejectedSamples != 0 {
		t.Fatal("GPU count overflow dropped ordinary observations")
	}
	minute, err := AggregateMinute(start, c.History(start, clock.Now()))
	if err != nil || minute.Validate() != nil || len(minute.GPU.Devices) != MaxGPUDevices {
		t.Fatal("GPU aggregate exceeded record bound", err)
	}
	// A hot-plugged ninth identity must not exceed the durable record bound.
	source := *minute.GPU
	source.Devices = append([]GPUDeviceGauges(nil), source.Devices...)
	source.Devices[0].ID = "GPU-ff"
	mergeGPU(&minute.GPU, &source)
	if len(minute.GPU.Devices) != MaxGPUDevices || minute.GPU.Freshness.Reason == nil || *minute.GPU.Freshness.Reason != RecordLimit || minute.Validate() != nil {
		t.Fatal("hot-plug identity bound not enforced")
	}
}

func gpuMinuteFixture(t *testing.T) Minute {
	t.Helper()
	samples, start := minuteFixture(t)
	for i := range samples {
		at := samples[i].ObservedAt
		samples[i].GPU = &GPU{Freshness: freshness(at.Add(-SampleInterval), at, 1, "", true), Devices: []GPUDevice{
			{ID: "GPU-aa", Name: "GPU A", UtilizationPercent: ptr(float64(i * 10)), MemoryUsedBytes: ptr(uint64(i) << 20)},
			{ID: "GPU-bb", Name: "GPU B", UtilizationPercent: ptr(float64(100 - i*10))},
		}}
	}
	// Reordering must not mix cards; a missing reading must not count as zero.
	samples[2].GPU.Devices[0], samples[2].GPU.Devices[1] = samples[2].GPU.Devices[1], samples[2].GPU.Devices[0]
	samples[3].GPU = &GPU{Freshness: freshness(samples[2].ObservedAt, samples[3].ObservedAt, 1, ReadFailed, false), Devices: []GPUDevice{}}
	minute, err := AggregateMinute(start, samples)
	if err != nil {
		t.Fatal(err)
	}
	return minute
}

func TestGPUHistoryPreservesIdentityCoverageAndOmissions(t *testing.T) {
	minute := gpuMinuteFixture(t)
	if err := minute.Validate(); err != nil {
		t.Fatal(err)
	}
	devices := minute.GPU.Devices
	if len(devices) != 2 || devices[0].UtilizationPercent.Samples != 3 || devices[0].UtilizationPercent.Min != 10 || devices[0].UtilizationPercent.Max != 40 || devices[1].UtilizationPercent.Last != 60 || devices[0].PowerWatts != nil {
		t.Fatalf("invalid GPU aggregation: %+v", devices)
	}
	var restored Minute
	raw, err := json.Marshal(minute)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &restored); err != nil || restored.Validate() != nil {
		t.Fatal("GPU history roundtrip failed", err)
	}
	var merged *GPUAggregate
	mergeGPU(&merged, restored.GPU)
	mergeGPU(&merged, restored.GPU)
	if merged.Devices[0].UtilizationPercent.Samples != 3 {
		t.Fatal("cached gauge counted twice")
	}
	if err := merged.Validate(minute.MinuteStart.Add(time.Minute)); err != nil {
		t.Fatal(err)
	}
}

func TestPostgresGPUHistory(t *testing.T) {
	working, store, _ := databaseFixture(t)
	minute := gpuMinuteFixture(t)
	now := minute.MinuteStart.Add(2 * time.Minute)
	if err := store.Flush(context.Background(), []Minute{minute}, now); err != nil {
		t.Fatal(err)
	}
	points, err := NewHistoryRepository(working, func() time.Time { return now }).Read(context.Background(), minute.MinuteStart, now, 5*time.Minute)
	if err != nil || len(points) != 1 {
		t.Fatalf("GPU history read: %+v %v", points, err)
	}
	if points[0].GPU == nil || points[0].GPU.Devices[0].UtilizationPercent.Samples != 3 || points[0].GPU.Devices[1].UtilizationPercent.Last != 60 {
		t.Fatal("GPU metrics lost in durable coarse history")
	}
}

func TestLocalNVIDIAGPU(t *testing.T) {
	if os.Getenv("CONTRACTOR_TEST_NVIDIA_GPU") != "1" {
		t.Skip("optional local GPU smoke test")
	}
	reader := NewNVIDIAGPUReader()
	if reader == nil {
		t.Fatal("nvidia-smi unavailable")
	}
	c := New(Options{ReadGPU: reader})
	c.Collect()
	view := c.Snapshot()
	if view.RejectedSamples != 0 || view.Current == nil || view.Current.GPU == nil || len(view.Current.GPU.Devices) == 0 {
		t.Fatalf("local GPU unavailable: %+v", view)
	}
	for _, d := range view.Current.GPU.Devices {
		t.Logf("GPU %s: %s", d.ID, d.Name)
	}
}
