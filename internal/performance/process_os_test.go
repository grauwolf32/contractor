//go:build linux || darwin || windows || freebsd || openbsd || solaris

package performance

import (
	"errors"
	"math"
	"testing"
	"time"

	"github.com/shirou/gopsutil/v4/cpu"
	"github.com/shirou/gopsutil/v4/process"
)

type testProcessMetrics struct {
	times                   *cpu.TimesStat
	memory                  *process.MemoryInfoStat
	timesError, memoryError error
	timesCalls, memoryCalls int
}

func (p *testProcessMetrics) Times() (*cpu.TimesStat, error) {
	p.timesCalls++
	return p.times, p.timesError
}

func (p *testProcessMetrics) MemoryInfo() (*process.MemoryInfoStat, error) {
	p.memoryCalls++
	return p.memory, p.memoryError
}

func TestPerformanceProcessReadFailuresPreserveIndependentFields(t *testing.T) {
	for _, tc := range []struct {
		name                    string
		times                   *cpu.TimesStat
		memory                  *process.MemoryInfoStat
		timesError, memoryError error
		user, system, rss       bool
	}{
		{name: "CPU error", timesError: errors.New("secret-canary"), memory: &process.MemoryInfoStat{RSS: 4096}, rss: true},
		{name: "RSS error", times: &cpu.TimesStat{User: 2.5, System: 1.5}, memoryError: errors.New("secret-canary"), user: true, system: true},
		{name: "nil CPU", memory: &process.MemoryInfoStat{RSS: 4096}, rss: true},
		{name: "nil RSS", times: &cpu.TimesStat{}, user: true, system: true},
		{name: "both errors", timesError: errors.New("secret-canary"), memoryError: errors.New("secret-canary")},
		{name: "invalid user", times: &cpu.TimesStat{User: math.NaN(), System: 1.5}, memory: &process.MemoryInfoStat{RSS: 4096}, system: true, rss: true},
		{name: "invalid system", times: &cpu.TimesStat{User: 2.5, System: -1}, memory: &process.MemoryInfoStat{RSS: 4096}, user: true, rss: true},
		{name: "overflow RSS", times: &cpu.TimesStat{}, memory: &process.MemoryInfoStat{RSS: 1 << 53}, user: true, system: true},
		{name: "zero-filled RSS", times: &cpu.TimesStat{}, memory: &process.MemoryInfoStat{}, user: true, system: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			current := &testProcessMetrics{times: tc.times, memory: tc.memory, timesError: tc.timesError, memoryError: tc.memoryError}
			value, reason := readProcessMetrics(current)
			if reason != ReadFailed || (value.CPUUserSeconds != nil) != tc.user || (value.CPUSystemSeconds != nil) != tc.system || (value.RSSBytes != nil) != tc.rss {
				t.Fatalf("wrong partial reading: %+v, %s", value, reason)
			}
			if current.timesCalls != 1 || current.memoryCalls != 1 {
				t.Fatal("reads were skipped or retried")
			}
		})
	}
}

func TestPerformanceProcessReadsCurrentRSSAndCPUTime(t *testing.T) {
	current := &testProcessMetrics{
		times:  &cpu.TimesStat{User: 2.5, System: 1.5, Idle: 1000},
		memory: &process.MemoryInfoStat{RSS: 12288, HWM: 999999999, VMS: 888888888},
	}
	value, reason := readProcessMetrics(current)
	if reason != "" || value.CPUUserSeconds == nil || *value.CPUUserSeconds != 2.5 || value.CPUSystemSeconds == nil || *value.CPUSystemSeconds != 1.5 || value.RSSBytes == nil || *value.RSSBytes != 12288 {
		t.Fatalf("wrong units or process fields: %+v, %s", value, reason)
	}
	// The published sample owns its values even if a reader reuses its records.
	current.times.User = 100
	current.memory.RSS = 100
	if *value.CPUUserSeconds != 2.5 || *value.RSSBytes != 12288 {
		t.Fatal("sample aliases a process record")
	}
}

func TestPerformanceRealProcessCPUAndRSSReachSnapshot(t *testing.T) {
	clock := newFakeClock()
	collector := New(Options{Clock: clock})
	collector.Collect()
	first := collector.Snapshot().Current.Process
	if first.RSSBytes == nil || *first.RSSBytes == 0 || first.CPUUserSeconds == nil || *first.CPUUserSeconds < 0 || first.CPUSystemSeconds == nil || *first.CPUSystemSeconds < 0 || first.CPUCores != nil || first.Freshness.Reason == nil || *first.Freshness.Reason != MissingBaseline {
		t.Fatalf("missing real-process baseline: %+v", first)
	}
	clock.advance(15 * time.Second)
	collector.Collect()
	next := collector.Snapshot().Current.Process
	if next.Freshness.Status != OK || next.CPUCores == nil || *next.CPUCores < 0 || next.RSSBytes == nil || *next.RSSBytes == 0 {
		t.Fatalf("real process did not reach current performance: %+v", next)
	}
}
