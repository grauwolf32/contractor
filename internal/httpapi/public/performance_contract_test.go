package public

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/performance"
)

func TestPerformancePublicContracts(t *testing.T) {
	document := loadPublicOpenAPI(t)
	for _, path := range []string{"/v1/operations/performance", "/v1/operations/performance/history", "/v1/operations/allocation-history"} {
		operation := document.Paths.Value(path).Get
		if operation == nil || operation.Extensions["x-contractor-implementation"] != "planned" {
			t.Fatalf("%s must remain a planned read-only contract", path)
		}
		if len(document.Paths.Value(path).Operations()) != 1 {
			t.Fatalf("unexpected writable performance route: %s", path)
		}
	}
	validate := func(name string, value any, valid bool) {
		t.Helper()
		encoded, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		var wire any
		if err := json.Unmarshal(encoded, &wire); err != nil {
			t.Fatal(err)
		}
		if err := document.Components.Schemas[name].Value.VisitJSON(wire); (err == nil) != valid {
			t.Fatalf("%s valid=%v: %v", name, valid, err)
		}
	}
	for _, name := range []string{"PerformanceSnapshot", "PerformanceHistory", "AllocationResourceSummary"} {
		validate(name, document.Components.Schemas[name].Value.Example, true)
	}
	collector := performance.New(performance.Options{})
	collector.Collect()
	if collector.Snapshot().Current == nil {
		t.Fatal("real process sample unavailable")
	}
	validate("PerformanceSample", collector.Snapshot().Current, true)
	at := time.Date(2026, 9, 6, 10, 0, 0, 0, time.UTC)
	for _, state := range []struct {
		status performance.Status
		reason performance.Reason
	}{
		{performance.Unavailable, performance.MissingBaseline},
		{performance.Partial, performance.ReadFailed},
		{performance.Unavailable, performance.CounterReset},
	} {
		freshness := performance.Freshness{Status: state.status, Reason: &state.reason, LastAttemptAt: at, IntervalSeconds: 15, Coverage: performance.Coverage{StartedAt: at.Add(-15 * time.Second), EndedAt: at, DurationSeconds: 15, ExpectedSamples: 1, ObservedSamples: 0}}
		sample := performance.Sample{Version: 1, Generation: "performance-new-generation", ObservedAt: at, Process: &performance.Process{Freshness: freshness}}
		validate("PerformanceSample", sample, true)
		snapshot := map[string]any{"enabled": true, "generation": sample.Generation, "observedAt": at, "sampleIntervalSeconds": 15, "databaseIntervalSeconds": 60, "databaseSizeIntervalSeconds": 300, "current": sample, "diagnostics": map[string]any{"skippedSamples": 0, "droppedMinutes": 0, "pendingMinutes": 0}}
		validate("PerformanceSnapshot", snapshot, true)
		snapshot["enabled"] = false
		validate("PerformanceSnapshot", snapshot, false)
		history := map[string]any{"from": at.Add(-time.Hour), "to": at, "step": "1m", "points": []performance.Sample{sample}}
		validate("PerformanceHistory", history, true)
		history["points"] = make([]performance.Sample, 1001)
		validate("PerformanceHistory", history, false)
	}
	validate("PerformanceProcess", map[string]any{"freshness": nil, "cpuCores": nil}, false)
	validate("RuntimeResourceSummary", contracts.RuntimeResources{Version: 1, Scope: "runtime_process", Status: contracts.ResourceUnavailable}, true)
	validate("RuntimeResourceSummary", contracts.RuntimeResources{Version: 1, Scope: "runtime_process", Status: contracts.ResourceComplete}, false)
	validate("RuntimeResourceSummary", map[string]any{"version": 1, "scope": "runtime_process", "status": "partial", "cpuUserSeconds": nil}, false)
	validate("RuntimeResourceSummary", map[string]any{"version": 1, "scope": "runtime_process", "status": "partial", "rssSampleCount": -1}, false)
}
