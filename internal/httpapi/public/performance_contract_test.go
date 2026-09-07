package public

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/performance"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

func TestPerformancePublicContracts(t *testing.T) {
	document := loadPublicOpenAPI(t)
	for _, path := range []string{"/v1/operations/performance", "/v1/operations/performance/history", "/v1/operations/allocation-history"} {
		operation := document.Paths.Value(path).Get
		if operation == nil || operation.Extensions["x-contractor-implementation"] != "implemented" {
			t.Fatalf("%s must be an implemented read-only contract", path)
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
		snapshot := map[string]any{"enabled": true, "generation": sample.Generation, "observedAt": at, "sampleIntervalSeconds": 15, "databaseIntervalSeconds": 60, "databaseSizeIntervalSeconds": 300, "current": sample, "diagnostics": map[string]any{"skippedSamples": 0, "rejectedSamples": 0, "skippedMinutes": 0, "droppedMinutes": 0, "pendingMinutes": 0}}
		validate("PerformanceSnapshot", snapshot, true)
		snapshot["enabled"] = false
		validate("PerformanceSnapshot", snapshot, false)
		history := map[string]any{"from": at.Add(-time.Hour), "to": at, "step": "15s", "points": []performance.FineHistoryPoint{{Kind: "sample", Sample: sample}}}
		validate("PerformanceHistory", history, true)
		history["points"] = make([]performance.FineHistoryPoint, 1001)
		validate("PerformanceHistory", history, false)
	}
	validate("PerformanceProcess", map[string]any{"freshness": nil, "cpuCores": nil}, false)
	aggregate := performance.AggregateHistoryPoint{
		Kind: "aggregate",
		HistoryPoint: performance.HistoryPoint{
			Minute: performance.Minute{
				Version: 1, Generation: "generation", MinuteStart: at.Add(-time.Minute),
				Status: performance.Partial, Process: performance.ProcessGauges{},
				Pool: performance.PoolGauges{},
			},
			StepSeconds: 60, ObservedMinutes: 1, ExpectedMinutes: 1,
		},
	}
	validate("PerformanceAggregateHistoryPoint", aggregate, true)
	validate("RuntimeResourceSummary", contracts.RuntimeResources{Version: 1, Scope: "runtime_process", Status: contracts.ResourceUnavailable}, true)
	validate("RuntimeResourceSummary", contracts.RuntimeResources{Version: 1, Scope: "runtime_process", Status: contracts.ResourceComplete}, false)
	validate("RuntimeResourceSummary", map[string]any{"version": 1, "scope": "runtime_process", "status": "partial", "cpuUserSeconds": nil}, false)
	validate("RuntimeResourceSummary", map[string]any{"version": 1, "scope": "runtime_process", "status": "partial", "rssSampleCount": -1}, false)
}

func TestPerformanceReadHandlersAndAllocationHistoryCursor(t *testing.T) {
	finished := time.Date(2026, 8, 29, 11, 0, 0, 0, time.UTC)
	fixture := newHandlerFixtureWithAuth(
		t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil,
		func(dependencies *Dependencies) {
			dependencies.AllocationResources = &fakeAllocationResourceReader{
				byStage: map[string][]telemetry.AllocationResourceSummary{},
				items: []telemetry.AllocationResourceSummary{
					{AllocationID: "allocation-2", RunID: "run-1", StageExecutionID: "stage-1", Stage: "analyze", LogicalAgent: "worker", Outcome: "succeeded", FinishedAt: finished, CollectionPolicy: contracts.PerformanceCollectionDisabled, Status: telemetry.AllocationResourceDisabled},
					{AllocationID: "allocation-1", RunID: "run-1", StageExecutionID: "stage-1", Stage: "analyze", LogicalAgent: "worker", Outcome: "succeeded", FinishedAt: finished.Add(-time.Second), CollectionPolicy: contracts.PerformanceCollectionUnsupported, Status: telemetry.AllocationResourceUnsupported},
				},
			}
		},
	)
	current := httptest.NewRecorder()
	fixture.handler.ServeHTTP(current, authenticatedRequest(http.MethodGet, "/v1/operations/performance", bytes.NewReader(nil)))
	if current.Code != http.StatusOK {
		t.Fatalf("current performance = %d %s", current.Code, current.Body.String())
	}
	if current.Header().Get("Cache-Control") != "no-store" {
		t.Fatalf("current performance cache policy = %q", current.Header().Get("Cache-Control"))
	}
	var snapshot performance.SnapshotResponse
	if json.Unmarshal(current.Body.Bytes(), &snapshot) != nil || snapshot.Enabled || snapshot.Current != nil {
		t.Fatalf("disabled performance snapshot = %+v", snapshot)
	}

	first := httptest.NewRecorder()
	fixture.handler.ServeHTTP(first, authenticatedRequest(http.MethodGet, "/v1/operations/allocation-history?limit=1", bytes.NewReader(nil)))
	if first.Code != http.StatusOK {
		t.Fatalf("allocation history = %d %s", first.Code, first.Body.String())
	}
	var page allocationResourcePageResponse
	if json.Unmarshal(first.Body.Bytes(), &page) != nil || len(page.Items) != 1 || !page.Page.HasMore || page.Page.NextCursor == nil {
		t.Fatalf("first allocation page = %+v", page)
	}
	cursor := *page.Page.NextCursor
	second := httptest.NewRecorder()
	fixture.handler.ServeHTTP(second, authenticatedRequest(http.MethodGet, "/v1/operations/allocation-history?limit=1&cursor="+cursor, bytes.NewReader(nil)))
	if second.Code != http.StatusOK {
		t.Fatalf("second allocation history = %d %s", second.Code, second.Body.String())
	}
	if json.Unmarshal(second.Body.Bytes(), &page) != nil || len(page.Items) != 1 || page.Items[0].AllocationID != "allocation-1" {
		t.Fatalf("second allocation page = %+v", page)
	}

	invalidCurrent := httptest.NewRecorder()
	fixture.handler.ServeHTTP(invalidCurrent, authenticatedRequest(http.MethodGet, "/v1/operations/performance?refresh=true", bytes.NewReader(nil)))
	if invalidCurrent.Code != http.StatusBadRequest {
		t.Fatalf("unknown current-performance query = %d %s", invalidCurrent.Code, invalidCurrent.Body.String())
	}
	filterMismatch := httptest.NewRecorder()
	fixture.handler.ServeHTTP(filterMismatch, authenticatedRequest(
		http.MethodGet,
		"/v1/operations/allocation-history?limit=1&runId=run-1&cursor="+cursor,
		bytes.NewReader(nil),
	))
	if filterMismatch.Code != http.StatusBadRequest {
		t.Fatalf("cross-filter allocation cursor = %d %s", filterMismatch.Code, filterMismatch.Body.String())
	}
}
