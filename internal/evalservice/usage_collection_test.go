package evalservice

import (
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"slices"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/evaldomain"
)

func TestEvalLegacyPassthroughCountersPreserveWorkerCompleteness(t *testing.T) {
	for _, tc := range []struct {
		name             string
		planner          resolved.PlannerRef
		missingWorker    bool
		unavailableUsage int
		incompleteReport bool
		wantIncomplete   bool
	}{
		{name: "legacy passthrough", planner: resolved.PlannerRef{PlannerID: "passthrough", Version: "1"}},
		{name: "unknown planner", wantIncomplete: true},
		{name: "model planner", planner: resolved.PlannerRef{PlannerID: "streamline", Version: "1"}, wantIncomplete: true},
		{name: "unknown passthrough version", planner: resolved.PlannerRef{PlannerID: "passthrough", Version: "2"}, wantIncomplete: true},
		{name: "missing Worker counter", planner: resolved.PlannerRef{PlannerID: "passthrough", Version: "1"}, missingWorker: true, wantIncomplete: true},
		{name: "unavailable Worker usage", planner: resolved.PlannerRef{PlannerID: "passthrough", Version: "1"}, unavailableUsage: 1, wantIncomplete: true},
		{name: "incomplete report", planner: resolved.PlannerRef{PlannerID: "passthrough", Version: "1"}, incompleteReport: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls, input, output, total := int64(3), int64(100), int64(20), int64(120)
			worker := reporting.ExecutionMetrics{
				ModelCalls: &calls, InputTokens: &input, OutputTokens: &output, TotalTokens: &total,
				Tools: map[string]reporting.ToolMetrics{},
			}
			if tc.missingWorker {
				worker.TotalTokens = nil
			}
			// JSON fixtures represent retained reports from before explicit Planner zeros.
			workerJSON, err := json.Marshal(worker)
			if err != nil {
				t.Fatal(err)
			}
			var metrics map[string]any
			if err := json.Unmarshal(workerJSON, &metrics); err != nil {
				t.Fatal(err)
			}
			if tc.unavailableUsage > 0 {
				metrics["workerBudget"] = map[string]any{"tokenUsageUnavailable": tc.unavailableUsage}
			}
			raw, err := json.Marshal(map[string]any{
				"planner": map[string]any{"complete": !tc.incompleteReport, "metrics": map[string]any{"tools": map[string]any{}}},
				"workers": map[string]any{"worker": map[string]any{"complete": true, "metrics": metrics}},
				"runtime": map[string]any{"worker": map[string]any{"complete": true}},
			})
			if err != nil {
				t.Fatal(err)
			}
			observed, err := attemptMetrics("run", "stage", raw, 1, tc.planner)
			if err != nil {
				t.Fatal(err)
			}
			if got := slices.Contains(observed.IncompleteMetrics, "totalTokens"); got != tc.wantIncomplete {
				t.Fatalf("token completeness = %v, want incomplete=%t", observed.IncompleteMetrics, tc.wantIncomplete)
			}
			if !tc.missingWorker && observed.Metrics["totalTokens"] != total {
				t.Fatalf("Worker tokens changed: %+v", observed)
			}
			if observed.Metrics["modelCalls"] != calls || observed.ReportsComplete == tc.incompleteReport {
				t.Fatalf("Worker calls or report completeness changed: %+v", observed)
			}
		})
	}
}

// A snapshot's counters do not depend on which Worker or tool holds which
// contribution. A counter whose known contributions exceed the int64 range is
// overflowed and has no value in every assignment, a sum exactly at the range
// is kept, and a missing contribution leaves only its own counter incomplete
// (EVM-R240).
func TestEvalAttemptMetricsOverflowIsOrderIndependent(t *testing.T) {
	const top = math.MaxInt64
	overflowing := []int64{top - 100, 200, 0}
	exact := []int64{top - 100, 100, 0}
	toolCalls := []int64{top, 1}
	value := func(v int64) *int64 { return &v }
	for _, workers := range [][3]int{{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}} {
		for _, tools := range [][2]int{{0, 1}, {1, 0}} {
			reports := map[string]any{}
			runtime := map[string]any{}
			for i, contribution := range workers {
				metrics := reporting.ExecutionMetrics{
					ModelCalls: value(exact[contribution]), InputTokens: value(overflowing[contribution]),
					OutputTokens: value(0), TotalTokens: value(overflowing[contribution]),
					Tools: map[string]reporting.ToolMetrics{},
				}
				if contribution == 2 {
					metrics.OutputTokens = nil
				}
				if i == 0 {
					metrics.Tools["a"] = reporting.ToolMetrics{Calls: value(toolCalls[tools[0]]), Failed: value(1)}
					metrics.Tools["b"] = reporting.ToolMetrics{Calls: value(toolCalls[tools[1]]), Failed: value(0)}
				}
				name := fmt.Sprintf("worker-%d", i)
				reports[name] = map[string]any{"complete": true, "metrics": metrics}
				runtime[name] = map[string]any{"complete": true}
			}
			raw, err := json.Marshal(map[string]any{"workers": reports, "runtime": runtime})
			if err != nil {
				t.Fatal(err)
			}
			observed, err := attemptMetrics("run", "stage", raw, 3, resolved.PlannerRef{PlannerID: "streamline", Version: "1"})
			if err != nil {
				t.Fatal(err)
			}
			// The source digest covers the exact bytes, which name contributions differently.
			observed.SourceSHA256 = ""
			want := evaldomain.AttemptObservation{
				RunID: "run", StageExecutionID: "stage", ReportsComplete: true,
				Metrics:           map[string]int64{"modelCalls": top, "outputTokens": 0, "toolFailures": 1},
				IncompleteMetrics: []string{"outputTokens"},
				OverflowMetrics:   []string{"inputTokens", "toolCalls", "totalTokens"},
			}
			if !reflect.DeepEqual(observed, want) {
				t.Fatalf("workers %v, tools %v:\n got %+v\nwant %+v", workers, tools, observed, want)
			}
		}
	}
}
