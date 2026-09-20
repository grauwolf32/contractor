package evalservice

import (
	"encoding/json"
	"slices"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestEvalLegacyPassthroughCountersPreserveWorkerCompleteness(t *testing.T) {
	for _, tc := range []struct {
		name             string
		planner          config.PlannerRef
		missingWorker    bool
		unavailableUsage int
		incompleteReport bool
		wantIncomplete   bool
	}{
		{name: "legacy passthrough", planner: config.PlannerRef{PlannerID: "passthrough", Version: "1"}},
		{name: "unknown planner", wantIncomplete: true},
		{name: "model planner", planner: config.PlannerRef{PlannerID: "streamline", Version: "1"}, wantIncomplete: true},
		{name: "unknown passthrough version", planner: config.PlannerRef{PlannerID: "passthrough", Version: "2"}, wantIncomplete: true},
		{name: "missing Worker counter", planner: config.PlannerRef{PlannerID: "passthrough", Version: "1"}, missingWorker: true, wantIncomplete: true},
		{name: "unavailable Worker usage", planner: config.PlannerRef{PlannerID: "passthrough", Version: "1"}, unavailableUsage: 1, wantIncomplete: true},
		{name: "incomplete report", planner: config.PlannerRef{PlannerID: "passthrough", Version: "1"}, incompleteReport: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls, input, output, total := int64(3), int64(100), int64(20), int64(120)
			worker := contracts.ExecutionMetrics{
				ModelCalls: &calls, InputTokens: &input, OutputTokens: &output, TotalTokens: &total,
				Tools: map[string]contracts.ToolMetrics{},
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
