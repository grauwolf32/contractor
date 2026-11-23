package scheduler

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// modelCounters returns the model calls and token counters of metrics.
func modelCounters(metrics reporting.ExecutionMetrics) map[string]*int64 {
	return map[string]*int64{
		"modelCalls": metrics.ModelCalls, "inputTokens": metrics.InputTokens,
		"outputTokens": metrics.OutputTokens, "totalTokens": metrics.TotalTokens,
	}
}

// The incomplete placeholder of a Planner that provided no report states
// zero model usage only for a model-free Planner, whose identity rules out
// model work; a model-backed Planner's counters stay unknown (OBS-R37,
// EVM-R81). No reader derives these zeros later.
func TestPlannerReportPlaceholderStatesZeroModelUsageOnlyForModelFreePlanners(t *testing.T) {
	for _, tc := range []struct {
		planner   resolved.PlannerRef
		modelFree bool
	}{
		{resolved.PlannerRef{PlannerID: "passthrough", Version: "1"}, true},
		{resolved.PlannerRef{PlannerID: "scan-plan", Version: "1"}, true},
		{resolved.PlannerRef{PlannerID: "streamline", Version: "1"}, false},
		{resolved.PlannerRef{PlannerID: "router", Version: "1"}, false},
		{resolved.PlannerRef{PlannerID: "passthrough", Version: "2"}, false},
		{resolved.PlannerRef{}, false},
	} {
		t.Run(tc.planner.PlannerID+"@"+tc.planner.Version, func(t *testing.T) {
			report := plannerReportPlaceholder("session-1", tc.planner)
			if report.ReportID != "planner-missing-session-1" || report.Complete {
				t.Fatalf("placeholder = %+v, want the incomplete placeholder", report)
			}
			if err := report.Validate(); err != nil {
				t.Fatalf("placeholder is invalid: %v", err)
			}
			for name, counter := range modelCounters(report.Metrics) {
				if tc.modelFree && (counter == nil || *counter != 0) {
					t.Fatalf("model-free Planner placeholder %s = %v, want an explicit zero", name, counter)
				}
				if !tc.modelFree && counter != nil {
					t.Fatalf("model-backed Planner placeholder %s = %d, want unknown", name, *counter)
				}
			}
		})
	}
}

// A passthrough@1 Stage whose Planner instance provides no report records the
// placeholder through the Scheduler's report path with the zeros the
// passthrough Planner's own report states.
func TestSchedulerRecordsZeroModelUsageForPassthroughPlaceholder(t *testing.T) {
	harness := newSchedulerHarness(t)
	worked, err := harness.scheduler.RunOnce(t.Context())
	if err != nil || !worked || harness.store.run.State != runstore.RunSucceeded {
		t.Fatalf("Run = (%t, %v), state=%s", worked, err, harness.store.run.State)
	}
	if len(harness.store.plannerReports) != 1 {
		t.Fatalf("Planner reports = %+v", harness.store.plannerReports)
	}
	report := harness.store.plannerReports[0].Report
	if report.ReportID != "planner-missing-session-1" || report.Complete {
		t.Fatalf("Planner report = %+v, want the placeholder of a Planner without a report", report)
	}
	for name, counter := range modelCounters(report.Metrics) {
		if counter == nil || *counter != 0 {
			t.Fatalf("passthrough@1 placeholder %s = %v, want an explicit zero", name, counter)
		}
	}
}
