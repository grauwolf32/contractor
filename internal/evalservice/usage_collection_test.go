package evalservice

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/evaldomain"
)

// validMetricSnapshot is a projection as metricSnapshotsQuery builds it from
// a stored StageMetrics document that passed validation on write.
const validMetricSnapshot = `{"planner":{"complete":true,"truncated":false,"metrics":{` +
	`"modelCalls":1,"inputTokens":2,"outputTokens":3,"totalTokens":5,` +
	`"tools":{"read":{"calls":1,"succeeded":1,"failed":0}}}},` +
	`"workers":{"builder":{"complete":true,"truncated":false,"metrics":{"modelCalls":4,"tools":{}}}},` +
	`"runtime":{"builder":{"complete":true}}}`

// A stored projection that breaks the contract its reports were validated
// against on write is a data-integrity failure of collection, never a missing
// or zero counter: a duplicate or unknown member, a missing or null one, a
// counter outside 0..2^53-1 and any other metric rule (EVM-R81, OBS-R110,
// OBS-R111).
func TestEvalAttemptMetricsRejectStoredMetricsThatBreakTheirContract(t *testing.T) {
	if _, err := attemptMetrics("run", "stage", []byte(validMetricSnapshot), 1); err != nil {
		t.Fatalf("valid projection: %v", err)
	}
	replace := func(old, new string) string {
		t.Helper()
		if !strings.Contains(validMetricSnapshot, old) {
			t.Fatalf("fixture has no %s", old)
		}
		return strings.Replace(validMetricSnapshot, old, new, 1)
	}
	plannerHead := `"complete":true,"truncated":false,"metrics":{"modelCalls":1`
	workerReport := `{"complete":true,"truncated":false,"metrics":{"modelCalls":4,"tools":{}}}`
	workerMetrics := `"metrics":{"modelCalls":4,"tools":{}}`
	for _, tc := range []struct{ name, document string }{
		{"duplicate report member", replace(plannerHead, `"complete":true,`+plannerHead)},
		{"duplicate counter", replace(`"modelCalls":1,`, `"modelCalls":1,"modelCalls":1,`)},
		{"duplicate Worker", replace(`"workers":{"builder":`, `"workers":{"builder":`+workerReport+`,"builder":`)},
		{"duplicate snapshot member", replace(`"runtime":`, `"runtime":{},"runtime":`)},
		{"unknown snapshot member", replace(`"runtime":`, `"summary":{},"runtime":`)},
		{"unknown report member", replace(plannerHead, `"prompt":"x",`+plannerHead)},
		{"unknown counter", replace(`"modelCalls":1,`, `"modelCalls":1,"cachedTokens":1,`)},
		{"case alias of a counter", replace(`"modelCalls":1,`, `"ModelCalls":1,`)},
		{"unknown tool member", replace(`"failed":0}`, `"failed":0,"retries":0}`)},
		{"unknown Runtime member", replace(`{"builder":{"complete":true}}`, `{"builder":{"complete":true,"durationMs":1}}`)},
		{"negative model counter", replace(`"modelCalls":1,`, `"modelCalls":-1,`)},
		{"negative token counter", replace(`"totalTokens":5`, `"totalTokens":-5`)},
		{"negative tool counter", replace(`"calls":1,`, `"calls":-1,`)},
		{"counter above 2^53-1", replace(`"totalTokens":5`, `"totalTokens":9007199254740992`)},
		{"counter beyond int64", replace(`"totalTokens":5`, `"totalTokens":9223372036854775808`)},
		{"fractional counter", replace(`"totalTokens":5`, `"totalTokens":5.5`)},
		{"Worker budget beyond its limit", replace(workerMetrics, `"metrics":{"modelCalls":4,"tools":{},"workerBudget":{`+
			`"maxModelCalls":1,"maxToolCalls":1,"maxTotalTokens":1,"observedModelCalls":2,`+
			`"observedToolCalls":0,"observedTotalTokens":0,"tokenUsageUnavailable":0}}`)},
		{"missing Planner member", replace(`"planner":{"complete":true,"truncated":false,"metrics":{`+
			`"modelCalls":1,"inputTokens":2,"outputTokens":3,"totalTokens":5,`+
			`"tools":{"read":{"calls":1,"succeeded":1,"failed":0}}}},`, ``)},
		{"missing Workers", replace(`"workers":{"builder":`+workerReport+`},`, ``)},
		{"missing truncated", replace(`"truncated":false,`+workerMetrics, workerMetrics)},
		{"missing tools", replace(workerMetrics, `"metrics":{"modelCalls":4}`)},
		{"missing Runtime completeness", replace(`{"builder":{"complete":true}}`, `{"builder":{}}`)},
		{"null Workers", replace(`"workers":{"builder":`+workerReport+`}`, `"workers":null`)},
		{"null metrics", replace(workerMetrics, `"metrics":null`)},
		{"null counter", replace(`"modelCalls":4,`, `"modelCalls":null,`)},
		{"null tools", replace(`"tools":{}`, `"tools":null`)},
		{"null completeness", replace(plannerHead, `"complete":null,"truncated":false,"metrics":{"modelCalls":1`)},
		{"blank Worker name", replace(`"workers":{"builder":`, `"workers":{" ":`)},
		{"blank tool name", replace(`"tools":{"read":`, `"tools":{"":`)},
		{"blank Runtime name", replace(`"runtime":{"builder":`, `"runtime":{"\t":`)},
		{"trailing data", validMetricSnapshot + `{}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			observed, err := attemptMetrics("run", "stage", []byte(tc.document), 1)
			if !errors.Is(err, errStoredMetrics) {
				t.Fatalf("attemptMetrics(%s) = (%+v, %v), want an integrity error", tc.document, observed, err)
			}
		})
	}
}

// Every counter is read as reported, whatever the Stage's Planner: a counter
// absent by contract leaves its counter incomplete without hiding the other
// counters (EVM-R79, EVM-R81), and a report that reports zeros contributes
// zeros.
func TestEvalAttemptMetricsReadCountersAsReported(t *testing.T) {
	value := func(v int64) *int64 { return &v }
	for _, tc := range []struct {
		name             string
		plannerCounters  bool
		missingWorker    bool
		unavailableUsage bool
		incompleteReport bool
		wantIncomplete   []string
	}{
		{name: "Planner reports zero model counters", plannerCounters: true},
		{name: "Planner without model counters", wantIncomplete: []string{"inputTokens", "modelCalls", "outputTokens", "totalTokens"}},
		{name: "missing Worker counter", plannerCounters: true, missingWorker: true, wantIncomplete: []string{"totalTokens"}},
		{name: "unavailable Worker usage", plannerCounters: true, unavailableUsage: true, wantIncomplete: []string{"inputTokens", "outputTokens", "totalTokens"}},
		{name: "incomplete Planner report", plannerCounters: true, incompleteReport: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			planner := reporting.ExecutionMetrics{Tools: map[string]reporting.ToolMetrics{
				"a2a.invoke": {Calls: value(1), Succeeded: value(1), Failed: value(0)},
			}}
			if tc.plannerCounters {
				planner.ModelCalls, planner.InputTokens = value(0), value(0)
				planner.OutputTokens, planner.TotalTokens = value(0), value(0)
			}
			worker := reporting.ExecutionMetrics{
				ModelCalls: value(3), InputTokens: value(100), OutputTokens: value(20), TotalTokens: value(120),
				Tools: map[string]reporting.ToolMetrics{},
			}
			if tc.missingWorker {
				worker.TotalTokens = nil
			}
			if tc.unavailableUsage {
				worker.WorkerBudget = &reporting.WorkerBudgetMetrics{
					MaxModelCalls: 10, MaxToolCalls: 10, MaxTotalTokens: 1000,
					ObservedModelCalls: 3, ObservedTotalTokens: 120, TokenUsageUnavailable: 1,
				}
			}
			raw, err := json.Marshal(map[string]any{
				"planner": map[string]any{"complete": !tc.incompleteReport, "truncated": false, "metrics": planner},
				"workers": map[string]any{"worker": map[string]any{"complete": true, "truncated": false, "metrics": worker}},
				"runtime": map[string]any{"worker": map[string]any{"complete": true}},
			})
			if err != nil {
				t.Fatal(err)
			}
			observed, err := attemptMetrics("run", "stage", raw, 1)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(observed.IncompleteMetrics, tc.wantIncomplete) {
				t.Fatalf("incomplete counters = %v, want %v", observed.IncompleteMetrics, tc.wantIncomplete)
			}
			if observed.Metrics["modelCalls"] != 3 || observed.Metrics["toolCalls"] != 1 ||
				!tc.missingWorker && observed.Metrics["totalTokens"] != 120 {
				t.Fatalf("reported counters changed: %+v", observed)
			}
			if observed.ReportsComplete == tc.incompleteReport {
				t.Fatalf("report completeness = %t, want %t", observed.ReportsComplete, !tc.incompleteReport)
			}
		})
	}
}

// A snapshot's counters do not depend on which Worker holds which
// contribution. Every contribution lies within 2^53-1 (OBS-R111), so a sum
// leaves the int64 range only past 1,024 of them: 1,024 at the bound and one
// of 1,024 overflow and have no value wherever the odd one is read, 1,024 at
// the bound and one of 1,023 are exactly math.MaxInt64 and kept, and a
// missing contribution leaves only its own counter incomplete (EVM-R240).
func TestEvalAttemptMetricsOverflowIsOrderIndependent(t *testing.T) {
	const workers = 1025
	value := func(v int64) *int64 { return &v }
	for _, odd := range []int{0, workers / 2, workers - 1} {
		reports := map[string]any{}
		runtime := map[string]any{}
		for i := range workers {
			exact, overflowing, failed := int64(contracts.MaxExactInteger), int64(contracts.MaxExactInteger), int64(0)
			if i == odd {
				exact, overflowing, failed = 1023, 1024, 1
			}
			metrics := reporting.ExecutionMetrics{
				ModelCalls: value(exact), InputTokens: value(overflowing),
				OutputTokens: value(0), TotalTokens: value(overflowing),
				Tools: map[string]reporting.ToolMetrics{"read": {Calls: value(overflowing), Failed: value(failed)}},
			}
			if i == odd {
				metrics.OutputTokens = nil
			}
			name := fmt.Sprintf("worker-%04d", i)
			reports[name] = map[string]any{"complete": true, "truncated": false, "metrics": metrics}
			runtime[name] = map[string]any{"complete": true}
		}
		raw, err := json.Marshal(map[string]any{"planner": nil, "workers": reports, "runtime": runtime})
		if err != nil {
			t.Fatal(err)
		}
		observed, err := attemptMetrics("run", "stage", raw, workers)
		if err != nil {
			t.Fatal(err)
		}
		// The source digest covers the exact bytes, which name contributions differently.
		observed.SourceSHA256 = ""
		want := evaldomain.AttemptObservation{
			RunID: "run", StageExecutionID: "stage", ReportsComplete: true,
			Metrics:           map[string]int64{"modelCalls": math.MaxInt64, "outputTokens": 0, "toolFailures": 1},
			IncompleteMetrics: []string{"outputTokens"},
			OverflowMetrics:   []string{"inputTokens", "toolCalls", "totalTokens"},
		}
		if !reflect.DeepEqual(observed, want) {
			t.Fatalf("odd contribution at Worker %d:\n got %+v\nwant %+v", odd, observed, want)
		}
	}
}
