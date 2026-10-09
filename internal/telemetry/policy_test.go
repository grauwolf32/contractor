package telemetry

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/reporting"
)

const recognizableSecret = "telemetry-secret-that-must-not-survive"

func TestNormalizeAllocationResourcesHonorsPinnedPolicy(t *testing.T) {
	invalid := reporting.ResourceInvalidReport
	for _, test := range []struct {
		name   string
		policy reporting.PerformanceCollectionPolicy
		report reporting.RuntimeReport
		want   *reporting.ResourceReason
	}{
		{name: "unrequested", policy: reporting.PerformanceCollectionUnsupported, report: reporting.RuntimeReport{Resources: &reporting.RuntimeResources{Version: 1, Scope: "runtime_process", Status: reporting.ResourcePartial}}},
		{name: "malformed", policy: reporting.PerformanceCollectionRequested, report: reporting.RuntimeReport{ResourcesError: &invalid}, want: &invalid},
	} {
		t.Run(test.name, func(t *testing.T) {
			normalizeAllocationResources(&test.report, test.policy)
			if test.want == nil {
				if test.report.Resources != nil || test.report.ResourcesError != nil {
					t.Fatalf("unrequested resources survived: %+v", test.report)
				}
				return
			}
			if test.report.Resources == nil || test.report.Resources.Reason == nil || *test.report.Resources.Reason != *test.want || test.report.ResourcesError != nil {
				t.Fatalf("malformed resource projection = %+v", test.report)
			}
		})
	}
}

func TestMetricsPolicyKeepsCompactUTF8ArgumentsAndReports(t *testing.T) {
	path := strings.Repeat("<>&\u2028\u2029", 300)
	arguments := map[string]any{"path": path}
	if size := encodedSize(arguments); size != 2711 {
		t.Fatalf("compact argument size = %d, want 2711", size)
	}
	worker := reporting.ExecutionReport{
		ReportID: "worker-size", Complete: true,
		Metrics:   reporting.ExecutionMetrics{Tools: map[string]reporting.ToolMetrics{}},
		ToolCalls: make([]reporting.ToolCallRecord, 0, 150), Errors: []reporting.ExecutionError{},
	}
	for index := range 150 {
		worker.ToolCalls = append(worker.ToolCalls, reporting.ToolCallRecord{
			CallID: fmt.Sprintf("call-%d", index), Tool: "read_source_file",
			Arguments: arguments, Outcome: reporting.ToolCallSucceeded,
		})
	}
	now := time.Now().UTC()
	source := reporting.AllocationFinalReport{
		ReportID: "allocation-size", AllocationID: "allocation-size",
		StartedAt: now.Add(-time.Second), FinishedAt: now,
		Worker: worker, Runtime: reporting.RuntimeReport{Complete: true},
	}
	normalized, err := NewPolicy().NormalizeAllocationReport(source)
	if err != nil {
		t.Fatalf("valid Runtime report was rejected by the persistence policy: %v", err)
	}
	if len(normalized.Worker.ToolCalls) != 150 || normalized.Worker.Truncated {
		t.Fatalf("valid compact report was truncated: calls=%d truncated=%t", len(normalized.Worker.ToolCalls), normalized.Worker.Truncated)
	}
	for _, call := range normalized.Worker.ToolCalls {
		if call.ArgumentsTruncated || call.Arguments["path"] != path {
			t.Fatalf("valid compact arguments were changed: %+v", call)
		}
	}
}

func TestMetricsPolicyBoundsDetailPreservesAggregatesAndRedactsSecrets(t *testing.T) {
	calls, succeeded, failed := int64(1005), int64(804), int64(201)
	report := reporting.ExecutionReport{
		ReportID: "worker-allocation-1", Complete: true,
		Metrics: reporting.ExecutionMetrics{
			ModelCalls: int64Ptr(2), InputTokens: int64Ptr(11), OutputTokens: int64Ptr(7),
			TotalTokens: int64Ptr(18),
			Tools: map[string]reporting.ToolMetrics{
				"probe": {Calls: &calls, Succeeded: &succeeded, Failed: &failed},
			},
		},
		ToolCalls: make([]reporting.ToolCallRecord, 0, MaxToolRecords+5),
		Errors:    make([]reporting.ExecutionError, 0, MaxErrorRecords+5),
	}
	for index := 0; index < MaxToolRecords+5; index++ {
		report.ToolCalls = append(report.ToolCalls, reporting.ToolCallRecord{
			CallID: fmt.Sprintf("call-%04d", index), Tool: "probe",
			Arguments: map[string]any{
				"authorization":             "Bearer " + recognizableSecret,
				"accessToken":               "opaque credential",
				"contentBase64":             "encoded artifact bytes",
				"callback":                  "https://user:password@example.test/path?token=secret",
				"value":                     strings.Repeat("x", 5000),
				"key-" + recognizableSecret: "secret-bearing key",
			},
			Outcome: reporting.ToolCallSucceeded,
		})
	}
	for index := 0; index < MaxErrorRecords+5; index++ {
		report.Errors = append(report.Errors, reporting.ExecutionError{
			Code: "provider_error", Message: fmt.Sprintf("error %d: %s", index, recognizableSecret),
		})
	}

	normalized, err := NewPolicy(recognizableSecret).NormalizeExecutionReport(
		report, MaxReportJSONBytes,
	)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(normalized)
	if err != nil {
		t.Fatal(err)
	}
	if len(encoded) > MaxReportJSONBytes || strings.Contains(string(encoded), recognizableSecret) {
		t.Fatalf("unsafe normalized report size=%d secret=%t", len(encoded), strings.Contains(string(encoded), recognizableSecret))
	}
	if !normalized.Truncated || len(normalized.ToolCalls) > MaxToolRecords ||
		len(normalized.Errors) != MaxErrorRecords {
		t.Fatalf("unbounded normalized report: calls=%d errors=%d truncated=%t", len(normalized.ToolCalls), len(normalized.Errors), normalized.Truncated)
	}
	if normalized.Metrics.Tools["probe"].Calls == nil ||
		*normalized.Metrics.Tools["probe"].Calls != 1005 {
		t.Fatalf("aggregate calls changed: %+v", normalized.Metrics.Tools["probe"])
	}
	last := normalized.ToolCalls[len(normalized.ToolCalls)-1]
	if last.CallID != "call-1004" || !last.ArgumentsTruncated || encodedSize(last.Arguments) > MaxArgumentBytes {
		t.Fatalf("last bounded call = %+v", last)
	}
}

func TestMetricsPolicyRedactsDerivedSensitiveKeys(t *testing.T) {
	report := reporting.ExecutionReport{
		ReportID: "worker-derived-keys", Complete: true,
		Metrics: reporting.ExecutionMetrics{Tools: map[string]reporting.ToolMetrics{}},
		ToolCalls: []reporting.ToolCallRecord{{
			CallID: "call-derived-keys", Tool: "probe",
			Arguments: map[string]any{
				"accessToken":   "opaque credential",
				"contentBase64": "encoded artifact bytes",
			},
			Outcome: reporting.ToolCallSucceeded,
		}},
		Errors: []reporting.ExecutionError{},
	}
	normalized, err := NewPolicy().NormalizeExecutionReport(report, MaxReportJSONBytes)
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"accessToken", "contentBase64"} {
		value, ok := normalized.ToolCalls[0].Arguments[key].(map[string]any)
		if !ok || value["redacted"] != true {
			t.Fatalf("sensitive argument %q was not redacted: %#v", key, value)
		}
	}
}

func TestMetricsPolicyPreservesAndClonesWorkerBudget(t *testing.T) {
	exhausted := "tool_calls"
	report := reporting.ExecutionReport{
		ReportID: "worker-budget", Complete: true,
		Metrics: reporting.ExecutionMetrics{
			Tools: map[string]reporting.ToolMetrics{},
			WorkerBudget: &reporting.WorkerBudgetMetrics{
				MaxModelCalls: 8, MaxToolCalls: 2, MaxTotalTokens: 32768,
				ObservedModelCalls: 3, ObservedToolCalls: 2, ObservedTotalTokens: 30,
				Exhausted: &exhausted,
			},
		},
		ToolCalls: []reporting.ToolCallRecord{}, Errors: []reporting.ExecutionError{},
	}
	normalized, err := NewPolicy().NormalizeExecutionReport(report, MaxReportJSONBytes)
	if err != nil {
		t.Fatal(err)
	}
	report.Metrics.WorkerBudget.MaxToolCalls = 99
	*report.Metrics.WorkerBudget.Exhausted = "model_calls"
	if normalized.Metrics.WorkerBudget == nil ||
		normalized.Metrics.WorkerBudget.MaxToolCalls != 2 ||
		normalized.Metrics.WorkerBudget.Exhausted == nil ||
		*normalized.Metrics.WorkerBudget.Exhausted != "tool_calls" {
		t.Fatalf("normalized Worker budget was aliased: %+v", normalized.Metrics.WorkerBudget)
	}
}

func TestMetricsPolicyPreservesAndClonesWorkerSummarizer(t *testing.T) {
	report := reporting.ExecutionReport{
		ReportID: "worker-summary", Complete: true,
		Metrics: reporting.ExecutionMetrics{
			Tools: map[string]reporting.ToolMetrics{},
			Summarizer: &reporting.WorkerSummarizerMetrics{
				Attempts: 1, Failed: 1, ModelCalls: 1, TokenUsageUnavailable: 1,
				FailureCodes: map[string]uint64{"gateway_unavailable": 1},
			},
		},
		ToolCalls: []reporting.ToolCallRecord{}, Errors: []reporting.ExecutionError{},
	}
	normalized, err := NewPolicy().NormalizeExecutionReport(report, MaxReportJSONBytes)
	if err != nil {
		t.Fatal(err)
	}
	report.Metrics.Summarizer.Attempts = 99
	report.Metrics.Summarizer.FailureCodes["gateway_unavailable"] = 99
	if normalized.Metrics.Summarizer == nil ||
		normalized.Metrics.Summarizer.Attempts != 1 ||
		normalized.Metrics.Summarizer.FailureCodes["gateway_unavailable"] != 1 {
		t.Fatalf("normalized Worker summarizer was aliased: %+v", normalized.Metrics.Summarizer)
	}
}

func TestAllocationPolicyRedactsRuntimeAndSummaryExposesOnlyAggregates(t *testing.T) {
	modelCalls, toolCalls, toolFailures := int64(3), int64(4), int64(1)
	stopReason := "gateway returned " + recognizableSecret
	now := time.Now().UTC()
	source := reporting.AllocationFinalReport{
		ReportID: "allocation-final-1", AllocationID: "allocation-1",
		StartedAt: now.Add(-time.Second), FinishedAt: now,
		Worker: reporting.ExecutionReport{
			ReportID: "worker-1", Complete: true,
			Metrics: reporting.ExecutionMetrics{
				ModelCalls: &modelCalls,
				Tools: map[string]reporting.ToolMetrics{
					"write": {Calls: &toolCalls, Failed: &toolFailures},
				},
			},
			ToolCalls: []reporting.ToolCallRecord{},
			Errors: []reporting.ExecutionError{{
				Code: "gateway_error", Message: "provider " + recognizableSecret,
			}},
		},
		Runtime: reporting.RuntimeReport{Complete: true, StopReason: &stopReason},
	}
	normalized, err := NewPolicy(recognizableSecret).NormalizeAllocationReport(source)
	if err != nil {
		t.Fatal(err)
	}
	metrics := BuildStageMetrics(nil, map[string]reporting.AllocationFinalReport{"builder": normalized})
	summary := Summarize(metrics)
	encoded, _ := json.Marshal(summary)
	if strings.Contains(string(encoded), recognizableSecret) || strings.Contains(string(encoded), "gateway_error") {
		t.Fatalf("public summary contains diagnostic detail: %s", encoded)
	}
	if !summary.ReportsComplete || summary.ModelCalls != 3 || summary.ToolCalls != 4 ||
		summary.ToolFailures != 1 || summary.ErrorCount != 1 {
		t.Fatalf("summary = %+v", summary)
	}
	allocationJSON, _ := json.Marshal(normalized)
	if strings.Contains(string(allocationJSON), recognizableSecret) {
		t.Fatalf("allocation report contains configured secret: %s", allocationJSON)
	}
}

func int64Ptr(value int64) *int64 { return &value }
