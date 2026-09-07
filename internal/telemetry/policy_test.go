package telemetry

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

const recognizableSecret = "telemetry-secret-that-must-not-survive"

func TestNormalizeAllocationResourcesHonorsPinnedPolicy(t *testing.T) {
	invalid := contracts.ResourceInvalidReport
	for _, test := range []struct {
		name   string
		policy contracts.PerformanceCollectionPolicy
		report contracts.RuntimeReport
		want   *contracts.ResourceReason
	}{
		{name: "unrequested", policy: contracts.PerformanceCollectionUnsupported, report: contracts.RuntimeReport{Resources: &contracts.RuntimeResources{Version: 1, Scope: "runtime_process", Status: contracts.ResourcePartial}}},
		{name: "malformed", policy: contracts.PerformanceCollectionRequested, report: contracts.RuntimeReport{ResourcesError: &invalid}, want: &invalid},
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

func TestMetricsPolicyBoundsDetailPreservesAggregatesAndRedactsSecrets(t *testing.T) {
	calls, succeeded, failed := int64(1005), int64(804), int64(201)
	report := contracts.ExecutionReport{
		ReportID: "worker-allocation-1", Complete: true,
		Metrics: contracts.ExecutionMetrics{
			ModelCalls: int64Ptr(2), InputTokens: int64Ptr(11), OutputTokens: int64Ptr(7),
			TotalTokens: int64Ptr(18),
			Tools: map[string]contracts.ToolMetrics{
				"probe": {Calls: &calls, Succeeded: &succeeded, Failed: &failed},
			},
		},
		ToolCalls: make([]contracts.ToolCallRecord, 0, MaxToolRecords+5),
		Errors:    make([]contracts.ExecutionError, 0, MaxErrorRecords+5),
	}
	for index := 0; index < MaxToolRecords+5; index++ {
		report.ToolCalls = append(report.ToolCalls, contracts.ToolCallRecord{
			CallID: fmt.Sprintf("call-%04d", index), Tool: "probe",
			Arguments: map[string]any{
				"authorization":             "Bearer " + recognizableSecret,
				"accessToken":               "opaque credential",
				"contentBase64":             "encoded artifact bytes",
				"callback":                  "https://user:password@example.test/path?token=secret",
				"value":                     strings.Repeat("x", 5000),
				"key-" + recognizableSecret: "secret-bearing key",
			},
			Outcome: contracts.ToolCallSucceeded,
		})
	}
	for index := 0; index < MaxErrorRecords+5; index++ {
		report.Errors = append(report.Errors, contracts.ExecutionError{
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
	report := contracts.ExecutionReport{
		ReportID: "worker-derived-keys", Complete: true,
		Metrics: contracts.ExecutionMetrics{Tools: map[string]contracts.ToolMetrics{}},
		ToolCalls: []contracts.ToolCallRecord{{
			CallID: "call-derived-keys", Tool: "probe",
			Arguments: map[string]any{
				"accessToken":   "opaque credential",
				"contentBase64": "encoded artifact bytes",
			},
			Outcome: contracts.ToolCallSucceeded,
		}},
		Errors: []contracts.ExecutionError{},
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
	report := contracts.ExecutionReport{
		ReportID: "worker-budget", Complete: true,
		Metrics: contracts.ExecutionMetrics{
			Tools: map[string]contracts.ToolMetrics{},
			WorkerBudget: &contracts.WorkerBudgetMetrics{
				MaxModelCalls: 8, MaxToolCalls: 2, MaxTotalTokens: 32768,
				ObservedModelCalls: 3, ObservedToolCalls: 2, ObservedTotalTokens: 30,
				Exhausted: &exhausted,
			},
		},
		ToolCalls: []contracts.ToolCallRecord{}, Errors: []contracts.ExecutionError{},
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
	report := contracts.ExecutionReport{
		ReportID: "worker-summary", Complete: true,
		Metrics: contracts.ExecutionMetrics{
			Tools: map[string]contracts.ToolMetrics{},
			Summarizer: &contracts.WorkerSummarizerMetrics{
				Attempts: 1, Failed: 1, ModelCalls: 1, TokenUsageUnavailable: 1,
				FailureCodes: map[string]uint64{"gateway_unavailable": 1},
			},
		},
		ToolCalls: []contracts.ToolCallRecord{}, Errors: []contracts.ExecutionError{},
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
	source := contracts.AllocationFinalReport{
		ReportID: "allocation-final-1", AllocationID: "allocation-1",
		StartedAt: now.Add(-time.Second), FinishedAt: now,
		Worker: contracts.ExecutionReport{
			ReportID: "worker-1", Complete: true,
			Metrics: contracts.ExecutionMetrics{
				ModelCalls: &modelCalls,
				Tools: map[string]contracts.ToolMetrics{
					"write": {Calls: &toolCalls, Failed: &toolFailures},
				},
			},
			ToolCalls: []contracts.ToolCallRecord{},
			Errors: []contracts.ExecutionError{{
				Code: "gateway_error", Message: "provider " + recognizableSecret,
			}},
		},
		Runtime: contracts.RuntimeReport{Complete: true, StopReason: &stopReason},
	}
	normalized, err := NewPolicy(recognizableSecret).NormalizeAllocationReport(source)
	if err != nil {
		t.Fatal(err)
	}
	metrics := BuildStageMetrics(nil, map[string]contracts.AllocationFinalReport{"builder": normalized})
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
