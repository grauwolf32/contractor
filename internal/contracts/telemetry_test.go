package contracts

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestAllocationRuntimeAdapterMetricsAreTypedAndBounded(t *testing.T) {
	t.Parallel()

	value, err := DecodeStrict[AllocationFinalResponse](
		contractstest.ReadFixture(t, "valid", "allocation-final-response.json"),
	)
	if err != nil {
		t.Fatal(err)
	}
	value.Report.Runtime.Adapters = map[RuntimeAdapterRef]RuntimeAdapterMetrics{
		RuntimeAdapterOTLPHTTP: {Operations: 2, FailedOperations: 1},
	}
	if err := value.Validate(); err != nil {
		t.Fatalf("valid adapter metrics were rejected: %v", err)
	}
	value.Report.Runtime.Adapters[RuntimeAdapterOTLPHTTP] = RuntimeAdapterMetrics{
		Operations: 1, FailedOperations: 2,
	}
	if err := value.Validate(); err == nil {
		t.Fatal("adapter metrics with failures above operations were accepted")
	}
	value.Report.Runtime.Adapters = map[RuntimeAdapterRef]RuntimeAdapterMetrics{
		"unknown@1": {Operations: 1},
	}
	if err := value.Validate(); err == nil {
		t.Fatal("unknown Runtime adapter metric key was accepted")
	}
}

func TestExecutionReportValidatesWorkerSummarizerMetrics(t *testing.T) {
	t.Parallel()

	valid := WorkerSummarizerMetrics{
		Attempts: 2, Succeeded: 1, Failed: 1, ModelCalls: 2,
		InputTokens: 10, OutputTokens: 4, TotalTokens: 14,
		FailureCodes: map[string]uint64{"gateway_unavailable": 1},
	}
	report := ExecutionReport{
		ReportID: "worker-summary", Complete: true,
		Metrics:   ExecutionMetrics{Tools: map[string]ToolMetrics{}, Summarizer: &valid},
		ToolCalls: []ToolCallRecord{}, Errors: []ExecutionError{},
	}
	if err := report.Validate(); err != nil {
		t.Fatalf("valid Worker summarizer metrics were rejected: %v", err)
	}

	tests := map[string]func(*WorkerSummarizerMetrics){
		"zero attempts":        func(value *WorkerSummarizerMetrics) { value.Attempts = 0 },
		"terminal mismatch":    func(value *WorkerSummarizerMetrics) { value.Failed = 0 },
		"too many calls":       func(value *WorkerSummarizerMetrics) { value.ModelCalls = 3 },
		"missing usage excess": func(value *WorkerSummarizerMetrics) { value.TokenUsageUnavailable = 3 },
		"nil failure map":      func(value *WorkerSummarizerMetrics) { value.FailureCodes = nil },
		"failure mismatch":     func(value *WorkerSummarizerMetrics) { value.FailureCodes = map[string]uint64{} },
		"invalid failure code": func(value *WorkerSummarizerMetrics) {
			value.FailureCodes = map[string]uint64{"bad-code": 1}
		},
	}
	for name, mutate := range tests {
		name, mutate := name, mutate
		t.Run(name, func(t *testing.T) {
			candidate := valid
			candidate.FailureCodes = map[string]uint64{"gateway_unavailable": 1}
			mutate(&candidate)
			report.Metrics.Summarizer = &candidate
			if err := report.Validate(); err == nil {
				t.Fatal("inconsistent Worker summarizer metrics were accepted")
			}
		})
	}
}

func TestMalformedRuntimeAdapterMetricsBecomeIncompleteInsteadOfBlockingReport(t *testing.T) {
	t.Parallel()

	var wire map[string]any
	if err := json.Unmarshal(contractstest.ReadFixture(t, "valid", "allocation-final-response.json"), &wire); err != nil {
		t.Fatal(err)
	}
	runtime := wire["report"].(map[string]any)["runtime"].(map[string]any)
	runtime["adapters"] = map[string]any{
		"otlp-http@1": map[string]any{
			"operations": 1, "failedOperations": 2,
		},
		"unknown@1": map[string]any{
			"operations": -1, "failedOperations": 0, "secretField": "must-not-reflect",
		},
	}
	encoded, err := json.Marshal(wire)
	if err != nil {
		t.Fatal(err)
	}
	value, err := DecodeStrict[AllocationFinalResponse](encoded)
	if err != nil {
		t.Fatalf("semantic final report was blocked by optional adapter metrics: %v", err)
	}
	if value.Report.Runtime.Complete || len(value.Report.Runtime.Adapters) != 0 {
		t.Fatalf("malformed metrics were retained: %+v", value.Report.Runtime)
	}
}
