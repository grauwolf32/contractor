package reporting

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
)

func TestSharedAllocationRuntimeAdapterMetricsAreTypedAndBounded(t *testing.T) {
	t.Parallel()

	value, err := contracts.DecodePrivateStrict[AllocationFinalResponse](
		contractstest.ReadFixture(t, "valid", "allocation-final-response.json"),
	)
	if err != nil {
		t.Fatal(err)
	}
	value.Report.Runtime.Adapters = map[runtimesettings.RuntimeAdapterRef]RuntimeAdapterMetrics{
		runtimesettings.RuntimeAdapterOTLPHTTP: {Operations: 2, FailedOperations: 1},
	}
	if err := value.Validate(); err != nil {
		t.Fatalf("valid adapter metrics were rejected: %v", err)
	}
	value.Report.Runtime.Adapters[runtimesettings.RuntimeAdapterOTLPHTTP] = RuntimeAdapterMetrics{
		Operations: 1, FailedOperations: 2,
	}
	if err := value.Validate(); err == nil {
		t.Fatal("adapter metrics with failures above operations were accepted")
	}
	value.Report.Runtime.Adapters = map[runtimesettings.RuntimeAdapterRef]RuntimeAdapterMetrics{
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

func TestSharedMalformedRuntimeAdapterMetricsBecomeIncompleteInsteadOfBlockingReport(t *testing.T) {
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
	value, err := contracts.DecodePrivateStrict[AllocationFinalResponse](encoded)
	if err != nil {
		t.Fatalf("semantic final report was blocked by optional adapter metrics: %v", err)
	}
	if value.Report.Runtime.Complete || len(value.Report.Runtime.Adapters) != 0 {
		t.Fatalf("malformed metrics were retained: %+v", value.Report.Runtime)
	}
}

// The Runtime bounds an argument summary by its own encoding, which writes a
// number as Python does (1e+16, 5e-05, -0.0, 1.0); the Server keeps those
// literals, so a summary at the 4096-byte bound stays within it and integers
// beyond 2^53 keep their value (OBS-R42).
func TestToolArgumentsKeepTheRuntimeNumberLiterals(t *testing.T) {
	t.Parallel()

	prefix := `{"large":1e+16,"small":5e-05,"negativeZero":-0.0,"integral":1.0,"integer":9007199254740993,"pad":"`
	arguments := prefix + strings.Repeat("a", 4096-len(prefix)-2) + `"}`
	record := `{"callId":"tool-1","tool":"rank","arguments":` + arguments +
		`,"argumentsTruncated":false,"outcome":"succeeded"}`
	report, err := contracts.DecodePrivateStrict[ExecutionReport]([]byte(
		`{"reportId":"worker-1","complete":true,"metrics":{"tools":{}},"toolCalls":[` + record +
			`],"errors":[],"truncated":false}`))
	if err != nil {
		t.Fatalf("a summary at the bound was rejected: %v", err)
	}
	decoded := report.ToolCalls[0].Arguments
	if size, err := contracts.ResultJSONSize(decoded); err != nil || size != 4096 {
		t.Fatalf("summary measures %d bytes (%v), want 4096", size, err)
	}
	for key, want := range map[string]json.Number{
		"large": "1e+16", "small": "5e-05", "negativeZero": "-0.0", "integral": "1.0", "integer": "9007199254740993",
	} {
		if decoded[key] != want {
			t.Fatalf("%s = %#v, want literal %s", key, decoded[key], want)
		}
	}

	var state WorkerStateToolCall
	if err := contracts.DecodePrivate([]byte(record), &state); err != nil {
		t.Fatal(err)
	}
	if err := validateStateToolCall(state); err != nil || state.Arguments["large"] != json.Number("1e+16") {
		t.Fatalf("Worker State arguments = %#v (%v)", state.Arguments, err)
	}

	var empty ToolCallRecord
	if err := contracts.DecodePrivate([]byte(`{"callId":"tool-2","tool":"list","arguments":{},"argumentsTruncated":false,"outcome":"succeeded"}`), &empty); err != nil {
		t.Fatal(err)
	}
	if encoded, err := json.Marshal(empty); err != nil || strings.Contains(string(encoded), "arguments\"") {
		t.Fatalf("an empty summary encodes as %s (%v), want it absent", encoded, err)
	}
	var null ToolCallRecord
	if err := contracts.DecodePrivate([]byte(`{"callId":"tool-3","tool":"list","arguments":null,"argumentsTruncated":false,"outcome":"succeeded"}`), &null); err == nil {
		t.Fatal("null arguments were accepted")
	}
}
