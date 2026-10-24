package reporting

import (
	"encoding/json"
	"errors"
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
// TestToolArgumentsRejectNumbersOutsideTheFloat64Range pins LIM-OBS-16: a
// number is finite as a float64 and writes an exponent between -324 and 308,
// at any depth of the summary and in both message types that carry one.
func TestToolArgumentsRejectNumbersOutsideTheFloat64Range(t *testing.T) {
	t.Parallel()

	accepted := []string{
		"5e-324", "-5e-324", "0e-324", "1e308", "1E+0308", "0e308", "-0.0",
		"1.7976931348623157e+308", "1180591620717411303424", "1" + strings.Repeat("0", 308),
		`[1,{"nested":2.5e-300}]`,
	}
	rejected := []string{
		"1e999999", "-1e400", "1.7976931348623159e308", "1" + strings.Repeat("0", 309),
		"0e-325", "1e-400", "1e-20000", "0e-16383", "0e309", "0e2147483648",
		`[1,{"nested":1e999999}]`,
	}
	validate := func(literal string) []error {
		record := `{"callId":"tool-1","tool":"rank","arguments":{"value":` + literal +
			`},"argumentsTruncated":false,"outcome":"succeeded"}`
		var call ToolCallRecord
		var state WorkerStateToolCall
		errs := []error{
			contracts.DecodePrivate([]byte(record), &call),
			contracts.DecodePrivate([]byte(record), &state),
		}
		if errs[0] == nil && errs[1] == nil {
			report := ExecutionReport{
				ReportID: "worker-1", Complete: true,
				Metrics:   ExecutionMetrics{Tools: map[string]ToolMetrics{}},
				ToolCalls: []ToolCallRecord{call}, Errors: []ExecutionError{},
			}
			errs = []error{report.Validate(), validateStateToolCall(state)}
		}
		return errs
	}
	for _, literal := range accepted {
		for _, err := range validate(literal) {
			if err != nil {
				t.Errorf("%.24s was rejected: %v", literal, err)
			}
		}
	}
	for _, literal := range rejected {
		for _, err := range validate(literal) {
			if !errors.Is(err, contracts.ErrValidation) {
				t.Errorf("%.24s error = %v, want an invariant violation", literal, err)
			}
		}
	}
	// Invalid JSON numbers stay a decoding error.
	for _, literal := range []string{"NaN", "01", "-01"} {
		var call ToolCallRecord
		err := contracts.DecodePrivate([]byte(`{"callId":"tool-1","tool":"rank","arguments":{"value":`+
			literal+`},"argumentsTruncated":false,"outcome":"succeeded"}`), &call)
		var private *contracts.PrivateProtocolError
		if !errors.As(err, &private) || private.Class != contracts.PrivateProtocolErrorSchema {
			t.Errorf("%s error = %v, want a schema rejection", literal, err)
		}
	}
}

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
