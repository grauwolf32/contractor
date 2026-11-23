package reporting

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
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
// TestToolArgumentsRejectNumbersOutsideTheirRange pins LIM-OBS-16 on the
// literal as received, in both message types that carry a summary: a number is
// zero or its leading digit's decimal exponent lies between -7 and 19, and it
// writes no exponent outside that range. The written-exponent rule is the
// Server's alone, since the Runtime never writes such a form.
func TestToolArgumentsRejectNumbersOutsideTheirRange(t *testing.T) {
	t.Parallel()

	accepted := []string{
		"0", "-0.0", "0e-7", "0E+19", "1e-7", "-1e-07", "1E+0019", "0.0000001", "1.0e-7",
		"9.999999999999998e+19", "99999999999999999999", "-99999999999999999999",
		"0.00012e3", "0.00000000000001e7", "123456789012345678.9", `[1,{"nested":2.5e-7}]`,
	}
	rejected := []string{
		"1e-8", "-1e-08", "0.00000001", "1e+20", "100000000000000000000", "10e19",
		"0e-8", "0e20", "1e-400", "0e-16383", "1e999999", "5e-324", "1.7976931348623157e+308",
		"0.000000000000001e7", `[1,{"nested":1e+20}]`,
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

// TestToolArgumentsRejectRepeatedMembersItself pins that the summary rejects
// a repeated member name at any depth and data after the value even when its
// caller decodes with plain encoding/json.
func TestToolArgumentsRejectRepeatedMembersItself(t *testing.T) {
	t.Parallel()

	for _, raw := range []string{
		`{"path":"a","path":"b"}`,
		`{"options":{"depth":1,"depth":2}}`,
		`{"items":[{"name":"a","name":"b"}]}`,
	} {
		var arguments ToolArguments
		if err := json.Unmarshal([]byte(raw), &arguments); !errors.Is(err, strictjson.ErrDuplicateKey) {
			t.Errorf("json.Unmarshal(%s) error = %v, want ErrDuplicateKey", raw, err)
		}
	}
	var arguments ToolArguments
	if err := arguments.UnmarshalJSON([]byte(`{"path":"a"} {"path":"b"}`)); !errors.Is(err, strictjson.ErrTrailingData) {
		t.Errorf("a second value: error = %v, want ErrTrailingData", err)
	}
	if err := json.Unmarshal([]byte(`{"path":"a","size":1e+16}`), &arguments); err != nil || arguments["size"] != json.Number("1e+16") {
		t.Fatalf("a valid summary decoded as %#v (%v)", arguments, err)
	}
}

func TestSharedCallWindowUsageStaysCumulativeAcrossStateAndReports(t *testing.T) {
	t.Parallel()
	state, err := contracts.DecodePrivateStrict[AgentStateSnapshot](
		contractstest.ReadFixture(t, "valid", "agent-state-compacted-tool-budget.json"),
	)
	if err != nil {
		t.Fatal(err)
	}
	final, err := contracts.DecodePrivateStrict[AllocationFinalResponse](
		contractstest.ReadFixture(t, "valid", "allocation-final-response-metric-integer-bound.json"),
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, dimension := range []string{"model_calls", "tool_calls"} {
		t.Run(dimension, func(t *testing.T) {
			live := *state.State.Metrics.WorkerBudget
			retained := *final.Report.Worker.Metrics.WorkerBudget
			live.Exhausted, retained.Exhausted = &dimension, &dimension
			metrics := ExecutionMetrics{Tools: map[string]ToolMetrics{}, WorkerBudget: &retained}
			if err := live.validate(); err != nil {
				t.Fatalf("cumulative live usage rejected: %v", err)
			}
			if err := metrics.Validate(); err != nil {
				t.Fatalf("cumulative retained usage rejected: %v", err)
			}
			if dimension == "model_calls" {
				live.ObservedModelCalls = live.MaxModelCalls - 1
				retained.ObservedModelCalls = retained.MaxModelCalls - 1
			} else {
				live.ObservedToolCalls = live.MaxToolCalls - 1
				retained.ObservedToolCalls = retained.MaxToolCalls - 1
			}
			if live.validate() == nil || metrics.Validate() == nil {
				t.Fatal("exhaustion below a full window was accepted")
			}
		})
	}
}
