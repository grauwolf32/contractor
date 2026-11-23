package telemetry

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

func TestSharedToolDiagnosticValuesRemainExact(t *testing.T) {
	var fixture struct {
		Cases []struct {
			Value    string
			Accepted bool
		}
		ArgumentsJSON string
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "public", "tool-diagnostic-values.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	for _, item := range fixture.Cases {
		t.Run(item.Value, func(t *testing.T) {
			var report reporting.ExecutionReport
			data := fmt.Appendf(nil, `{"reportId":"private","complete":true,"metrics":{"tools":{}},"toolCalls":[{"callId":"private-call","tool":"probe","outcome":"succeeded","argumentsTruncated":false,"durationMs":%s}],"errors":[],"truncated":false}`, item.Value)
			valid := strictjson.DecodeNonNull(data, &report) == nil && report.Validate() == nil
			if valid != item.Accepted {
				t.Fatalf("accepted=%v want %v", valid, item.Accepted)
			}
			if valid {
				got := projectToolCallDiagnostic(ToolDiagnosticParams{StageExecutionID: "stage", Participant: "planner"}, "generation", 0, report.ToolCalls[0])
				if got.DurationMS == nil || *got.DurationMS != item.Value {
					t.Fatalf("rounded: %+v", got)
				}
			}
		})
	}
	var args reporting.ToolArguments
	if err := strictjson.DecodeNonNull([]byte(fixture.ArgumentsJSON), &args); err != nil {
		t.Fatal(err)
	}
	got := projectToolCallDiagnostic(ToolDiagnosticParams{StageExecutionID: "stage", Participant: "planner"}, "generation", 0, reporting.ToolCallRecord{CallID: "private", Tool: "probe", Arguments: args, Outcome: reporting.ToolCallSucceeded})
	if got.ArgumentsJSON == nil || *got.ArgumentsJSON != fixture.ArgumentsJSON {
		t.Fatalf("argument literals changed: %+v", got)
	}
}

func TestToolDiagnosticProjectionPrivacyBoundsAndUTF8(t *testing.T) {
	call := reporting.ToolCallRecord{CallID: strings.Repeat("private-call", 100000), Tool: strings.Repeat("tool", 260000), Outcome: reporting.ToolCallFailed,
		Arguments: reporting.ToolArguments{"nested": map[string]any{"authorization": "secret-sentinel", "location": "https://url-sentinel.invalid"}, "large": json.Number("9007199254740993")},
		Error:     &reporting.ExecutionError{Code: "https://code-sentinel.invalid", Message: strings.Repeat("a://b ", 680)}}
	got := projectToolCallDiagnostic(ToolDiagnosticParams{StageExecutionID: "stage", Participant: "worker", LogicalAgent: "Scanner.v2"}, "generation", 0, call)
	data, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.CallReference) != 64 || !got.ToolTruncated || got.DurationMS != nil || got.ResultSizeBytes != nil || got.Error == nil || !got.Error.MessageTruncated || len(got.Error.Message) > 4096 || got.ArgumentsJSON == nil || !got.ArgumentsTruncated {
		t.Fatalf("missing bounded facts: %+v", got)
	}
	for _, sentinel := range []string{"private-call", "secret-sentinel", "url-sentinel.invalid", "code-sentinel.invalid"} {
		if strings.Contains(string(data), sentinel) {
			t.Fatalf("private content leaked: %s", sentinel)
		}
	}
	if len(data) > MaxDiagnosticPageBytes-4096 {
		t.Fatalf("one record cannot advance: %d", len(data))
	}
	args := map[string]any{"line": "\u2028\u2029", "literal": `\u2028`, "html": "<tag>"}
	text := diagnosticArgumentJSON(args)
	size, err := contracts.ResultJSONSize(args)
	if err != nil {
		t.Fatal(err)
	}
	if len(text) != size || !strings.Contains(text, "<tag>") || !strings.Contains(text, `\\u2028`) || !strings.Contains(text, "\u2028") {
		t.Fatalf("shared UTF8 measure changed: %q size %d", text, size)
	}
}

func TestToolDiagnosticRetainsAnExistingArgumentSummary(t *testing.T) {
	call := reporting.ToolCallRecord{CallID: "private", Tool: "probe", Outcome: reporting.ToolCallSucceeded, ArgumentsTruncated: true, Arguments: reporting.ToolArguments{"summary": "[TRUNCATED]", "originalSizeBytes": json.Number("8192")}}
	got := projectToolCallDiagnostic(ToolDiagnosticParams{StageExecutionID: "stage", Participant: "planner"}, "generation", 0, call)
	if got.ArgumentsJSON == nil || *got.ArgumentsJSON != `{"originalSizeBytes":8192,"summary":"[TRUNCATED]"}` || !got.ArgumentsTruncated {
		t.Fatalf("existing safe summary changed: %+v", got)
	}
}
