package telemetry

import (
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/reporting"
)

func TestAttemptDiagnosticsAreDeterministicBoundedAndNewest(t *testing.T) {
	planner := diagnosticReport("planner", 30)
	planner.Truncated = true
	metrics := reporting.StageMetrics{
		Planner: &planner,
		Workers: map[string]reporting.ExecutionReport{
			"zeta":  diagnosticReport("zeta", 100),
			"alpha": diagnosticReport("alpha", 100),
		},
		Runtime: map[string]reporting.RuntimeReport{},
	}

	diagnostics := ProjectAttemptDiagnostics(metrics)

	if !diagnostics.Truncated || len(diagnostics.Items) != MaxAttemptDiagnosticRecords {
		t.Fatalf("bounded diagnostics = %+v", diagnostics)
	}
	first := diagnostics.Items[0]
	if first.Participant != AttemptDiagnosticWorker || first.LogicalAgent != "alpha" ||
		first.Code != "alpha_072" {
		t.Fatalf("first retained diagnostic = %+v", first)
	}
	last := diagnostics.Items[len(diagnostics.Items)-1]
	if last.LogicalAgent != "zeta" || last.Code != "zeta_099" {
		t.Fatalf("last retained diagnostic = %+v", last)
	}
	again := ProjectAttemptDiagnostics(metrics)
	for index := range diagnostics.Items {
		if diagnostics.Items[index].Code != again.Items[index].Code ||
			diagnostics.Items[index].LogicalAgent != again.Items[index].LogicalAgent {
			t.Fatalf("diagnostic order changed at %d", index)
		}
	}
}

func TestAttemptDiagnosticsExposeOnlyNormalizedSafeFields(t *testing.T) {
	const secret = "diagnostic-secret-canary"
	retryable := true
	source := reporting.ExecutionReport{
		ReportID: "worker-private-report-id", Complete: true,
		Metrics: reporting.ExecutionMetrics{Tools: map[string]reporting.ToolMetrics{}},
		ToolCalls: []reporting.ToolCallRecord{{
			CallID: "private-call-id", Tool: "probe",
			Arguments: map[string]any{"token": secret},
			Outcome:   reporting.ToolCallSucceeded,
		}},
		Errors: []reporting.ExecutionError{
			{
				Code:      "worker_result_schema_json_invalid",
				Message:   "Worker result did not match StageContentResult",
				Retryable: &retryable,
			},
			{
				Code:    "gateway_error",
				Message: "provider " + secret + " at https://provider.example.test/v1",
			},
		},
	}
	normalized, err := NewPolicy(secret).NormalizeExecutionReport(source, MaxReportJSONBytes)
	if err != nil {
		t.Fatal(err)
	}
	diagnostics := ProjectAttemptDiagnostics(reporting.StageMetrics{
		Workers: map[string]reporting.ExecutionReport{"builder": normalized},
		Runtime: map[string]reporting.RuntimeReport{},
	})
	encoded := fmt.Sprintf("%+v", diagnostics)
	for _, forbidden := range []string{
		secret, "https://provider.example.test", "private-call-id", "private-report-id",
	} {
		if strings.Contains(encoded, forbidden) {
			t.Fatalf("diagnostics leaked %q: %s", forbidden, encoded)
		}
	}
	if len(diagnostics.Items) != 2 ||
		diagnostics.Items[0].Participant != AttemptDiagnosticWorker ||
		diagnostics.Items[0].LogicalAgent != "builder" ||
		diagnostics.Items[0].Code != "worker_result_schema_json_invalid" ||
		diagnostics.Items[0].Retryable == nil || !*diagnostics.Items[0].Retryable ||
		!strings.Contains(diagnostics.Items[1].Message, "[REDACTED_URL]") {
		t.Fatalf("safe diagnostics = %+v", diagnostics)
	}
}

func diagnosticReport(prefix string, count int) reporting.ExecutionReport {
	errors := make([]reporting.ExecutionError, 0, count)
	for index := range count {
		errors = append(errors, reporting.ExecutionError{
			Code: fmt.Sprintf("%s_%03d", prefix, index), Message: "safe normalized failure",
		})
	}
	return reporting.ExecutionReport{
		ReportID: prefix + "-report", Complete: true,
		Metrics:   reporting.ExecutionMetrics{Tools: map[string]reporting.ToolMetrics{}},
		ToolCalls: []reporting.ToolCallRecord{}, Errors: errors,
	}
}
