package telemetry

import (
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestCompletionDiagnosticsRetainFactsWithoutInventingUsageOrAcceptance(t *testing.T) {
	for _, phase := range []string{"collecting", "sealed", "publishing", "published", "failed"} {
		t.Run(phase, func(t *testing.T) {
			source := contracts.ExecutionReport{ReportID: "report", Complete: true,
				Metrics:   contracts.ExecutionMetrics{Tools: map[string]contracts.ToolMetrics{}},
				ToolCalls: []contracts.ToolCallRecord{}, Errors: []contracts.ExecutionError{},
				Completion: &contracts.WorkerCompletionDiagnostics{Kind: contracts.AuditCheckResultsV1,
					Phase: phase, AcceptedCount: 2, TotalCount: 2, ReminderCount: 1}}
			if phase == "failed" {
				source.Completion.FailureCode = "audit_result_publication_conflict"
			}
			normalized, err := NewPolicy().NormalizeExecutionReport(source, MaxReportJSONBytes)
			if err != nil {
				t.Fatal(err)
			}
			source.Completion.AcceptedCount = 0
			if normalized.Completion.AcceptedCount != 2 {
				t.Fatal("report retained a mutable pointer")
			}
			metrics := contracts.StageMetrics{Workers: map[string]contracts.ExecutionReport{"checker": normalized}, Runtime: map[string]contracts.RuntimeReport{}}
			diagnostics := ProjectAttemptDiagnostics(metrics)
			if len(diagnostics.Items) != 1 || diagnostics.Items[0].Completion == nil {
				t.Fatal("completion facts were lost")
			}
			if phase == "published" && !strings.Contains(diagnostics.Items[0].Message, "acceptance is separate") {
				t.Fatal("publication was confused with acceptance")
			}
			if phase == "failed" && !strings.Contains(diagnostics.Items[0].Message, "different bytes") {
				t.Fatal("write conflict was confused with missing results")
			}
			diagnostics.Items[0].Completion.AcceptedCount = 0
			if normalized.Completion.AcceptedCount != 2 {
				t.Fatal("projection retained a mutable pointer")
			}
			if len(normalized.Errors) != 0 || len(normalized.ToolCalls) != 0 || normalized.Metrics.ModelCalls != nil {
				t.Fatal("phase facts invented calls/errors")
			}
		})
	}
}
