package telemetry

import (
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
)

const MaxAttemptDiagnosticRecords = 128

type AttemptDiagnosticParticipant string

const (
	AttemptDiagnosticPlanner AttemptDiagnosticParticipant = "planner"
	AttemptDiagnosticWorker  AttemptDiagnosticParticipant = "worker"
)

type AttemptDiagnostic struct {
	Completion   *reporting.WorkerCompletionDiagnostics `json:"completion,omitempty"`
	Participant  AttemptDiagnosticParticipant           `json:"participant"`
	LogicalAgent string                                 `json:"logicalAgent,omitempty"`
	Code         string                                 `json:"code"`
	Message      string                                 `json:"message"`
	Retryable    *bool                                  `json:"retryable,omitempty"`
}

type AttemptDiagnostics struct {
	Items     []AttemptDiagnostic `json:"items"`
	Truncated bool                `json:"truncated"`
}

var (
	diagnosticURLPattern = regexp.MustCompile(`(?i)\b[a-z][a-z0-9+.-]*://[^\s<>"']+`)
)

// Summary is the aggregate telemetry projection exposed through the public API.
// It contains no arguments, error messages, participant IDs, or provider URLs.
// Each counter is the saturating sum of its reports' counters: at most
// math.MaxInt64, never wrapped negative (OBS-R112).
type Summary struct {
	ReportsComplete bool  `json:"reportsComplete" strictjson:"required"`
	ModelCalls      int64 `json:"modelCalls" strictjson:"required"`
	InputTokens     int64 `json:"inputTokens" strictjson:"required"`
	OutputTokens    int64 `json:"outputTokens" strictjson:"required"`
	TotalTokens     int64 `json:"totalTokens" strictjson:"required"`
	ToolCalls       int64 `json:"toolCalls" strictjson:"required"`
	ToolFailures    int64 `json:"toolFailures" strictjson:"required"`
	ErrorCount      int64 `json:"errorCount" strictjson:"required"`
	Truncated       bool  `json:"truncated" strictjson:"required"`
}

// Validate checks a summary that Summarize did not just build, read back from
// storage or held by the Operations registry: a saturating sum of validated
// reports never has a negative counter (OBS-R112).
func (s Summary) Validate() error {
	for _, value := range []int64{
		s.ModelCalls, s.InputTokens, s.OutputTokens, s.TotalTokens,
		s.ToolCalls, s.ToolFailures, s.ErrorCount,
	} {
		if value < 0 {
			return fmt.Errorf("%w: summary counters must not be negative", ErrInvalid)
		}
	}
	return nil
}

func BuildStageMetrics(
	planner *reporting.ExecutionReport,
	allocations map[string]reporting.AllocationFinalReport,
) reporting.StageMetrics {
	result := reporting.StageMetrics{
		Planner: planner,
		Workers: make(map[string]reporting.ExecutionReport, len(allocations)),
		Runtime: make(map[string]reporting.RuntimeReport, len(allocations)),
	}
	for name, report := range allocations {
		result.Workers[name] = report.Worker
		result.Runtime[name] = report.Runtime
	}
	return result
}

func Summarize(metrics reporting.StageMetrics) Summary {
	result := Summary{ReportsComplete: true}
	if metrics.Planner != nil {
		addExecutionReport(&result, *metrics.Planner)
	}
	for _, report := range metrics.Workers {
		addExecutionReport(&result, report)
	}
	for _, report := range metrics.Runtime {
		result.ReportsComplete = result.ReportsComplete && report.Complete
	}
	return result
}

// ProjectAttemptDiagnostics returns only already normalized Planner and Worker
// report errors and bounded completion facts. Reports are ordered Planner first and then by logical Agent;
// error order within each report is preserved. If the public cap is exceeded,
// the tail of that deterministic sequence is retained because report policy
// itself also retains newest records.
func ProjectAttemptDiagnostics(metrics reporting.StageMetrics) AttemptDiagnostics {
	result := AttemptDiagnostics{Items: make([]AttemptDiagnostic, 0)}
	if metrics.Planner != nil {
		appendAttemptDiagnostics(&result, AttemptDiagnosticPlanner, "", *metrics.Planner)
	}
	workers := make([]string, 0, len(metrics.Workers))
	for logicalAgent := range metrics.Workers {
		workers = append(workers, logicalAgent)
	}
	sort.Strings(workers)
	for _, logicalAgent := range workers {
		appendAttemptDiagnostics(
			&result, AttemptDiagnosticWorker, logicalAgent, metrics.Workers[logicalAgent],
		)
	}
	if len(result.Items) > MaxAttemptDiagnosticRecords {
		start := len(result.Items) - MaxAttemptDiagnosticRecords
		items := make([]AttemptDiagnostic, MaxAttemptDiagnosticRecords)
		copy(items, result.Items[start:])
		result.Items = items
		result.Truncated = true
	}
	return result
}

func appendAttemptDiagnostics(
	target *AttemptDiagnostics,
	participant AttemptDiagnosticParticipant,
	logicalAgent string,
	report reporting.ExecutionReport,
) {
	target.Truncated = target.Truncated || report.Truncated
	if source := report.Completion; source != nil && source.Validate() == nil {
		value := *source
		code := "audit_result_" + value.Phase
		message := fmt.Sprintf("Audit results: %d/%d recorded; reminders: %d. ", value.AcceptedCount, value.TotalCount, value.ReminderCount)
		switch value.Phase {
		case "published":
			message += "Result package published; Audit evidence acceptance is separate."
		case "failed":
			code = value.FailureCode
			switch code {
			case "audit_result_incomplete":
				message += "Required results remain missing."
			case "audit_result_publication_conflict":
				message += "The Run result binding contains different bytes; existing output was preserved."
			case "audit_result_publication_failed":
				message += "Result publication could not be verified."
			default:
				message += "Worker completion failed."
			}
		default:
			message += "Result collection/publication is in progress."
		}
		target.Items = append(target.Items, AttemptDiagnostic{
			Participant: participant, LogicalAgent: logicalAgent, Code: code, Message: message, Completion: &value,
		})
	}
	for _, source := range report.Errors {
		code := source.Code
		if !contracts.ValidName(code) {
			code = "execution_error"
		}
		message := diagnosticURLPattern.ReplaceAllString(source.Message, "[REDACTED_URL]")
		message = truncateUTF8(message, MaxErrorMessageBytes)
		if strings.TrimSpace(message) == "" {
			message = "Execution failed"
		}
		var retryable *bool
		if source.Retryable != nil {
			value := *source.Retryable
			retryable = &value
		}
		target.Items = append(target.Items, AttemptDiagnostic{
			Participant: participant, LogicalAgent: logicalAgent,
			Code: code, Message: message, Retryable: retryable,
		})
	}
}

func addExecutionReport(summary *Summary, report reporting.ExecutionReport) {
	summary.ReportsComplete = summary.ReportsComplete && report.Complete
	addOptional(&summary.ModelCalls, report.Metrics.ModelCalls)
	addOptional(&summary.InputTokens, report.Metrics.InputTokens)
	addOptional(&summary.OutputTokens, report.Metrics.OutputTokens)
	addOptional(&summary.TotalTokens, report.Metrics.TotalTokens)
	for _, tool := range report.Metrics.Tools {
		addOptional(&summary.ToolCalls, tool.Calls)
		addOptional(&summary.ToolFailures, tool.Failed)
	}
	summary.ErrorCount = saturatingAdd(summary.ErrorCount, int64(len(report.Errors)))
	summary.Truncated = summary.Truncated || report.Truncated
}

func addOptional(target *int64, value *int64) {
	if value != nil {
		*target = saturatingAdd(*target, *value)
	}
}

// saturatingAdd adds two non-negative counters, which validated reports
// guarantee, and returns math.MaxInt64 for a sum beyond it (OBS-R112).
func saturatingAdd(total, value int64) int64 {
	if value > math.MaxInt64-total {
		return math.MaxInt64
	}
	return total + value
}
