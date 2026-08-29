package telemetry

import "github.com/grauwolf32/contractor/internal/contracts"

// Summary is the only telemetry projection exposed through the public API.
// It contains no arguments, error messages, participant IDs, or provider URLs.
type Summary struct {
	ReportsComplete bool  `json:"reportsComplete"`
	ModelCalls      int64 `json:"modelCalls"`
	InputTokens     int64 `json:"inputTokens"`
	OutputTokens    int64 `json:"outputTokens"`
	TotalTokens     int64 `json:"totalTokens"`
	ToolCalls       int64 `json:"toolCalls"`
	ToolFailures    int64 `json:"toolFailures"`
	ErrorCount      int64 `json:"errorCount"`
	Truncated       bool  `json:"truncated"`
}

func BuildStageMetrics(
	planner *contracts.ExecutionReport,
	allocations map[string]contracts.AllocationFinalReport,
) contracts.StageMetrics {
	result := contracts.StageMetrics{
		Planner: planner,
		Workers: make(map[string]contracts.ExecutionReport, len(allocations)),
		Runtime: make(map[string]contracts.RuntimeReport, len(allocations)),
	}
	for name, report := range allocations {
		result.Workers[name] = report.Worker
		result.Runtime[name] = report.Runtime
	}
	return result
}

func Summarize(metrics contracts.StageMetrics) Summary {
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

func MergeSummaries(summaries ...Summary) Summary {
	result := Summary{ReportsComplete: true}
	for _, summary := range summaries {
		result.ReportsComplete = result.ReportsComplete && summary.ReportsComplete
		result.ModelCalls += summary.ModelCalls
		result.InputTokens += summary.InputTokens
		result.OutputTokens += summary.OutputTokens
		result.TotalTokens += summary.TotalTokens
		result.ToolCalls += summary.ToolCalls
		result.ToolFailures += summary.ToolFailures
		result.ErrorCount += summary.ErrorCount
		result.Truncated = result.Truncated || summary.Truncated
	}
	return result
}

func addExecutionReport(summary *Summary, report contracts.ExecutionReport) {
	summary.ReportsComplete = summary.ReportsComplete && report.Complete
	addOptional(&summary.ModelCalls, report.Metrics.ModelCalls)
	addOptional(&summary.InputTokens, report.Metrics.InputTokens)
	addOptional(&summary.OutputTokens, report.Metrics.OutputTokens)
	addOptional(&summary.TotalTokens, report.Metrics.TotalTokens)
	for _, tool := range report.Metrics.Tools {
		addOptional(&summary.ToolCalls, tool.Calls)
		addOptional(&summary.ToolFailures, tool.Failed)
	}
	summary.ErrorCount += int64(len(report.Errors))
	summary.Truncated = summary.Truncated || report.Truncated
}

func addOptional(target *int64, value *int64) {
	if value != nil {
		*target += *value
	}
}
