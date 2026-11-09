package streamline

import (
	"fmt"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// maxReportErrors is the report's error bound (LIM-OBS-5). Rejected tool
// selections leave its last two entries to the failure that ended the
// invocation and the error the Planner returns, so the report always ends
// with the returned error (PLN-R120).
const maxReportErrors = 100

const maxRejectionErrors = maxReportErrors - 2

type executionState struct {
	mu   sync.Mutex
	cost reporting.LLMCostCollector

	limits Limits

	modelCalls   int64
	inputTokens  int64
	outputTokens int64
	workerCalls  int64
	completion   *stageresult.StageContentResult
	failure      *planner.Error

	toolMetrics map[string]toolCounters
	toolCalls   []reporting.ToolCallRecord
	// rejections holds the errors of rejected tool selections; failure is
	// reported after them.
	rejections []reporting.ExecutionError
	nextCall   int64
}

type toolCounters struct {
	calls     int64
	succeeded int64
	failed    int64
}

func newExecutionState(limits Limits) *executionState {
	return &executionState{limits: limits, toolMetrics: map[string]toolCounters{}}
}

func (s *executionState) beforeModel() *planner.Error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.failure != nil {
		return s.failure
	}
	if s.completion != nil {
		return planner.NewError("planner_already_completed", "Planner already produced a candidate", false, nil)
	}
	if s.modelCalls >= int64(s.limits.MaxModelCalls) {
		return s.setFailureLocked(limitError(
			"planner_model_call_limit", "Planner exhausted its model-call limit",
		))
	}
	if s.inputTokens+s.outputTokens >= s.limits.MaxTokens {
		return s.setFailureLocked(limitError(
			"planner_token_limit", "Planner exhausted its cumulative token limit",
		))
	}
	s.modelCalls++
	return nil
}

func (s *executionState) afterModel(response *model.LLMResponse, allowed map[string]struct{}) (*model.LLMResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if response == nil {
		return nil, s.setFailureLocked(planner.NewError(
			"planner_gateway_invalid_response", "Planner Gateway returned no response", true, nil,
		))
	}
	if response.UsageMetadata == nil || response.UsageMetadata.PromptTokenCount < 0 ||
		response.UsageMetadata.CandidatesTokenCount < 0 {
		return nil, s.setFailureLocked(planner.NewError(
			"planner_gateway_invalid_response", "Planner Gateway returned invalid token usage", true, nil,
		))
	}
	s.inputTokens += int64(response.UsageMetadata.PromptTokenCount)
	s.outputTokens += int64(response.UsageMetadata.CandidatesTokenCount)
	if s.inputTokens+s.outputTokens > s.limits.MaxTokens {
		return nil, s.setFailureLocked(limitError(
			"planner_token_limit", "Planner exhausted its cumulative token limit",
		))
	}
	functionCount := 0
	unknown := false
	if response.Content != nil {
		for _, part := range response.Content.Parts {
			if part == nil || part.FunctionCall == nil {
				continue
			}
			functionCount++
			if _, ok := allowed[part.FunctionCall.Name]; !ok {
				unknown = true
			}
		}
	}
	if functionCount <= 1 && !unknown {
		return nil, nil
	}
	if len(s.rejections) < maxRejectionErrors {
		s.rejections = append(s.rejections, executionError(planner.Failure{
			Code:    "planner_tool_selection_invalid",
			Message: "Planner model selected an unknown or concurrent tool call",
		}))
	}
	replacement := *response
	replacement.Content = genai.NewContentFromText(
		"The tool selection was rejected. Call exactly one of the declared tools, then continue.",
		genai.RoleModel,
	)
	return &replacement, nil
}

func (s *executionState) providerFailure() *planner.Error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.setFailureLocked(planner.NewError(
		"planner_gateway_unavailable", "Planner Gateway request failed", true, nil,
	))
}

// rejectedGatewayRequest records a permanent Gateway failure, such as denied
// access or an over-long prompt, which another Stage attempt would repeat.
// code is the Gateway's safe classification.
func (s *executionState) rejectedGatewayRequest(code string) *planner.Error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.setFailureLocked(planner.NewError(
		"planner_gateway_rejected", "Planner Gateway rejected the request ("+code+")", false, nil,
	))
}

func (s *executionState) rejectedGatewayResponse(rejected *gatewayResponseRejected) *planner.Error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if rejected.usage != nil {
		s.inputTokens += int64(rejected.usage.PromptTokenCount)
		s.outputTokens += int64(rejected.usage.CandidatesTokenCount)
		if s.inputTokens+s.outputTokens > s.limits.MaxTokens {
			return s.setFailureLocked(limitError(
				"planner_token_limit", "Planner exhausted its cumulative token limit",
			))
		}
	}
	return s.setFailureLocked(planner.NewError(
		"planner_gateway_invalid_response", "Planner Gateway returned an invalid response", true, nil,
	))
}

// chargeUsage charges the known usage of an accepted compactor response to
// the cumulative Planner token budget, once (PLN-R127).
func (s *executionState) chargeUsage(usage *genai.GenerateContentResponseUsageMetadata) *planner.Error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if usage == nil {
		return s.setFailureLocked(planner.NewError(
			"planner_gateway_invalid_response", "Planner Gateway returned invalid token usage", true, nil,
		))
	}
	s.inputTokens += int64(usage.PromptTokenCount)
	s.outputTokens += int64(usage.CandidatesTokenCount)
	if s.inputTokens+s.outputTokens > s.limits.MaxTokens {
		return s.setFailureLocked(limitError(
			"planner_token_limit", "Planner exhausted its cumulative token limit",
		))
	}
	return nil
}

func (s *executionState) reserveWorkerCall() *planner.Error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.failure != nil {
		return s.failure
	}
	if s.workerCalls >= int64(s.limits.MaxWorkerCalls) {
		return s.setFailureLocked(limitError(
			"planner_worker_call_limit", "Planner exhausted its Worker-call limit",
		))
	}
	s.workerCalls++
	return nil
}

func (s *executionState) recordTool(
	toolName string,
	arguments map[string]any,
	succeeded bool,
	duration time.Duration,
	resultSize int,
	failure *planner.Failure,
) {
	s.mu.Lock()
	defer s.mu.Unlock()
	counters := s.toolMetrics[toolName]
	counters.calls++
	if succeeded {
		counters.succeeded++
	} else {
		counters.failed++
	}
	s.toolMetrics[toolName] = counters
	s.nextCall++
	durationMS := max(0, duration.Milliseconds())
	resultBytes := int64(max(0, resultSize))
	record := reporting.ToolCallRecord{
		CallID: fmt.Sprintf("planner_tool_%04d", s.nextCall), Tool: toolName,
		Arguments: arguments, Outcome: reporting.ToolCallSucceeded,
		DurationMS: &durationMS, ResultSizeBytes: &resultBytes,
	}
	if !succeeded {
		record.Outcome = reporting.ToolCallFailed
		retryable := false
		code, message := "planner_tool_rejected", "Planner tool call was rejected"
		if failure != nil {
			code, message, retryable = failure.Code, failure.Message, failure.Retryable
		}
		record.Error = &reporting.ExecutionError{
			Code: code, Message: message, Retryable: &retryable,
		}
	}
	s.toolCalls = append(s.toolCalls, record)
}

func (s *executionState) setCompletion(result stageresult.StageContentResult) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.failure != nil || s.completion != nil {
		return false
	}
	cloned := result.Clone()
	s.completion = &cloned
	return true
}

func (s *executionState) terminal() (*stageresult.StageContentResult, *planner.Error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var result *stageresult.StageContentResult
	if s.completion != nil {
		cloned := s.completion.Clone()
		result = &cloned
	}
	return result, s.failure
}

func (s *executionState) exhaustedAfterTurn() *planner.Error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.completion != nil || s.failure != nil {
		return s.failure
	}
	if s.modelCalls >= int64(s.limits.MaxModelCalls) {
		return s.setFailureLocked(limitError(
			"planner_model_call_limit", "Planner exhausted its model-call limit",
		))
	}
	if s.inputTokens+s.outputTokens >= s.limits.MaxTokens {
		return s.setFailureLocked(limitError(
			"planner_token_limit", "Planner exhausted its cumulative token limit",
		))
	}
	return nil
}

func (s *executionState) setExternalFailure(value *planner.Error) *planner.Error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.setFailureLocked(value)
}

func (s *executionState) setFailureLocked(value *planner.Error) *planner.Error {
	if s.failure == nil {
		s.failure = value
	}
	return s.failure
}

func executionError(failure planner.Failure) reporting.ExecutionError {
	retryable := failure.Retryable
	return reporting.ExecutionError{Code: failure.Code, Message: failure.Message, Retryable: &retryable}
}

// report builds the ExecutionReport of the invocation. Its errors are the
// rejected tool selections, then the failure that ended the invocation, then
// returned, the error the Planner returns, when that differs: a failed
// session write after the Planner had its outcome (PLN-R106, PLN-R120).
func (s *executionState) report(
	reportID string, duration time.Duration, returned *planner.Failure,
) reporting.ExecutionReport {
	s.mu.Lock()
	defer s.mu.Unlock()
	modelCalls, inputTokens, outputTokens := s.modelCalls, s.inputTokens, s.outputTokens
	totalTokens := inputTokens + outputTokens
	durationMS := max(0, duration.Milliseconds())
	tools := make(map[string]reporting.ToolMetrics, len(s.toolMetrics))
	for name, counters := range s.toolMetrics {
		calls, succeeded, failed := counters.calls, counters.succeeded, counters.failed
		tools[name] = reporting.ToolMetrics{
			Calls: &calls, Succeeded: &succeeded, Failed: &failed,
		}
	}
	report := reporting.ExecutionReport{
		ReportID: reportID, Complete: true,
		Metrics: reporting.ExecutionMetrics{
			LLMCost:    s.cost.Snapshot(modelCalls),
			DurationMS: &durationMS, ModelCalls: &modelCalls,
			InputTokens: &inputTokens, OutputTokens: &outputTokens,
			TotalTokens: &totalTokens, Tools: tools,
		},
		ToolCalls: append([]reporting.ToolCallRecord(nil), s.toolCalls...),
		Errors:    append([]reporting.ExecutionError(nil), s.rejections...),
	}
	if s.failure != nil {
		report.Errors = append(report.Errors, executionError(s.failure.Failure))
	}
	if returned != nil && (s.failure == nil || s.failure.Failure != *returned) {
		report.Errors = append(report.Errors, executionError(*returned))
	}
	return report
}

func limitError(code, message string) *planner.Error {
	return planner.NewError(code, message, true, nil)
}
