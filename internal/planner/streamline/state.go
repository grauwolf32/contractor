package streamline

import (
	"fmt"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

const maxReportErrors = 100

type executionState struct {
	mu sync.Mutex

	limits Limits

	modelCalls   int64
	inputTokens  int64
	outputTokens int64
	workerCalls  int64
	completion   *contracts.StageContentResult
	failure      *planner.Error

	toolMetrics map[string]toolCounters
	toolCalls   []contracts.ToolCallRecord
	errors      []contracts.ExecutionError
	nextCall    int64
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
	s.appendErrorLocked(
		"planner_tool_selection_invalid",
		"Planner model selected an unknown or concurrent tool call",
		false,
	)
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
	record := contracts.ToolCallRecord{
		CallID: fmt.Sprintf("planner_tool_%04d", s.nextCall), Tool: toolName,
		Arguments: arguments, Outcome: contracts.ToolCallSucceeded,
		DurationMS: &durationMS, ResultSizeBytes: &resultBytes,
	}
	if !succeeded {
		record.Outcome = contracts.ToolCallFailed
		retryable := false
		code, message := "planner_tool_rejected", "Planner tool call was rejected"
		if failure != nil {
			code, message, retryable = failure.Code, failure.Message, failure.Retryable
		}
		record.Error = &contracts.ExecutionError{
			Code: code, Message: message, Retryable: &retryable,
		}
	}
	s.toolCalls = append(s.toolCalls, record)
}

func (s *executionState) setCompletion(result contracts.StageContentResult) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.failure != nil || s.completion != nil {
		return false
	}
	cloned := planner.CloneStageResult(result)
	s.completion = &cloned
	return true
}

func (s *executionState) terminal() (*contracts.StageContentResult, *planner.Error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var result *contracts.StageContentResult
	if s.completion != nil {
		cloned := planner.CloneStageResult(*s.completion)
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
	if s.workerCalls >= int64(s.limits.MaxWorkerCalls) {
		return s.setFailureLocked(limitError(
			"planner_worker_call_limit", "Planner exhausted its Worker-call limit",
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
		s.appendErrorLocked(value.Code, value.Message, value.Retryable)
	}
	return s.failure
}

func (s *executionState) appendErrorLocked(code, message string, retryable bool) {
	if len(s.errors) >= maxReportErrors {
		return
	}
	retryableCopy := retryable
	s.errors = append(s.errors, contracts.ExecutionError{
		Code: code, Message: message, Retryable: &retryableCopy,
	})
}

func (s *executionState) report(reportID string, duration time.Duration) contracts.ExecutionReport {
	s.mu.Lock()
	defer s.mu.Unlock()
	modelCalls, inputTokens, outputTokens := s.modelCalls, s.inputTokens, s.outputTokens
	totalTokens := inputTokens + outputTokens
	durationMS := max(0, duration.Milliseconds())
	tools := make(map[string]contracts.ToolMetrics, len(s.toolMetrics))
	for name, counters := range s.toolMetrics {
		calls, succeeded, failed := counters.calls, counters.succeeded, counters.failed
		tools[name] = contracts.ToolMetrics{
			Calls: &calls, Succeeded: &succeeded, Failed: &failed,
		}
	}
	report := contracts.ExecutionReport{
		ReportID: reportID, Complete: true,
		Metrics: contracts.ExecutionMetrics{
			DurationMS: &durationMS, ModelCalls: &modelCalls,
			InputTokens: &inputTokens, OutputTokens: &outputTokens,
			TotalTokens: &totalTokens, Tools: tools,
		},
		ToolCalls: append([]contracts.ToolCallRecord(nil), s.toolCalls...),
		Errors:    append([]contracts.ExecutionError(nil), s.errors...),
	}
	return report
}

func limitError(code, message string) *planner.Error {
	return planner.NewError(code, message, true, nil)
}
