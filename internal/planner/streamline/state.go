package streamline

import (
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/stageresult"
)

// maxReportErrors is the report's error bound (LIM-OBS-5). Rejected tool
// selections leave its last two entries to the failure that ended the
// invocation and the error the Planner returns, so the report always ends
// with the returned error (PLN-R120).
const maxReportErrors = 100

const maxRejectionErrors = maxReportErrors - 2

type executionState struct {
	mu          sync.Mutex
	runtimeCost *reporting.LLMCost

	materials                    map[string]map[string]contracts.ArtifactRef
	materialCount, materialBytes int

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

type modelUsage struct{ PromptTokenCount, CandidatesTokenCount, CachedContentTokenCount, TotalTokenCount int32 }

func (s *executionState) afterModel(usage *modelUsage, calls []string, allowed map[string]struct{}) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if usage == nil || usage.PromptTokenCount < 0 || usage.CandidatesTokenCount < 0 {
		return false, s.setFailureLocked(planner.NewError("planner_gateway_invalid_response", "Planner Gateway returned invalid token usage", true, nil))
	}
	s.inputTokens += int64(usage.PromptTokenCount)
	s.outputTokens += int64(usage.CandidatesTokenCount)
	if s.inputTokens+s.outputTokens > s.limits.MaxTokens {
		return false, s.setFailureLocked(limitError("planner_token_limit", "Planner exhausted its cumulative token limit"))
	}
	unknown := false
	for _, name := range calls {
		if _, ok := allowed[name]; !ok {
			unknown = true
		}
	}
	if len(calls) <= 1 && !unknown {
		return false, nil
	}
	if len(s.rejections) < maxRejectionErrors {
		s.rejections = append(s.rejections, executionError(planner.Failure{Code: "planner_tool_selection_invalid", Message: "Planner model selected an unknown or concurrent tool call"}))
	}
	return true, nil
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

func (s *executionState) rejectedGatewayResponse(usage *modelUsage) *planner.Error {
	if usage != nil {
		if failure := s.chargeUsage(usage); failure != nil {
			return failure
		}
	}
	return s.setExternalFailure(planner.NewError("planner_gateway_invalid_response", "Planner Gateway returned an invalid response", true, nil))
}

func (s *executionState) chargeUsage(usage *modelUsage) *planner.Error {
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
	cost := new(reporting.LLMCostCollector).Snapshot(modelCalls)
	if s.runtimeCost != nil {
		value := *s.runtimeCost
		value.Calls = max(value.Calls, modelCalls)
		cost = &value
	}
	report := reporting.ExecutionReport{
		ReportID: reportID, Complete: true,
		Metrics: reporting.ExecutionMetrics{
			LLMCost:    cost,
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

// Keep only each logical Worker's latest accepted refs, independently of ADK compaction.
func (s *executionState) selectWorkerArtifacts(worker string, refs map[string]contracts.ArtifactRef) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.materials == nil {
		s.materials = map[string]map[string]contracts.ArtifactRef{}
	}
	previous := s.materials[worker]
	oldBytes, _ := json.Marshal(previous)
	newBytes, _ := json.Marshal(refs)
	count := s.materialCount + len(refs) - len(previous)
	oldSize, newSize := 0, 0
	if len(previous) != 0 {
		oldSize = len(oldBytes) + len(worker) + 3
	}
	if len(refs) != 0 {
		newSize = len(newBytes) + len(worker) + 3
	}
	bytes := s.materialBytes + newSize - oldSize
	if count > 4096 || bytes > 4*1024*1024 {
		return false
	}
	if len(refs) == 0 {
		delete(s.materials, worker)
	} else {
		s.materials[worker] = contracts.CloneArtifactRefs(refs)
	}
	s.materialCount, s.materialBytes = count, bytes
	return true
}
