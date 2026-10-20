package reporting

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"

	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

type ExecutionError struct {
	Code      string `json:"code" strictjson:"required"`
	Message   string `json:"message" strictjson:"required"`
	Retryable *bool  `json:"retryable,omitempty"`
}

type ToolMetrics struct {
	Calls     *int64 `json:"calls,omitempty"`
	Succeeded *int64 `json:"succeeded,omitempty"`
	Failed    *int64 `json:"failed,omitempty"`
}

type WorkerBudgetMetrics struct {
	MaxModelCalls         int64   `json:"maxModelCalls" strictjson:"required"`
	MaxToolCalls          int64   `json:"maxToolCalls" strictjson:"required"`
	MaxTotalTokens        int64   `json:"maxTotalTokens" strictjson:"required"`
	ObservedModelCalls    int64   `json:"observedModelCalls" strictjson:"required"`
	ObservedToolCalls     int64   `json:"observedToolCalls" strictjson:"required"`
	ObservedTotalTokens   int64   `json:"observedTotalTokens" strictjson:"required"`
	TokenUsageUnavailable int64   `json:"tokenUsageUnavailable" strictjson:"required"`
	Exhausted             *string `json:"exhausted,omitempty"`
}

type WorkerSummarizerMetrics struct {
	Attempts              uint64            `json:"attempts" strictjson:"required"`
	Succeeded             uint64            `json:"succeeded" strictjson:"required"`
	Failed                uint64            `json:"failed" strictjson:"required"`
	ModelCalls            uint64            `json:"modelCalls" strictjson:"required"`
	InputTokens           uint64            `json:"inputTokens" strictjson:"required"`
	OutputTokens          uint64            `json:"outputTokens" strictjson:"required"`
	TotalTokens           uint64            `json:"totalTokens" strictjson:"required"`
	TokenUsageUnavailable uint64            `json:"tokenUsageUnavailable" strictjson:"required"`
	FailureCodes          map[string]uint64 `json:"failureCodes" strictjson:"required"`
}

type ExecutionMetrics struct {
	DurationMS   *int64                   `json:"durationMs,omitempty"`
	ModelCalls   *int64                   `json:"modelCalls,omitempty"`
	InputTokens  *int64                   `json:"inputTokens,omitempty"`
	OutputTokens *int64                   `json:"outputTokens,omitempty"`
	TotalTokens  *int64                   `json:"totalTokens,omitempty"`
	Tools        map[string]ToolMetrics   `json:"tools" strictjson:"required"`
	WorkerBudget *WorkerBudgetMetrics     `json:"workerBudget,omitempty"`
	Summarizer   *WorkerSummarizerMetrics `json:"summarizer,omitempty"`
}

type ToolCallOutcome string

const (
	ToolCallSucceeded ToolCallOutcome = "succeeded"
	ToolCallFailed    ToolCallOutcome = "failed"
)

type ToolCallRecord struct {
	CallID             string          `json:"callId" strictjson:"required"`
	Tool               string          `json:"tool" strictjson:"required"`
	Arguments          map[string]any  `json:"arguments,omitempty"`
	ArgumentsTruncated bool            `json:"argumentsTruncated" strictjson:"required"`
	Outcome            ToolCallOutcome `json:"outcome" strictjson:"required"`
	DurationMS         *int64          `json:"durationMs,omitempty"`
	ResultSizeBytes    *int64          `json:"resultSizeBytes,omitempty"`
	Error              *ExecutionError `json:"error,omitempty"`
}

type ExecutionReport struct {
	Completion *WorkerCompletionDiagnostics `json:"completion,omitempty"`
	ReportID   string                       `json:"reportId" strictjson:"required"`
	Complete   bool                         `json:"complete" strictjson:"required"`
	Metrics    ExecutionMetrics             `json:"metrics" strictjson:"required"`
	ToolCalls  []ToolCallRecord             `json:"toolCalls" strictjson:"required"`
	Errors     []ExecutionError             `json:"errors" strictjson:"required"`
	Truncated  bool                         `json:"truncated" strictjson:"required"`
}

type RuntimeReport struct {
	Complete       bool                                                        `json:"complete" strictjson:"required"`
	DurationMS     *int64                                                      `json:"durationMs,omitempty" strictjson:"nullable"`
	StopReason     *string                                                     `json:"stopReason,omitempty" strictjson:"nullable"`
	Adapters       map[runtimesettings.RuntimeAdapterRef]RuntimeAdapterMetrics `json:"adapters" strictjson:"required"`
	Resources      *RuntimeResources                                           `json:"resources,omitempty"`
	ResourcesError *ResourceReason                                             `json:"-"`
}

func (r RuntimeReport) MarshalJSON() ([]byte, error) {
	type wireRuntimeReport RuntimeReport
	copy := wireRuntimeReport(r)
	if copy.Adapters == nil {
		copy.Adapters = map[runtimesettings.RuntimeAdapterRef]RuntimeAdapterMetrics{}
	}
	return json.Marshal(copy)
}

// UnmarshalJSON requires the members RuntimeReport tags required, but decodes
// each adapter entry and the resources block on its own: a malformed one is
// discarded as incomplete telemetry instead of failing the report (OBS-R36,
// CTL-R136).
func (r *RuntimeReport) UnmarshalJSON(data []byte) error {
	type wireRuntimeReport struct {
		Complete   bool                       `json:"complete" strictjson:"required"`
		DurationMS *int64                     `json:"durationMs,omitempty" strictjson:"nullable"`
		StopReason *string                    `json:"stopReason,omitempty" strictjson:"nullable"`
		Adapters   map[string]json.RawMessage `json:"adapters" strictjson:"required"`
		Resources  json.RawMessage            `json:"resources" strictjson:"nullable"`
	}
	var wire wireRuntimeReport
	if err := strictjson.DecodeNonNull(data, &wire); errors.Is(err, strictjson.ErrDuplicateKey) {
		return contracts.Invalidf("invalid runtime report JSON")
	} else if err != nil {
		return err
	}
	*r = RuntimeReport{
		Complete: wire.Complete, DurationMS: wire.DurationMS, StopReason: wire.StopReason,
		Adapters: map[runtimesettings.RuntimeAdapterRef]RuntimeAdapterMetrics{},
	}
	r.Resources, r.ResourcesError = decodeOptionalResources(wire.Resources)
	for rawRef, rawMetrics := range wire.Adapters {
		ref := runtimesettings.RuntimeAdapterRef(rawRef)
		metrics, err := decodeRuntimeAdapterMetrics(rawMetrics)
		if err != nil || ref.Validate() != nil {
			r.Complete = false
			continue
		}
		r.Adapters[ref] = metrics
	}
	return nil
}

func decodeRuntimeAdapterMetrics(data []byte) (RuntimeAdapterMetrics, error) {
	var metrics RuntimeAdapterMetrics
	if err := strictjson.DecodeNonNull(data, &metrics); err != nil {
		return RuntimeAdapterMetrics{}, err
	}
	if err := metrics.Validate(); err != nil {
		return RuntimeAdapterMetrics{}, err
	}
	return metrics, nil
}

type StageMetrics struct {
	Planner *ExecutionReport           `json:"planner,omitempty"`
	Workers map[string]ExecutionReport `json:"workers"`
	Runtime map[string]RuntimeReport   `json:"runtime"`
}

type AllocationFinalReport struct {
	ReportID     string          `json:"reportId" strictjson:"required"`
	AllocationID string          `json:"allocationId" strictjson:"required"`
	StartedAt    time.Time       `json:"startedAt" strictjson:"required"`
	FinishedAt   time.Time       `json:"finishedAt" strictjson:"required"`
	Worker       ExecutionReport `json:"worker" strictjson:"required"`
	Runtime      RuntimeReport   `json:"runtime" strictjson:"required"`
}

func (r *AllocationFinalReport) UnmarshalJSON(data []byte) error {
	// Check the original report before optional invalid resources are removed.
	if len(data) > 1024*1024 {
		return contracts.Invalidf("allocation final report exceeds 1 MiB")
	}
	type wireReport AllocationFinalReport
	var value wireReport
	if err := strictjson.DecodeNonNull(data, &value); errors.Is(err, strictjson.ErrDuplicateKey) {
		return contracts.Invalidf("invalid allocation report JSON")
	} else if err != nil {
		return err
	}
	*r = AllocationFinalReport(value)
	return nil
}

func (r ExecutionReport) Validate() error {
	if r.Completion != nil {
		if err := r.Completion.Validate(); err != nil {
			return err
		}
	}
	if err := contracts.ValidateOpaqueID("reportId", r.ReportID); err != nil {
		return err
	}
	if r.Metrics.Tools == nil || r.ToolCalls == nil || r.Errors == nil {
		return contracts.Invalidf("execution report maps and lists must not be null")
	}
	for name, metrics := range r.Metrics.Tools {
		if err := contracts.ValidateOpaqueID("metrics tool name", name); err != nil {
			return err
		}
		for _, value := range []*int64{metrics.Calls, metrics.Succeeded, metrics.Failed} {
			if value != nil && *value < 0 {
				return contracts.Invalidf("tool metrics must be non-negative")
			}
		}
	}
	for _, value := range []*int64{
		r.Metrics.DurationMS, r.Metrics.ModelCalls, r.Metrics.InputTokens,
		r.Metrics.OutputTokens, r.Metrics.TotalTokens,
	} {
		if value != nil && *value < 0 {
			return contracts.Invalidf("execution metrics must be non-negative")
		}
	}
	if budget := r.Metrics.WorkerBudget; budget != nil {
		if budget.MaxModelCalls <= 0 || budget.MaxModelCalls > contracts.MaxWorkerModelCalls ||
			budget.MaxToolCalls <= 0 || budget.MaxToolCalls > contracts.MaxWorkerToolCalls ||
			budget.MaxTotalTokens <= 0 || budget.MaxTotalTokens > contracts.MaxWorkerTotalTokens {
			return contracts.Invalidf("worker budget limits are invalid")
		}
		if budget.ObservedModelCalls < 0 || budget.ObservedModelCalls > budget.MaxModelCalls ||
			budget.ObservedToolCalls < 0 || budget.ObservedToolCalls > budget.MaxToolCalls ||
			budget.ObservedTotalTokens < 0 || budget.TokenUsageUnavailable < 0 {
			return contracts.Invalidf("worker budget observations are invalid")
		}
		if budget.Exhausted != nil {
			switch *budget.Exhausted {
			case "model_calls":
				if budget.ObservedModelCalls != budget.MaxModelCalls {
					return contracts.Invalidf("worker model-call exhaustion is inconsistent")
				}
			case "tool_calls":
				if budget.ObservedToolCalls != budget.MaxToolCalls {
					return contracts.Invalidf("worker tool-call exhaustion is inconsistent")
				}
			case "total_tokens":
				if budget.ObservedTotalTokens < budget.MaxTotalTokens {
					return contracts.Invalidf("worker token exhaustion is inconsistent")
				}
			default:
				return contracts.Invalidf("worker budget exhausted dimension is invalid")
			}
		}
	}
	if summarizer := r.Metrics.Summarizer; summarizer != nil {
		if summarizer.Attempts == 0 || summarizer.Succeeded > summarizer.Attempts ||
			summarizer.Failed > summarizer.Attempts ||
			^uint64(0)-summarizer.Succeeded < summarizer.Failed ||
			summarizer.Succeeded+summarizer.Failed != summarizer.Attempts ||
			summarizer.ModelCalls > summarizer.Attempts ||
			summarizer.TokenUsageUnavailable > summarizer.ModelCalls ||
			summarizer.FailureCodes == nil || len(summarizer.FailureCodes) > 64 {
			return contracts.Invalidf("worker summarizer metrics are inconsistent")
		}
		var failures uint64
		for code, count := range summarizer.FailureCodes {
			if !contracts.ValidWorkerFailureCode(code) || count == 0 || ^uint64(0)-failures < count {
				return contracts.Invalidf("worker summarizer failure metrics are invalid")
			}
			failures += count
		}
		if failures != summarizer.Failed {
			return contracts.Invalidf("worker summarizer failure metrics are inconsistent")
		}
	}
	for index, call := range r.ToolCalls {
		if err := contracts.ValidateOpaqueID("tool call ID", call.CallID); err != nil {
			return err
		}
		if err := contracts.ValidateOpaqueID("tool call name", call.Tool); err != nil {
			return err
		}
		if call.Arguments == nil && call.ArgumentsTruncated {
			return contracts.Invalidf("tool call %d marks absent arguments truncated", index)
		}
		if call.Arguments != nil {
			size, err := contracts.ResultJSONSize(call.Arguments)
			if err != nil || size > 4096 {
				return contracts.Invalidf("tool call %d arguments exceed 4096 bytes", index)
			}
		}
		if call.Outcome != ToolCallSucceeded && call.Outcome != ToolCallFailed {
			return contracts.Invalidf("unknown tool call outcome %q", call.Outcome)
		}
		if (call.DurationMS != nil && *call.DurationMS < 0) ||
			(call.ResultSizeBytes != nil && *call.ResultSizeBytes < 0) {
			return contracts.Invalidf("tool call measurements must be non-negative")
		}
		if (call.Outcome == ToolCallSucceeded && call.Error != nil) ||
			(call.Outcome == ToolCallFailed && call.Error == nil) {
			return contracts.Invalidf("tool call outcome and error are inconsistent")
		}
		if call.Error != nil {
			if err := call.Error.Validate(); err != nil {
				return err
			}
		}
	}
	for _, item := range r.Errors {
		if err := item.Validate(); err != nil {
			return err
		}
	}
	if len(r.ToolCalls) > 1000 || len(r.Errors) > 100 {
		return contracts.Invalidf("execution report detail exceeds bounded record limits")
	}
	size, err := contracts.ResultJSONSize(r)
	if err != nil {
		return contracts.Invalidf("execution report cannot be encoded")
	}
	if size > 1024*1024 {
		return contracts.Invalidf("execution report exceeds 1 MiB")
	}
	return nil
}

func (e ExecutionError) Validate() error {
	if strings.TrimSpace(e.Code) == "" || strings.TrimSpace(e.Message) == "" {
		return contracts.Invalidf("execution error code and message are required")
	}
	if len([]byte(e.Message)) > 4096 {
		return contracts.Invalidf("execution error message exceeds 4096 bytes")
	}
	return nil
}

func (m StageMetrics) Validate() error {
	if m.Workers == nil || m.Runtime == nil {
		return contracts.Invalidf("StageMetrics maps must not be null")
	}
	if m.Planner != nil {
		if err := m.Planner.Validate(); err != nil {
			return fmt.Errorf("planner report: %w", err)
		}
	}
	for name, report := range m.Workers {
		if err := contracts.ValidateOpaqueID("worker logical Agent name", name); err != nil {
			return err
		}
		if err := report.Validate(); err != nil {
			return fmt.Errorf("worker report %q: %w", name, err)
		}
	}
	for name, report := range m.Runtime {
		if err := contracts.ValidateOpaqueID("runtime logical Agent name", name); err != nil {
			return err
		}
		if report.DurationMS != nil && *report.DurationMS < 0 {
			return contracts.Invalidf("runtime duration must be non-negative")
		}
		if err := report.validateAdapters(); err != nil {
			return err
		}
	}
	return nil
}

func (r AllocationFinalReport) Validate() error {
	if err := contracts.ValidateOpaqueID("allocation final report ID", r.ReportID); err != nil {
		return err
	}
	if err := contracts.ValidateOpaqueID("allocationId", r.AllocationID); err != nil {
		return err
	}
	if r.StartedAt.IsZero() || r.FinishedAt.IsZero() || r.FinishedAt.Before(r.StartedAt) {
		return contracts.Invalidf("allocation final report timestamps are invalid")
	}
	if err := r.Worker.Validate(); err != nil {
		return fmt.Errorf("worker report: %w", err)
	}
	if r.Runtime.DurationMS != nil && *r.Runtime.DurationMS < 0 {
		return contracts.Invalidf("runtime duration must be non-negative")
	}
	if r.Runtime.StopReason != nil && strings.TrimSpace(*r.Runtime.StopReason) == "" {
		return contracts.Invalidf("runtime stopReason must not be empty")
	}
	if err := r.Runtime.validateAdapters(); err != nil {
		return err
	}
	size, err := contracts.ResultJSONSize(r)
	if err != nil {
		return contracts.Invalidf("allocation final report cannot be encoded")
	}
	if size > 1024*1024 {
		return contracts.Invalidf("allocation final report exceeds 1 MiB")
	}
	return nil
}

func (r RuntimeReport) validateAdapters() error {
	if r.Resources != nil {
		if err := r.Resources.Validate(); err != nil {
			return err
		}
	}
	if len(r.Adapters) > 64 {
		return contracts.Invalidf("runtime adapter metrics exceed 64 entries")
	}
	for ref, metrics := range r.Adapters {
		if err := ref.Validate(); err != nil {
			return err
		}
		if err := metrics.Validate(); err != nil {
			return err
		}
	}
	return nil
}

var runtimeAdapterErrorCodes = map[string]struct{}{
	"close_failed": {}, "delivery_failed": {}, "flush_failed": {},
	"flush_timeout": {}, "queue_overflow": {}, "request_failed": {},
}

type RuntimeAdapterMetrics struct {
	Operations       uint64             `json:"operations" strictjson:"required"`
	FailedOperations uint64             `json:"failedOperations" strictjson:"required"`
	FlushAttempted   *bool              `json:"flushAttempted,omitempty" strictjson:"nullable"`
	FlushSucceeded   *bool              `json:"flushSucceeded,omitempty" strictjson:"nullable"`
	LastErrorCode    *string            `json:"lastErrorCode,omitempty" strictjson:"nullable"`
	DroppedSpans     *DroppedSpanCounts `json:"droppedSpans,omitempty" strictjson:"nullable"`
}

func (m RuntimeAdapterMetrics) Validate() error {
	if m.FailedOperations > m.Operations {
		return contracts.Invalidf("Runtime adapter failures exceed operations")
	}
	if (m.FlushAttempted == nil) != (m.FlushSucceeded == nil) {
		return contracts.Invalidf("Runtime adapter flush fields must be present together")
	}
	if m.FlushAttempted != nil && !*m.FlushAttempted && *m.FlushSucceeded {
		return contracts.Invalidf("Runtime adapter flush cannot succeed when not attempted")
	}
	if m.LastErrorCode != nil {
		if _, allowed := runtimeAdapterErrorCodes[*m.LastErrorCode]; !allowed {
			return contracts.Invalidf("Runtime adapter lastErrorCode is invalid")
		}
	}
	return nil
}

func (r RuntimeReport) Validate() error {
	if r.Resources != nil {
		if err := r.Resources.Validate(); err != nil {
			return err
		}
	}
	if r.DurationMS != nil && *r.DurationMS < 0 {
		return contracts.Invalidf("runtime duration must be non-negative")
	}
	if r.StopReason != nil && strings.TrimSpace(*r.StopReason) == "" {
		return contracts.Invalidf("runtime stopReason must not be empty")
	}
	if r.Adapters == nil || len(r.Adapters) > 64 {
		return contracts.Invalidf("runtime adapter metrics map is invalid")
	}
	for ref, metrics := range r.Adapters {
		if err := ref.Validate(); err != nil {
			return err
		}
		if err := metrics.Validate(); err != nil {
			return err
		}
	}
	return nil
}

type AllocationFinalResponse struct {
	APIVersion string                `json:"apiVersion" strictjson:"required"`
	Report     AllocationFinalReport `json:"report" strictjson:"required"`
}

func (r AllocationFinalResponse) Validate() error {
	if err := contracts.ValidateAPIVersion(r.APIVersion); err != nil {
		return err
	}
	return r.Report.Validate()
}
