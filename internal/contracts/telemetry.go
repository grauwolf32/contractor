package contracts

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"
)

type ExecutionError struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	Retryable *bool  `json:"retryable,omitempty"`
}

type ToolMetrics struct {
	Calls     *int64 `json:"calls,omitempty"`
	Succeeded *int64 `json:"succeeded,omitempty"`
	Failed    *int64 `json:"failed,omitempty"`
}

type WorkerBudgetMetrics struct {
	MaxModelCalls         int64   `json:"maxModelCalls"`
	MaxToolCalls          int64   `json:"maxToolCalls"`
	MaxTotalTokens        int64   `json:"maxTotalTokens"`
	ObservedModelCalls    int64   `json:"observedModelCalls"`
	ObservedToolCalls     int64   `json:"observedToolCalls"`
	ObservedTotalTokens   int64   `json:"observedTotalTokens"`
	TokenUsageUnavailable int64   `json:"tokenUsageUnavailable"`
	Exhausted             *string `json:"exhausted,omitempty"`
}

type WorkerSummarizerMetrics struct {
	Attempts              uint64            `json:"attempts"`
	Succeeded             uint64            `json:"succeeded"`
	Failed                uint64            `json:"failed"`
	ModelCalls            uint64            `json:"modelCalls"`
	InputTokens           uint64            `json:"inputTokens"`
	OutputTokens          uint64            `json:"outputTokens"`
	TotalTokens           uint64            `json:"totalTokens"`
	TokenUsageUnavailable uint64            `json:"tokenUsageUnavailable"`
	FailureCodes          map[string]uint64 `json:"failureCodes"`
}

type ExecutionMetrics struct {
	DurationMS   *int64                   `json:"durationMs,omitempty"`
	ModelCalls   *int64                   `json:"modelCalls,omitempty"`
	InputTokens  *int64                   `json:"inputTokens,omitempty"`
	OutputTokens *int64                   `json:"outputTokens,omitempty"`
	TotalTokens  *int64                   `json:"totalTokens,omitempty"`
	Tools        map[string]ToolMetrics   `json:"tools"`
	WorkerBudget *WorkerBudgetMetrics     `json:"workerBudget,omitempty"`
	Summarizer   *WorkerSummarizerMetrics `json:"summarizer,omitempty"`
}

type ToolCallOutcome string

const (
	ToolCallSucceeded ToolCallOutcome = "succeeded"
	ToolCallFailed    ToolCallOutcome = "failed"
)

type ToolCallRecord struct {
	CallID             string          `json:"callId"`
	Tool               string          `json:"tool"`
	Arguments          map[string]any  `json:"arguments,omitempty"`
	ArgumentsTruncated bool            `json:"argumentsTruncated"`
	Outcome            ToolCallOutcome `json:"outcome"`
	DurationMS         *int64          `json:"durationMs,omitempty"`
	ResultSizeBytes    *int64          `json:"resultSizeBytes,omitempty"`
	Error              *ExecutionError `json:"error,omitempty"`
}

type ExecutionReport struct {
	ReportID  string           `json:"reportId"`
	Complete  bool             `json:"complete"`
	Metrics   ExecutionMetrics `json:"metrics"`
	ToolCalls []ToolCallRecord `json:"toolCalls"`
	Errors    []ExecutionError `json:"errors"`
	Truncated bool             `json:"truncated"`
}

type RuntimeReport struct {
	Complete       bool                                          `json:"complete"`
	DurationMS     *int64                                        `json:"durationMs,omitempty"`
	StopReason     *string                                       `json:"stopReason,omitempty"`
	Adapters       map[RuntimeAdapterRef]RuntimeAdapterMetricsV2 `json:"adapters"`
	Resources      *RuntimeResources                             `json:"resources,omitempty"`
	ResourcesError *ResourceReason                               `json:"-"`
}

func (r RuntimeReport) MarshalJSON() ([]byte, error) {
	type wireRuntimeReport RuntimeReport
	copy := wireRuntimeReport(r)
	if copy.Adapters == nil {
		copy.Adapters = map[RuntimeAdapterRef]RuntimeAdapterMetricsV2{}
	}
	return json.Marshal(copy)
}

func (r *RuntimeReport) UnmarshalJSON(data []byte) error {
	if err := rejectDuplicateJSONKeys(data); err != nil {
		return invalidf("invalid runtime report JSON")
	}
	type wireRuntimeReport struct {
		Complete   bool                       `json:"complete"`
		DurationMS *int64                     `json:"durationMs,omitempty"`
		StopReason *string                    `json:"stopReason,omitempty"`
		Adapters   map[string]json.RawMessage `json:"adapters"`
		Resources  json.RawMessage            `json:"resources"`
	}
	var wire wireRuntimeReport
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&wire); err != nil {
		return err
	}
	if err := requireTelemetryJSONEOF(decoder); err != nil {
		return err
	}
	*r = RuntimeReport{
		Complete: wire.Complete, DurationMS: wire.DurationMS, StopReason: wire.StopReason,
		Adapters: map[RuntimeAdapterRef]RuntimeAdapterMetricsV2{},
	}
	r.Resources, r.ResourcesError = decodeOptionalResources(wire.Resources)
	if wire.Adapters == nil {
		r.Complete = false
		return nil
	}
	for rawRef, rawMetrics := range wire.Adapters {
		ref := RuntimeAdapterRef(rawRef)
		metrics, err := decodeRuntimeAdapterMetrics(rawMetrics)
		if err != nil || ref.Validate() != nil {
			r.Complete = false
			continue
		}
		r.Adapters[ref] = metrics
	}
	return nil
}

func decodeRuntimeAdapterMetrics(data []byte) (RuntimeAdapterMetricsV2, error) {
	var metrics RuntimeAdapterMetricsV2
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&metrics); err != nil {
		return RuntimeAdapterMetricsV2{}, err
	}
	if err := requireTelemetryJSONEOF(decoder); err != nil {
		return RuntimeAdapterMetricsV2{}, err
	}
	if err := metrics.Validate(); err != nil {
		return RuntimeAdapterMetricsV2{}, err
	}
	return metrics, nil
}

func requireTelemetryJSONEOF(decoder *json.Decoder) error {
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			return fmt.Errorf("unexpected trailing JSON value")
		}
		return err
	}
	return nil
}

type StageMetrics struct {
	Planner *ExecutionReport           `json:"planner,omitempty"`
	Workers map[string]ExecutionReport `json:"workers"`
	Runtime map[string]RuntimeReport   `json:"runtime"`
}

type AllocationFinalReport struct {
	ReportID     string          `json:"reportId"`
	AllocationID string          `json:"allocationId"`
	StartedAt    time.Time       `json:"startedAt"`
	FinishedAt   time.Time       `json:"finishedAt"`
	Worker       ExecutionReport `json:"worker"`
	Runtime      RuntimeReport   `json:"runtime"`
}

func (r *AllocationFinalReport) UnmarshalJSON(data []byte) error {
	// Check the original report before optional invalid resources are removed.
	if len(data) > 1024*1024 {
		return invalidf("allocation final report exceeds 1 MiB")
	}
	if err := rejectDuplicateJSONKeys(data); err != nil {
		return invalidf("invalid allocation report JSON")
	}
	type wireReport AllocationFinalReport
	var value wireReport
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&value); err != nil {
		return err
	}
	*r = AllocationFinalReport(value)
	return nil
}

func (r ExecutionReport) Validate() error {
	if err := validateOpaqueID("reportId", r.ReportID); err != nil {
		return err
	}
	if r.Metrics.Tools == nil || r.ToolCalls == nil || r.Errors == nil {
		return invalidf("execution report maps and lists must not be null")
	}
	for name, metrics := range r.Metrics.Tools {
		if err := validateOpaqueID("metrics tool name", name); err != nil {
			return err
		}
		for _, value := range []*int64{metrics.Calls, metrics.Succeeded, metrics.Failed} {
			if value != nil && *value < 0 {
				return invalidf("tool metrics must be non-negative")
			}
		}
	}
	for _, value := range []*int64{
		r.Metrics.DurationMS, r.Metrics.ModelCalls, r.Metrics.InputTokens,
		r.Metrics.OutputTokens, r.Metrics.TotalTokens,
	} {
		if value != nil && *value < 0 {
			return invalidf("execution metrics must be non-negative")
		}
	}
	if budget := r.Metrics.WorkerBudget; budget != nil {
		if budget.MaxModelCalls <= 0 || budget.MaxModelCalls > MaxWorkerModelCalls ||
			budget.MaxToolCalls <= 0 || budget.MaxToolCalls > MaxWorkerToolCalls ||
			budget.MaxTotalTokens <= 0 || budget.MaxTotalTokens > MaxWorkerTotalTokens {
			return invalidf("worker budget limits are invalid")
		}
		if budget.ObservedModelCalls < 0 || budget.ObservedModelCalls > budget.MaxModelCalls ||
			budget.ObservedToolCalls < 0 || budget.ObservedToolCalls > budget.MaxToolCalls ||
			budget.ObservedTotalTokens < 0 || budget.TokenUsageUnavailable < 0 {
			return invalidf("worker budget observations are invalid")
		}
		if budget.Exhausted != nil {
			switch *budget.Exhausted {
			case "model_calls":
				if budget.ObservedModelCalls != budget.MaxModelCalls {
					return invalidf("worker model-call exhaustion is inconsistent")
				}
			case "tool_calls":
				if budget.ObservedToolCalls != budget.MaxToolCalls {
					return invalidf("worker tool-call exhaustion is inconsistent")
				}
			case "total_tokens":
				if budget.ObservedTotalTokens < budget.MaxTotalTokens {
					return invalidf("worker token exhaustion is inconsistent")
				}
			default:
				return invalidf("worker budget exhausted dimension is invalid")
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
			return invalidf("worker summarizer metrics are inconsistent")
		}
		var failures uint64
		for code, count := range summarizer.FailureCodes {
			if !validWorkerFailureCode(code) || count == 0 || ^uint64(0)-failures < count {
				return invalidf("worker summarizer failure metrics are invalid")
			}
			failures += count
		}
		if failures != summarizer.Failed {
			return invalidf("worker summarizer failure metrics are inconsistent")
		}
	}
	for index, call := range r.ToolCalls {
		if err := validateOpaqueID("tool call ID", call.CallID); err != nil {
			return err
		}
		if err := validateOpaqueID("tool call name", call.Tool); err != nil {
			return err
		}
		if call.Arguments == nil && call.ArgumentsTruncated {
			return invalidf("tool call %d marks absent arguments truncated", index)
		}
		if call.Arguments != nil {
			encoded, err := json.Marshal(call.Arguments)
			if err != nil || len(encoded) > 4096 {
				return invalidf("tool call %d arguments exceed 4096 bytes", index)
			}
		}
		if call.Outcome != ToolCallSucceeded && call.Outcome != ToolCallFailed {
			return invalidf("unknown tool call outcome %q", call.Outcome)
		}
		if (call.DurationMS != nil && *call.DurationMS < 0) ||
			(call.ResultSizeBytes != nil && *call.ResultSizeBytes < 0) {
			return invalidf("tool call measurements must be non-negative")
		}
		if (call.Outcome == ToolCallSucceeded && call.Error != nil) ||
			(call.Outcome == ToolCallFailed && call.Error == nil) {
			return invalidf("tool call outcome and error are inconsistent")
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
		return invalidf("execution report detail exceeds bounded record limits")
	}
	encoded, err := json.Marshal(r)
	if err != nil {
		return invalidf("execution report cannot be encoded")
	}
	if len(encoded) > 1024*1024 {
		return invalidf("execution report exceeds 1 MiB")
	}
	return nil
}

func validWorkerFailureCode(value string) bool {
	return workerFailureCode.MatchString(value)
}

func (e ExecutionError) Validate() error {
	if strings.TrimSpace(e.Code) == "" || strings.TrimSpace(e.Message) == "" {
		return invalidf("execution error code and message are required")
	}
	if len([]byte(e.Message)) > 4096 {
		return invalidf("execution error message exceeds 4096 bytes")
	}
	return nil
}

func (m StageMetrics) Validate() error {
	if m.Workers == nil || m.Runtime == nil {
		return invalidf("StageMetrics maps must not be null")
	}
	if m.Planner != nil {
		if err := m.Planner.Validate(); err != nil {
			return fmt.Errorf("planner report: %w", err)
		}
	}
	for name, report := range m.Workers {
		if err := validateOpaqueID("worker logical Agent name", name); err != nil {
			return err
		}
		if err := report.Validate(); err != nil {
			return fmt.Errorf("worker report %q: %w", name, err)
		}
	}
	for name, report := range m.Runtime {
		if err := validateOpaqueID("runtime logical Agent name", name); err != nil {
			return err
		}
		if report.DurationMS != nil && *report.DurationMS < 0 {
			return invalidf("runtime duration must be non-negative")
		}
		if err := report.validateAdapters(); err != nil {
			return err
		}
	}
	return nil
}

func (r AllocationFinalReport) Validate() error {
	if err := validateOpaqueID("allocation final report ID", r.ReportID); err != nil {
		return err
	}
	if err := validateOpaqueID("allocationId", r.AllocationID); err != nil {
		return err
	}
	if r.StartedAt.IsZero() || r.FinishedAt.IsZero() || r.FinishedAt.Before(r.StartedAt) {
		return invalidf("allocation final report timestamps are invalid")
	}
	if err := r.Worker.Validate(); err != nil {
		return fmt.Errorf("worker report: %w", err)
	}
	if r.Runtime.DurationMS != nil && *r.Runtime.DurationMS < 0 {
		return invalidf("runtime duration must be non-negative")
	}
	if r.Runtime.StopReason != nil && strings.TrimSpace(*r.Runtime.StopReason) == "" {
		return invalidf("runtime stopReason must not be empty")
	}
	if err := r.Runtime.validateAdapters(); err != nil {
		return err
	}
	encoded, err := json.Marshal(r)
	if err != nil {
		return invalidf("allocation final report cannot be encoded")
	}
	if len(encoded) > 1024*1024 {
		return invalidf("allocation final report exceeds 1 MiB")
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
		return invalidf("runtime adapter metrics exceed 64 entries")
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
