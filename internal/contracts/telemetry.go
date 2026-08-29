package contracts

import (
	"encoding/json"
	"fmt"
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

type ExecutionMetrics struct {
	DurationMS   *int64                 `json:"durationMs,omitempty"`
	ModelCalls   *int64                 `json:"modelCalls,omitempty"`
	InputTokens  *int64                 `json:"inputTokens,omitempty"`
	OutputTokens *int64                 `json:"outputTokens,omitempty"`
	TotalTokens  *int64                 `json:"totalTokens,omitempty"`
	Tools        map[string]ToolMetrics `json:"tools"`
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
	Complete   bool    `json:"complete"`
	DurationMS *int64  `json:"durationMs,omitempty"`
	StopReason *string `json:"stopReason,omitempty"`
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
	encoded, err := json.Marshal(r)
	if err != nil {
		return invalidf("allocation final report cannot be encoded")
	}
	if len(encoded) > 1024*1024 {
		return invalidf("allocation final report exceeds 1 MiB")
	}
	return nil
}
