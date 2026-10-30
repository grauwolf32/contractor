package reporting

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
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

// ToolArguments is the bounded, redacted argument summary of one tool call;
// an empty summary encodes as an absent member, as the Runtime writes it. A
// decoded number keeps the literal its producer wrote (json.Number), so the
// Server encodes, bounds and measures the bytes the Runtime measured
// (OBS-R42), and an integer beyond 2^53 keeps its value.
type ToolArguments map[string]any

// UnmarshalJSON decodes the summary with its number literals kept. It
// rejects a repeated member name at any depth and data after the value
// itself, whatever decoder its caller used (AGENTS.md section 6); the private
// codec rejects null, and validation checks the numbers (LIM-OBS-16) and the
// size.
func (a *ToolArguments) UnmarshalJSON(data []byte) error {
	if err := strictjson.RejectDuplicateKeys(data); err != nil {
		return err
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var value map[string]any
	if err := decoder.Decode(&value); err != nil {
		return err
	}
	*a = value
	return nil
}

// Tool argument numbers (LIM-OBS-16) keep only a small decimal exponent:
// PostgreSQL stores a jsonb number as its full decimal expansion, so a stored
// number renders at most 20 integer digits and 7 decimal places beyond its own
// fraction digits. -7 keeps probabilities to one in ten million and float32
// epsilon; 19 keeps every 64-bit integer, which includes millisecond and
// nanosecond timestamps, counts and sizes.
const (
	minArgumentNumberExponent = -7
	maxArgumentNumberExponent = 19
	// ArgumentNumberMarker replaces a number a summary cannot keep: one
	// outside LIM-OBS-16, NaN or an infinity.
	ArgumentNumberMarker = "[NUMBER_OUT_OF_RANGE]"
)

// numbersInRange reports whether every number in the summary stays within
// LIM-OBS-16.
func (a ToolArguments) numbersInRange() bool {
	return argumentNumbersInRange(map[string]any(a))
}

func argumentNumbersInRange(value any) bool {
	switch typed := value.(type) {
	case map[string]any:
		for _, item := range typed {
			if !argumentNumbersInRange(item) {
				return false
			}
		}
	case []any:
		for _, item := range typed {
			if !argumentNumbersInRange(item) {
				return false
			}
		}
	default:
		return !ArgumentNumberOutOfRange(value)
	}
	return true
}

// ArgumentNumberOutOfRange reports whether value is a number that a tool
// argument summary cannot keep (LIM-OBS-16), judged by the JSON it is written
// as: the literal a producer wrote (json.Number) or encoding/json's text for a
// number the Server built. A number is kept when it is zero or the decimal
// exponent of its leading digit lies between minArgumentNumberExponent and
// maxArgumentNumberExponent, and it writes no exponent outside that range.
// Any other value is no number and reports false.
func ArgumentNumberOutOfRange(value any) bool {
	var literal string
	switch typed := value.(type) {
	case json.Number:
		literal = string(typed)
	case float64, float32, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		encoded, err := json.Marshal(typed)
		if err != nil { // NaN or an infinity
			return true
		}
		literal = string(encoded)
	default:
		return false
	}
	return !argumentLiteralInRange(literal)
}

func argumentLiteralInRange(literal string) bool {
	mantissa, exponentText, hasExponent := strings.Cut(strings.ToLower(literal), "e")
	exponent := 0
	if hasExponent {
		negative := strings.HasPrefix(exponentText, "-")
		digits := strings.TrimLeft(strings.TrimLeft(exponentText, "+-"), "0")
		if len(digits) > 2 {
			return false
		}
		exponent, _ = strconv.Atoi("0" + digits)
		if negative {
			exponent = -exponent
		}
		if exponent < minArgumentNumberExponent || exponent > maxArgumentNumberExponent {
			return false
		}
	}
	integer, fraction, _ := strings.Cut(strings.TrimPrefix(mantissa, "-"), ".")
	lead := 0
	if index := strings.IndexFunc(integer, isNonZeroDigit); index >= 0 {
		lead = len(integer) - 1 - index
	} else if index := strings.IndexFunc(fraction, isNonZeroDigit); index >= 0 {
		lead = -(index + 1)
	} else {
		return true // zero, whose written exponent is already in range
	}
	return lead+exponent >= minArgumentNumberExponent && lead+exponent <= maxArgumentNumberExponent
}

func isNonZeroDigit(r rune) bool {
	return r >= '1' && r <= '9'
}

type ToolCallRecord struct {
	CallID             string          `json:"callId" strictjson:"required"`
	Tool               string          `json:"tool" strictjson:"required"`
	Arguments          ToolArguments   `json:"arguments,omitempty"`
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

// UnmarshalJSON strictly decodes a report wherever it comes from: the private
// response that carries it, which bounds its received bytes, or the jsonb
// column that stores it, which renders it longer than it was received
// (OBS-R110).
func (r *AllocationFinalReport) UnmarshalJSON(data []byte) error {
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

// Validate checks a report as its producer sent it, including the byte bounds
// its compact JSON must meet (LIM-OBS-6, OBS-R40).
func (r ExecutionReport) Validate() error {
	return r.validate(true)
}

// validateStored checks a report read back from storage. It applies every
// rule of Validate except the byte bounds: jsonb renders each number as its
// full decimal expansion, so the stored report no longer measures the bytes it
// met when it was received and trimmed (OBS-R110).
func (r ExecutionReport) validateStored() error {
	return r.validate(false)
}

// metricIntegerInRange reports whether value may be an integer of
// ExecutionMetrics: every one lies between 0 and contracts.MaxExactInteger, so the
// sums the Server builds from them stay far from the int64 range and every
// JSON consumer reads them exactly (OBS-R111, LIM-OBS-17).
func metricIntegerInRange(value int64) bool {
	return value >= 0 && value <= contracts.MaxExactInteger
}

func (r ExecutionReport) validate(received bool) error {
	if r.Completion != nil {
		if err := r.Completion.Validate(); err != nil {
			return err
		}
	}
	if err := contracts.ValidateOpaqueID("reportId", r.ReportID); err != nil {
		return err
	}
	if r.ToolCalls == nil || r.Errors == nil {
		return contracts.Invalidf("execution report lists must not be null")
	}
	if err := r.Metrics.Validate(); err != nil {
		return err
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
			if !call.Arguments.numbersInRange() {
				return contracts.Invalidf("tool call %d arguments hold a number outside their range", index)
			}
			if size, err := contracts.ResultJSONSize(call.Arguments); err != nil || received && size > 4096 {
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
	if received && size > 1024*1024 {
		return contracts.Invalidf("execution report exceeds 1 MiB")
	}
	return nil
}

// Validate checks execution metrics wherever they are read: in a received
// report, in a stored one and in the counters an Eval reads from stored
// StageMetrics (EVM-R81). Tool names are opaque IDs, every integer lies
// between 0 and 2^53-1 (OBS-R111), and the Worker budget and the summarizer
// aggregate are consistent.
func (m ExecutionMetrics) Validate() error {
	if m.Tools == nil {
		return contracts.Invalidf("execution metrics tools must not be null")
	}
	for name, metrics := range m.Tools {
		if err := contracts.ValidateOpaqueID("metrics tool name", name); err != nil {
			return err
		}
		for _, value := range []*int64{metrics.Calls, metrics.Succeeded, metrics.Failed} {
			if value != nil && !metricIntegerInRange(*value) {
				return contracts.Invalidf("tool metrics must lie between 0 and 2^53-1")
			}
		}
	}
	for _, value := range []*int64{
		m.DurationMS, m.ModelCalls, m.InputTokens,
		m.OutputTokens, m.TotalTokens,
	} {
		if value != nil && !metricIntegerInRange(*value) {
			return contracts.Invalidf("execution metrics must lie between 0 and 2^53-1")
		}
	}
	if budget := m.WorkerBudget; budget != nil {
		if budget.MaxModelCalls <= 0 || budget.MaxModelCalls > contracts.MaxWorkerModelCalls ||
			budget.MaxToolCalls <= 0 || budget.MaxToolCalls > contracts.MaxWorkerToolCalls ||
			budget.MaxTotalTokens <= 0 || budget.MaxTotalTokens > contracts.MaxWorkerTotalTokens {
			return contracts.Invalidf("worker budget limits are invalid")
		}
		if budget.ObservedModelCalls < 0 || budget.ObservedModelCalls > budget.MaxModelCalls ||
			budget.ObservedToolCalls < 0 || budget.ObservedToolCalls > budget.MaxToolCalls ||
			!metricIntegerInRange(budget.ObservedTotalTokens) ||
			!metricIntegerInRange(budget.TokenUsageUnavailable) {
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
	if summarizer := m.Summarizer; summarizer != nil {
		for _, value := range []uint64{
			summarizer.Attempts, summarizer.Succeeded, summarizer.Failed, summarizer.ModelCalls,
			summarizer.InputTokens, summarizer.OutputTokens, summarizer.TotalTokens,
			summarizer.TokenUsageUnavailable,
		} {
			if value > contracts.MaxExactInteger {
				return contracts.Invalidf("worker summarizer metrics must not exceed 2^53-1")
			}
		}
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

// Validate checks StageMetrics built from stored reports, so each report is
// checked as a stored report (validateStored).
func (m StageMetrics) Validate() error {
	if m.Workers == nil || m.Runtime == nil {
		return contracts.Invalidf("StageMetrics maps must not be null")
	}
	if m.Planner != nil {
		if err := m.Planner.validateStored(); err != nil {
			return fmt.Errorf("planner report: %w", err)
		}
	}
	for name, report := range m.Workers {
		if err := contracts.ValidateOpaqueID("worker logical Agent name", name); err != nil {
			return err
		}
		if err := report.validateStored(); err != nil {
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

// UnmarshalJSON bounds the report a Runtime sent at 1 MiB (LIM-OBS-7) before
// any of it is decoded, so the original bytes are measured before optional
// invalid resources and adapter metrics are discarded. The bound applies to
// these received bytes only; a stored report is decoded without it.
func (r *AllocationFinalResponse) UnmarshalJSON(data []byte) error {
	var wire struct {
		APIVersion string          `json:"apiVersion" strictjson:"required"`
		Report     json.RawMessage `json:"report" strictjson:"required"`
	}
	if err := strictjson.DecodeNonNull(data, &wire); err != nil {
		return err
	}
	if len(wire.Report) > 1024*1024 {
		return contracts.Invalidf("allocation final report exceeds 1 MiB")
	}
	var report AllocationFinalReport
	if err := strictjson.DecodeNonNull(wire.Report, &report); err != nil {
		return err
	}
	*r = AllocationFinalResponse{APIVersion: wire.APIVersion, Report: report}
	return nil
}

func (r AllocationFinalResponse) Validate() error {
	if err := contracts.ValidateAPIVersion(r.APIVersion); err != nil {
		return err
	}
	return r.Report.Validate()
}
