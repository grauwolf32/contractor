package telemetry

import (
	"errors"
	"fmt"
	"net/url"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
)

const (
	MaxToolRecords        = 1000
	MaxErrorRecords       = 100
	MaxArgumentBytes      = 4096
	MaxErrorMessageBytes  = 4096
	MaxReportJSONBytes    = 1024 * 1024
	redactedValue         = "[REDACTED]"
	redactedURL           = "[REDACTED_URL]"
	truncatedValue        = "[TRUNCATED]"
	allocationReserveSize = 4096
)

var sensitiveKeys = map[string]struct{}{
	"apikey": {}, "authorization": {}, "body": {}, "bytes": {}, "content": {},
	"cookie": {}, "data": {}, "databaseurl": {}, "llmgatewaytoken": {},
	"password": {}, "payload": {}, "proxyauthorization": {}, "secret": {},
	"setcookie": {}, "token": {},
}

// Policy is the mandatory Control Plane persistence-boundary sanitizer.
// Secrets are copied so callers can erase their own configuration independently.
type Policy struct {
	secrets []string
}

func NewPolicy(secrets ...string) Policy {
	filtered := make([]string, 0, len(secrets))
	for _, secret := range secrets {
		if secret != "" {
			filtered = append(filtered, secret)
		}
	}
	return Policy{secrets: filtered}
}

func (p Policy) NormalizeAllocationReport(
	source reporting.AllocationFinalReport,
) (reporting.AllocationFinalReport, error) {
	result := source
	worker, err := p.NormalizeExecutionReport(source.Worker, MaxReportJSONBytes-allocationReserveSize)
	if err != nil {
		return reporting.AllocationFinalReport{}, err
	}
	result.Worker = worker
	if result.Runtime.StopReason != nil {
		value := p.redactText(*result.Runtime.StopReason)
		result.Runtime.StopReason = &value
	}
	worker, fits, err := trimDetail(result.Worker, MaxReportJSONBytes,
		func(candidate reporting.ExecutionReport) int {
			report := result
			report.Worker = candidate
			return encodedSize(report)
		})
	if err != nil {
		return reporting.AllocationFinalReport{}, fmt.Errorf("normalize allocation report: %w", err)
	}
	if !fits {
		return reporting.AllocationFinalReport{}, errors.New("allocation report aggregate exceeds 1 MiB")
	}
	result.Worker = worker
	if err := result.Validate(); err != nil {
		return reporting.AllocationFinalReport{}, fmt.Errorf("normalize allocation report: %w", err)
	}
	return result, nil
}

func (p Policy) NormalizeExecutionReport(
	source reporting.ExecutionReport,
	maximumBytes int,
) (reporting.ExecutionReport, error) {
	if maximumBytes <= 0 || maximumBytes > MaxReportJSONBytes {
		return reporting.ExecutionReport{}, errors.New("invalid execution report byte limit")
	}
	result := source
	result.Metrics.Tools = cloneToolMetrics(source.Metrics.Tools)
	if source.Completion != nil {
		value := *source.Completion
		result.Completion = &value
	}
	result.Metrics.WorkerBudget = cloneWorkerBudget(source.Metrics.WorkerBudget)
	result.Metrics.Summarizer = cloneWorkerSummarizer(source.Metrics.Summarizer)
	if result.Metrics.Tools == nil {
		result.Metrics.Tools = map[string]reporting.ToolMetrics{}
	}
	result.ToolCalls = make([]reporting.ToolCallRecord, 0, min(len(source.ToolCalls), MaxToolRecords))
	start := max(0, len(source.ToolCalls)-MaxToolRecords)
	if start > 0 {
		result.Truncated = true
	}
	for _, sourceCall := range source.ToolCalls[start:] {
		call := sourceCall
		if sourceCall.Arguments != nil {
			arguments, changed := p.sanitizeMap(sourceCall.Arguments)
			call.Arguments = arguments
			call.ArgumentsTruncated = call.ArgumentsTruncated || changed
			if size := encodedSize(arguments); size > MaxArgumentBytes {
				call.Arguments = map[string]any{
					"summary":           truncatedValue,
					"originalSizeBytes": size,
				}
				call.ArgumentsTruncated = true
			}
		}
		if call.Error != nil {
			normalized := p.normalizeError(*call.Error)
			call.Error = &normalized
		}
		result.ToolCalls = append(result.ToolCalls, call)
	}
	result.Errors = make([]reporting.ExecutionError, 0, min(len(source.Errors), MaxErrorRecords))
	errorStart := max(0, len(source.Errors)-MaxErrorRecords)
	if errorStart > 0 {
		result.Truncated = true
	}
	for _, item := range source.Errors[errorStart:] {
		result.Errors = append(result.Errors, p.normalizeError(item))
	}
	result, fits, err := trimDetail(result, maximumBytes, func(candidate reporting.ExecutionReport) int {
		return encodedSize(candidate)
	})
	if err != nil {
		return reporting.ExecutionReport{}, fmt.Errorf("normalize execution report: %w", err)
	}
	if !fits {
		return reporting.ExecutionReport{}, errors.New("execution report aggregate exceeds byte limit")
	}
	if err := result.Validate(); err != nil {
		return reporting.ExecutionReport{}, fmt.Errorf("normalize execution report: %w", err)
	}
	return result, nil
}

func cloneWorkerSummarizer(
	source *reporting.WorkerSummarizerMetrics,
) *reporting.WorkerSummarizerMetrics {
	if source == nil {
		return nil
	}
	result := *source
	result.FailureCodes = make(map[string]uint64, len(source.FailureCodes))
	for code, count := range source.FailureCodes {
		result.FailureCodes[code] = count
	}
	return &result
}

func (p Policy) normalizeError(source reporting.ExecutionError) reporting.ExecutionError {
	result := source
	result.Code = p.redactText(source.Code)
	result.Message = p.RedactMessage(source.Message)
	return result
}

// RedactMessage applies the rule of an execution-report error message to
// text: known secrets and credential-bearing URLs are removed (OBS-R30) and
// the result is at most MaxErrorMessageBytes long (LIM-OBS-6). The Scheduler
// logs the cause of a Run it could not progress through it (SCH-R170).
func (p Policy) RedactMessage(text string) string {
	return truncateUTF8(p.redactText(text), MaxErrorMessageBytes)
}

func (p Policy) sanitizeMap(source map[string]any) (map[string]any, bool) {
	result := make(map[string]any, len(source))
	changed := false
	for key, value := range source {
		targetKey := truncateUTF8(p.redactText(key), MaxArgumentBytes)
		changed = changed || targetKey != key
		if isSensitiveKey(key) {
			result[targetKey] = map[string]any{"redacted": true, "size": valueSize(value)}
			changed = true
			continue
		}
		normalized, itemChanged := p.sanitizeValue(value, 0)
		result[targetKey] = normalized
		changed = changed || itemChanged
	}
	return result, changed
}

func (p Policy) sanitizeValue(source any, depth int) (any, bool) {
	if depth >= 4 {
		return truncatedValue, true
	}
	switch value := source.(type) {
	case nil, bool, float64, float32, int, int8, int16, int32, int64,
		uint, uint8, uint16, uint32, uint64:
		return value, false
	case string:
		result := p.redactText(value)
		return truncateUTF8(result, MaxArgumentBytes), result != value || len(result) > MaxArgumentBytes
	case []byte:
		return map[string]any{"bytes": len(value)}, true
	case map[string]any:
		result := make(map[string]any, len(value))
		changed := false
		for key, item := range value {
			targetKey := truncateUTF8(p.redactText(key), MaxArgumentBytes)
			changed = changed || targetKey != key
			if isSensitiveKey(key) {
				result[targetKey] = map[string]any{"redacted": true, "size": valueSize(item)}
				changed = true
				continue
			}
			normalized, itemChanged := p.sanitizeValue(item, depth+1)
			result[targetKey] = normalized
			changed = changed || itemChanged
		}
		return result, changed
	case []any:
		limit := min(len(value), 32)
		result := make([]any, 0, limit+1)
		changed := len(value) > limit
		for _, item := range value[:limit] {
			normalized, itemChanged := p.sanitizeValue(item, depth+1)
			result = append(result, normalized)
			changed = changed || itemChanged
		}
		if len(value) > limit {
			result = append(result, truncatedValue)
		}
		return result, changed
	default:
		return "<" + fmt.Sprintf("%T", source) + ">", true
	}
}

func (p Policy) redactText(source string) string {
	result := source
	for _, secret := range p.secrets {
		result = strings.ReplaceAll(result, secret, redactedValue)
	}
	if credentialBearingURL(result) {
		return redactedURL
	}
	return result
}

func credentialBearingURL(value string) bool {
	if !strings.Contains(value, "://") {
		return false
	}
	parsed, err := url.Parse(value)
	if err == nil && parsed.Scheme != "" && parsed.Host != "" {
		return parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != ""
	}
	return strings.Contains(value, "@") || strings.Contains(value, "?") || strings.Contains(value, "#")
}

func normalizeKey(value string) string {
	return strings.NewReplacer("_", "", "-", "", " ", "").Replace(strings.ToLower(value))
}

func isSensitiveKey(value string) bool {
	normalized := normalizeKey(value)
	if _, sensitive := sensitiveKeys[normalized]; sensitive {
		return true
	}
	for _, fragment := range []string{
		"apikey", "authorization", "base64", "cookie", "password", "secret", "token",
	} {
		if strings.Contains(normalized, fragment) {
			return true
		}
	}
	for _, suffix := range []string{"body", "bytes", "content", "data", "payload"} {
		if strings.HasSuffix(normalized, suffix) {
			return true
		}
	}
	return false
}

func cloneToolMetrics(source map[string]reporting.ToolMetrics) map[string]reporting.ToolMetrics {
	result := make(map[string]reporting.ToolMetrics, len(source))
	for name, metrics := range source {
		result[name] = metrics
	}
	return result
}

func cloneWorkerBudget(source *reporting.WorkerBudgetMetrics) *reporting.WorkerBudgetMetrics {
	if source == nil {
		return nil
	}
	result := *source
	if source.Exhausted != nil {
		exhausted := *source.Exhausted
		result.Exhausted = &exhausted
	}
	return &result
}

// trimDetail bounds a report whose complete JSON, as measure counts it, may
// exceed maximum. Overflow removes the oldest tool-call record first and the
// oldest error only after no tool-call record remains, and marks the report
// truncated (OBS-R41): the result is the first report of that removal
// sequence that fits. The JSON of a list is its elements' JSON joined by
// commas, so each record is measured once and every removal is subtracted
// from the size; measure runs at most three times, for the candidate, for the
// envelope without detail and for the retained result, however many records
// are removed (OBS-R110). fits is false when even the envelope exceeds
// maximum. An error means the computed and the measured size of the retained
// result disagree, which the JSON encoding rules exclude.
func trimDetail(
	report reporting.ExecutionReport,
	maximum int,
	measure func(reporting.ExecutionReport) int,
) (result reporting.ExecutionReport, fits bool, err error) {
	if measure(report) <= maximum {
		return report, true, nil
	}
	envelope := report
	envelope.ToolCalls = []reporting.ToolCallRecord{}
	envelope.Errors = []reporting.ExecutionError{}
	envelope.Truncated = true
	callSizes := make([]int, len(report.ToolCalls))
	for index, call := range report.ToolCalls {
		callSizes[index] = encodedSize(call)
	}
	errorSizes := make([]int, len(report.Errors))
	for index, item := range report.Errors {
		errorSizes[index] = encodedSize(item)
	}
	size := measure(envelope) + listSize(callSizes) + listSize(errorSizes)
	// The candidate does not fit, so at least one record goes even when the
	// shorter "truncated":true alone would make it fit.
	calls, errorsFrom := 0, 0
	for removed := false; !removed || size > maximum; removed = true {
		switch {
		case calls < len(callSizes):
			size -= removalSize(callSizes[calls:])
			calls++
		case errorsFrom < len(errorSizes):
			size -= removalSize(errorSizes[errorsFrom:])
			errorsFrom++
		default:
			return reporting.ExecutionReport{}, false, nil
		}
	}
	result = report
	result.ToolCalls = report.ToolCalls[calls:]
	result.Errors = report.Errors[errorsFrom:]
	result.Truncated = true
	if measured := measure(result); measured != size {
		return reporting.ExecutionReport{}, false, fmt.Errorf(
			"retained report measures %d bytes, not the computed %d", measured, size)
	}
	return result, true, nil
}

// listSize is the JSON size of a list's elements and the commas between them;
// the brackets belong to the envelope's empty list.
func listSize(sizes []int) int {
	total := max(0, len(sizes)-1)
	for _, size := range sizes {
		total += size
	}
	return total
}

// removalSize is what removing the first element of a list saves: the element
// and, unless it is the last one, the comma after it.
func removalSize(remaining []int) int {
	if len(remaining) > 1 {
		return remaining[0] + 1
	}
	return remaining[0]
}

// jsonSize is the shared compact UTF-8 JSON measure (OBS-R42). Tests replace
// it to count the measurements a normalization performs (OBS-R110).
var jsonSize = contracts.ResultJSONSize

// encodedSize is jsonSize, except that a value which cannot be encoded counts
// as larger than any report bound, so trimming removes the record carrying it
// and a report that still carries it is rejected.
func encodedSize(value any) int {
	size, err := jsonSize(value)
	if err != nil {
		return MaxReportJSONBytes + 1
	}
	return size
}

func valueSize(value any) *int {
	size := encodedSize(value)
	return &size
}

func truncateUTF8(value string, maximum int) string {
	if len(value) <= maximum {
		return value
	}
	const suffix = "…"
	result := value[:maximum-len(suffix)]
	for !utf8.ValidString(result) {
		result = result[:len(result)-1]
	}
	return result + suffix
}
