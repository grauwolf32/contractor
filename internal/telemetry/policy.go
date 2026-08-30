package telemetry

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts"
)

const (
	MaxToolRecords        = 1000
	MaxErrorRecords       = 100
	MaxArgumentBytes      = 4096
	MaxErrorMessageBytes  = 4096
	MaxReportJSONBytes    = 1024 * 1024
	DefaultRetentionDays  = 30
	DefaultCleanupBatch   = 500
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
	source contracts.AllocationFinalReport,
) (contracts.AllocationFinalReport, error) {
	result := source
	worker, err := p.NormalizeExecutionReport(source.Worker, MaxReportJSONBytes-allocationReserveSize)
	if err != nil {
		return contracts.AllocationFinalReport{}, err
	}
	result.Worker = worker
	if result.Runtime.StopReason != nil {
		value := p.redactText(*result.Runtime.StopReason)
		result.Runtime.StopReason = &value
	}
	for encodedSize(result) > MaxReportJSONBytes {
		if len(result.Worker.ToolCalls) > 0 {
			result.Worker.ToolCalls = result.Worker.ToolCalls[1:]
			result.Worker.Truncated = true
			continue
		}
		if len(result.Worker.Errors) > 0 {
			result.Worker.Errors = result.Worker.Errors[1:]
			result.Worker.Truncated = true
			continue
		}
		return contracts.AllocationFinalReport{}, errors.New("allocation report aggregate exceeds 1 MiB")
	}
	if err := result.Validate(); err != nil {
		return contracts.AllocationFinalReport{}, fmt.Errorf("normalize allocation report: %w", err)
	}
	return result, nil
}

func (p Policy) NormalizeExecutionReport(
	source contracts.ExecutionReport,
	maximumBytes int,
) (contracts.ExecutionReport, error) {
	if maximumBytes <= 0 || maximumBytes > MaxReportJSONBytes {
		return contracts.ExecutionReport{}, errors.New("invalid execution report byte limit")
	}
	result := source
	result.Metrics.Tools = cloneToolMetrics(source.Metrics.Tools)
	result.Metrics.WorkerBudget = cloneWorkerBudget(source.Metrics.WorkerBudget)
	if result.Metrics.Tools == nil {
		result.Metrics.Tools = map[string]contracts.ToolMetrics{}
	}
	result.ToolCalls = make([]contracts.ToolCallRecord, 0, min(len(source.ToolCalls), MaxToolRecords))
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
			if encodedSize(arguments) > MaxArgumentBytes {
				call.Arguments = map[string]any{
					"summary":           truncatedValue,
					"originalSizeBytes": encodedSize(arguments),
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
	result.Errors = make([]contracts.ExecutionError, 0, min(len(source.Errors), MaxErrorRecords))
	errorStart := max(0, len(source.Errors)-MaxErrorRecords)
	if errorStart > 0 {
		result.Truncated = true
	}
	for _, item := range source.Errors[errorStart:] {
		result.Errors = append(result.Errors, p.normalizeError(item))
	}
	for encodedSize(result) > maximumBytes {
		if len(result.ToolCalls) > 0 {
			result.ToolCalls = result.ToolCalls[1:]
			result.Truncated = true
			continue
		}
		if len(result.Errors) > 0 {
			result.Errors = result.Errors[1:]
			result.Truncated = true
			continue
		}
		return contracts.ExecutionReport{}, errors.New("execution report aggregate exceeds byte limit")
	}
	if err := result.Validate(); err != nil {
		return contracts.ExecutionReport{}, fmt.Errorf("normalize execution report: %w", err)
	}
	return result, nil
}

func (p Policy) normalizeError(source contracts.ExecutionError) contracts.ExecutionError {
	result := source
	result.Code = p.redactText(source.Code)
	result.Message = truncateUTF8(p.redactText(source.Message), MaxErrorMessageBytes)
	return result
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

func cloneToolMetrics(source map[string]contracts.ToolMetrics) map[string]contracts.ToolMetrics {
	result := make(map[string]contracts.ToolMetrics, len(source))
	for name, metrics := range source {
		result[name] = metrics
	}
	return result
}

func cloneWorkerBudget(source *contracts.WorkerBudgetMetrics) *contracts.WorkerBudgetMetrics {
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

func encodedSize(value any) int {
	encoded, err := json.Marshal(value)
	if err != nil {
		return MaxReportJSONBytes + 1
	}
	return len(encoded)
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
