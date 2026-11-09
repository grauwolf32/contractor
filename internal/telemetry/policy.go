package telemetry

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"regexp"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/helpers/secretmatch"
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

// A name containing one of sensitiveFragments or ending in one of
// sensitiveSuffixes, after normalizeKey, is sensitive as well.
var (
	sensitiveFragments = []string{"apikey", "authorization", "base64", "cookie", "password", "secret", "token"}
	sensitiveSuffixes  = []string{"body", "bytes", "content", "data", "payload"}
)

// safeDerivedSizeKeys name the sizes the Runtime derives from content it
// does not report. A non-negative integer under one of them is kept although
// the name ends in "bytes"; the Runtime's SAFE_DERIVED_SIZE_METRIC_KEYS lists
// the same names, and telemetry-report-trim-cases.json pins both lists.
var safeDerivedSizeKeys = map[string]struct{}{
	"content_bytes": {}, "description_bytes": {},
	"result_content_bytes": {}, "result_description_bytes": {},
}

// Policy is the mandatory Control Plane persistence-boundary sanitizer.
// Secrets are copied so callers can erase their own configuration independently.
type Policy struct {
	secrets []string
	// size is the compact UTF-8 JSON measure (OBS-R42). NewPolicy sets
	// contracts.ResultJSONSize; a test builds a Policy with a measure that
	// counts, so no package state changes and tests may run in parallel.
	size func(any) (int, error)
}

// Format redacts the policy as a whole for every verb (SEC-R88): its secrets field holds secret values in plain text.
func (Policy) Format(state fmt.State, _ rune) {
	_, _ = io.WriteString(state, "telemetry.Policy([REDACTED])")
}

func NewPolicy(secrets ...string) Policy {
	filtered := make([]string, 0, len(secrets))
	for _, secret := range secrets {
		if secret != "" {
			filtered = append(filtered, secret)
		}
	}
	return Policy{secrets: filtered, size: contracts.ResultJSONSize}
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
	worker, fits, err := p.trimDetail(result.Worker, MaxReportJSONBytes,
		func(candidate reporting.ExecutionReport) int {
			report := result
			report.Worker = candidate
			return p.encodedSize(report)
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
	if source.Metrics.LLMCost != nil {
		if err := source.Metrics.LLMCost.Validate(); err != nil {
			return reporting.ExecutionReport{}, err
		}
		cost := *source.Metrics.LLMCost
		cost.Groups = make([]reporting.LLMCostGroup, 0, len(cost.Groups))
		for _, group := range source.Metrics.LLMCost.Groups {
			// A secret-bearing alias is not safe diagnostic metadata. Keep the
			// physical attempt but leave its monetary attribution unavailable.
			if p.redactText(group.Model) == group.Model {
				cost.Groups = append(cost.Groups, group)
			}
		}
		result.Metrics.LLMCost = &cost
	}
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
		if sourceCall.Arguments != nil && !isArgumentSummary(sourceCall.Arguments) {
			arguments, changed := p.sanitizeMap(sourceCall.Arguments)
			call.Arguments = arguments
			call.ArgumentsTruncated = call.ArgumentsTruncated || changed
			if size := p.encodedSize(arguments); size > MaxArgumentBytes {
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
	result, fits, err := p.trimDetail(result, maximumBytes, func(candidate reporting.ExecutionReport) int {
		return p.encodedSize(candidate)
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
	result.Code = p.normalizeCode(source.Code)
	result.Message = p.RedactMessage(source.Message)
	return result
}

// normalizeCode keeps an error code that is an identifier and holds no secret
// and otherwise replaces it with tool_call_failed: a code is never partially
// redacted (OBS-R32).
func (p Policy) normalizeCode(code string) string {
	if !contracts.ValidName(code) {
		return fallbackErrorCode
	}
	for _, secret := range secretmatch.Long(p.secrets) {
		if strings.Contains(code, secret) {
			return fallbackErrorCode
		}
	}
	return code
}

// fallbackErrorCode replaces a code that is not an identifier in the shared
// name grammar (the Runtime's SAFE_ERROR_CODE) or contains a long secret
// (OBS-R32, FAIL-OBS-3).
const fallbackErrorCode = "tool_call_failed"

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
		targetKey, normalized, itemChanged := p.sanitizeMember(key, value, 0)
		result[targetKey] = normalized
		changed = changed || itemChanged
	}
	return result, changed
}

// sanitizeMember redacts one member of an argument object (OBS-R27). A
// sensitive name replaces its value with the redaction placeholder, except a
// size the Runtime derives and a value that already is the placeholder, so a
// record the Runtime redacted passes unchanged and keeps its size (OBS-R110).
// Every name isSensitiveKey accepts the Runtime treats as sensitive too.
func (p Policy) sanitizeMember(key string, value any, depth int) (string, any, bool) {
	targetKey := truncateUTF8(p.redactText(key), MaxArgumentBytes)
	changed := targetKey != key
	if isSafeDerivedSize(key, value) {
		return targetKey, value, changed
	}
	if isSensitiveKey(key) {
		if isRedactionPlaceholder(value) {
			return targetKey, value, changed
		}
		return targetKey, map[string]any{"redacted": true, "size": p.encodedSize(value)}, true
	}
	normalized, itemChanged := p.sanitizeValue(value, depth)
	return targetKey, normalized, changed || itemChanged
}

// isArgumentSummary reports whether arguments already are the summary that
// replaces arguments above MaxArgumentBytes; its size member is no content.
func isArgumentSummary(arguments map[string]any) bool {
	return len(arguments) == 2 && arguments["summary"] == truncatedValue &&
		isSizeInRange(arguments["originalSizeBytes"])
}

// isRedactionPlaceholder reports whether value is exactly the placeholder a
// sensitive member's value becomes: redacted true and a size that is a
// non-negative integer within LIM-OBS-16 or null.
func isRedactionPlaceholder(value any) bool {
	placeholder, ok := value.(map[string]any)
	if !ok || len(placeholder) != 2 || placeholder["redacted"] != true {
		return false
	}
	size, present := placeholder["size"]
	return present && (size == nil || isSizeInRange(size))
}

func isSafeDerivedSize(key string, value any) bool {
	_, safe := safeDerivedSizeKeys[metricIdentifier(key)]
	return safe && isSizeInRange(value)
}

// isSizeInRange reports whether value is a size a Runtime record may carry:
// a non-negative integer that LIM-OBS-16 keeps.
func isSizeInRange(value any) bool {
	return isNonNegativeInteger(value) && !reporting.ArgumentNumberOutOfRange(value)
}

// isNonNegativeInteger accepts a decoded literal of digits only, which JSON
// writes for a non-negative integer, and a Go integer that is not negative.
func isNonNegativeInteger(value any) bool {
	switch typed := value.(type) {
	case json.Number:
		return typed != "" && strings.Trim(string(typed), "0123456789") == ""
	case int:
		return typed >= 0
	case int64:
		return typed >= 0
	}
	return false
}

var metricIdentifierSeparators = regexp.MustCompile(`[^a-z0-9_]+`)

// metricIdentifier is the Runtime's _metric_identifier: the lower-case name
// with every run of characters other than a-z, 0-9 and _ replaced by _,
// without leading or trailing _, cut to 64 bytes.
func metricIdentifier(value string) string {
	normalized := strings.Trim(metricIdentifierSeparators.ReplaceAllString(strings.ToLower(value), "_"), "_")
	return normalized[:min(len(normalized), 64)]
}

func (p Policy) sanitizeValue(source any, depth int) (any, bool) {
	if depth >= 4 {
		return truncatedValue, true
	}
	switch value := source.(type) {
	case nil, bool:
		return value, false
	case json.Number, float64, float32, int, int8, int16, int32, int64,
		uint, uint8, uint16, uint32, uint64:
		// A number the summary cannot keep becomes the marker the Runtime
		// writes for it (LIM-OBS-16); a Runtime number is already in range.
		if reporting.ArgumentNumberOutOfRange(value) {
			return reporting.ArgumentNumberMarker, true
		}
		return value, false
	case string:
		result := p.redactText(value)
		return truncateUTF8(result, MaxArgumentBytes), result != value || len(result) > MaxArgumentBytes
	case []byte:
		return map[string]any{"redacted": true, "size": len(value)}, true
	case map[string]any:
		result := make(map[string]any, len(value))
		changed := false
		for key, item := range value {
			targetKey, normalized, itemChanged := p.sanitizeMember(key, item, depth+1)
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

// redactText replaces known secrets under the shared rule (OBS-R31,
// LIM-OBS-3): a complete string equal to a secret, and a secret of at least
// secretmatch.MinSubstringBytes wherever it occurs, longest first.
func (p Policy) redactText(source string) string {
	result := secretmatch.Redact(source, p.secrets, redactedValue)
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
	for _, fragment := range sensitiveFragments {
		if strings.Contains(normalized, fragment) {
			return true
		}
	}
	for _, suffix := range sensitiveSuffixes {
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
func (p Policy) trimDetail(
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
		callSizes[index] = p.encodedSize(call)
	}
	errorSizes := make([]int, len(report.Errors))
	for index, item := range report.Errors {
		errorSizes[index] = p.encodedSize(item)
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

// encodedSize is the policy's JSON measure, except that a value which cannot
// be encoded counts as larger than any report bound, so trimming removes the
// record carrying it and a report that still carries it is rejected.
func (p Policy) encodedSize(value any) int {
	size, err := p.size(value)
	if err != nil {
		return MaxReportJSONBytes + 1
	}
	return size
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
