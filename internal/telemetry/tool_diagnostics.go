package telemetry

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/jackc/pgx/v5"
)

const (
	DefaultToolDiagnosticLimit   = 25
	MaxToolDiagnosticLimit       = 100
	maxToolDiagnosticReportBytes = 8 << 20
	maxDiagnosticToolBytes       = 128
)

// ToolDiagnosticParams selects one retained participant, never a physical report
// or allocation identity. Offset is meaningful only in its pinned generation.
type ToolDiagnosticParams struct {
	OwnerID, RunID, StageExecutionID string
	Participant, LogicalAgent        string
	Generation                       string
	Offset, Limit                    int
}

type ToolCallDiagnosticError struct {
	Code             string `json:"code"`
	Message          string `json:"message"`
	MessageTruncated bool   `json:"messageTruncated"`
	Retryable        *bool  `json:"retryable,omitempty"`
}

type ToolCallDiagnostic struct {
	CallReference      string                    `json:"callReference"`
	Tool               string                    `json:"tool"`
	ToolTruncated      bool                      `json:"toolTruncated"`
	Outcome            reporting.ToolCallOutcome `json:"outcome"`
	ArgumentsJSON      *string                   `json:"argumentsJson,omitempty"`
	ArgumentsTruncated bool                      `json:"argumentsTruncated"`
	DurationMS         *string                   `json:"durationMs"`
	ResultSizeBytes    *string                   `json:"resultSizeBytes"`
	Error              *ToolCallDiagnosticError  `json:"error,omitempty"`
}

type ToolDiagnosticPage struct {
	Status          string               `json:"status"`
	Reason          string               `json:"reason,omitempty"`
	Participant     string               `json:"participant"`
	LogicalAgent    string               `json:"logicalAgent,omitempty"`
	Complete        *bool                `json:"complete,omitempty"`
	DetailTruncated bool                 `json:"detailTruncated"`
	Items           []ToolCallDiagnostic `json:"items"`
	Generation      string               `json:"-"`
	More            bool                 `json:"-"`
	NextOffset      int                  `json:"-"`
}

func (r *Repository) ReadToolCallDiagnostics(ctx context.Context, p ToolDiagnosticParams) (ToolDiagnosticPage, error) {
	result := ToolDiagnosticPage{Status: "available", Participant: p.Participant, LogicalAgent: p.LogicalAgent, Items: []ToolCallDiagnostic{}}
	if err := validateDiagnosticIdentity(p.OwnerID, p.RunID, p.StageExecutionID); err != nil {
		return result, err
	}
	if (p.Participant != "planner" && p.Participant != "worker") ||
		(p.Participant == "planner" && p.LogicalAgent != "") ||
		(p.Participant == "worker" && !contracts.ValidName(p.LogicalAgent)) ||
		p.Limit < 1 || p.Limit > MaxToolDiagnosticLimit || p.Offset < 0 || p.Offset > MaxToolRecords ||
		(p.Offset == 0) != (p.Generation == "") {
		return result, fmt.Errorf("%w: invalid tool diagnostic selection", ErrInvalid)
	}
	var terminal, oversized bool
	var version, tuple, kind *string
	var updated *time.Time
	var data []byte
	err := r.db.QueryRow(ctx, toolDiagnosticReportSQL, p.OwnerID, p.RunID, p.StageExecutionID, p.Participant, p.LogicalAgent, maxToolDiagnosticReportBytes).
		Scan(&terminal, &version, &updated, &tuple, &kind, &oversized, &data)
	if errors.Is(err, pgx.ErrNoRows) {
		return result, ErrNotFound
	}
	if err != nil {
		return result, fmt.Errorf("read tool diagnostics: %w", err)
	}
	if version == nil || updated == nil || tuple == nil {
		result.Status, result.Reason = diagnosticUnavailable(terminal)
		return result, nil
	}
	result.Generation = diagnosticGeneration(*updated, *tuple)
	if p.Generation != "" && p.Generation != result.Generation {
		return result, fmt.Errorf("%w: diagnostic generation changed", ErrConflict)
	}
	if kind == nil {
		result.Status, result.Reason = diagnosticUnavailable(terminal)
		return result, nil
	}
	if oversized {
		result.Status, result.Reason = "unavailable", "report_oversized"
		return result, nil
	}
	var report reporting.ExecutionReport
	if *version != contracts.APIVersion || *kind != "object" || len(data) == 0 || len(data) > maxToolDiagnosticReportBytes ||
		strictjson.DecodeNonNull(data, &report) != nil ||
		(reporting.StageMetrics{Planner: &report, Workers: map[string]reporting.ExecutionReport{}, Runtime: map[string]reporting.RuntimeReport{}}).Validate() != nil {
		result.Status, result.Reason = "unavailable", "invalid_report"
		return result, nil
	}
	if p.Offset > len(report.ToolCalls) {
		return result, fmt.Errorf("%w: invalid diagnostic offset", ErrInvalid)
	}
	result.Complete, result.DetailTruncated = &report.Complete, report.Truncated
	used := 0
	for index := p.Offset; index < len(report.ToolCalls); index++ {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		if len(result.Items) == p.Limit {
			result.More = true
			break
		}
		entry := projectToolCallDiagnostic(p, result.Generation, index, report.ToolCalls[index])
		encoded, err := json.Marshal(entry)
		if err != nil {
			return result, fmt.Errorf("encode tool diagnostic: %w", err)
		}
		if used+len(encoded)+1 > MaxDiagnosticPageBytes-4096 {
			result.More = true
			break
		}
		used += len(encoded) + 1
		result.Items = append(result.Items, entry)
		result.NextOffset = index + 1
	}
	return result, ctx.Err()
}

func projectToolCallDiagnostic(p ToolDiagnosticParams, generation string, index int, call reporting.ToolCallRecord) ToolCallDiagnostic {
	digest := sha256.Sum256([]byte(p.StageExecutionID + "\x00" + p.Participant + "\x00" + p.LogicalAgent + "\x00" + generation + "\x00" + strconv.Itoa(index)))
	result := ToolCallDiagnostic{CallReference: hex.EncodeToString(digest[:]), Tool: call.Tool, Outcome: call.Outcome,
		ArgumentsTruncated: call.ArgumentsTruncated, DurationMS: diagnosticInt64(call.DurationMS), ResultSizeBytes: diagnosticInt64(call.ResultSizeBytes)}
	// Oversized or malformed names are not content summaries: keep the row and
	// disclose that its vocabulary name cannot safely be shown.
	if len(call.Tool) > maxDiagnosticToolBytes || !contracts.ValidName(call.Tool) {
		result.Tool, result.ToolTruncated = "[TOOL_NAME_UNAVAILABLE]", true
	}
	if call.Arguments != nil {
		arguments := map[string]any(call.Arguments)
		changed := false
		if !isArgumentSummary(arguments) {
			arguments, changed = NewPolicy().sanitizeMap(arguments)
		}
		before := diagnosticArgumentJSON(arguments)
		arguments = publicArgumentValue(arguments).(map[string]any)
		encoded := diagnosticArgumentJSON(arguments)
		changed = changed || before != encoded
		if len(encoded) > MaxArgumentBytes {
			encoded = `{"summary":"[TRUNCATED]"}`
			changed = true
		}
		result.ArgumentsJSON = &encoded
		result.ArgumentsTruncated = result.ArgumentsTruncated || changed
	}
	if call.Error != nil {
		code := NewPolicy().normalizeCode(call.Error.Code)
		message := diagnosticURLPattern.ReplaceAllString(NewPolicy().RedactMessage(call.Error.Message), redactedURL)
		result.Error = &ToolCallDiagnosticError{Code: code, Message: truncateUTF8(message, MaxErrorMessageBytes), MessageTruncated: len(message) > MaxErrorMessageBytes, Retryable: call.Error.Retryable}
		if strings.TrimSpace(result.Error.Message) == "" {
			result.Error.Message = "Tool call failed"
		}
	}
	return result
}

func diagnosticInt64(value *int64) *string {
	if value == nil {
		return nil
	}
	text := strconv.FormatInt(*value, 10)
	return &text
}

// publicArgumentValue supplements the mandatory persistence policy with the
// public diagnostic URL rule, on keys as well as nested values. Numeric literals
// remain json.Number; they never pass through float64 or a browser JSON parser.
func publicArgumentValue(value any) any {
	switch typed := value.(type) {
	case string:
		return diagnosticURLPattern.ReplaceAllString(typed, redactedURL)
	case map[string]any:
		result := make(map[string]any, len(typed))
		keys := make([]string, 0, len(typed))
		for key := range typed {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			result[diagnosticURLPattern.ReplaceAllString(key, redactedURL)] = publicArgumentValue(typed[key])
		}
		return result
	case []any:
		result := make([]any, len(typed))
		for i, item := range typed {
			result[i] = publicArgumentValue(item)
		}
		return result
	default:
		return value
	}
}

func diagnosticArgumentJSON(value any) string {
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return `{"summary":"[UNAVAILABLE]"}`
	}
	encoded := strings.TrimSuffix(buffer.String(), "\n")
	var result strings.Builder
	for index := 0; index < len(encoded); index++ {
		if encoded[index] == '\\' && index+1 < len(encoded) {
			if index+6 <= len(encoded) && (encoded[index:index+6] == `\u2028` || encoded[index:index+6] == `\u2029`) {
				if encoded[index+5] == '8' {
					result.WriteRune('\u2028')
				} else {
					result.WriteRune('\u2029')
				}
				index += 5
				continue
			}
			result.WriteByte(encoded[index])
			index++
			result.WriteByte(encoded[index])
			continue
		}
		result.WriteByte(encoded[index])
	}
	return result.String()
}
