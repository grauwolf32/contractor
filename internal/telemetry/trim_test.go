package telemetry

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
)

// The cases of api/testdata/v1alpha1/telemetry-report-trim-cases.json, which
// the Runtime's trim_report_detail checks as well.
type reportTrimCase struct {
	Name   string          `json:"name"`
	Bytes  int             `json:"bytes"`
	Report json.RawMessage `json:"report"`
	Limits []struct {
		MaximumBytes      int  `json:"maximumBytes"`
		ToolCallsFrom     int  `json:"toolCallsFrom"`
		ErrorsFrom        int  `json:"errorsFrom"`
		Truncated         bool `json:"truncated"`
		Bytes             int  `json:"bytes"`
		AggregateTooLarge bool `json:"aggregateTooLarge"`
	} `json:"limits"`
}

func TestSharedReportTrimKeepsTheSharedSuffixAtEveryBoundary(t *testing.T) {
	var fixture struct {
		Description string           `json:"description"`
		Cases       []reportTrimCase `json:"cases"`
	}
	raw := contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "telemetry-report-trim-cases.json")
	if err := json.Unmarshal(raw, &fixture); err != nil || len(fixture.Cases) == 0 {
		t.Fatalf("decode trim cases: %v (%d cases)", err, len(fixture.Cases))
	}
	for _, test := range fixture.Cases {
		t.Run(test.Name, func(t *testing.T) {
			report, err := contracts.DecodePrivateStrict[reporting.ExecutionReport](test.Report)
			if err != nil {
				t.Fatalf("decode report: %v", err)
			}
			if size := encodedSize(report); size != test.Bytes {
				t.Fatalf("report size = %d, Runtime measures %d", size, test.Bytes)
			}
			for _, limit := range test.Limits {
				normalized, err := NewPolicy().NormalizeExecutionReport(report, limit.MaximumBytes)
				if limit.AggregateTooLarge {
					if err == nil || !strings.Contains(err.Error(), "aggregate exceeds byte limit") {
						t.Fatalf("limit %d: error = %v, want aggregate too large", limit.MaximumBytes, err)
					}
					continue
				}
				if err != nil {
					t.Fatalf("limit %d: %v", limit.MaximumBytes, err)
				}
				want := report
				want.ToolCalls = report.ToolCalls[limit.ToolCallsFrom:]
				want.Errors = report.Errors[limit.ErrorsFrom:]
				want.Truncated = limit.Truncated
				if !bytes.Equal(mustJSON(t, normalized), mustJSON(t, want)) {
					t.Fatalf("limit %d: normalized = %s\nwant %s", limit.MaximumBytes, mustJSON(t, normalized), mustJSON(t, want))
				}
				if size := encodedSize(normalized); size != limit.Bytes || size > limit.MaximumBytes {
					t.Fatalf("limit %d: retained size = %d, want %d", limit.MaximumBytes, size, limit.Bytes)
				}
			}
		})
	}
}

// previousTrim is the loop trimDetail replaced: it measures the complete
// report again after removing each record.
func previousTrim(
	report reporting.ExecutionReport,
	maximum int,
	measure func(reporting.ExecutionReport) int,
) (reporting.ExecutionReport, bool) {
	for measure(report) > maximum {
		if len(report.ToolCalls) > 0 {
			report.ToolCalls = report.ToolCalls[1:]
			report.Truncated = true
			continue
		}
		if len(report.Errors) > 0 {
			report.Errors = report.Errors[1:]
			report.Truncated = true
			continue
		}
		return reporting.ExecutionReport{}, false
	}
	return report, true
}

// removalSequence is every report previousTrim may return, in order.
func removalSequence(report reporting.ExecutionReport) []reporting.ExecutionReport {
	states := []reporting.ExecutionReport{report}
	for len(report.ToolCalls) > 0 || len(report.Errors) > 0 {
		if len(report.ToolCalls) > 0 {
			report.ToolCalls = report.ToolCalls[1:]
		} else {
			report.Errors = report.Errors[1:]
		}
		report.Truncated = true
		states = append(states, report)
	}
	return states
}

// trimBoundaries are the sizes of every report of the removal sequence, one
// byte less and one byte more.
func trimBoundaries(report reporting.ExecutionReport, measure func(reporting.ExecutionReport) int) []int {
	var result []int
	for _, state := range removalSequence(report) {
		size := measure(state)
		result = append(result, size-1, size, size+1)
	}
	return result
}

var trimUnits = []string{"a", "ж", "界", "😀", " ", " ", "<>&", `"\`, "\n\t\x01", "\x7f", "\xff"}

// multibyteReport holds ASCII, multibyte, escaped and line-separator detail;
// Go counts U+2028 and U+2029 as their UTF-8 bytes although it escapes them.
func multibyteReport(calls, errorCount int) reporting.ExecutionReport {
	retryable := true
	count := int64(calls)
	report := reporting.ExecutionReport{
		ReportID: "worker-differential", Complete: true,
		Metrics: reporting.ExecutionMetrics{
			ModelCalls: int64Ptr(3),
			Tools:      map[string]reporting.ToolMetrics{"search_text": {Calls: &count}},
		},
		ToolCalls: make([]reporting.ToolCallRecord, 0, calls),
		Errors:    make([]reporting.ExecutionError, 0, errorCount),
	}
	for index := range calls {
		unit := trimUnits[index%len(trimUnits)]
		call := reporting.ToolCallRecord{
			CallID: fmt.Sprintf("call-%02d", index), Tool: "search_text",
			Arguments: map[string]any{"pattern": strings.Repeat(unit, index*7+1), "limit": index},
			Outcome:   reporting.ToolCallSucceeded,
		}
		if index%3 == 0 {
			call.Outcome = reporting.ToolCallFailed
			call.Error = &reporting.ExecutionError{Code: "tool_failed", Message: "failed " + strings.Repeat(unit, index*11+1)}
		}
		report.ToolCalls = append(report.ToolCalls, call)
	}
	for index := range errorCount {
		unit := trimUnits[(index+3)%len(trimUnits)]
		report.Errors = append(report.Errors, reporting.ExecutionError{
			Code: "model_call_failed", Message: "Model call failed " + strings.Repeat(unit, index*5+1),
			Retryable: &retryable,
		})
	}
	return report
}

func allocationMeasure(stopReason string) func(reporting.ExecutionReport) int {
	finished := time.Date(2026, 10, 23, 12, 0, 0, 0, time.UTC)
	return func(worker reporting.ExecutionReport) int {
		return encodedSize(reporting.AllocationFinalReport{
			ReportID: "allocation-final-1", AllocationID: "allocation-1",
			StartedAt: finished.Add(-time.Minute), FinishedAt: finished,
			Worker:  worker,
			Runtime: reporting.RuntimeReport{Complete: true, StopReason: &stopReason},
		})
	}
}

func TestReportTrimMatchesThePreviousLoopAtEveryBoundary(t *testing.T) {
	unencodable := multibyteReport(12, 4)
	unencodable.ToolCalls[7].Arguments = map[string]any{"value": math.NaN()}
	for _, test := range []struct {
		name    string
		report  reporting.ExecutionReport
		measure func(reporting.ExecutionReport) int
	}{
		{name: "execution report", report: multibyteReport(24, 10), measure: func(report reporting.ExecutionReport) int { return encodedSize(report) }},
		{name: "already truncated", report: func() reporting.ExecutionReport {
			report := multibyteReport(9, 3)
			report.Truncated = true
			return report
		}(), measure: func(report reporting.ExecutionReport) int { return encodedSize(report) }},
		{name: "errors only", report: multibyteReport(0, 10), measure: func(report reporting.ExecutionReport) int { return encodedSize(report) }},
		// RuntimeReport renders through json.Marshal, which escapes <, > and
		// &; the envelope keeps that size and the records keep theirs.
		{name: "allocation envelope", report: multibyteReport(24, 10), measure: allocationMeasure("stopped <early> & " + strings.Repeat("界", 40))},
		{name: "unencodable record", report: unencodable, measure: func(report reporting.ExecutionReport) int { return encodedSize(report) }},
	} {
		t.Run(test.name, func(t *testing.T) {
			boundaries := trimBoundaries(test.report, test.measure)
			if len(boundaries) < 3*(len(test.report.ToolCalls)+len(test.report.Errors)) {
				t.Fatalf("only %d boundaries", len(boundaries))
			}
			for _, maximum := range boundaries {
				if maximum > MaxReportJSONBytes {
					continue // No report bound is larger; an unencodable record measures above it.
				}
				want, wantFits := previousTrim(test.report, maximum, test.measure)
				got, fits, err := trimDetail(test.report, maximum, test.measure)
				if err != nil {
					t.Fatalf("maximum %d: %v", maximum, err)
				}
				if fits != wantFits || !bytes.Equal(mustJSON(t, got), mustJSON(t, want)) {
					t.Fatalf("maximum %d: fits=%t %s\nprevious loop fits=%t %s", maximum, fits, mustJSON(t, got), wantFits, mustJSON(t, want))
				}
			}
		})
	}
}

type measurementCounts struct {
	reports, allocations, records, other int
}

// countMeasurements counts the JSON measurements the policy performs, by
// what it measures, until the test ends.
func countMeasurements(t *testing.T) *measurementCounts {
	t.Helper()
	counts := &measurementCounts{}
	previous := jsonSize
	jsonSize = func(value any) (int, error) {
		switch value.(type) {
		case reporting.ExecutionReport:
			counts.reports++
		case reporting.AllocationFinalReport:
			counts.allocations++
		case reporting.ToolCallRecord, reporting.ExecutionError:
			counts.records++
		default:
			counts.other++
		}
		return previous(value)
	}
	t.Cleanup(func() { jsonSize = previous })
	return counts
}

// maximumDetailReport has the retained-record maxima: 1000 tool calls of
// about 4 KiB, the newest 100 failed, and 100 errors of 4096 bytes each.
func maximumDetailReport() reporting.ExecutionReport {
	message := strings.Repeat("界", 1365) + "a"
	calls, failed := int64(MaxToolRecords), int64(MaxErrorRecords)
	report := reporting.ExecutionReport{
		ReportID: "worker-maximum-detail", Complete: true,
		Metrics: reporting.ExecutionMetrics{Tools: map[string]reporting.ToolMetrics{
			"read_source_file": {Calls: &calls, Failed: &failed},
		}},
		ToolCalls: make([]reporting.ToolCallRecord, 0, MaxToolRecords),
		Errors:    make([]reporting.ExecutionError, 0, MaxErrorRecords),
	}
	for index := range MaxToolRecords {
		call := reporting.ToolCallRecord{
			CallID: fmt.Sprintf("tool-%08d", index+1), Tool: "read_source_file",
			Arguments: map[string]any{"path": fmt.Sprintf("%04d", index) + strings.Repeat("ж", 1990)},
			Outcome:   reporting.ToolCallSucceeded,
		}
		if index >= MaxToolRecords-MaxErrorRecords {
			call.Outcome = reporting.ToolCallFailed
			call.Error = &reporting.ExecutionError{Code: "tool_input_invalid", Message: message}
			report.Errors = append(report.Errors, reporting.ExecutionError{Code: "tool_read_source_file_failed", Message: message})
		}
		report.ToolCalls = append(report.ToolCalls, call)
	}
	return report
}

// assertFirstFit checks that trimmed keeps a suffix of source's tool calls
// and every error, and that keeping one more older call would not fit.
func assertFirstFit(
	t *testing.T,
	source, trimmed reporting.ExecutionReport,
	maximum int,
	measure func(reporting.ExecutionReport) int,
) {
	t.Helper()
	removed := len(source.ToolCalls) - len(trimmed.ToolCalls)
	if removed <= 0 || removed >= len(source.ToolCalls) || !trimmed.Truncated {
		t.Fatalf("removed %d of %d tool calls, truncated=%t", removed, len(source.ToolCalls), trimmed.Truncated)
	}
	if !bytes.Equal(mustJSON(t, trimmed.ToolCalls), mustJSON(t, source.ToolCalls[removed:])) ||
		!bytes.Equal(mustJSON(t, trimmed.Errors), mustJSON(t, source.Errors)) {
		t.Fatal("trimmed report does not keep the newest detail")
	}
	if size := measure(trimmed); size > maximum {
		t.Fatalf("trimmed report measures %d bytes, above %d", size, maximum)
	}
	oneMore := trimmed
	oneMore.ToolCalls = source.ToolCalls[removed-1:]
	if size := measure(oneMore); size <= maximum {
		t.Fatalf("one more tool call measures %d bytes and fits %d", size, maximum)
	}
}

func TestReportTrimMeasuresTheCompleteReportAtMostThreeTimes(t *testing.T) {
	executionSize := func(report reporting.ExecutionReport) int { return encodedSize(report) }
	t.Run("Planner report at the 1 MiB report bound", func(t *testing.T) {
		source := maximumDetailReport()
		counts := countMeasurements(t)
		normalized, err := NewPolicy().NormalizeExecutionReport(source, MaxReportJSONBytes)
		if err != nil {
			t.Fatal(err)
		}
		// The candidate, the envelope without detail and the retained result;
		// each record once; each argument summary once.
		want := measurementCounts{reports: 3, records: MaxToolRecords + MaxErrorRecords, other: MaxToolRecords}
		if *counts != want {
			t.Fatalf("measurements = %+v, want %+v", *counts, want)
		}
		jsonSize = contracts.ResultJSONSize
		assertFirstFit(t, source, normalized, MaxReportJSONBytes, executionSize)
		if *normalized.Metrics.Tools["read_source_file"].Calls != MaxToolRecords {
			t.Fatalf("aggregate changed: %+v", normalized.Metrics.Tools)
		}
	})
	t.Run("allocation report trimmed at both bounds", func(t *testing.T) {
		source := maximumDetailReport()
		stopReason := strings.Repeat("s", 2*allocationReserveSize)
		finished := time.Date(2026, 10, 23, 12, 0, 0, 0, time.UTC)
		allocation := reporting.AllocationFinalReport{
			ReportID: "allocation-final-1", AllocationID: "allocation-1",
			StartedAt: finished.Add(-time.Minute), FinishedAt: finished, Worker: source,
			Runtime: reporting.RuntimeReport{Complete: true, StopReason: &stopReason},
		}
		workerBound, err := NewPolicy().NormalizeExecutionReport(source, MaxReportJSONBytes-allocationReserveSize)
		if err != nil {
			t.Fatal(err)
		}
		counts := countMeasurements(t)
		normalized, err := NewPolicy().NormalizeAllocationReport(allocation)
		if err != nil {
			t.Fatal(err)
		}
		// Three complete measurements at each bound; every record once at
		// the Worker bound and every record it kept once more at 1 MiB.
		want := measurementCounts{
			reports: 3, allocations: 3, other: MaxToolRecords,
			records: MaxToolRecords + MaxErrorRecords + len(workerBound.ToolCalls) + len(workerBound.Errors),
		}
		if *counts != want || len(normalized.Worker.ToolCalls) >= len(workerBound.ToolCalls) {
			t.Fatalf("measurements = %+v, want %+v; kept %d of %d", *counts, want,
				len(normalized.Worker.ToolCalls), len(workerBound.ToolCalls))
		}
		jsonSize = contracts.ResultJSONSize
		assertFirstFit(t, source, normalized.Worker, MaxReportJSONBytes, allocationMeasure(stopReason))
	})
}

func TestReportTrimMeasuresAFittingReportOnce(t *testing.T) {
	report := multibyteReport(3, 2)
	counts := countMeasurements(t)
	normalized, err := NewPolicy().NormalizeExecutionReport(report, MaxReportJSONBytes)
	if err != nil {
		t.Fatal(err)
	}
	if counts.reports != 1 || counts.records != 0 || normalized.Truncated || len(normalized.ToolCalls) != 3 {
		t.Fatalf("measurements = %+v, truncated=%t", *counts, normalized.Truncated)
	}
}

// A report the Runtime trimmed below the private 1 MiB bound can grow at the
// Server, because a secret shorter than "[REDACTED]" expands; the Server then
// trims the redacted records, never sizing the unredacted ones.
func TestAllocationNormalizationTrimsRedactionExpansionOfABoundedReport(t *testing.T) {
	const secret = "k3y"
	value := strings.Repeat(secret+"/aaaaaa", 90) // 900 bytes, 90 secrets
	finished := time.Date(2026, 10, 23, 12, 0, 0, 0, time.UTC)
	calls := int64(1000)
	worker := reporting.ExecutionReport{
		ReportID: "worker-redaction", Complete: true,
		Metrics:   reporting.ExecutionMetrics{Tools: map[string]reporting.ToolMetrics{"read_source_file": {Calls: &calls}}},
		ToolCalls: make([]reporting.ToolCallRecord, 0, calls),
		Errors:    []reporting.ExecutionError{{Code: "model_call_failed", Message: "gateway said " + value}},
	}
	for index := range calls {
		worker.ToolCalls = append(worker.ToolCalls, reporting.ToolCallRecord{
			CallID: fmt.Sprintf("tool-%08d", index+1), Tool: "read_source_file",
			Arguments: map[string]any{"path": value}, Outcome: reporting.ToolCallSucceeded,
		})
	}
	body := mustJSON(t, reporting.AllocationFinalResponse{
		APIVersion: contracts.APIVersion,
		Report: reporting.AllocationFinalReport{
			ReportID: "allocation-final-1", AllocationID: "allocation-1",
			StartedAt: finished.Add(-time.Minute), FinishedAt: finished, Worker: worker,
			Runtime: reporting.RuntimeReport{Complete: true},
		},
	})
	if len(body) > MaxReportJSONBytes {
		t.Fatalf("Runtime response is %d bytes, above the private bound", len(body))
	}
	response, err := contracts.DecodePrivateStrict[reporting.AllocationFinalResponse](body)
	if err != nil {
		t.Fatalf("bounded Runtime report rejected at the private boundary: %v", err)
	}

	counts := countMeasurements(t)
	normalized, err := NewPolicy(secret).NormalizeAllocationReport(response.Report)
	if err != nil {
		t.Fatal(err)
	}
	if counts.reports != 3 || counts.allocations != 1 {
		t.Fatalf("measurements = %+v", *counts)
	}
	jsonSize = contracts.ResultJSONSize
	encoded := mustJSON(t, normalized)
	if bytes.Contains(encoded, []byte(secret)) {
		t.Fatal("normalized report contains the secret")
	}
	redacted := response.Report.Worker
	redacted.ToolCalls = make([]reporting.ToolCallRecord, 0, len(worker.ToolCalls))
	for _, call := range response.Report.Worker.ToolCalls {
		call.Arguments = map[string]any{"path": strings.ReplaceAll(value, secret, redactedValue)}
		call.ArgumentsTruncated = true
		redacted.ToolCalls = append(redacted.ToolCalls, call)
	}
	redacted.Errors = []reporting.ExecutionError{{Code: "model_call_failed", Message: "gateway said " + strings.ReplaceAll(value, secret, redactedValue)}}
	assertFirstFit(t, redacted, normalized.Worker, MaxReportJSONBytes-allocationReserveSize,
		func(report reporting.ExecutionReport) int { return encodedSize(report) })
}

func TestReportNormalizationRejectsAnAggregateAboveItsBound(t *testing.T) {
	report := multibyteReport(4, 2)
	envelope := report
	envelope.ToolCalls, envelope.Errors, envelope.Truncated = []reporting.ToolCallRecord{}, []reporting.ExecutionError{}, true
	limit := encodedSize(envelope)
	if _, err := NewPolicy().NormalizeExecutionReport(report, limit); err != nil {
		t.Fatalf("envelope-sized limit: %v", err)
	}
	if _, err := NewPolicy().NormalizeExecutionReport(report, limit-1); err == nil ||
		err.Error() != "execution report aggregate exceeds byte limit" {
		t.Fatalf("below the envelope: %v", err)
	}

	stopReason := strings.Repeat("s", MaxReportJSONBytes)
	finished := time.Date(2026, 10, 23, 12, 0, 0, 0, time.UTC)
	_, err := NewPolicy().NormalizeAllocationReport(reporting.AllocationFinalReport{
		ReportID: "allocation-final-1", AllocationID: "allocation-1",
		StartedAt: finished.Add(-time.Minute), FinishedAt: finished, Worker: report,
		Runtime: reporting.RuntimeReport{Complete: true, StopReason: &stopReason},
	})
	if err == nil || err.Error() != "allocation report aggregate exceeds 1 MiB" {
		t.Fatalf("oversized allocation aggregate: %v", err)
	}
}

func mustJSON(t *testing.T, value any) []byte {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}
