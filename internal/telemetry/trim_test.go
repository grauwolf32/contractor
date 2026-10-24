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

type reportTrimFixture struct {
	Description string `json:"description"`
	Redaction   struct {
		SensitiveNames       []string `json:"sensitiveNames"`
		SafeDerivedSizeNames []string `json:"safeDerivedSizeNames"`
	} `json:"redaction"`
	ArgumentNumbers struct {
		MinExponent int      `json:"minExponent"`
		MaxExponent int      `json:"maxExponent"`
		Marker      string   `json:"marker"`
		Kept        []string `json:"kept"`
		Replaced    []string `json:"replaced"`
	} `json:"argumentNumbers"`
	Cases []reportTrimCase `json:"cases"`
}

func readReportTrimFixture(t *testing.T) reportTrimFixture {
	t.Helper()
	var fixture reportTrimFixture
	raw := contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "telemetry-report-trim-cases.json")
	if err := json.Unmarshal(raw, &fixture); err != nil || len(fixture.Cases) == 0 {
		t.Fatalf("decode trim cases: %v (%d cases)", err, len(fixture.Cases))
	}
	return fixture
}

// TestSharedArgumentNumbersAgreeWithTheRuntime checks LIM-OBS-16 on the
// numbers the Runtime writes: a kept literal passes receipt and the Server's
// sanitizer unchanged; a replaced one fails receipt and becomes the shared
// marker wherever the Server builds a summary itself.
func TestSharedArgumentNumbersAgreeWithTheRuntime(t *testing.T) {
	numbers := readReportTrimFixture(t).ArgumentNumbers
	if numbers.Marker != reporting.ArgumentNumberMarker || len(numbers.Kept) == 0 || len(numbers.Replaced) == 0 {
		t.Fatalf("shared argument numbers = %+v, marker %q", numbers, reporting.ArgumentNumberMarker)
	}
	for exponent, want := range map[int]bool{
		numbers.MinExponent - 1: true, numbers.MinExponent: false,
		numbers.MaxExponent: false, numbers.MaxExponent + 1: true,
	} {
		if got := reporting.ArgumentNumberOutOfRange(json.Number(fmt.Sprintf("1e%d", exponent))); got != want {
			t.Fatalf("1e%d out of range = %t, want %t", exponent, got, want)
		}
	}
	policy := NewPolicy()
	check := func(literal string, kept bool) {
		t.Helper()
		value := json.Number(literal)
		sanitized, changed := policy.sanitizeValue(value, 1)
		// Receipt decodes the record as a Runtime sends it.
		_, err := contracts.DecodePrivateStrict[reporting.ExecutionReport]([]byte(
			`{"reportId":"worker-numbers","complete":true,"metrics":{"tools":{}},"toolCalls":[` +
				`{"callId":"call-1","tool":"rank","arguments":{"value":` + literal + `},` +
				`"argumentsTruncated":false,"outcome":"succeeded"}],"errors":[],"truncated":false}`))
		if kept && (changed || sanitized != value || err != nil) {
			t.Fatalf("kept %s became %v (changed %t, validation %v)", literal, sanitized, changed, err)
		}
		if !kept && (!changed || sanitized != reporting.ArgumentNumberMarker || err == nil) {
			t.Fatalf("replaced %s became %v (changed %t, validation %v)", literal, sanitized, changed, err)
		}
	}
	for _, literal := range numbers.Kept {
		check(literal, true)
	}
	for _, literal := range numbers.Replaced {
		check(literal, false)
	}
}

// TestPolicyMarksNumbersTheServerBuiltOutsideTheirRange covers a Planner
// summary, whose numbers are Go values judged by the JSON encoding/json writes
// for them: the Server writes the marker the Runtime writes and the report
// stays valid instead of being lost.
func TestPolicyMarksNumbersTheServerBuiltOutsideTheirRange(t *testing.T) {
	report := reporting.ExecutionReport{
		ReportID: "planner-numbers", Complete: true,
		Metrics: reporting.ExecutionMetrics{Tools: map[string]reporting.ToolMetrics{}},
		ToolCalls: []reporting.ToolCallRecord{{
			CallID: "call-1", Tool: "rank", Outcome: reporting.ToolCallSucceeded,
			Arguments: map[string]any{
				"kept": []any{1e-7, 65535.0, int64(math.MaxInt64), uint64(math.MaxUint64), 9.999999999999998e19},
				"tiny": 1e-9, "huge": 1e20, "nan": math.NaN(), "infinite": math.Inf(-1),
				"content_bytes": json.Number("100000000000000000000"),
			},
		}},
		Errors: []reporting.ExecutionError{},
	}
	normalized, err := NewPolicy().NormalizeExecutionReport(report, MaxReportJSONBytes)
	if err != nil {
		t.Fatal(err)
	}
	call := normalized.ToolCalls[0]
	if !call.ArgumentsTruncated {
		t.Fatal("a replaced number did not mark the arguments truncated")
	}
	for _, key := range []string{"tiny", "huge", "nan", "infinite"} {
		if call.Arguments[key] != reporting.ArgumentNumberMarker {
			t.Fatalf("%s = %#v, want the marker", key, call.Arguments[key])
		}
	}
	if kept := fmt.Sprint(call.Arguments["kept"]); kept != fmt.Sprint(report.ToolCalls[0].Arguments["kept"]) {
		t.Fatalf("in-range numbers became %s", kept)
	}
	if placeholder, ok := call.Arguments["content_bytes"].(map[string]any); !ok || placeholder["redacted"] != true {
		t.Fatalf("an out-of-range derived size = %#v, want the redaction placeholder", call.Arguments["content_bytes"])
	}
}

// The Runtime checks that it treats every listed name as sensitive, so the
// Server never redacts a member the Runtime kept; the list must hold a sample
// of every rule isSensitiveKey applies (OBS-R27).
func TestSharedRedactionNamesAgreeWithTheRuntime(t *testing.T) {
	fixture := readReportTrimFixture(t)
	listed := map[string]bool{}
	for _, name := range fixture.Redaction.SensitiveNames {
		if !isSensitiveKey(name) {
			t.Fatalf("listed name %q is not sensitive", name)
		}
		listed[name] = true
	}
	var required []string
	for name := range sensitiveKeys {
		required = append(required, name)
	}
	for _, fragment := range sensitiveFragments {
		required = append(required, "x"+fragment+"x")
	}
	for _, suffix := range sensitiveSuffixes {
		required = append(required, "x"+suffix)
	}
	for _, name := range required {
		if !listed[name] {
			t.Fatalf("sensitive name rule %q has no sample in the shared list", name)
		}
	}
	safe := map[string]struct{}{}
	for _, name := range fixture.Redaction.SafeDerivedSizeNames {
		safe[name] = struct{}{}
	}
	if fmt.Sprint(safe) != fmt.Sprint(safeDerivedSizeKeys) {
		t.Fatalf("derived size names = %v, shared list %v", safeDerivedSizeKeys, safe)
	}
}

func TestSharedReportTrimKeepsTheSharedSuffixAtEveryBoundary(t *testing.T) {
	fixture := readReportTrimFixture(t)
	for _, test := range fixture.Cases {
		t.Run(test.Name, func(t *testing.T) {
			report, err := contracts.DecodePrivateStrict[reporting.ExecutionReport](test.Report)
			if err != nil {
				t.Fatalf("decode report: %v", err)
			}
			// Both sides measure the same bytes: an empty arguments object
			// counts as absent and a number as the literal the Runtime wrote.
			if size := jsonBytes(report); size != test.Bytes {
				t.Fatalf("report size = %d, Runtime measures %d", size, test.Bytes)
			}
			// Redacting the Runtime's records again changes none of them.
			normalized, err := NewPolicy().NormalizeExecutionReport(report, MaxReportJSONBytes)
			if err != nil {
				t.Fatal(err)
			}
			if len(normalized.ToolCalls) != len(report.ToolCalls) {
				t.Fatalf("normalization kept %d of %d tool calls", len(normalized.ToolCalls), len(report.ToolCalls))
			}
			for index, call := range report.ToolCalls {
				if got, want := mustJSON(t, normalized.ToolCalls[index]), mustJSON(t, call); !bytes.Equal(got, want) {
					t.Fatalf("normalization changed tool call %d:\n%s\nwant %s", index, got, want)
				}
			}
			if !bytes.Equal(mustJSON(t, normalized), mustJSON(t, report)) {
				t.Fatalf("normalization changed the report:\n%s\nwant %s", mustJSON(t, normalized), mustJSON(t, report))
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
				if size := jsonBytes(normalized); size != limit.Bytes || size > limit.MaximumBytes {
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
		return jsonBytes(reporting.AllocationFinalReport{
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
		{name: "execution report", report: multibyteReport(24, 10), measure: func(report reporting.ExecutionReport) int { return jsonBytes(report) }},
		{name: "already truncated", report: func() reporting.ExecutionReport {
			report := multibyteReport(9, 3)
			report.Truncated = true
			return report
		}(), measure: func(report reporting.ExecutionReport) int { return jsonBytes(report) }},
		{name: "errors only", report: multibyteReport(0, 10), measure: func(report reporting.ExecutionReport) int { return jsonBytes(report) }},
		// RuntimeReport renders through json.Marshal, which escapes <, > and
		// &; the envelope keeps that size and the records keep theirs.
		{name: "allocation envelope", report: multibyteReport(24, 10), measure: allocationMeasure("stopped <early> & " + strings.Repeat("界", 40))},
		{name: "unencodable record", report: unencodable, measure: func(report reporting.ExecutionReport) int { return jsonBytes(report) }},
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
				got, fits, err := NewPolicy().trimDetail(test.report, maximum, test.measure)
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

// jsonBytes is the policy's JSON measure of value.
func jsonBytes(value any) int {
	return NewPolicy().encodedSize(value)
}

// countingPolicy is NewPolicy(secrets...) with a measure that counts what it
// measures; the counts belong to the returned policy alone.
func countingPolicy(secrets ...string) (Policy, *measurementCounts) {
	counts := &measurementCounts{}
	policy := NewPolicy(secrets...)
	measure := policy.size
	policy.size = func(value any) (int, error) {
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
		return measure(value)
	}
	return policy, counts
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
	executionSize := func(report reporting.ExecutionReport) int { return jsonBytes(report) }
	t.Run("Planner report at the 1 MiB report bound", func(t *testing.T) {
		source := maximumDetailReport()
		policy, counts := countingPolicy()
		normalized, err := policy.NormalizeExecutionReport(source, MaxReportJSONBytes)
		if err != nil {
			t.Fatal(err)
		}
		// The candidate, the envelope without detail and the retained result;
		// each record once; each argument summary once.
		want := measurementCounts{reports: 3, records: MaxToolRecords + MaxErrorRecords, other: MaxToolRecords}
		if *counts != want {
			t.Fatalf("measurements = %+v, want %+v", *counts, want)
		}
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
		policy, counts := countingPolicy()
		normalized, err := policy.NormalizeAllocationReport(allocation)
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
		assertFirstFit(t, source, normalized.Worker, MaxReportJSONBytes, allocationMeasure(stopReason))
	})
}

func TestReportTrimMeasuresAFittingReportOnce(t *testing.T) {
	report := multibyteReport(3, 2)
	policy, counts := countingPolicy()
	normalized, err := policy.NormalizeExecutionReport(report, MaxReportJSONBytes)
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
	// A secret shorter than LIM-OBS-3 is redacted only where it is a complete
	// value (OBS-R31), and only a value shorter than the marker grows, so every
	// argument value is the secret: 75 members of 12 bytes each grow by 7.
	const secret = "k3y"
	arguments := func(value string) map[string]any {
		result := make(map[string]any, 75)
		for index := range 75 {
			result[fmt.Sprintf("p%02d", index)] = value
		}
		return result
	}
	finished := time.Date(2026, 10, 23, 12, 0, 0, 0, time.UTC)
	calls := int64(1000)
	worker := reporting.ExecutionReport{
		ReportID: "worker-redaction", Complete: true,
		Metrics:   reporting.ExecutionMetrics{Tools: map[string]reporting.ToolMetrics{"read_source_file": {Calls: &calls}}},
		ToolCalls: make([]reporting.ToolCallRecord, 0, calls),
		Errors:    []reporting.ExecutionError{{Code: "model_call_failed", Message: secret}},
	}
	for index := range calls {
		worker.ToolCalls = append(worker.ToolCalls, reporting.ToolCallRecord{
			CallID: fmt.Sprintf("tool-%08d", index+1), Tool: "read_source_file",
			Arguments: arguments(secret), Outcome: reporting.ToolCallSucceeded,
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

	policy, counts := countingPolicy(secret)
	normalized, err := policy.NormalizeAllocationReport(response.Report)
	if err != nil {
		t.Fatal(err)
	}
	if counts.reports != 3 || counts.allocations != 1 {
		t.Fatalf("measurements = %+v", *counts)
	}
	encoded := mustJSON(t, normalized)
	if bytes.Contains(encoded, []byte(secret)) {
		t.Fatal("normalized report contains the secret")
	}
	redacted := response.Report.Worker
	redacted.ToolCalls = make([]reporting.ToolCallRecord, 0, len(worker.ToolCalls))
	for _, call := range response.Report.Worker.ToolCalls {
		call.Arguments = arguments(redactedValue)
		call.ArgumentsTruncated = true
		redacted.ToolCalls = append(redacted.ToolCalls, call)
	}
	redacted.Errors = []reporting.ExecutionError{{Code: "model_call_failed", Message: redactedValue}}
	assertFirstFit(t, redacted, normalized.Worker, MaxReportJSONBytes-allocationReserveSize,
		func(report reporting.ExecutionReport) int { return jsonBytes(report) })
}

func TestReportNormalizationRejectsAnAggregateAboveItsBound(t *testing.T) {
	report := multibyteReport(4, 2)
	envelope := report
	envelope.ToolCalls, envelope.Errors, envelope.Truncated = []reporting.ToolCallRecord{}, []reporting.ExecutionError{}, true
	limit := jsonBytes(envelope)
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

// A measure that disagrees with the per-record arithmetic is an accounting
// failure: trimDetail reports it instead of returning an unverified report.
func TestReportTrimRejectsInconsistentSizeAccounting(t *testing.T) {
	report := multibyteReport(6, 2)
	full := jsonBytes(report)
	inflated := func(candidate reporting.ExecutionReport) int {
		size := jsonBytes(candidate)
		if len(candidate.ToolCalls) > 0 && len(candidate.ToolCalls) < len(report.ToolCalls) {
			size += 7
		}
		return size
	}
	got, fits, err := NewPolicy().trimDetail(report, full-1, inflated)
	if err == nil || fits || got.ReportID != "" || !strings.Contains(err.Error(), "not the computed") {
		t.Fatalf("inconsistent accounting = (%q, %t, %v)", got.ReportID, fits, err)
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
