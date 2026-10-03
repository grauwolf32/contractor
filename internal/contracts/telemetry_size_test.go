package contracts

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

type telemetrySizeCase struct {
	Name         string `json:"name"`
	Unit         string `json:"unit"`
	Repeat       int    `json:"repeat"`
	CompactBytes int    `json:"compact_bytes"`
	Valid        bool   `json:"valid"`
}

func telemetrySizeCases(t *testing.T) []telemetrySizeCase {
	t.Helper()
	raw, err := os.ReadFile("../../api/testdata/v1alpha1/telemetry-json-size-cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []telemetrySizeCase
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	return cases
}

func telemetrySizeReport(arguments map[string]any, count int) ExecutionReport {
	report := ExecutionReport{
		ReportID: "worker-size", Complete: true,
		Metrics:   ExecutionMetrics{Tools: map[string]ToolMetrics{}},
		ToolCalls: make([]ToolCallRecord, 0, count), Errors: []ExecutionError{},
	}
	for index := range count {
		report.ToolCalls = append(report.ToolCalls, ToolCallRecord{
			CallID: fmt.Sprintf("call-%d", index), Tool: "read_source_file",
			Arguments: arguments, Outcome: ToolCallSucceeded,
		})
	}
	return report
}

func TestTelemetryArgumentSizeMatchesRuntimeCases(t *testing.T) {
	for _, test := range telemetrySizeCases(t) {
		t.Run(test.Name, func(t *testing.T) {
			arguments := map[string]any{"path": strings.Repeat(test.Unit, test.Repeat)}
			size, err := ResultJSONSize(arguments)
			if err != nil || size != test.CompactBytes {
				t.Fatalf("compact argument size = %d, %v; want %d", size, err, test.CompactBytes)
			}
			report := telemetrySizeReport(arguments, 1)
			if err := report.Validate(); (err == nil) != test.Valid {
				t.Fatalf("valid=%t: %v", test.Valid, err)
			}
		})
	}
}

func TestTelemetryAggregatesUseCompactUTF8Size(t *testing.T) {
	test := telemetrySizeCases(t)[0]
	arguments := map[string]any{"path": strings.Repeat(test.Unit, test.Repeat)}
	worker := telemetrySizeReport(arguments, 150)
	compact, err := ResultJSONSize(worker)
	if err != nil || compact > 1024*1024 {
		t.Fatalf("compact worker report size = %d, %v", compact, err)
	}
	escaped, err := json.Marshal(worker)
	if err != nil || len(escaped) <= 1024*1024 {
		t.Fatalf("HTML-escaped worker report size = %d, %v", len(escaped), err)
	}
	if err := worker.Validate(); err != nil {
		t.Fatalf("valid Runtime worker report rejected: %v", err)
	}
	now := time.Now().UTC()
	final := AllocationFinalReport{
		ReportID: "allocation-size", AllocationID: "allocation-size",
		StartedAt: now.Add(-time.Second), FinishedAt: now,
		Worker: worker, Runtime: RuntimeReport{Complete: true},
	}
	compact, err = ResultJSONSize(final)
	if err != nil || compact > 1024*1024 {
		t.Fatalf("compact allocation report size = %d, %v", compact, err)
	}
	escaped, err = json.Marshal(final)
	if err != nil || len(escaped) <= 1024*1024 {
		t.Fatalf("HTML-escaped allocation report size = %d, %v", len(escaped), err)
	}
	if err := final.Validate(); err != nil {
		t.Fatalf("valid Runtime allocation report rejected: %v", err)
	}
	var wire bytes.Buffer
	encoder := json.NewEncoder(&wire)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(final); err != nil {
		t.Fatal(err)
	}
	if wire.Len() > 1024*1024 {
		t.Fatalf("unescaped allocation report wire size = %d", wire.Len())
	}
	var decoded AllocationFinalReport
	if err := json.Unmarshal(wire.Bytes(), &decoded); err != nil {
		t.Fatalf("valid Runtime report wire rejected: %v", err)
	}
	if err := decoded.Validate(); err != nil {
		t.Fatalf("decoded Runtime report rejected: %v", err)
	}
	oversized := telemetrySizeReport(arguments, 400)
	if size, err := ResultJSONSize(oversized); err != nil || size <= 1024*1024 {
		t.Fatalf("oversized compact worker report size = %d, %v", size, err)
	}
	if err := oversized.Validate(); err == nil {
		t.Fatal("oversized compact worker report was accepted")
	}
	final.Worker = oversized
	if err := final.Validate(); err == nil {
		t.Fatal("oversized compact allocation report was accepted")
	}
}
