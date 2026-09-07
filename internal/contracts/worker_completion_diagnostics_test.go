package contracts

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSharedWorkerCompletionDiagnostics(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "..", "api", "testdata", "audit-completion", "diagnostics.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Name  string
		Value json.RawMessage
		Valid bool
		Known bool
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	for _, item := range cases {
		t.Run(item.Name, func(t *testing.T) {
			liveRaw := `{"counters":{},"toolCalls":[],"errors":[],"finalOutcome":null,"truncated":false,"completion":` + string(item.Value) + `}`
			var live WorkerAllocationState
			liveErr := json.Unmarshal([]byte(liveRaw), &live)
			if liveErr == nil {
				liveErr = live.validate()
			}
			if (liveErr == nil) != item.Valid {
				t.Fatalf("live valid=%v want=%v: %v", liveErr == nil, item.Valid, liveErr)
			}
			raw := `{"reportId":"report-1","complete":true,"metrics":{"tools":{}},"toolCalls":[],"errors":[],"truncated":false,"completion":` + string(item.Value) + `}`
			var report ExecutionReport
			err := json.Unmarshal([]byte(raw), &report)
			if err == nil {
				err = report.Validate()
			}
			if (err == nil) != item.Valid {
				t.Fatalf("valid=%v want=%v: %v", err == nil, item.Valid, err)
			}
			if !item.Valid {
				return
			}
			if (report.Completion != nil) != item.Known || (live.Completion != nil) != item.Known {
				t.Fatal("unknown diagnostics were interpreted")
			}
			encoded, err := json.Marshal(report)
			if err != nil || strings.Contains(string(encoded), "private-evidence-marker") {
				t.Fatal("unsafe report roundtrip")
			}
			encoded, err = json.Marshal(live)
			if err != nil || strings.Contains(string(encoded), "private-evidence-marker") {
				t.Fatal("unsafe live State roundtrip")
			}
		})
	}
}

func TestCompletionOmissionAndStrictDuplicateFields(t *testing.T) {
	var report ExecutionReport
	if err := json.Unmarshal([]byte(`{"reportId":"old","complete":true,"metrics":{"tools":{}},"toolCalls":[],"errors":[],"truncated":false}`), &report); err != nil {
		t.Fatal(err)
	}
	if report.Completion != nil {
		t.Fatal("legacy report acquired completion")
	}
	encoded, _ := json.Marshal(report)
	if strings.Contains(string(encoded), "completion") {
		t.Fatal("legacy omission changed")
	}
	if _, err := decodeWorkerCompletionDiagnostics([]byte(`{"kind":"audit-check-results@1","phase":"published","phase":"failed"}`)); err == nil {
		t.Fatal("accepted duplicate phase")
	}
}
